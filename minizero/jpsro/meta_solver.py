"""No-regret CCE solver for small empirical normal-form games.

The implementation intentionally uses only the Python standard library.  It
accumulates the product strategy played by simultaneous Hedge learners.  The
time-average is an approximate coarse-correlated equilibrium (CCE).
"""

import itertools
import math


def _profiles(policy_sets):
    return itertools.product(*(range(len(items)) for items in policy_sets))


def _softmax(log_weights):
    offset = max(log_weights)
    weights = [math.exp(value - offset) for value in log_weights]
    total = sum(weights)
    return [value / total for value in weights]


def cce_gap(distribution, policy_sets, payoffs):
    """Return the maximum unconditional pure-deviation gain and its witness."""
    best_gain = 0.0
    witness = None
    for player, actions in enumerate(policy_sets):
        for deviation in range(len(actions)):
            gain = 0.0
            for profile, probability in distribution.items():
                deviated = list(profile)
                deviated[player] = deviation
                gain += probability * (
                    payoffs[tuple(deviated)][player] - payoffs[profile][player]
                )
            if gain > best_gain:
                best_gain = gain
                witness = {"player": player, "policy": actions[deviation]}
    return max(0.0, best_gain), witness


def compact_cce_support(distribution, policy_sets, payoffs, tolerance):
    """Return the smallest top-mass prefix that still passes the CCE check."""
    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    for size in range(1, len(ordered) + 1):
        selected = ordered[:size]
        total = sum(probability for _, probability in selected)
        candidate = {profile: probability / total for profile, probability in selected}
        gap, witness = cce_gap(candidate, policy_sets, payoffs)
        if gap <= tolerance:
            return candidate, gap, witness
    gap, witness = cce_gap(distribution, policy_sets, payoffs)
    return distribution, gap, witness


def solve_cce(policy_sets, payoffs, iterations=5000, tolerance=0.01,
              eta=None, utility_min=-1.0, utility_max=1.0,
              check_interval=100):
    """Solve a complete restricted game with simultaneous full-info Hedge.

    Args:
        policy_sets: policy IDs available to each player.
        payoffs: mapping from index profile tuples to payoff vectors.
        iterations: maximum number of no-regret updates.
        tolerance: stop once the measured empirical CCE gap is below this.
        eta: Hedge learning rate; a horizon-dependent default is used if None.
        utility_min/utility_max: known payoff range used for normalization.

    Returns a JSON-serializable dictionary containing the joint distribution,
    empirical gap, deviation witness, iterations, and a conservative external
    regret bound in original utility units.
    """
    if len(policy_sets) < 2 or any(not items for items in policy_sets):
        raise ValueError("CCE requires at least two players with non-empty policy sets")
    if iterations < 1 or utility_max <= utility_min:
        raise ValueError("invalid solver iterations or utility range")
    all_profiles = list(_profiles(policy_sets))
    missing = [profile for profile in all_profiles if profile not in payoffs]
    if missing:
        raise ValueError(f"payoff table is incomplete: {len(missing)} profiles missing")
    for profile in all_profiles:
        if len(payoffs[profile]) != len(policy_sets):
            raise ValueError(f"payoff vector length mismatch for {profile}")

    max_actions = max(len(items) for items in policy_sets)
    if eta is None:
        eta = math.sqrt(8.0 * math.log(max(2, max_actions)) / iterations)
    if eta <= 0:
        raise ValueError("eta must be positive")

    scale = utility_max - utility_min
    log_weights = [[0.0] * len(items) for items in policy_sets]
    average = {profile: 0.0 for profile in all_profiles}
    answer_gap = math.inf
    answer_witness = None
    completed = 0

    for step in range(1, iterations + 1):
        mixed = [_softmax(values) for values in log_weights]
        joint = {}
        for profile in all_profiles:
            probability = math.prod(mixed[p][action] for p, action in enumerate(profile))
            joint[profile] = probability
            average[profile] += probability

        action_values = [[0.0] * len(items) for items in policy_sets]
        for player, actions in enumerate(policy_sets):
            for profile in all_profiles:
                opponents_probability = math.prod(
                    mixed[p][action] for p, action in enumerate(profile) if p != player
                )
                action_values[player][profile[player]] += (
                    opponents_probability * payoffs[profile][player]
                )
            for action in range(len(actions)):
                normalized = (action_values[player][action] - utility_min) / scale
                log_weights[player][action] += eta * normalized

        completed = step
        if step >= check_interval and (step % check_interval == 0 or step == iterations):
            distribution = {profile: mass / step for profile, mass in average.items()}
            answer_gap, answer_witness = cce_gap(distribution, policy_sets, payoffs)
            if answer_gap <= tolerance:
                break

    distribution = {profile: mass / completed for profile, mass in average.items()}
    distribution = {profile: mass for profile, mass in distribution.items() if mass >= 1e-12}
    total = sum(distribution.values())
    distribution = {profile: mass / total for profile, mass in distribution.items()}
    raw_support_size = len(distribution)
    distribution, answer_gap, answer_witness = compact_cce_support(
        distribution, policy_sets, payoffs, tolerance)
    regret_bound = scale * (
        math.log(max(2, max_actions)) / (eta * completed) + eta / 8.0
    )
    return {
        "distribution": distribution,
        "gap": answer_gap,
        "witness": answer_witness,
        "iterations": completed,
        "eta": eta,
        "external_regret_bound": regret_bound,
        "raw_support_size": raw_support_size,
        "support_size": len(distribution),
    }
