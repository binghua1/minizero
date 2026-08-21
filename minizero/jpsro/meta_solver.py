"""No-regret CCE solver for small empirical normal-form games.

The implementation intentionally uses only the Python standard library.  It
accumulates the product strategy played by simultaneous Hedge learners.  The
time-average is an approximate coarse-correlated equilibrium (CCE).
"""

import itertools
import math

try:
    import numpy as np
except ImportError:  # Keep the solver usable in minimal Python installations.
    np = None


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
    # For a top-mass prefix, each deviation gain is a cumulative weighted sum.
    # Updating those sums once per added profile changes this from O(N^2) to
    # O(N * players * actions), which matters for 4,096+ profile games.
    gains = [[0.0] * len(actions) for actions in policy_sets]
    total = 0.0
    for size, (profile, probability) in enumerate(ordered, 1):
        total += probability
        for player, actions in enumerate(policy_sets):
            baseline = payoffs[profile][player]
            for deviation in range(len(actions)):
                deviated = list(profile)
                deviated[player] = deviation
                gains[player][deviation] += probability * (
                    payoffs[tuple(deviated)][player] - baseline
                )
        best_gain = 0.0
        witness = None
        for player, actions in enumerate(policy_sets):
            for deviation, numerator in enumerate(gains[player]):
                gain = numerator / total
                if gain > best_gain:
                    best_gain = gain
                    witness = {"player": player, "policy": actions[deviation]}
        if best_gain <= tolerance:
            selected = ordered[:size]
            candidate = {
                profile: probability / total for profile, probability in selected
            }
            return candidate, max(0.0, best_gain), witness
    gap, witness = cce_gap(distribution, policy_sets, payoffs)
    return distribution, gap, witness


def solve_cce(policy_sets, payoffs, iterations=5000, tolerance=0.01,
              eta=None, utility_min=-1.0, utility_max=1.0,
              check_interval=100, progress=None):
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

    # The restricted game is naturally a dense tensor.  Vectorizing these
    # contractions avoids tens of millions of Python-level operations for an
    # 8^4 JPSRO population, while retaining the old implementation as a
    # fallback when NumPy is unavailable.
    payoff_tensor = None
    average_tensor = None
    if np is not None:
        shape = tuple(len(items) for items in policy_sets)
        payoff_tensor = np.empty(shape + (len(policy_sets),), dtype=np.float64)
        for profile in all_profiles:
            payoff_tensor[profile] = payoffs[profile]
        average_tensor = np.zeros(shape, dtype=np.float64)

    for step in range(1, iterations + 1):
        mixed = [_softmax(values) for values in log_weights]
        action_values = [[0.0] * len(items) for items in policy_sets]
        if payoff_tensor is not None:
            mixed_arrays = [np.asarray(values) for values in mixed]
            joint_tensor = mixed_arrays[0]
            for values in mixed_arrays[1:]:
                joint_tensor = np.multiply.outer(joint_tensor, values)
            average_tensor += joint_tensor
            for player, actions in enumerate(policy_sets):
                weighted = payoff_tensor[..., player]
                for opponent, probabilities in enumerate(mixed_arrays):
                    if opponent != player:
                        reshape = [1] * len(policy_sets)
                        reshape[opponent] = len(probabilities)
                        weighted = weighted * probabilities.reshape(reshape)
                axes = tuple(axis for axis in range(len(policy_sets)) if axis != player)
                action_values[player] = np.sum(weighted, axis=axes).tolist()
        else:
            for profile in all_profiles:
                probability = math.prod(mixed[p][action] for p, action in enumerate(profile))
                average[profile] += probability
            for player, actions in enumerate(policy_sets):
                for profile in all_profiles:
                    opponents_probability = math.prod(
                        mixed[p][action] for p, action in enumerate(profile) if p != player
                    )
                    action_values[player][profile[player]] += (
                        opponents_probability * payoffs[profile][player]
                    )

        for player, actions in enumerate(policy_sets):
            for action in range(len(actions)):
                normalized = (action_values[player][action] - utility_min) / scale
                log_weights[player][action] += eta * normalized

        completed = step
        if step >= check_interval and (step % check_interval == 0 or step == iterations):
            if average_tensor is not None:
                distribution = {
                    profile: float(average_tensor[profile] / step)
                    for profile in all_profiles
                }
            else:
                distribution = {profile: mass / step for profile, mass in average.items()}
            answer_gap, answer_witness = cce_gap(distribution, policy_sets, payoffs)
            if progress is not None:
                progress(step, iterations, answer_gap)
            if answer_gap <= tolerance:
                break

    if average_tensor is not None:
        distribution = {
            profile: float(average_tensor[profile] / completed)
            for profile in all_profiles
        }
    else:
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
