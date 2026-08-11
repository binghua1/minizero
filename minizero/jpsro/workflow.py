"""JPSRO orchestration helpers shared by the command-line tool and tests."""

import itertools
import json
import math
import sys
from pathlib import Path


PLAYER_CODES = ("b", "w", "r", "g", "y", "p")
DEFAULT_MAX_MOVES = {"blokus": 400, "blokus10": 160, "blokus15": 220}


def create_evaluation_manifest(state, profiles, game, config_path,
                               executable, repo_root, games_per_profile=1,
                               simulations=None, search_type="maxn", noise=False,
                               seed=0, command_timeout=300):
    """Create a fixed-seat arena manifest for ordered empirical profiles."""
    config_path = str(Path(config_path).resolve())
    executable = str(Path(executable).resolve())
    repo_root = str(Path(repo_root).resolve())
    used = list(dict.fromkeys(policy for profile in profiles for policy in profile))
    agents = []
    for policy_id in used:
        overrides = {
            "nn_file_name": state.model_path(policy_id),
            "actor_multiplayer_search_type": search_type,
            "actor_use_dirichlet_noise": str(bool(noise)).lower(),
            "actor_use_gumbel": "false",
            "actor_use_gumbel_noise": "false",
            "actor_use_random_rotation_features": str(bool(noise)).lower(),
            "actor_select_action_by_count": "true",
            "actor_select_action_by_softmax_count": "false",
            "zero_disable_resign_ratio": "1",
            "program_auto_seed": "false",
            "program_seed": "{seed}",
        }
        if simulations is not None:
            overrides["actor_num_simulation"] = str(simulations)
        conf_str = ":".join(f"{key}={value}" for key, value in overrides.items())
        agents.append({
            "name": policy_id,
            "model_path": state.model_path(policy_id),
            "cwd": repo_root,
            "command": [executable, "-mode", "console", "-conf_file", config_path,
                        "-conf_str", conf_str],
        })
    return {
        "game": game,
        "players": list(PLAYER_CODES[:state.num_players]),
        "agents": agents,
        "lineups": [list(profile) for profile in profiles],
        "jpsro_profiles": [list(profile) for profile in profiles],
        "seat_mode": "fixed",
        "share_agent_engines": True,
        "games_per_seating": int(games_per_profile),
        "max_moves": DEFAULT_MAX_MOVES.get(game, 2048),
        "terminal_passes": state.num_players if game.startswith("blokus") else 1,
        "pass_mode": "elimination" if game.startswith("blokus") else "terminal",
        "command_timeout": float(command_timeout),
        "seed": int(seed),
        "batched_evaluation": {
            "executable": executable,
            "config_file": config_path,
            "cwd": repo_root,
            "num_simulations": simulations,
            "search_type": search_type,
            "noise": bool(noise),
        },
    }


def ingest_evaluation(state, manifest_path, games_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    profiles = manifest.get("jpsro_profiles", manifest.get("lineups", []))
    accepted = 0
    errors = 0
    with Path(games_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            game = json.loads(line)
            if game.get("error") or not game.get("returns"):
                errors += 1
                continue
            lineup_id = int(game["lineup_id"])
            if lineup_id < 0 or lineup_id >= len(profiles):
                raise ValueError(f"invalid lineup_id {lineup_id}")
            profile = tuple(profiles[lineup_id])
            # A fixed-seat JPSRO manifest preserves ordered profile semantics.
            if list(game.get("seating", [])) != list(profile):
                raise ValueError("JPSRO payoff ingestion requires seat_mode=fixed")
            sample_id = f"{manifest_path}:{int(game['game_id'])}"
            accepted += int(state.payoffs.add(profile, game["returns"], sample_id))
    state.save()
    return accepted, errors


def create_oracle_profiles(state, responders=None, response_id="CURRENT"):
    """Marginalize the CCE recommendation at one responder seat at a time."""
    meta = state.load_meta_strategy()
    if responders is None:
        responders = list(range(state.num_players))
    responders = [int(player) for player in responders]
    aggregate = {}
    for item in meta["distribution"]:
        probability = float(item["probability"])
        for responder in responders:
            profile = list(item["profile"])
            profile[responder] = response_id
            mask = tuple(player == responder for player in range(state.num_players))
            key = (tuple(profile), mask)
            aggregate[key] = aggregate.get(key, 0.0) + probability / len(responders)
    return [(weight, profile, mask) for (profile, mask), weight in aggregate.items()
            if weight >= 1e-12]


def write_oracle_plan(state, output_path, responders=None, response_id="CURRENT"):
    """Write the compact TSV consumed by ZeroServer's JPSRO profile loader."""
    rows = create_oracle_profiles(state, responders, response_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# weight\tprofile_id\tpolicy_ids\ttrainable_mask\tmodel_paths"]
    for index, (weight, profile, mask) in enumerate(rows):
        paths = ["CURRENT" if trainable else state.model_path(policy_id)
                 for policy_id, trainable in zip(profile, mask)]
        lines.append("\t".join([
            f"{weight:.17g}", f"oracle_{index}", ",".join(profile),
            ",".join("1" if value else "0" for value in mask), ",".join(paths),
        ]))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output_path)
    return len(rows)


def _guided_masks(num_players, minimum, maximum, balance_seats):
    """Return (mask, probability) with equal trainable-position mass by default."""
    minimum = max(1, int(minimum))
    maximum = min(num_players - 1, int(maximum))
    if minimum > maximum:
        raise ValueError("invalid guided current-seat range")
    rows = []
    for count in range(minimum, maximum + 1):
        combinations = list(itertools.combinations(range(num_players), count))
        count_mass = 1.0 / count if balance_seats else 1.0
        for seats in combinations:
            mask = tuple(player in seats for player in range(num_players))
            rows.append((mask, count_mass / len(combinations)))
    total = sum(weight for _, weight in rows)
    return [(mask, weight / total) for mask, weight in rows]


def _softmax_weighted(items, temperature):
    if temperature <= 0:
        raise ValueError("guided hard temperature must be positive")
    offset = max(score / temperature + math.log(max(1e-12, prior))
                 for _, score, prior in items)
    values = [math.exp(score / temperature + math.log(max(1e-12, prior)) - offset)
              for _, score, prior in items]
    total = sum(values)
    return [(item[0], value / total) for item, value in zip(items, values)]


def create_guided_profiles(state, candidate_id=None, hard_ratio=0.6,
                           cce_ratio=0.1, history_ratio=0.15,
                           temperature=0.2, confidence=1.0,
                           current_seat_min=1, current_seat_max=None,
                           balance_seats=True, response_id="CURRENT"):
    """Mix hard, CCE and history profiles for one continuing CURRENT model.

    The ratios are global training masses; all-current self-play is scheduled by
    ZeroServer separately.  Hardness uses a candidate's role-specific payoff
    against the current CCE support.  Missing estimates safely fall back to the
    CCE prior instead of inventing a return.
    """
    ratios = (float(hard_ratio), float(cce_ratio), float(history_ratio))
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("guided profile ratios must be non-negative with positive total")
    if candidate_id is not None and candidate_id not in state.policies:
        raise ValueError(f"unknown guided candidate: {candidate_id}")
    meta = state.load_meta_strategy()
    support = [(tuple(item["profile"]), float(item["probability"]))
               for item in meta["distribution"] if float(item["probability"]) > 0]
    if not support:
        raise ValueError("CCE meta strategy has empty support")
    maximum = state.num_players - 1 if current_seat_max is None else current_seat_max
    masks = _guided_masks(state.num_players, current_seat_min, maximum, balance_seats)

    aggregate = {}

    def add(source, frozen, mask, weight):
        if weight <= 0:
            return
        policies = tuple(response_id if trainable else policy
                         for policy, trainable in zip(frozen, mask))
        key = (policies, mask)
        row = aggregate.setdefault(key, {
            "profile": policies, "mask": mask, "components": {}, "weight": 0.0,
        })
        row["components"][source] = row["components"].get(source, 0.0) + weight
        row["weight"] += weight

    for frozen, probability in support:
        for mask, mask_probability in masks:
            add("cce", frozen, mask, ratios[1] * probability * mask_probability)

    hard_items = []
    for frozen, probability in support:
        for mask, mask_probability in masks:
            estimates = []
            if candidate_id is not None:
                for player, trainable in enumerate(mask):
                    if not trainable:
                        continue
                    deviated = list(frozen)
                    deviated[player] = candidate_id
                    entry = state.payoffs.get(tuple(deviated))
                    if not entry:
                        continue
                    error = state.payoffs.stderr(tuple(deviated))[player]
                    exploration = confidence * error if math.isfinite(error) else 0.0
                    estimates.append(-float(entry["mean"][player]) + exploration)
            hardness = sum(estimates) / len(estimates) if estimates else 0.0
            hard_items.append(((frozen, mask), hardness, probability * mask_probability))
    for (frozen, mask), probability in _softmax_weighted(hard_items, temperature):
        add("hard", frozen, mask, ratios[0] * probability)

    history = {profile for profile, _ in support}
    for policy_id, policy in state.policies.items():
        if all(player in policy.players for player in range(state.num_players)):
            history.add(tuple([policy_id] * state.num_players))
    history_rows = [(profile, mask, mask_probability)
                    for profile in sorted(history) for mask, mask_probability in masks]
    profile_mass = 1.0 / len(history)
    for profile, mask, mask_probability in history_rows:
        add("history", profile, mask, ratios[2] * profile_mass * mask_probability)

    rows = sorted(aggregate.values(), key=lambda row: (-row["weight"], row["profile"], row["mask"]))
    return rows


def write_guided_plan(state, output_path, **kwargs):
    rows = create_guided_profiles(state, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# weight\tprofile_id\tpolicy_ids\ttrainable_mask\tmodel_paths"]
    summary = []
    for index, row in enumerate(rows):
        profile_id = f"guided_{index:03d}"
        paths = ["CURRENT" if trainable else state.model_path(policy_id)
                 for policy_id, trainable in zip(row["profile"], row["mask"])]
        lines.append("\t".join([
            f"{row['weight']:.17g}", profile_id, ",".join(row["profile"]),
            ",".join("1" if value else "0" for value in row["mask"]),
            ",".join(paths),
        ]))
        summary.append({
            "profile_id": profile_id,
            "weight": row["weight"],
            "profile": list(row["profile"]),
            "trainable_mask": list(row["mask"]),
            "components": row["components"],
        })
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output_path)
    atomic_json(output_path.with_suffix(output_path.suffix + ".json"), {
        "candidate": kwargs.get("candidate_id"),
        "profiles": summary,
        "total_weight": sum(row["weight"] for row in rows),
    })
    return len(rows)


def meta_profile_probabilities(meta):
    return {tuple(item["profile"]): float(item["probability"])
            for item in meta["distribution"]}


def required_deviation_profiles(state, candidate_id):
    """Profiles needed to measure every player's gain from a new policy."""
    meta = state.load_meta_strategy()
    result = set()
    for item in meta["distribution"]:
        for player in range(state.num_players):
            profile = list(item["profile"])
            profile[player] = candidate_id
            result.add(tuple(profile))
    return sorted(result)


def candidate_deviation_gains(state, candidate_id):
    """Estimate unilateral gain of a frozen oracle against the previous CCE."""
    meta = state.load_meta_strategy()
    result = []
    for player in range(state.num_players):
        gain = 0.0
        variance = 0.0
        games = 0
        for item in meta["distribution"]:
            probability = float(item["probability"])
            baseline = tuple(item["profile"])
            deviated = list(baseline)
            deviated[player] = candidate_id
            deviated = tuple(deviated)
            baseline_entry = state.payoffs.get(baseline)
            deviation_entry = state.payoffs.get(deviated)
            if not baseline_entry or not deviation_entry:
                raise ValueError(f"missing candidate payoff for player {player}: {deviated}")
            gain += probability * (
                deviation_entry["mean"][player] - baseline_entry["mean"][player]
            )
            base_error = state.payoffs.stderr(baseline)[player]
            dev_error = state.payoffs.stderr(deviated)[player]
            variance += probability * probability * (base_error * base_error + dev_error * dev_error)
            games += deviation_entry["count"]
        result.append({
            "player": player,
            "gain": gain,
            "standard_error": variance ** 0.5,
            "candidate_games": games,
        })
    return result


def admit_candidate(state, candidate_id, min_gain=0.02, confidence=2.0):
    """Keep a candidate only in player pools with positive lower-bound gain."""
    gains = candidate_deviation_gains(state, candidate_id)
    eligible = [
        item["player"] for item in gains
        if item["gain"] - confidence * item["standard_error"] > min_gain
    ]
    result = {
        "candidate": candidate_id,
        "accepted": bool(eligible),
        "eligible_players": eligible,
        "confidence_multiplier": confidence,
        "minimum_gain": min_gain,
        "per_player": gains,
    }
    if eligible:
        state.set_policy_players(candidate_id, eligible, {"admission": result})
    else:
        state.remove_policy(candidate_id)
    return result


def prune_population(state, maximum, protected=()):
    """Apply a soft cap while never deleting a policy in the CCE support."""
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("population maximum must be positive")
    meta = state.load_meta_strategy()
    support = {policy_id for item in meta["distribution"] for policy_id in item["profile"]}
    protected = set(protected)
    unknown = protected.difference(state.policies)
    if unknown:
        raise ValueError(f"cannot protect unknown policies: {sorted(unknown)}")
    keep = set(support).union(protected)
    candidates = sorted(
        (policy for policy in state.policies.values() if policy.policy_id not in keep),
        key=lambda policy: (policy.generation, policy.policy_id), reverse=True,
    )
    for policy in candidates[:max(0, maximum - len(keep))]:
        keep.add(policy.policy_id)
    removed = []
    for policy_id in list(state.policies):
        if policy_id not in keep:
            state.remove_policy(policy_id)
            removed.append(policy_id)
    return removed


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
