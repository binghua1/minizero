#!/usr/bin/env python3
"""Summarize a complete checkpoint cross-play matrix and detect 3-cycles."""

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def percentile(sorted_values, probability):
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_distribution(records, samples, seed):
    """Bootstrap games within each exact model-to-seat assignment."""
    strata = defaultdict(list)
    for record in records:
        strata[record["seat_signature"]].append(record["score"])
    rng = random.Random(seed)
    total = len(records)
    estimates = []
    for _ in range(samples):
        score_sum = 0.0
        for values in strata.values():
            score_sum += sum(rng.choice(values) for _ in values)
        estimates.append(score_sum / total)
    return estimates


def read_latest_records(path):
    latest = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if "game_id" not in record:
                raise ValueError(f"missing game_id in {path}:{line_number}")
            latest[int(record["game_id"])] = record
    return [latest[game_id] for game_id in sorted(latest)]


def load_pair(root, first, second, samples, seed, familywise_tail):
    pair_dir = root / f"{first}_vs_{second}"
    arena_path = pair_dir / "arena.json"
    games_path = pair_dir / "games.jsonl"
    if not arena_path.is_file() or not games_path.is_file():
        raise FileNotFoundError(f"missing arena.json or games.jsonl in {pair_dir}")

    arena = json.loads(arena_path.read_text())
    expected_first = f"iter_{first}"
    expected_second = f"iter_{second}"
    model_fight = arena.get("model_fight", {})
    model_a = model_fight.get("model_a", expected_first)
    model_b = model_fight.get("model_b", expected_second)
    if (model_a, model_b) != (expected_first, expected_second):
        raise ValueError(
            f"unexpected agents in {arena_path}: {(model_a, model_b)}, "
            f"expected {(expected_first, expected_second)}"
        )

    valid = []
    errors = 0
    for record in read_latest_records(games_path):
        if record.get("error"):
            errors += 1
            continue
        seating = record.get("seating", [])
        returns = record.get("returns", [])
        if not seating or len(seating) != len(returns):
            raise ValueError(f"invalid seating/returns in {games_path}, game {record['game_id']}")
        unknown = set(seating) - {model_a, model_b}
        if unknown:
            raise ValueError(f"unexpected agents {sorted(unknown)} in {games_path}")
        if model_a not in seating or model_b not in seating:
            raise ValueError(f"non-mixed seating in {games_path}, game {record['game_id']}")

        top_return = max(float(value) for value in returns)
        top_models = {
            seating[index]
            for index, value in enumerate(returns)
            if float(value) == top_return
        }
        if top_models == {model_a}:
            score = 1.0
        elif top_models == {model_b}:
            score = 0.0
        elif top_models == {model_a, model_b}:
            score = 0.5
        else:
            raise ValueError(f"invalid top-model set in {games_path}: {sorted(top_models)}")
        valid.append({
            "score": score,
            "seat_signature": tuple(agent == model_a for agent in seating),
        })

    if not valid:
        raise ValueError(f"no valid games in {pair_dir}")
    distribution = bootstrap_distribution(valid, samples, seed)
    sorted_distribution = sorted(distribution)
    score = sum(record["score"] for record in valid) / len(valid)
    scheduled = int(arena.get("num_games", len(valid) + errors))
    return {
        "pair": pair_dir.name,
        "first": first,
        "second": second,
        "scheduled_games": scheduled,
        "valid_games": len(valid),
        "errors": errors,
        "complete": errors == 0 and len(valid) >= scheduled,
        "first_wins": sum(record["score"] == 1.0 for record in valid),
        "draws": sum(record["score"] == 0.5 for record in valid),
        "first_losses": sum(record["score"] == 0.0 for record in valid),
        "score": score,
        "ci_low": percentile(sorted_distribution, 0.025),
        "ci_high": percentile(sorted_distribution, 0.975),
        "simultaneous_ci_low": percentile(sorted_distribution, familywise_tail),
        "simultaneous_ci_high": percentile(sorted_distribution, 1.0 - familywise_tail),
        "bootstrap": distribution,
    }


def directed_value(pair, source, key):
    if source == pair["first"]:
        return pair[key]
    if key.endswith("_low"):
        opposite_key = key[:-4] + "_high"
        return 1.0 - pair[opposite_key]
    if key.endswith("_high"):
        opposite_key = key[:-5] + "_low"
        return 1.0 - pair[opposite_key]
    if key == "bootstrap":
        return [1.0 - value for value in pair[key]]
    return 1.0 - pair[key]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.8f}"


def make_pair_rows(pairs):
    rows = []
    for pair in pairs:
        rows.append({
            "pair": pair["pair"],
            "row_checkpoint": pair["first"],
            "column_checkpoint": pair["second"],
            "scheduled_games": pair["scheduled_games"],
            "valid_games": pair["valid_games"],
            "errors": pair["errors"],
            "complete": str(pair["complete"]).lower(),
            "row_wins": pair["first_wins"],
            "shared_top_draws": pair["draws"],
            "row_losses": pair["first_losses"],
            "row_model_score": fmt(pair["score"]),
            "row_score_ci95_low": fmt(pair["ci_low"]),
            "row_score_ci95_high": fmt(pair["ci_high"]),
            "row_score_simultaneous95_low": fmt(pair["simultaneous_ci_low"]),
            "row_score_simultaneous95_high": fmt(pair["simultaneous_ci_high"]),
        })
    return rows


def build_lookup(pairs):
    lookup = {}
    for pair in pairs:
        lookup[frozenset((pair["first"], pair["second"]))] = pair
    return lookup


def value(lookup, source, target, key="score"):
    if source == target:
        return 0.5
    pair = lookup[frozenset((source, target))]
    return directed_value(pair, source, key)


def make_matrix_rows(checkpoints, lookup):
    rows = []
    for source in checkpoints:
        row = {"row_beats_column": source}
        for target in checkpoints:
            row[str(target)] = fmt(value(lookup, source, target))
        rows.append(row)
    return rows


def cycle_rows(checkpoints, lookup, samples):
    rows = []
    for triple in itertools.combinations(checkpoints, 3):
        for order in ((triple[0], triple[1], triple[2]),
                      (triple[0], triple[2], triple[1])):
            edges = ((order[0], order[1]), (order[1], order[2]), (order[2], order[0]))
            edge_scores = [value(lookup, source, target) for source, target in edges]
            ci_lows = [value(lookup, source, target, "ci_low") for source, target in edges]
            simultaneous_lows = [
                value(lookup, source, target, "simultaneous_ci_low")
                for source, target in edges
            ]
            complete = all(lookup[frozenset(edge)]["complete"] for edge in edges)
            distributions = [value(lookup, source, target, "bootstrap") for source, target in edges]
            joint_probability = sum(
                all(distribution[index] > 0.5 for distribution in distributions)
                for index in range(samples)
            ) / samples
            observed = complete and all(score > 0.5 for score in edge_scores)
            supported_95 = observed and all(low > 0.5 for low in ci_lows)
            familywise_supported_95 = observed and all(low > 0.5 for low in simultaneous_lows)
            rows.append({
                "cycle": f"{order[0]} > {order[1]} > {order[2]} > {order[0]}",
                "edge_1": f"{edges[0][0]}>{edges[0][1]}",
                "edge_1_score": fmt(edge_scores[0]),
                "edge_2": f"{edges[1][0]}>{edges[1][1]}",
                "edge_2_score": fmt(edge_scores[1]),
                "edge_3": f"{edges[2][0]}>{edges[2][1]}",
                "edge_3_score": fmt(edge_scores[2]),
                "minimum_edge_score": fmt(min(edge_scores)),
                "minimum_edge_ci95_low": fmt(min(ci_lows)),
                "minimum_edge_simultaneous95_low": fmt(min(simultaneous_lows)),
                "joint_bootstrap_probability": fmt(joint_probability),
                "complete": str(complete).lower(),
                "observed_cycle": str(observed).lower(),
                "all_edges_ci95_above_50": str(supported_95).lower(),
                "familywise95_supported": str(familywise_supported_95).lower(),
            })
    rows.sort(
        key=lambda row: (
            row["familywise95_supported"] == "true",
            row["observed_cycle"] == "true",
            float(row["minimum_edge_score"]),
            float(row["joint_bootstrap_probability"]),
        ),
        reverse=True,
    )
    return rows


def make_heatmap(root, checkpoints, lookup, game_name, method_name, games_per_pair):
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("warning: matplotlib unavailable; crossplay_heatmap.png was not generated", file=sys.stderr)
        return

    matrix = [[value(lookup, row, column) for column in checkpoints] for row in checkpoints]
    size = max(7, 1.15 * len(checkpoints))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))
    axis.set_xticks(range(len(checkpoints)), [str(item) for item in checkpoints], rotation=45, ha="right")
    axis.set_yticks(range(len(checkpoints)), [str(item) for item in checkpoints])
    axis.set_xlabel("Column checkpoint")
    axis.set_ylabel("Row checkpoint")
    axis.set_title(f"{game_name} — {method_name} ({games_per_pair} games/pair)")
    for row_index, row in enumerate(checkpoints):
        for column_index, column in enumerate(checkpoints):
            score = matrix[row_index][column_index]
            if row == column:
                axis.text(column_index, row_index, "—", ha="center", va="center",
                          color="black", fontsize=12)
                continue
            ci_low = value(lookup, row, column, "ci_low")
            ci_high = value(lookup, row, column, "ci_high")
            color = "white" if score < 0.25 or score > 0.75 else "black"
            ci_margin = 50.0 * (ci_high - ci_low)
            label = f"{100 * score:.1f}% ± {ci_margin:.1f}%"
            axis.text(column_index, row_index, label,
                      ha="center", va="center", color=color, fontsize=7.5)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Model-level W + 0.5D")
    figure.tight_layout()
    figure.savefig(root / "crossplay_heatmap.png", dpi=200)
    figure.savefig(root / "crossplay_heatmap.pdf")
    plt.close(figure)


def write_report(root, checkpoints, pairs, cycles, samples):
    observed = [row for row in cycles if row["observed_cycle"] == "true"]
    pairwise_supported = [row for row in cycles if row["all_edges_ci95_above_50"] == "true"]
    familywise_supported = [row for row in cycles if row["familywise95_supported"] == "true"]
    incomplete = [pair["pair"] for pair in pairs if not pair["complete"]]
    best = cycles[0] if cycles else None
    lines = [
        "Checkpoint strategy-cycle analysis",
        "==================================",
        f"Checkpoints: {', '.join(map(str, checkpoints))}",
        f"Complete pair evaluations: {len(pairs) - len(incomplete)}/{len(pairs)}",
        f"Bootstrap samples: {samples}",
        "Primary score: model-level W + 0.5D; each physical game is one observation.",
        "Bootstrap: stratified by exact model-to-seat assignment.",
        "",
        f"Observed directed 3-cycles: {len(observed)}",
        f"Cycles with every ordinary pairwise 95% CI above 50%: {len(pairwise_supported)}",
        f"Cycles supported by simultaneous 95% intervals: {len(familywise_supported)}",
    ]
    if incomplete:
        lines.extend(("", "INCOMPLETE PAIRS (do not claim a cycle using these):", *incomplete))
    if familywise_supported:
        lines.extend(("", "Strongest simultaneous-95%-supported cycle:", familywise_supported[0]["cycle"]))
    elif observed:
        lines.extend((
            "",
            "A cycle is visible, but it is not supported by the simultaneous 95% criterion.",
            f"Strongest observed cycle: {observed[0]['cycle']}",
            "Treat it as exploratory and confirm its three edges with an independent seed.",
        ))
    elif best:
        lines.extend((
            "",
            "No directed 3-cycle was observed among these checkpoints.",
            f"Closest candidate orientation: {best['cycle']}",
            f"Weakest edge score: {100 * float(best['minimum_edge_score']):.2f}%",
        ))
    (root / "cycle_report.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--game-name", default="Game")
    parser.add_argument("--method-name", default="Checkpoint cross-play")
    parser.add_argument("--games-per-pair", type=int)
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if len(checkpoints) < 3:
        parser.error("at least three distinct checkpoints are required")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    root = args.evaluation_dir.resolve()
    pair_count = len(checkpoints) * (len(checkpoints) - 1) // 2
    # Two-sided Bonferroni intervals cover every pair at least 95% jointly.
    familywise_tail = 0.05 / (2 * pair_count)
    pairs = []
    for index, (first, second) in enumerate(itertools.combinations(checkpoints, 2)):
        pairs.append(load_pair(
            root, first, second, args.bootstrap_samples,
            args.seed + 1009 * index, familywise_tail,
        ))
    lookup = build_lookup(pairs)
    pair_rows = make_pair_rows(pairs)
    matrix_rows = make_matrix_rows(checkpoints, lookup)
    cycles = cycle_rows(checkpoints, lookup, args.bootstrap_samples)

    write_csv(root / "pair_summary.csv", list(pair_rows[0]), pair_rows)
    write_csv(root / "crossplay_matrix.csv", list(matrix_rows[0]), matrix_rows)
    write_csv(root / "cycle_candidates.csv", list(cycles[0]), cycles)
    write_report(root, checkpoints, pairs, cycles, args.bootstrap_samples)
    games_per_pair = args.games_per_pair
    if games_per_pair is None:
        games_per_pair = pairs[0]["scheduled_games"] if pairs else 0
    make_heatmap(root, checkpoints, lookup, args.game_name, args.method_name,
                 games_per_pair)
    print((root / "cycle_report.txt").read_text(), end="")
    print(f"Heatmap: {root / 'crossplay_heatmap.png'}")
    print(f"Cycle candidates: {root / 'cycle_candidates.csv'}")


if __name__ == "__main__":
    main()
