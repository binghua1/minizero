#!/usr/bin/env python3
"""Create paper-oriented two-model summaries without changing arena outputs.

The arena's agent_summary.csv counts seat appearances.  That is useful for
diagnostics, but its win rate is not a game-level model win rate when a model
occupies multiple seats.  This script reads games.jsonl directly and writes a
separate report in which every physical game contributes one unit of
first-place credit, split equally among tied first-place seats.
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


PAIR_RE = re.compile(r"^(\d+)_vs_(\d+)$")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


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


def bootstrap_ci(records, value_key, samples, seed):
    """Stratified game bootstrap, conditional on every exact seat assignment."""
    strata = defaultdict(list)
    for record in records:
        strata[record["seat_signature"]].append(record[value_key])
    rng = random.Random(seed)
    estimates = []
    total = len(records)
    for _ in range(samples):
        sampled_sum = 0.0
        for values in strata.values():
            sampled_sum += sum(rng.choice(values) for _ in values)
        estimates.append(sampled_sum / total)
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def ranks(values):
    """One-based descending midranks; tied values receive their mean rank."""
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        midrank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[order[position]] = midrank
        start = end
    return result


def load_pair(pair_dir):
    match = PAIR_RE.match(pair_dir.name)
    if not match:
        raise ValueError(f"pair directory must be NEWER_vs_OLDER: {pair_dir}")
    target_iteration, reference_iteration = map(int, match.groups())
    arena_path = pair_dir / "arena.json"
    arena = json.loads(arena_path.read_text()) if arena_path.exists() else {}
    self_eval = arena.get("self_eval", {})
    target_name = f"iter_{self_eval.get('newer_iteration', target_iteration)}"
    reference_name = f"iter_{self_eval.get('older_iteration', reference_iteration)}"
    scheduled_games = arena.get("num_games")

    valid = []
    errors = 0
    with (pair_dir / "games.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("error"):
                errors += 1
                continue
            seating = raw["seating"]
            returns = [float(value) for value in raw["returns"]]
            if len(seating) != len(returns):
                raise ValueError(f"seat/return length mismatch in {pair_dir}, line {line_number}")
            unknown = set(seating) - {target_name, reference_name}
            if unknown:
                raise ValueError(f"unexpected agents {sorted(unknown)} in {pair_dir}, line {line_number}")

            top = max(returns)
            top_seats = [index for index, value in enumerate(returns) if value == top]
            credit_per_top_seat = 1.0 / len(top_seats)
            target_credit = sum(credit_per_top_seat for seat in top_seats if seating[seat] == target_name)
            target_indices = [index for index, agent in enumerate(seating) if agent == target_name]
            reference_indices = [index for index, agent in enumerate(seating) if agent == reference_name]
            if not target_indices or not reference_indices:
                raise ValueError(f"non-mixed game in {pair_dir}, line {line_number}")
            seat_ranks = ranks(returns)
            target_mean_return = mean([returns[index] for index in target_indices])
            reference_mean_return = mean([returns[index] for index in reference_indices])
            target_mean_rank = mean([seat_ranks[index] for index in target_indices])
            reference_mean_rank = mean([seat_ranks[index] for index in reference_indices])
            target_seat_fraction = len(target_indices) / len(seating)
            target_present_at_top = any(seating[index] == target_name for index in top_seats)
            reference_present_at_top = any(seating[index] == reference_name for index in top_seats)
            model_score = 0.5 if target_present_at_top and reference_present_at_top else float(target_present_at_top)

            valid.append({
                "raw": raw,
                "seating": seating,
                "returns": returns,
                "ranks": seat_ranks,
                "top_seats": top_seats,
                "target_seats": len(target_indices),
                "seat_signature": tuple(agent == target_name for agent in seating),
                "target_credit": target_credit,
                "reference_credit": 1.0 - target_credit,
                # A diagnostic centered at .5 even for an incomplete/unbalanced schedule.
                "seat_count_adjusted_score": 0.5 + target_credit - target_seat_fraction,
                "model_score": model_score,
                "target_mean_return": target_mean_return,
                "reference_mean_return": reference_mean_return,
                "return_advantage": target_mean_return - reference_mean_return,
                "target_mean_rank": target_mean_rank,
                "reference_mean_rank": reference_mean_rank,
                "rank_advantage": reference_mean_rank - target_mean_rank,
            })
    return {
        "pair": pair_dir.name,
        "target_iteration": target_iteration,
        "reference_iteration": reference_iteration,
        "target": target_name,
        "reference": reference_name,
        "scheduled_games": scheduled_games,
        "errors": errors,
        "records": valid,
    }


def rounded(value):
    return "" if math.isnan(value) else f"{value:.8f}"


def summarize_pair(pair, bootstrap_samples, seed):
    records = pair["records"]
    if not records:
        raise ValueError(f"no valid games in {pair['pair']}")
    game_count = len(records)
    seat_count = len(records[0]["seating"])
    target_exposures = sum(record["target_seats"] for record in records)
    total_exposures = game_count * seat_count
    target_credits = sum(record["target_credit"] for record in records)
    model_wins = sum(record["model_score"] == 1.0 for record in records)
    model_draws = sum(record["model_score"] == 0.5 for record in records)
    model_losses = sum(record["model_score"] == 0.0 for record in records)

    metrics = {}
    for offset, key in enumerate(("target_credit", "seat_count_adjusted_score", "model_score", "return_advantage", "rank_advantage")):
        estimate = mean([record[key] for record in records])
        low, high = bootstrap_ci(records, key, bootstrap_samples, seed + offset)
        metrics[key] = (estimate, low, high)

    scheduled = pair["scheduled_games"]
    # A paper-ready pair must contain the requested number of valid games.
    # Merely having an error record for every scheduled game is not complete.
    complete = pair["errors"] == 0 and (scheduled is None or game_count >= scheduled)
    return {
        "pair": pair["pair"],
        "target": pair["target"],
        "reference": pair["reference"],
        "scheduled_games": "" if scheduled is None else scheduled,
        "valid_games": game_count,
        "errors": pair["errors"],
        "complete": str(complete).lower(),
        "target_seat_exposures": target_exposures,
        "reference_seat_exposures": total_exposures - target_exposures,
        "target_seat_fraction": rounded(target_exposures / total_exposures),
        "target_first_place_credit": rounded(target_credits),
        "reference_first_place_credit": rounded(game_count - target_credits),
        "target_first_place_share": rounded(metrics["target_credit"][0]),
        "target_first_place_ci_low": rounded(metrics["target_credit"][1]),
        "target_first_place_ci_high": rounded(metrics["target_credit"][2]),
        "target_adjusted_score": rounded(metrics["seat_count_adjusted_score"][0]),
        "target_adjusted_ci_low": rounded(metrics["seat_count_adjusted_score"][1]),
        "target_adjusted_ci_high": rounded(metrics["seat_count_adjusted_score"][2]),
        "target_model_wins": model_wins,
        "shared_model_top_draws": model_draws,
        "target_model_losses": model_losses,
        "target_model_score": rounded(metrics["model_score"][0]),
        "target_model_score_ci_low": rounded(metrics["model_score"][1]),
        "target_model_score_ci_high": rounded(metrics["model_score"][2]),
        "target_game_mean_return": rounded(mean([record["target_mean_return"] for record in records])),
        "reference_game_mean_return": rounded(mean([record["reference_mean_return"] for record in records])),
        "return_advantage": rounded(metrics["return_advantage"][0]),
        "return_advantage_ci_low": rounded(metrics["return_advantage"][1]),
        "return_advantage_ci_high": rounded(metrics["return_advantage"][2]),
        "target_game_mean_rank": rounded(mean([record["target_mean_rank"] for record in records])),
        "reference_game_mean_rank": rounded(mean([record["reference_mean_rank"] for record in records])),
        "rank_advantage": rounded(metrics["rank_advantage"][0]),
        "rank_advantage_ci_low": rounded(metrics["rank_advantage"][1]),
        "rank_advantage_ci_high": rounded(metrics["rank_advantage"][2]),
    }


def composition_rows(pair):
    rows = []
    groups = defaultdict(list)
    for record in pair["records"]:
        groups[record["target_seats"]].append(record)
    num_seats = len(pair["records"][0]["seating"])
    for target_seats, records in sorted(groups.items()):
        wins = sum(record["model_score"] == 1.0 for record in records)
        draws = sum(record["model_score"] == 0.5 for record in records)
        losses = sum(record["model_score"] == 0.0 for record in records)
        share = mean([record["target_credit"] for record in records])
        null_share = target_seats / num_seats
        rows.append({
            "pair": pair["pair"],
            "target": pair["target"],
            "reference": pair["reference"],
            "target_seats": target_seats,
            "reference_seats": num_seats - target_seats,
            "games": len(records),
            "target_first_place_share": rounded(share),
            "equal_model_null_share": rounded(null_share),
            "excess_first_place_share": rounded(share - null_share),
            "target_model_wins": wins,
            "shared_model_top_draws": draws,
            "target_model_losses": losses,
            "target_model_score": rounded((wins + 0.5 * draws) / len(records)),
            "target_game_mean_return": rounded(mean([record["target_mean_return"] for record in records])),
            "reference_game_mean_return": rounded(mean([record["reference_mean_return"] for record in records])),
            "return_advantage": rounded(mean([record["return_advantage"] for record in records])),
        })
    return rows


def seat_rows(pair):
    stats = defaultdict(lambda: {"games": 0, "credit": 0.0, "return": 0.0, "rank": 0.0})
    for record in pair["records"]:
        for seat, (agent, value, rank) in enumerate(zip(record["seating"], record["returns"], record["ranks"])):
            key = (agent, seat)
            stats[key]["games"] += 1
            stats[key]["return"] += value
            stats[key]["rank"] += rank
            if seat in record["top_seats"]:
                stats[key]["credit"] += 1.0 / len(record["top_seats"])
    rows = []
    for (agent, seat), values in sorted(stats.items()):
        games = values["games"]
        rows.append({
            "pair": pair["pair"],
            "agent": agent,
            "seat_index": seat,
            "seat": pair["records"][0]["raw"]["players"][seat].upper(),
            "games": games,
            "first_place_credit": rounded(values["credit"]),
            "seat_top1_rate": rounded(values["credit"] / games),
            "mean_return": rounded(values["return"] / games),
            "mean_rank": rounded(values["rank"] / games),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path, source, bootstrap_samples):
    text = f"""# Paper-oriented multiplayer summary

Source: `{source}`

This directory is derived only from the existing `games.jsonl` files. No arena
result was modified.

## Recommended reporting

Use `target_first_place_share` in `pair_summary.csv` as the primary game-level
outcome for complete, seat-balanced pairs. Every physical game contributes
exactly one unit of first-place credit. A unique winner receives 1; tied
first-place seats split that unit equally. Therefore, if the lineup is ABBB and
a B seat wins, model B receives one game-level credit, not one win and two
losses. The equal-strength null is 0.5 only after the complementary mixed
lineups and seat assignments are balanced.

Report `return_advantage` as a co-primary score-margin outcome. Within each
game it is mean(target-controlled seat return) minus
mean(reference-controlled seat return), so a physical game remains the unit of
analysis even when one model controls several seats.

The 95% confidence intervals use a deterministic {bootstrap_samples}-sample
bootstrap stratified by exact seat assignment. This preserves lineup
composition and seat balance and does not treat seats from one game as
independent samples.

`target_model_score` is conventional W + 0.5D at model level: target-only top
seats are a win, reference-only top seats a loss, and a top shared by both
models a draw. `target_adjusted_score` is a diagnostic centered at 0.5 after
subtracting the target's seat fraction in every game; prefer it when inspecting
an incomplete schedule, but do not mix incomplete and complete pairs in the
main paper table.

`composition_summary.csv` reports results separately by the number of target
seats. `seat_diagnostics.csv` is intended for detecting seat bias; its
`seat_top1_rate` is not a model-level match win rate.
"""
    with path.open("x") as stream:
        stream.write(text)


def make_plot(output_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; CSV reports were still created", file=sys.stderr)
        return
    complete = [row for row in rows if row["complete"] == "true"]
    if not complete:
        return
    x = list(range(len(complete)))
    labels = [row["pair"] for row in complete]
    shares = [float(row["target_first_place_share"]) for row in complete]
    share_low = [float(row["target_first_place_ci_low"]) for row in complete]
    share_high = [float(row["target_first_place_ci_high"]) for row in complete]
    advantages = [float(row["return_advantage"]) for row in complete]
    advantage_low = [float(row["return_advantage_ci_low"]) for row in complete]
    advantage_high = [float(row["return_advantage_ci_high"]) for row in complete]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(x) * 0.7), 8), sharex=True)
    axes[0].errorbar(x, shares,
                     yerr=[[value - low for value, low in zip(shares, share_low)],
                           [high - value for value, high in zip(shares, share_high)]],
                     marker="o", capsize=3, linestyle="-")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Newer-model first-place share")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, advantages,
                     yerr=[[value - low for value, low in zip(advantages, advantage_low)],
                           [high - value for value, high in zip(advantages, advantage_high)]],
                     marker="o", capsize=3, linestyle="-")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Newer minus older return")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Seat-balanced Blokus checkpoint comparisons (95% stratified bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_comparisons.png", dpi=200)
    fig.savefig(output_dir / "checkpoint_comparisons.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path,
                        help="directory containing NEWER_vs_OLDER subdirectories")
    parser.add_argument("--output", type=Path,
                        help="new output directory (default: EVALUATION_DIR/paper_summary_v1)")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    source = args.evaluation_dir.resolve()
    output = args.output.resolve() if args.output else source / "paper_summary_v1"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    pair_dirs = sorted(
        (path for path in source.iterdir() if path.is_dir() and PAIR_RE.match(path.name)
         and (path / "games.jsonl").exists()),
        key=lambda path: tuple(map(int, PAIR_RE.match(path.name).groups())),
    )
    if not pair_dirs:
        raise FileNotFoundError(f"no NEWER_vs_OLDER/games.jsonl pairs under {source}")

    # mkdir is intentionally delayed until all source pairs parse successfully.
    pairs = [load_pair(path) for path in pair_dirs]
    summaries = [summarize_pair(pair, args.bootstrap_samples, args.seed + index * 100)
                 for index, pair in enumerate(pairs)]
    compositions = [row for pair in pairs for row in composition_rows(pair)]
    seats = [row for pair in pairs for row in seat_rows(pair)]

    output.mkdir(parents=True)
    write_csv(output / "pair_summary.csv", summaries)
    write_csv(output / "composition_summary.csv", compositions)
    write_csv(output / "seat_diagnostics.csv", seats)
    write_readme(output / "README.md", source, args.bootstrap_samples)
    make_plot(output, summaries)
    print(f"created {output}")
    print(f"summarized {len(pairs)} checkpoint pairs and {sum(len(pair['records']) for pair in pairs)} valid games")


if __name__ == "__main__":
    main()
