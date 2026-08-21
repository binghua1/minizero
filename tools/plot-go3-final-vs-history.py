#!/usr/bin/env python3
"""Plot Go3 final-checkpoint scores against same-method history."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def percentile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def result(path, final_name, expected_games, samples, seed):
    latest = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                latest[int(record["game_id"])] = record
    records = [record for record in latest.values() if not record.get("error")]
    errors = sum(bool(record.get("error")) for record in latest.values())
    if len(records) != expected_games or errors:
        raise ValueError(
            f"{path}: expected {expected_games} valid games and no errors; "
            f"found {len(records)} valid and {errors} errors"
        )

    strata = defaultdict(list)
    wins = draws = losses = 0
    for record in records:
        returns = [float(value) for value in record["returns"]]
        top = max(returns)
        winners = {
            record["seating"][index]
            for index, value in enumerate(returns)
            if value == top
        }
        if winners == {final_name}:
            score = 1.0
            wins += 1
        elif final_name in winners:
            score = 0.5
            draws += 1
        else:
            score = 0.0
            losses += 1
        strata[tuple(record["seating"])].append(score)

    score = (wins + 0.5 * draws) / len(records)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        total = sum(sum(rng.choice(values) for _ in values) for values in strata.values())
        bootstrap.append(total / len(records))
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "ci95_low": percentile(bootstrap, 0.025),
        "ci95_high": percentile(bootstrap, 0.975),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--final-iteration", type=int, required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Plot historical checkpoint score against the final checkpoint",
    )
    args = parser.parse_args()

    methods = (
        ("maz", "Multiplayer AlphaZero", "#1f77b4", "^"),
        ("jpsro", "JPSRO + Rank", "#d62728", "s"),
    )
    rows = []
    for method_index, (key, label, _color, _marker) in enumerate(methods):
        final_name = f"{key}_final_i{args.final_iteration}"
        for iteration in args.iterations:
            path = (
                args.root / key
                / f"i{iteration}_vs_i{args.final_iteration}_n50_noise_{args.games}"
                / "games.jsonl"
            )
            stats = result(
                path, final_name, args.games, args.bootstrap_samples,
                args.seed + method_index * 100003 + iteration,
            )
            rows.append({
                "method": label,
                "history_iteration": iteration,
                "final_iteration": args.final_iteration,
                "games": args.games,
                **stats,
            })

    stem = f"i{args.final_iteration}_vs_history"
    csv_path = args.root / f"{stem}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for key, label, color, marker in methods:
        selected = [row for row in rows if row["method"] == label]
        x = [row["history_iteration"] for row in selected]
        y = [100 * row["score"] for row in selected]
        low = [100 * row["ci95_low"] for row in selected]
        high = [100 * row["ci95_high"] for row in selected]
        if args.reverse:
            y = [100 - value for value in y]
            low, high = (
                [100 - value for value in high],
                [100 - value for value in low],
            )
        axis.plot(x, y, color=color, marker=marker, linewidth=2, label=label)
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    axis.axhline(50, color="0.35", linestyle="--", linewidth=1.5, label="50%")
    axis.set_xlabel("Historical checkpoint")
    axis.set_ylabel(
        f"Historical checkpoint score vs i{args.final_iteration} (%)"
        if args.reverse else f"i{args.final_iteration} model-level score (%)"
    )
    axis.set_title(
        (
            f"Go3 historical checkpoints vs same-method i{args.final_iteration}\n"
            if args.reverse else f"Go3 i{args.final_iteration} vs same-method history\n"
        )
        + f"50 simulations, {args.games} games/pair"
    )
    axis.set_xticks(args.iterations)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    image_stem = (
        f"historical_checkpoints_vs_i{args.final_iteration}"
        if args.reverse else stem
    )
    figure.savefig(args.root / f"{image_stem}.png", dpi=200)
    figure.savefig(args.root / f"{image_stem}.pdf")
    plt.close(figure)
    print(f"Curve: {args.root / f'{image_stem}.png'}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
