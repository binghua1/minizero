#!/usr/bin/env python3
"""Plot Go3 MAZ and JPSRO learning curves against a fixed MAZ anchor."""

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


def checkpoint_result(path, candidate_prefix, samples, seed):
    latest = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                latest[int(record["game_id"])] = record
    records = [record for record in latest.values() if not record.get("error")]
    errors = sum(bool(record.get("error")) for record in latest.values())
    if errors or len(records) != 300:
        raise ValueError(
            f"{path}: expected 300 valid latest games and 0 errors; "
            f"found {len(records)} valid and {errors} errors"
        )

    strata = defaultdict(list)
    for record in records:
        candidate = next(
            (name for name in record["seating"] if name.startswith(candidate_prefix)),
            None,
        )
        if candidate is None:
            raise ValueError(f"{path}: candidate prefix {candidate_prefix!r} not found")
        top = max(float(value) for value in record["returns"])
        winners = {
            record["seating"][index]
            for index, value in enumerate(record["returns"])
            if float(value) == top
        }
        score = 1.0 if winners == {candidate} else 0.5 if candidate in winners else 0.0
        strata[tuple(record["seating"])].append(score)

    score = sum(sum(values) for values in strata.values()) / len(records)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        total = sum(sum(rng.choice(values) for _ in values) for values in strata.values())
        bootstrap.append(total / len(records))
    return score, percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--anchor-iteration", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    rows = []
    methods = [
        ("maz", "multiplayer_alphazero_i", "Multiplayer AlphaZero"),
        ("jpsro", "jpsro_rank_l075_i", "JPSRO + Rank"),
    ]
    for method_index, (directory, prefix, label) in enumerate(methods):
        for iteration in args.iterations:
            path = args.root / directory / f"i{iteration}_n50_noise_300" / "games.jsonl"
            score, low, high = checkpoint_result(
                path, prefix, args.bootstrap_samples,
                args.seed + method_index * 100003 + iteration,
            )
            rows.append({
                "method": label,
                "iteration": iteration,
                "score": score,
                "ci95_low": low,
                "ci95_high": high,
            })

    stem = f"fixed_maz{args.anchor_iteration}_anchor_curve"
    csv_path = args.root / f"{stem}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    styles = {
        "Multiplayer AlphaZero": ("#1f77b4", "^"),
        "JPSRO + Rank": ("#d62728", "s"),
    }
    for label in styles:
        selected = [row for row in rows if row["method"] == label]
        x = [row["iteration"] for row in selected]
        y = [100 * row["score"] for row in selected]
        low = [100 * row["ci95_low"] for row in selected]
        high = [100 * row["ci95_high"] for row in selected]
        color, marker = styles[label]
        axis.plot(x, y, color=color, marker=marker, linewidth=2, label=label)
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    axis.axhline(50, color="0.35", linestyle="--", linewidth=1.5, label="50%")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel(
        f"Model-level score vs MAZ i{args.anchor_iteration} anchor (%)"
    )
    axis.set_title(
        f"Go3 vs fixed Multiplayer AlphaZero i{args.anchor_iteration} anchor\n"
        "50 simulations, 300 games/checkpoint")
    axis.set_xticks(args.iterations)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.root / f"{stem}.png", dpi=200)
    figure.savefig(args.root / f"{stem}.pdf")
    plt.close(figure)
    print(f"Curve: {args.root / f'{stem}.png'}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
