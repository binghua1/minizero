#!/usr/bin/env python3
"""Plot one Blokus10 final checkpoint against regularly sampled history."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--final-iteration", type=int, required=True)
    parser.add_argument("--history-start", type=int, default=0)
    parser.add_argument("--history-end", type=int, required=True)
    parser.add_argument("--history-step", type=int, default=20)
    parser.add_argument("--games", type=int, default=700)
    parser.add_argument("--label", default="Multiplayer AlphaZero")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Plot historical checkpoint score against the final checkpoint",
    )
    args = parser.parse_args()

    final_steps = args.final_iteration * 500
    rng = np.random.default_rng(20260819 + args.final_iteration)
    rows = []
    for iteration in range(args.history_start, args.history_end + 1, args.history_step):
        step = iteration * 500
        path = args.root / f"{step}_vs_{final_steps}" / "games.jsonl"
        latest = {}
        for line in path.open():
            if line.strip():
                record = json.loads(line)
                latest[int(record["game_id"])] = record
        records = [record for record in latest.values() if not record.get("error")]
        errors = sum(bool(record.get("error")) for record in latest.values())
        if len(records) != args.games or errors:
            raise ValueError(f"{path}: {len(records)} valid, {errors} errors")

        strata = defaultdict(list)
        final_name = f"iter_{final_steps}"
        for record in records:
            best = max(float(value) for value in record["returns"])
            top = {
                agent for agent, value in zip(record["seating"], record["returns"])
                if float(value) == best
            }
            score = 1.0 if top == {final_name} else 0.5 if final_name in top else 0.0
            strata[tuple(record["seating"])].append(score)
        score = sum(map(sum, strata.values())) / len(records)
        bootstrap = []
        for _ in range(args.bootstrap_samples):
            total = sum(
                rng.choice(values, size=len(values), replace=True).sum()
                for values in strata.values()
            )
            bootstrap.append(total / len(records))
        low, high = np.quantile(bootstrap, (0.025, 0.975))
        rows.append({
            "historical_iteration": iteration,
            "historical_training_steps": step,
            "games": len(records),
            "final_model_score": score,
            "ci95_low": low,
            "ci95_high": high,
        })

    csv_path = args.root / f"i{args.final_iteration}_vs_history.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([row["historical_iteration"] for row in rows])
    y = 100 * np.asarray([row["final_model_score"] for row in rows])
    low = 100 * np.asarray([row["ci95_low"] for row in rows])
    high = 100 * np.asarray([row["ci95_high"] for row in rows])
    if args.reverse:
        y = 100 - y
        low, high = 100 - high, 100 - low
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(x, y, "o-", color="#1f77b4", linewidth=2, markersize=5,
              label=f"{args.label} i{args.final_iteration}")
    axis.fill_between(x, low, high, color="#1f77b4", alpha=0.16,
                      label="95% bootstrap CI")
    axis.axhline(50, color="0.3", linestyle="--", linewidth=1.3, label="50%")
    ylabel = (
        f"Historical checkpoint score vs i{args.final_iteration} (%)"
        if args.reverse else f"i{args.final_iteration} model-level score (%)"
    )
    direction = (
        f"historical checkpoints vs {args.label} i{args.final_iteration}"
        if args.reverse else f"{args.label} i{args.final_iteration} vs history"
    )
    axis.set(
        xlabel="Historical checkpoint iteration",
        ylabel=ylabel,
        xlim=(args.history_start - 5, args.history_end + 5),
        ylim=(0, 100),
        title=f"Blokus10 {direction}\n"
              f"50 simulations, {args.games} games/pair",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    image_stem = (
        f"historical_checkpoints_vs_i{args.final_iteration}"
        if args.reverse else f"i{args.final_iteration}_vs_history"
    )
    png_path = args.root / f"{image_stem}.png"
    figure.savefig(png_path, dpi=200)
    figure.savefig(args.root / f"{image_stem}.pdf")
    plt.close(figure)
    print(f"Figure: {png_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
