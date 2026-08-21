#!/usr/bin/env python3
"""Overlay final-vs-history curves for Blokus10 MAZ and JPSRO."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--jpsro-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-iteration", type=int, default=500)
    parser.add_argument("--games", type=int, default=700)
    parser.add_argument(
        "--reverse", action="store_true",
        help="Plot each historical checkpoint's complementary score vs final",
    )
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(11, 6))
    series = (
        ("Multiplayer AlphaZero", load(args.baseline_csv), "#1f77b4", "^"),
        ("JPSRO + Rank", load(args.jpsro_csv), "#d62728", "s"),
    )
    for label, rows, color, marker in series:
        x = np.asarray([int(row["historical_iteration"]) for row in rows])
        y = 100 * np.asarray([float(row["final_model_score"]) for row in rows])
        low = 100 * np.asarray([float(row["ci95_low"]) for row in rows])
        high = 100 * np.asarray([float(row["ci95_high"]) for row in rows])
        if args.reverse:
            y = 100 - y
            low, high = 100 - high, 100 - low
        axis.plot(x, y, marker=marker, color=color, linewidth=2.2,
                  markersize=6, label=label)
        axis.fill_between(x, low, high, color=color, alpha=0.15)

    axis.axhline(50, color="0.35", linestyle="--", linewidth=1.4, label="50%")
    ylabel = (
        f"Historical checkpoint score vs i{args.final_iteration} (%)"
        if args.reverse else f"i{args.final_iteration} model-level score (%)"
    )
    direction = "Historical checkpoints vs final model" if args.reverse else "Final model vs history"
    axis.set(
        xlabel="Historical checkpoint iteration",
        ylabel=ylabel,
        ylim=(0, 100),
        title=f"Blokus10: {direction}\n"
              f"50 simulations, {args.games} games/pair",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"Comparison figure: {args.output}")


if __name__ == "__main__":
    main()
