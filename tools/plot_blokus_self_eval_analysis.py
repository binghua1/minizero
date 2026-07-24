#!/usr/bin/env python3
"""Plot checkpoint-pair strength and return metrics for multiplayer self-eval."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    output = args.output or evaluation_dir / "self_eval_analysis.png"
    elo_rows = read_csv(evaluation_dir / "elo.csv")

    rows = []
    for elo_row in elo_rows:
        newer = int(elo_row["P1"])
        older = int(elo_row["P2"])
        summaries = read_csv(evaluation_dir / f"{newer}_vs_{older}" / "agent_summary.csv")
        by_agent = {row["agent"]: row for row in summaries}
        newer_summary = by_agent[f"iter_{newer}"]
        older_summary = by_agent[f"iter_{older}"]
        total = int(elo_row["Total"])
        score = float(elo_row["WinRate"])
        # A conservative binomial approximation; draws are scored as one half.
        score_se = math.sqrt(max(score * (1.0 - score), 0.0) / total)
        rows.append({
            "newer": newer,
            "older": older,
            "score": score,
            "score_ci": 1.96 * score_se,
            "elo": float(elo_row["P1 Elo"]),
            "newer_return": float(newer_summary["avg_return"]),
            "older_return": float(older_summary["avg_return"]),
            "newer_raw_win_rate": float(newer_summary["win_rate"]),
            "older_raw_win_rate": float(older_summary["win_rate"]),
        })

    x = [row["newer"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].errorbar(
        x,
        [100.0 * row["score"] for row in rows],
        yerr=[100.0 * row["score_ci"] for row in rows],
        marker="o",
        capsize=3,
        label="newer checkpoint score",
    )
    axes[0].axhline(50.0, color="black", linewidth=1, linestyle="--", label="equal strength")
    axes[0].set_ylabel("Pair score (%)")
    axes[0].set_title("Newer checkpoint vs immediately previous checkpoint")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [row["newer_return"] for row in rows], marker="o", label="newer avg return")
    axes[1].plot(x, [row["older_return"] for row in rows], marker="o", alpha=0.65, label="older avg return")
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Normalized Blokus return")
    axes[1].set_title("Average return within each mixed-checkpoint arena")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    elo_x = [rows[0]["older"]] + x
    elo_y = [0.0] + [row["elo"] for row in rows]
    axes[2].plot(elo_x, elo_y, marker="o", color="tab:purple", label="sequential Elo")
    axes[2].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[2].set_xlabel("Training step / checkpoint iteration")
    axes[2].set_ylabel("Chained Elo")
    axes[2].set_title("Sequential Elo (pairwise differences accumulated)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle("Blokus MaxN self-evaluation analysis", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)

    summary_path = output.with_suffix(".csv")
    fields = list(rows[0])
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(output)
    print(summary_path)


if __name__ == "__main__":
    main()
