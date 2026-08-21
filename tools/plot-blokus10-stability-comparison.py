#!/usr/bin/env python3
"""Combine matched cross-play matrices and quantify checkpoint regression."""

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/blokus10_common_stability")
protocol = os.environ.get("STABILITY_PROTOCOL", "MaxN, 50 simulations, 700 games/pair")
methods = (
    ("vanilla", "Multiplayer AlphaZero", os.environ.get("VANILLA_STABILITY_DIR")),
    ("pool_only", "Opponent Pool Only", os.environ.get("POOL_ONLY_STABILITY_DIR")),
    ("pool_rank", "Opponent Pool + Rank", os.environ.get("POOL_RANK_STABILITY_DIR")),
    ("strong_jpsro", "Strong JPSRO", os.environ.get("JPSRO_STABILITY_DIR")),
)
all_data = []
summary = []

for key, title, source_override in methods:
    source_dir = Path(source_override) if source_override else root / key
    path = source_dir / "pair_summary.csv"
    if not path.exists():
        continue
    rows = list(csv.DictReader(path.open()))
    checkpoints = sorted({int(r["row_checkpoint"]) for r in rows} | {int(r["column_checkpoint"]) for r in rows})
    index = {value: i for i, value in enumerate(checkpoints)}
    matrix = np.full((len(checkpoints), len(checkpoints)), .5)
    ordinary_low = np.full_like(matrix, .5)
    ordinary_high = np.full_like(matrix, .5)
    simultaneous_low = np.full_like(matrix, .5)
    simultaneous_high = np.full_like(matrix, .5)
    for row in rows:
        a, b = int(row["row_checkpoint"]), int(row["column_checkpoint"])
        i, j = index[a], index[b]
        score = float(row["row_model_score"])
        low, high = float(row["row_score_ci95_low"]), float(row["row_score_ci95_high"])
        slow, shigh = float(row["row_score_simultaneous95_low"]), float(row["row_score_simultaneous95_high"])
        matrix[i, j], matrix[j, i] = score, 1 - score
        ordinary_low[i, j], ordinary_high[i, j] = low, high
        ordinary_low[j, i], ordinary_high[j, i] = 1 - high, 1 - low
        simultaneous_low[i, j], simultaneous_high[i, j] = slow, shigh
        simultaneous_low[j, i], simultaneous_high[j, i] = 1 - shigh, 1 - slow
    later = [(i, j) for i in range(len(checkpoints)) for j in range(i)]
    scores = [matrix[i, j] for i, j in later]
    summary.append({
        "method": title,
        "pairs": len(later),
        "later_score_above_50": sum(v > .5 for v in scores),
        "ordinary_significant_regressions": sum(ordinary_high[i, j] < .5 for i, j in later),
        "simultaneous_significant_regressions": sum(simultaneous_high[i, j] < .5 for i, j in later),
        "mean_later_vs_earlier_score": f"{np.mean(scores):.4f}",
        "worst_later_vs_earlier_score": f"{min(scores):.4f}",
    })
    all_data.append((title, checkpoints, matrix, ordinary_low, ordinary_high,
                     simultaneous_low, simultaneous_high))

root.mkdir(parents=True, exist_ok=True)
with (root / "stability_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

if not all_data:
    raise SystemExit(f"no completed pair summaries under {root}")


def draw_heatmap(ax, title, checkpoints, matrix, ordinary_low, ordinary_high,
                 simultaneous_low, simultaneous_high):
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu")
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11)
                continue
            ci_margin = 50.0 * (ordinary_high[i, j] - ordinary_low[i, j])
            label = f"{100*matrix[i,j]:.1f}% ± {ci_margin:.1f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    labels = [f"i{x//500}" for x in checkpoints]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Blokus10 — {title} ({protocol.split(',')[-1].strip()})")
    ax.set_xlabel("Column checkpoint")
    ax.set_ylabel("Row checkpoint")
    return image


file_names = {
    "Multiplayer AlphaZero": "multiplayer_alphazero_heatmap",
    "Opponent Pool Only": "opponent_pool_only_heatmap",
    "Opponent Pool + Rank": "opponent_pool_rank_heatmap",
    "Strong JPSRO": "strong_jpsro_heatmap",
}
for (title, checkpoints, matrix, ordinary_low, ordinary_high,
     simultaneous_low, simultaneous_high) in all_data:
    fig, ax = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
    image = draw_heatmap(ax, title, checkpoints, matrix, ordinary_low, ordinary_high,
                         simultaneous_low, simultaneous_high)
    fig.colorbar(image, ax=ax, shrink=.82, label="Row model W + 0.5D")
    stem = file_names[title]
    png_path = root / f"{stem}.png"
    fig.savefig(png_path, dpi=180)
    fig.savefig(root / f"{stem}.pdf")
    plt.close(fig)
    print(png_path)
print(root / "stability_summary.csv")
