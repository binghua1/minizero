#!/usr/bin/env python3
"""Create the two compact figures used by the final Blokus10 report."""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/blokus10_final_results"
OUT.mkdir(parents=True, exist_ok=True)
PLOT_END = int(os.environ.get("PLOT_END", "300"))


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


# Figure 1: the failure case and the proposed method on the same checkpoints.
heatmap_specs = (
    (
        "Multiplayer AlphaZero",
        ROOT / "blokus10_maxn_01/evaluation/checkpoint_cycles_maxn_n50_noise_g700",
    ),
    (
        "Opponent Pool + Rank",
        ROOT / "runs/blokus10_poolrank_long_stability",
    ),
)

heatmap_data = []
stability_rows = []
for title, directory in heatmap_specs:
    rows = read_rows(directory / "pair_summary.csv")
    checkpoints = sorted(
        {int(row["row_checkpoint"]) for row in rows}
        | {int(row["column_checkpoint"]) for row in rows}
    )
    index = {checkpoint: i for i, checkpoint in enumerate(checkpoints)}
    matrix = np.full((len(checkpoints), len(checkpoints)), 0.5)
    ordinary_low = np.full_like(matrix, 0.5)
    ordinary_high = np.full_like(matrix, 0.5)
    simultaneous_low = np.full_like(matrix, 0.5)
    simultaneous_high = np.full_like(matrix, 0.5)
    for row in rows:
        a = int(row["row_checkpoint"])
        b = int(row["column_checkpoint"])
        i, j = index[a], index[b]
        score = float(row["row_model_score"])
        low = float(row["row_score_ci95_low"])
        high = float(row["row_score_ci95_high"])
        slow = float(row["row_score_simultaneous95_low"])
        shigh = float(row["row_score_simultaneous95_high"])
        matrix[i, j], matrix[j, i] = score, 1.0 - score
        ordinary_low[i, j], ordinary_low[j, i] = low, 1.0 - high
        ordinary_high[i, j], ordinary_high[j, i] = high, 1.0 - low
        simultaneous_low[i, j], simultaneous_low[j, i] = slow, 1.0 - shigh
        simultaneous_high[i, j], simultaneous_high[j, i] = shigh, 1.0 - slow
    heatmap_data.append((title, checkpoints, matrix, ordinary_low, ordinary_high,
                         simultaneous_low, simultaneous_high))
    later_scores = [matrix[i, j] for i in range(len(checkpoints)) for j in range(i)]
    stability_rows.append({
        "method": title,
        "pairs": len(later_scores),
        "later_checkpoint_wins": sum(score > 0.5 for score in later_scores),
        "mean_later_checkpoint_score": f"{np.mean(later_scores):.8f}",
        "worst_later_checkpoint_score": f"{min(later_scores):.8f}",
    })

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
image = None
for ax, (title, checkpoints, matrix, ordinary_low, ordinary_high,
         simultaneous_low, simultaneous_high) in zip(axes, heatmap_data):
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu")
    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11)
                continue
            ci_margin = 50.0 * (ordinary_high[i, j] - ordinary_low[i, j])
            label = f"{100 * matrix[i, j]:.1f}% ± {ci_margin:.1f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.8)
    labels = [f"{checkpoint // 1000}k" for checkpoint in checkpoints]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.set_xlabel("Column checkpoint")
axes[0].set_ylabel("Row checkpoint")
fig.colorbar(image, ax=axes, shrink=0.82, label="Row-model score")
fig.suptitle("Blokus10 — checkpoint cross-play (700 games/pair)")
fig.savefig(OUT / "checkpoint_stability_heatmaps.png", dpi=200)
fig.savefig(OUT / "checkpoint_stability_heatmaps.pdf")
plt.close(fig)

with (OUT / "checkpoint_stability_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=stability_rows[0].keys())
    writer.writeheader()
    writer.writerows(stability_rows)


# Figure 2: main fixed-anchor learning curve. Keep only the baseline and JPSRO;
# opponent-pool ablations belong in supplementary results, not the main figure.
old_curve = read_rows(ROOT / "runs/blokus10_i100_anchor_winrate/winrate_summary.csv")
new_curve = read_rows(ROOT / "runs/blokus10_i100_anchor_winrate/final_l025/winrate_summary.csv")
curve_specs = (
    ("Multiplayer AlphaZero", old_curve, "Multiplayer AlphaZero"),
    ("Strong JPSRO + Rank (search lambda=0.75)", new_curve, "Strong JPSRO + Rank (lambda=0.75)"),
)

selected_rows = []
fig, ax = plt.subplots(figsize=(10, 6))
styles = ("^-", "P-")
for style, (label, source, source_label) in zip(styles, curve_specs):
    rows = [row for row in source if row["method"] == source_label]
    rows.sort(key=lambda row: int(row["iteration"]))
    if not rows:
        raise SystemExit(f"missing curve: {source_label}")
    x = [int(row["iteration"]) for row in rows]
    score_key = "score" if "score" in rows[0] else "winrate"
    y = [100.0 * float(row[score_key]) for row in rows]
    error = [100.0 * float(row["ci95_half_width"]) for row in rows]
    line, = ax.plot(x, y, style, linewidth=2.0, markersize=6.5, label=label)
    ax.fill_between(
        x,
        [value - half for value, half in zip(y, error)],
        [value + half for value, half in zip(y, error)],
        color=line.get_color(),
        alpha=0.12,
        linewidth=0,
    )
    for row in rows:
        selected_rows.append({
            "method": label,
            "iteration": row["iteration"],
            "training_steps": row["training_steps"],
            "model_score": row[score_key],
            "ci95_half_width": row["ci95_half_width"],
        })

ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.7, label="50%")
ax.set(
    xlabel="Training iteration",
    ylabel="Model-level score (%)",
    xlim=(0, PLOT_END),
    ylim=(0, 100),
    title=f"Sample efficiency vs fixed Multiplayer AlphaZero i100, i0-i{PLOT_END} (700 games/checkpoint)",
)
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fixed_anchor_learning_curve.png", dpi=200)
fig.savefig(OUT / "fixed_anchor_learning_curve.pdf")
plt.close(fig)

with (OUT / "fixed_anchor_learning_curve.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=selected_rows[0].keys())
    writer.writeheader()
    writer.writerows(selected_rows)

print(OUT / "checkpoint_stability_heatmaps.png")
print(OUT / "fixed_anchor_learning_curve.png")
