#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CSV = ROOT / "runs/connect3x3_anchor_i50_sweep/anchor_sweep_summary.csv"
STRENGTH_ROOT = ROOT / (
    "runs/connect3x3_guided_pool_100_s0/evaluations/"
    "strength_check_guided_vs_alphazero"
)
OUTPUT_DIR = ROOT / "runs/connect3x3_jpsro_winrate"


def read_anchor():
    with ANCHOR_CSV.open() as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["version"] == "old"]


def read_strength():
    rows = []
    for iteration in (100, 200, 300, 400, 500):
        path = STRENGTH_ROOT / (
            f"maxn_guided_i300_vs_alphazero_i{iteration}/fight_summary.csv"
        )
        row = next(csv.DictReader(path.open()))
        valid = int(row["valid_games"])
        score = float(row["model_a_score"])
        rows.append({
            "baseline_iteration": iteration,
            "wins": int(row["model_a_wins"]),
            "losses": int(row["model_b_wins"]),
            "draws": int(row["draws"]),
            "valid_games": valid,
            "score": score,
            "ci95_half_width": 1.96 * math.sqrt(score * (1 - score) / valid),
        })
    return rows


anchor = read_anchor()
strength = read_strength()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with (OUTPUT_DIR / "guided_i300_vs_alphazero_checkpoints.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=strength[0].keys())
    writer.writeheader()
    writer.writerows(strength)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
x = [int(row["iteration"]) for row in anchor]
y = [100 * float(row["score"]) for row in anchor]
e = [100 * float(row["ci95_half_width"]) for row in anchor]
ax.errorbar(x, y, yerr=e, fmt="o-", capsize=4, linewidth=2,
            color="#1976d2", label="JPSRO-Guided")
ax.axhline(50, color="black", linestyle="--", linewidth=1, label="AlphaZero i50 = 50%")
ax.set_title("Training curve vs fixed AlphaZero i50")
ax.set_xlabel("JPSRO-Guided iteration")
ax.set_ylabel("Model-level score (%)")
ax.set_ylim(15, 75)
ax.grid(alpha=0.25)
ax.legend()

ax = axes[1]
x = [row["baseline_iteration"] for row in strength]
y = [100 * row["score"] for row in strength]
e = [100 * row["ci95_half_width"] for row in strength]
ax.errorbar(x, y, yerr=e, fmt="o-", capsize=4, linewidth=2,
            color="#d32f2f", label="JPSRO-Guided i300")
total = sum(row["valid_games"] for row in strength)
pooled_score = sum(row["wins"] + 0.5 * row["draws"] for row in strength) / total
pooled_ci = 1.96 * math.sqrt(pooled_score * (1 - pooled_score) / total)
ax.axhline(50, color="black", linestyle="--", linewidth=1, label="AlphaZero = 50%")
ax.axhline(100 * pooled_score, color="#d32f2f", linestyle=":", linewidth=1.5,
           label=f"Pooled: {100*pooled_score:.2f}% ± {100*pooled_ci:.2f}%")
ax.set_title("Final i300 vs AlphaZero checkpoints")
ax.set_xlabel("AlphaZero baseline iteration")
ax.set_ylabel("Model-level score (%)")
ax.set_ylim(40, 65)
ax.grid(alpha=0.25)
ax.legend()

fig.suptitle("Connect3x3 JPSRO-Guided win-rate evaluation (95% CI)")
fig.tight_layout()
output = OUTPUT_DIR / "connect3x3_jpsro_winrate.png"
fig.savefig(output, dpi=180)

print(f"Pooled final score: {100*pooled_score:.2f}% ± {100*pooled_ci:.2f}% ({total} games)")
print(f"PNG: {output}")
