#!/usr/bin/env python3
"""Plot the no-pool Rank-adaptive fixed-anchor learning curve."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = (
    ROOT
    / "blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50"
    / "evaluation"
)
OUT = ROOT / "runs" / "blokus10_final_results"
PATTERN = re.compile(
    r"balancedrank(\d+)_rank_rankonly_curve_l075_vs_nohead50000_maxn_n50_noise_700$"
)


def game_score(game):
    best = max(game["returns"])
    target_top = any(
        agent == "candidate_rank" and value == best
        for agent, value in zip(game["seating"], game["returns"])
    )
    anchor_top = any(
        agent == "nohead_maxn" and value == best
        for agent, value in zip(game["seating"], game["returns"])
    )
    if target_top and not anchor_top:
        return 1.0
    if target_top and anchor_top:
        return 0.5
    return 0.0


rng = np.random.default_rng(20260817)
rows = []
for directory in EVAL_ROOT.iterdir():
    match = PATTERN.fullmatch(directory.name)
    games_path = directory / "games.jsonl"
    if not match or not games_path.exists():
        continue
    games = [json.loads(line) for line in games_path.open()]
    scores = np.asarray(
        [game_score(game) for game in games if not game.get("error") and game.get("returns")],
        dtype=float,
    )
    if len(scores) != 700:
        continue
    bootstrap = scores[rng.integers(0, len(scores), size=(5000, len(scores)))].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    step = int(match.group(1))
    rows.append(
        {
            "iteration": step // 500,
            "training_steps": step,
            "model_score": scores.mean(),
            "ci95_low": low,
            "ci95_high": high,
            "games": len(scores),
        }
    )

rows.sort(key=lambda row: row["training_steps"])
if not rows:
    raise SystemExit(f"no completed Rank-adaptive anchor evaluations under {EVAL_ROOT}")

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "rank_adaptive_fixed_anchor_curve.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

x = np.asarray([row["iteration"] for row in rows])
y = 100 * np.asarray([row["model_score"] for row in rows])
low = 100 * np.asarray([row["ci95_low"] for row in rows])
high = 100 * np.asarray([row["ci95_high"] for row in rows])

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(
    x,
    y,
    "o-",
    color="#2ca02c",
    linewidth=2.2,
    markersize=6.5,
    label="No-pool Rank-adaptive",
)
ax.fill_between(x, low, high, color=line.get_color(), alpha=0.15, linewidth=0)
ax.axhline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="50%")
ax.set(
    xlabel="Training iteration",
    ylabel="Model-level score (%)",
    xlim=(0, 300),
    ylim=(0, 100),
    title=(
        "No-pool Rank-adaptive vs fixed Multiplayer AlphaZero i100\n"
        "Evaluation: fixed Rank utility lambda=0.75, 700 games/checkpoint"
    ),
)
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()

png_path = OUT / "rank_adaptive_fixed_anchor_curve.png"
fig.savefig(png_path, dpi=200)
fig.savefig(OUT / "rank_adaptive_fixed_anchor_curve.pdf")
plt.close(fig)

print(png_path)
print(csv_path)
