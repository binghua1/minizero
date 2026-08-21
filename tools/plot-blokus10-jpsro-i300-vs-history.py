#!/usr/bin/env python3
"""Plot JPSRO i300 against every 20-iteration historical checkpoint."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "blokus10_jpsro_l075_i300_stability"
OUT = ROOT / "runs" / "blokus10_final_results"
rng = np.random.default_rng(20260817)
rows = []

for step in range(0, 140001, 10000):
    path = RESULTS / f"{step}_vs_150000" / "games.jsonl"
    if not path.exists():
        continue
    scores = []
    for line in path.open():
        game = json.loads(line)
        if game.get("error") or not game.get("returns"):
            continue
        best = max(game["returns"])
        newer_top = any(
            agent == "iter_150000" and value == best
            for agent, value in zip(game["seating"], game["returns"])
        )
        older_top = any(
            agent == f"iter_{step}" and value == best
            for agent, value in zip(game["seating"], game["returns"])
        )
        scores.append(1.0 if newer_top and not older_top else 0.5 if newer_top and older_top else 0.0)
    values = np.asarray(scores)
    if len(values) != 700:
        continue
    bootstrap = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    rows.append({
        "opponent_iteration": step // 500,
        "opponent_training_steps": step,
        "games": len(values),
        "jpsro_i300_model_score": values.mean(),
        "ci95_low": low,
        "ci95_high": high,
        "historical_model_score": 1.0 - values.mean(),
        "historical_ci95_low": 1.0 - high,
        "historical_ci95_high": 1.0 - low,
    })

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "jpsro_i300_vs_history.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

x = np.asarray([row["opponent_iteration"] for row in rows])
y = 100 * np.asarray([row["historical_model_score"] for row in rows])
low = 100 * np.asarray([row["historical_ci95_low"] for row in rows])
high = 100 * np.asarray([row["historical_ci95_high"] for row in rows])

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(
    x,
    y,
    "o-",
    color="#d62728",
    linewidth=2,
    markersize=6,
    label="Historical JPSRO checkpoint",
)
ax.fill_between(x, low, high, color=line.get_color(), alpha=0.16, linewidth=0,
                label="95% bootstrap CI")
ax.axhline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="50%")
ax.set(
    xlabel="Historical JPSRO opponent iteration",
    ylabel="Historical checkpoint model-level score (%)",
    xlim=(-5, 285),
    ylim=(0, 60),
    title="Historical JPSRO checkpoints vs JPSRO i300\nRank utility lambda=0.75, 700 games/pair",
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
png_path = OUT / "jpsro_i300_vs_history.png"
fig.savefig(png_path, dpi=200)
fig.savefig(OUT / "jpsro_i300_vs_history.pdf")
plt.close(fig)

print(png_path)
print(csv_path)
