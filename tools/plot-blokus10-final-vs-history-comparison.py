#!/usr/bin/env python3
"""Compare MAZ and JPSRO i300 against their own historical checkpoints."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "blokus10_final_results"
METHODS = [
    ("Multiplayer AlphaZero", ROOT / "runs" / "blokus10_maz_i300_stability", "#1f77b4"),
    ("JPSRO + Rank", ROOT / "runs" / "blokus10_jpsro_l075_i300_stability", "#d62728"),
]
rng = np.random.default_rng(20260819)


def load_scores(path, older_name):
    latest = {}
    for line in path.open():
        if line.strip():
            record = json.loads(line)
            latest[int(record["game_id"])] = record
    records = [record for record in latest.values() if not record.get("error")]
    errors = sum(bool(record.get("error")) for record in latest.values())
    if len(records) != 700 or errors:
        raise ValueError(f"{path}: {len(records)} valid games, {errors} errors; expected 700/0")
    strata = defaultdict(list)
    for record in records:
        best = max(float(value) for value in record["returns"])
        top = {
            agent for agent, value in zip(record["seating"], record["returns"])
            if float(value) == best
        }
        newer = "iter_150000"
        score = 1.0 if top == {newer} else 0.5 if newer in top else 0.0
        strata[tuple(record["seating"])].append(score)
    values = np.asarray([score for group in strata.values() for score in group])
    bootstrap = np.empty(5000)
    for index in range(len(bootstrap)):
        bootstrap[index] = np.mean([
            score
            for group in strata.values()
            for score in rng.choice(group, size=len(group), replace=True)
        ])
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return values.mean(), low, high


rows = []
for method, root, _ in METHODS:
    for step in range(0, 140001, 10000):
        path = root / f"{step}_vs_150000" / "games.jsonl"
        if not path.is_file():
            continue
        score, low, high = load_scores(path, f"iter_{step}")
        rows.append({
            "method": method,
            "historical_iteration": step // 500,
            "historical_training_steps": step,
            "games": 700,
            "i300_score": score,
            "ci95_low": low,
            "ci95_high": high,
        })

for method, _, _ in METHODS:
    count = sum(row["method"] == method for row in rows)
    if count != 15:
        raise ValueError(f"{method}: found {count}/15 complete historical comparisons")

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "i300_vs_history_comparison.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

figure, axis = plt.subplots(figsize=(10, 6))
for method, _, color in METHODS:
    selected = [row for row in rows if row["method"] == method]
    x = np.asarray([row["historical_iteration"] for row in selected])
    y = 100 * np.asarray([row["i300_score"] for row in selected])
    low = 100 * np.asarray([row["ci95_low"] for row in selected])
    high = 100 * np.asarray([row["ci95_high"] for row in selected])
    axis.plot(x, y, "o-", color=color, linewidth=2, markersize=5, label=method)
    axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
axis.axhline(50, color="0.3", linestyle="--", linewidth=1.3, label="50%")
axis.set(
    xlabel="Historical checkpoint iteration",
    ylabel="i300 model-level score (%)",
    xlim=(-5, 285),
    ylim=(45, 100),
    title="Blokus10 final-checkpoint stability: i300 vs history\n"
          "50 simulations, 700 games/pair",
)
axis.grid(alpha=0.25)
axis.legend()
figure.tight_layout()
png_path = OUT / "i300_vs_history_comparison.png"
figure.savefig(png_path, dpi=200)
figure.savefig(OUT / "i300_vs_history_comparison.pdf")
plt.close(figure)
print(f"Figure: {png_path}")
print(f"CSV: {csv_path}")
