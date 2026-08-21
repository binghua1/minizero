#!/usr/bin/env python3
"""Plot the report learning curve using the selected lambda=.25 methods."""

import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASELINE_STEP = 50000
PLOT_END = int(os.environ.get("PLOT_END", "300"))
POOLRANK_EVAL = ROOT / "blokus10_balanced_population_rank_nofilm_l0p75_n50/evaluation"
selected_poolrank = list(POOLRANK_EVAL.glob(
    f"balancedrank*_rank_curve_l025_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700"
))
if len(selected_poolrank) >= 16:
    poolrank_label = "Opponent Pool + Rank (lambda=0.25)"
    poolrank_pattern = re.compile(
        rf"balancedrank(\d+)_rank_curve_l025_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"
    )
else:
    # Until the optional lambda=.25 curve is complete, retain the already
    # evaluated lambda=.75 Pool+Rank curve rather than dropping the method.
    poolrank_label = "Opponent Pool + Rank (lambda=0.75)"
    poolrank_pattern = re.compile(
        rf"balancedrank(\d+)_rank_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"
    )
SPECS = (
    ("Multiplayer AlphaZero", ROOT / "blokus10_maxn_01/evaluation",
     re.compile(rf"balancedrank(\d+)_maxn_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"), "candidate_maxn"),
    ("Opponent Pool Only", ROOT / "blokus10_balanced_population_maxn_nofilm_n50/evaluation",
     re.compile(rf"balancedrank(\d+)_maxn_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"), "candidate_maxn"),
    (poolrank_label, POOLRANK_EVAL, poolrank_pattern, "candidate_rank"),
    ("Strong JPSRO + Rank (lambda=0.25)", ROOT / "runs/blokus10_jpsro_strong_300_w4b32/training/evaluation",
     re.compile(rf"balancedrank(\d+)_rank_curve_l025_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"), "candidate_rank"),
    ("Strong JPSRO + Rank (lambda=0.75)", ROOT / "runs/blokus10_jpsro_strong_300_w4b32/training/evaluation",
     re.compile(rf"balancedrank(\d+)_rank_curve_l075_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"), "candidate_rank"),
)


def summarize(path, candidate):
    latest = {}
    for line in path.open():
        if line.strip():
            record = json.loads(line)
            latest[record["game_id"]] = record
    wins = draws = losses = errors = 0
    for record in latest.values():
        if record.get("error"):
            errors += 1
            continue
        top_value = max(record["returns"])
        top = {name for name, value in zip(record["seating"], record["returns"]) if value == top_value}
        if top == {candidate}:
            wins += 1
        elif candidate in top:
            draws += 1
        else:
            losses += 1
    games = wins + draws + losses
    if games != 700 or errors:
        return None
    score = (wins + .5 * draws) / games
    ci = 1.96 * math.sqrt(score * (1 - score) / games)
    return wins, draws, losses, score, ci


rows = []
for method, directory, pattern, candidate in SPECS:
    for result_dir in directory.iterdir():
        match = pattern.fullmatch(result_dir.name)
        games = result_dir / "games.jsonl"
        if not match or not games.exists():
            continue
        result = summarize(games, candidate)
        if result is None:
            continue
        wins, draws, losses, score, ci = result
        step = int(match.group(1))
        if step <= PLOT_END * 500:
            rows.append(dict(method=method, iteration=step // 500, training_steps=step,
                             wins=wins, draws=draws, losses=losses, games=700,
                             score=score, ci95_half_width=ci))

out = ROOT / "runs/blokus10_i100_anchor_winrate/final_l025"
out.mkdir(parents=True, exist_ok=True)
rows.sort(key=lambda row: (row["method"], row["iteration"]))
if not rows:
    raise SystemExit("no completed curve evaluations")
with (out / "winrate_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(10, 6))
styles = ("^-", "D-", "o-", "s-", "P-")
for style, (method, *_rest) in zip(styles, SPECS):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        continue
    x = [row["iteration"] for row in selected]
    y = [100 * row["score"] for row in selected]
    error = [100 * row["ci95_half_width"] for row in selected]
    line, = ax.plot(x, y, style, linewidth=1.8, label=method)
    ax.fill_between(x, [a - b for a, b in zip(y, error)], [a + b for a, b in zip(y, error)],
                    color=line.get_color(), alpha=.12, linewidth=0)
ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=.7, label="50%")
ax.set(xlabel="Training Iteration", ylabel="Model-level Score (%)", xlim=(0, PLOT_END), ylim=(0, 100),
       title="Blokus10 vs fixed Multiplayer AlphaZero i100 (700 games/checkpoint)")
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(out / "winrate_vs_alphazero_i100_final_l025.png", dpi=180)
fig.savefig(out / "winrate_vs_alphazero_i100_final_l025.pdf")
print(out / "winrate_vs_alphazero_i100_final_l025.png")
