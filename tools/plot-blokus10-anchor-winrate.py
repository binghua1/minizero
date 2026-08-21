#!/usr/bin/env python3

import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASELINE_STEP = 50000  # multiplayer AlphaZero i100
PLOT_END = int(os.environ.get("PLOT_END", "300"))
SPECS = (
    (
        "Multiplayer AlphaZero",
        ROOT / "blokus10_maxn_01/evaluation",
        re.compile(rf"balancedrank(\d+)_maxn_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"),
        "candidate_maxn",
        "nohead_maxn",
    ),
    (
        "Opponent Pool Only",
        ROOT / "blokus10_balanced_population_maxn_nofilm_n50/evaluation",
        re.compile(rf"balancedrank(\d+)_maxn_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"),
        "candidate_maxn",
        "nohead_maxn",
    ),
    (
        "Opponent Pool + Rank",
        ROOT / "blokus10_balanced_population_rank_nofilm_l0p75_n50/evaluation",
        re.compile(rf"balancedrank(\d+)_rank_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"),
        "candidate_rank",
        "nohead_maxn",
    ),
    (
        "JPSRO Pool",
        ROOT / "runs/blokus10_guided_pool_500_s0/training/evaluation",
        re.compile(rf"balancedrank(\d+)_maxn_vs_nohead{BASELINE_STEP}_maxn_n50_noise_700$"),
        "candidate_maxn",
        "nohead_maxn",
    ),
)


def summarize(path, candidate, baseline):
    wins = losses = draws = errors = 0
    latest = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            # Resume appends replacement records for failed game IDs. Match
            # multiplayer-eval.py: only the latest record for each ID counts.
            latest[record["game_id"]] = record
    candidate_returns = []
    baseline_returns = []
    for record in latest.values():
        if record.get("error"):
            errors += 1
            continue
        returns = record["returns"]
        seating = record["seating"]
        for name, value in zip(seating, returns):
            if name == candidate:
                candidate_returns.append(float(value))
            elif name == baseline:
                baseline_returns.append(float(value))
        best = max(returns)
        top = {name for name, value in zip(seating, returns) if value == best}
        if top == {candidate}:
            wins += 1
        elif top == {baseline}:
            losses += 1
        elif top == {candidate, baseline}:
            draws += 1
        else:
            raise ValueError(f"unexpected agents in {path}: {sorted(top)}")
    valid = wins + losses + draws
    if valid != 700 or errors:
        return None
    score = (wins + 0.5 * draws) / valid
    ci = 1.96 * math.sqrt(score * (1.0 - score) / valid)
    return (wins, losses, draws, valid, score, ci,
            sum(candidate_returns) / len(candidate_returns),
            sum(baseline_returns) / len(baseline_returns))


rows = []
for method, directory, pattern, candidate, baseline in SPECS:
    if not directory.is_dir():
        continue
    for result_dir in directory.iterdir():
        match = pattern.fullmatch(result_dir.name)
        games = result_dir / "games.jsonl"
        if not match or not games.is_file():
            continue
        result = summarize(games, candidate, baseline)
        if result is None:
            continue
        wins, losses, draws, valid, score, ci, candidate_return, baseline_return = result
        step = int(match.group(1))
        if step // 500 > PLOT_END:
            continue
        rows.append({
            "method": method,
            "iteration": step // 500,
            "training_steps": step,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "valid_games": valid,
            "winrate": score,
            "ci95_half_width": ci,
            "model_avg_return": candidate_return,
            "baseline_avg_return": baseline_return,
            "return_advantage": candidate_return - baseline_return,
        })

# The completed MAZ-i500-vs-history sweep already contains the exact i500 vs
# fixed-MAZ-i100 matchup under reversed directory/name ordering. Reuse it
# instead of spending another 700 games solely for the anchor curve.
if PLOT_END >= 500 and not any(
    row["method"] == "Multiplayer AlphaZero" and row["iteration"] == 500
    for row in rows
):
    reused = (
        ROOT / "runs/blokus10_maz_i500_stability/50000_vs_250000/games.jsonl"
    )
    if reused.is_file():
        result = summarize(reused, "iter_250000", "iter_50000")
        if result is not None:
            wins, losses, draws, valid, score, ci, candidate_return, baseline_return = result
            rows.append({
                "method": "Multiplayer AlphaZero",
                "iteration": 500,
                "training_steps": 250000,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "valid_games": valid,
                "winrate": score,
                "ci95_half_width": ci,
                "model_avg_return": candidate_return,
                "baseline_avg_return": baseline_return,
                "return_advantage": candidate_return - baseline_return,
            })

if not rows:
    raise SystemExit("no completed 700-game AlphaZero-i100 evaluations found")

rows.sort(key=lambda row: (row["method"], row["iteration"]))
output_dir = ROOT / "runs/blokus10_i100_anchor_winrate"
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "winrate_summary.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(10, 6))
styles = ("^-", "D-", "o-", "s-")
for style, (method, *_rest) in zip(styles, SPECS):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        continue
    x = [row["iteration"] for row in selected]
    y = [100.0 * row["winrate"] for row in selected]
    error = [100.0 * row["ci95_half_width"] for row in selected]
    lower = [value - half_width for value, half_width in zip(y, error)]
    upper = [value + half_width for value, half_width in zip(y, error)]
    line, = ax.plot(x, y, style, linewidth=1.8, label=method)
    ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.12,
                    linewidth=0)

ax.axhline(50.0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="50%")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Model-level Score (%)")
ax.set_title(f"Blokus10 vs Multiplayer AlphaZero i100, i0-i{PLOT_END} (700 games/checkpoint)")
ax.set_xlim(0, PLOT_END)
ax.set_ylim(0, 100)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
png_path = output_dir / "winrate_vs_alphazero_i100.png"
fig.savefig(png_path, dpi=180)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for style, (method, *_rest) in zip(styles, SPECS):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        continue
    x = [row["iteration"] for row in selected]
    axes[0].plot(x, [row["model_avg_return"] for row in selected], style,
                 linewidth=1.8, label=method)
    axes[1].plot(x, [row["return_advantage"] for row in selected], style,
                 linewidth=1.8, label=method)

axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
axes[0].set_title("Model average return")
axes[0].set_xlabel("Training Iteration")
axes[0].set_ylabel("Average per-seat return")
axes[0].set_xlim(0, PLOT_END)
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7,
                label="Equal return")
axes[1].set_title("Return advantage over fixed AlphaZero i100")
axes[1].set_xlabel("Training Iteration")
axes[1].set_ylabel("Model return - baseline return")
axes[1].set_xlim(0, PLOT_END)
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.suptitle(f"Blokus10 return evaluation, i0-i{PLOT_END} (700 games/checkpoint)")
fig.tight_layout()
return_png_path = output_dir / "return_vs_alphazero_i100.png"
fig.savefig(return_png_path, dpi=180)

print("iteration | method | W/L/D | score (95% CI)")
for row in rows:
    print(
        f'i{row["iteration"]:>3} | {row["method"]} | '
        f'{row["wins"]}/{row["losses"]}/{row["draws"]} | '
        f'{100*row["winrate"]:.2f}% ± {100*row["ci95_half_width"]:.2f}%'
    )
print(f"CSV: {csv_path}")
print(f"PNG: {png_path}")
print(f"Return PNG: {return_png_path}")
