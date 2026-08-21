#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

guided_iteration=${GUIDED_ITERATION:-500}
baseline_start=${BASELINE_START:-100}
baseline_end=${BASELINE_END:-500}
baseline_interval=${BASELINE_INTERVAL:-100}
games=${FIGHT_GAMES:-700}
gpu=${EVAL_GPU:-0123}
# multiplayer-eval creates this many workers per GPU.  GPU=0123 and 1 means
# four workers total, with one worker on each GPU.
threads_per_gpu=${EVAL_THREADS_PER_GPU:-1}
output_root=${OUTPUT_DIR:-runs/blokus10_guided_pool_500_s0/evaluations/alphazero_sweep}

die() { echo "error: $*" >&2; exit 1; }
[[ "$baseline_interval" =~ ^[1-9][0-9]*$ ]] || die "BASELINE_INTERVAL must be positive"
(( baseline_start <= baseline_end )) || die "BASELINE_START must not exceed BASELINE_END"

for ((iteration=baseline_start; iteration<=baseline_end; iteration+=baseline_interval)); do
  echo
  echo "Guided i${guided_iteration} vs multiplayer AlphaZero i${iteration}"
  GUIDED_ITERATION="$guided_iteration" \
  BASELINE_ITERATION="$iteration" \
  BASELINE_NAME=alphazero \
  FIGHT_GAMES="$games" \
  EVAL_GPU="$gpu" \
  EVAL_THREADS="$threads_per_gpu" \
  tools/eval-guided-final.sh blokus10 fight
done

mkdir -p "$output_root"
python3 - "$repo_root" "$guided_iteration" "$baseline_start" "$baseline_end" \
  "$baseline_interval" "$output_root" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

repo = Path(sys.argv[1])
guided_iteration = int(sys.argv[2])
start, end, interval = map(int, sys.argv[3:6])
output_root = Path(sys.argv[6])
strength = repo / "runs/blokus10_guided_pool_500_s0/evaluations/final_strength"

rows = []
for iteration in range(start, end + 1, interval):
    result_dir = strength / f"maxn_guided_i{guided_iteration}_vs_alphazero_i{iteration}"
    games_path = result_dir / "games.jsonl"
    summary_path = result_dir / "fight_summary.csv"
    if not games_path.is_file() or not summary_path.is_file():
        print(f"warning: missing completed result for AlphaZero i{iteration}", file=sys.stderr)
        continue

    fight = next(csv.DictReader(summary_path.open()))
    valid = int(fight["valid_games"])
    score = float(fight["model_a_score"])
    ci = 1.96 * math.sqrt(score * (1.0 - score) / valid)
    records = [json.loads(line) for line in games_path.open() if line.strip()]
    records = [record for record in records if not record.get("error")]

    composition_scores = {}
    guided_returns = []
    baseline_returns = []
    for lineup_id, label in ((0, "3v1"), (1, "2v2"), (2, "1v3")):
        subset = [record for record in records if record["lineup_id"] == lineup_id]
        wins = draws = 0
        for record in subset:
            best = max(record["returns"])
            top = {
                name for name, value in zip(record["seating"], record["returns"])
                if value == best
            }
            wins += top == {f"guided_i{guided_iteration}"}
            draws += top == {f"guided_i{guided_iteration}", f"alphazero_i{iteration}"}
        composition_scores[label] = (wins + 0.5 * draws) / len(subset)

    for record in records:
        for name, value in zip(record["seating"], record["returns"]):
            (guided_returns if name == f"guided_i{guided_iteration}" else baseline_returns).append(value)

    rows.append({
        "new_model": f"Guided i{guided_iteration}",
        "baseline": f"AlphaZero i{iteration}",
        "wins/losses/draws": f'{fight["model_a_wins"]}/{fight["model_b_wins"]}/{fight["draws"]}',
        "1v3": composition_scores["1v3"],
        "2v2": composition_scores["2v2"],
        "3v1": composition_scores["3v1"],
        "overall": score,
        "ci95_half_width": ci,
        "baseline_return": sum(baseline_returns) / len(baseline_returns),
        "new_model_return": sum(guided_returns) / len(guided_returns),
    })

output = output_root / "guided_i500_vs_alphazero_sweep.csv"
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== Blokus10 Guided strength sweep ==========")
print("New model | Baseline | W/L/D | 1v3 | 2v2 | 3v1 | Overall (95% CI) | Baseline Return | New Model Return")
for row in rows:
    print(
        f'{row["new_model"]} | {row["baseline"]} | {row["wins/losses/draws"]} | '
        f'{100*row["1v3"]:.2f}% | {100*row["2v2"]:.2f}% | {100*row["3v1"]:.2f}% | '
        f'{100*row["overall"]:.2f}% ± {100*row["ci95_half_width"]:.2f}% | '
        f'{row["baseline_return"]:.4f} | {row["new_model_return"]:.4f}'
    )
print(f"CSV: {output}")
PY
