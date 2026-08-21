#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

run_dir=$(readlink -m "${RUN_DIR:-runs/connect3x3_guided_pool_100_s0}")
baseline_dir=$(readlink -m "${BASELINE_DIR:-connect3x3_maxn_smoke_01}")
start=${SWEEP_START:-0}
end=${SWEEP_END:-100}
interval=${SWEEP_INTERVAL:-10}
baseline_iteration=${BASELINE_ITERATION:-100}
games=${FIGHT_GAMES:-600}
gpu=${EVAL_GPU:-0}
threads=${EVAL_THREADS:-4}
simulations=${EVAL_SIMULATIONS:-50}
seed=${EVAL_SEED:-0}
output_root=$(readlink -m "${OUTPUT_DIR:-$run_dir/evaluations/i0_i100_vs_alphazero_i100}")

die() { echo "error: $*" >&2; exit 1; }

guided_cfg="$run_dir/training/training.cfg"
baseline_cfg=$(find "$baseline_dir" -maxdepth 1 -type f -name '*.cfg' -print | sort | tail -n 1)
baseline_model="$baseline_dir/model/weight_iter_$((baseline_iteration * 500)).pt"
[[ -f "$guided_cfg" ]] || die "guided config not found: $guided_cfg"
[[ -n "$baseline_cfg" && -f "$baseline_cfg" ]] || die "baseline config not found in $baseline_dir"
[[ -f "$baseline_model" ]] || die "baseline checkpoint not found: $baseline_model"
[[ -x build/connect3x3/minizero_connect3x3 ]] || die "connect3x3 executable is not built"
mkdir -p "$output_root"

for ((iteration=start; iteration<=end; iteration+=interval)); do
    model="$run_dir/training/model/weight_iter_$((iteration * 500)).pt"
    output="$output_root/guided_i${iteration}_vs_alphazero_i${baseline_iteration}"
    [[ -f "$model" ]] || die "guided checkpoint not found: $model"
    echo
    echo "Guided i${iteration} vs AlphaZero i${baseline_iteration}: ${games} games"
    python3 tools/multiplayer-eval.py model-fight connect3x3 \
        "$model" "$baseline_model" \
        --conf-file-a "$guided_cfg" \
        --conf-file-b "$baseline_cfg" \
        --names "guided_i${iteration}" "alphazero_i${baseline_iteration}" \
        --games "$games" \
        --output "$output" \
        --search-type maxn \
        --num-simulations "$simulations" \
        --noise \
        --seed "$seed" \
        -g "$gpu" \
        --num_threads "$threads" \
        --resume
done

python3 - "$output_root" "$baseline_iteration" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_iteration = int(sys.argv[2])
rows = []
for result_dir in root.glob(f"guided_i*_vs_alphazero_i{baseline_iteration}"):
    summary_path = result_dir / "fight_summary.csv"
    games_path = result_dir / "games.jsonl"
    if not summary_path.is_file() or not games_path.is_file():
        continue
    fight = next(csv.DictReader(summary_path.open()))
    if int(fight["errors"]) or int(fight["valid_games"]) == 0:
        continue
    iteration = int(fight["model_a"].rsplit("_i", 1)[1])
    candidate, baseline = fight["model_a"], fight["model_b"]
    by_count = {1: [0, 0, 0], 2: [0, 0, 0]}
    for line in games_path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue
        count = record["seating"].count(candidate)
        if count not in by_count:
            continue
        best = max(record["returns"])
        top = {name for name, value in zip(record["seating"], record["returns"]) if value == best}
        if top == {candidate}:
            by_count[count][0] += 1
        elif top == {baseline}:
            by_count[count][1] += 1
        else:
            by_count[count][2] += 1

    def score(count):
        wins, losses, draws = by_count[count]
        total = wins + losses + draws
        return (wins + 0.5 * draws) / total

    valid = int(fight["valid_games"])
    overall = float(fight["model_a_score"])
    ci = 1.96 * math.sqrt(overall * (1.0 - overall) / valid)
    rows.append({
        "new_model": f"Guided i{iteration}",
        "baseline": f"AlphaZero i{baseline_iteration}",
        "wins/losses/draws": f'{fight["model_a_wins"]}/{fight["model_b_wins"]}/{fight["draws"]}',
        "1v2": score(1), "2v1": score(2), "overall": overall,
        "ci95_half_width": ci,
    })

rows.sort(key=lambda row: int(row["new_model"].split("i")[-1]))
if not rows:
    raise SystemExit("no completed evaluations to summarize")
csv_path = root / "guided_i0_i100_vs_alphazero_i100.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("\n========== Connect3x3 vs fixed AlphaZero i100 ==========")
print("New model | Baseline | W/L/D | 1v2 | 2v1 | Overall (95% CI)")
for row in rows:
    print(f'{row["new_model"]} | {row["baseline"]} | {row["wins/losses/draws"]} | '
          f'{100*row["1v2"]:.2f}% | {100*row["2v1"]:.2f}% | '
          f'{100*row["overall"]:.2f}% ± {100*row["ci95_half_width"]:.2f}%')
print(f"CSV: {csv_path}")
PY
