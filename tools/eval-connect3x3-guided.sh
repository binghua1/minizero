#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

run_dir=$(readlink -m "${1:-runs/connect3x3_guided_pool_100_s0}")
alphazero_dir=$(readlink -m "${2:-connect3x3_maxn_smoke_01}")
training_dir="$run_dir/training"
config="$training_dir/training.cfg"
model_dir="$training_dir/model"
settings="$run_dir/guided_pool.settings"
alphazero_model_dir="$alphazero_dir/model"

interval_iterations=${INTERVAL_ITERATIONS:-10}
self_eval_games=${SELF_EVAL_GAMES:-120}
fight_games=${FIGHT_GAMES:-600}
simulations=${EVAL_SIMULATIONS:-50}
gpu=${EVAL_GPU:-0}
threads=${EVAL_THREADS:-4}
seed=${EVAL_SEED:-0}
milestones=${EVAL_MILESTONES:-"100 200 300 400 500"}
guided_fight_iteration=${GUIDED_FIGHT_ITERATION:-300}

die() { echo "error: $*" >&2; exit 1; }

extend_arena_games() {
    local manifest=$1
    local target=$2
    [[ -f "$manifest" ]] || return 0
    python3 - "$manifest" "$target" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
data = json.loads(path.read_text())
current = int(data.get("num_games", 0))
if current > target:
    raise SystemExit(f"refusing to reduce {path} from {current} to {target} games")
if current < target:
    data["num_games"] = target
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    print(f"extended {path}: {current} -> {target} games")
PY
}

[[ -d "$run_dir" ]] || die "run directory not found: $run_dir"
[[ -d "$model_dir" ]] || die "model directory not found: $model_dir"
[[ -f "$config" ]] || die "training config not found: $config"
[[ -d "$alphazero_model_dir" ]] || die "AlphaZero model directory not found: $alphazero_model_dir"
[[ -x build/connect3x3/minizero_connect3x3 ]] || die "connect3x3 engine is not built"

alphazero_config=$(find "$alphazero_dir" -maxdepth 1 -name '*.cfg' -type f -print | sort | tail -n 1)
[[ -n "$alphazero_config" ]] || die "AlphaZero config not found in $alphazero_dir"

training_steps=500
if [[ -f "$settings" ]]; then
    saved_steps=$(sed -nE 's/.*(^| )steps=([0-9]+)( |$).*/\2/p' "$settings")
    [[ -z "$saved_steps" ]] || training_steps=$saved_steps
fi

interval_checkpoints=$interval_iterations

mapfile -t checkpoints < <(find "$model_dir" -maxdepth 1 -name 'weight_iter_*.pt' -print | sort -V)
(( ${#checkpoints[@]} > 1 )) || die "not enough checkpoints in $model_dir"
latest_model=${checkpoints[-1]}
latest_step=${latest_model##*weight_iter_}
latest_step=${latest_step%.pt}
latest_iteration=$((latest_step / training_steps))

alphazero_steps=$(sed -nE 's/^[[:space:]]*learner_training_step=([0-9]+).*/\1/p' "$alphazero_config" | head -n 1)
[[ -n "$alphazero_steps" ]] || alphazero_steps=500
mapfile -t alphazero_checkpoints < <(find "$alphazero_model_dir" -maxdepth 1 -name 'weight_iter_*.pt' -print | sort -V)
(( ${#alphazero_checkpoints[@]} > 1 )) || die "not enough AlphaZero checkpoints in $alphazero_model_dir"
alphazero_latest_model=${alphazero_checkpoints[-1]}
alphazero_latest_step=${alphazero_latest_model##*weight_iter_}
alphazero_latest_step=${alphazero_latest_step%.pt}
alphazero_latest_iteration=$((alphazero_latest_step / alphazero_steps))

output_root="$run_dir/evaluations/strength_check_guided_vs_alphazero"
self_output="$output_root/self_eval_maxn_every_${interval_iterations}i"
mkdir -p "$output_root"

guided_fight_model="$model_dir/weight_iter_$((guided_fight_iteration * training_steps)).pt"
[[ -f "$guided_fight_model" ]] || die "fixed guided checkpoint not found: $guided_fight_model"

guided_rank_head=$(sed -nE 's/^[[:space:]]*nn_use_rank_head=([^ #]+).*/\1/p' "$config" | head -n 1)
alphazero_rank_head=$(sed -nE 's/^[[:space:]]*nn_use_rank_head=([^ #]+).*/\1/p' "$alphazero_config" | head -n 1)
rank_available=false
if [[ "${guided_rank_head,,}" == true && "${alphazero_rank_head,,}" == true ]]; then
    rank_available=true
fi

echo "Run:              $run_dir"
echo "AlphaZero run:    $alphazero_dir"
echo "Guided latest:    iteration $latest_iteration ($latest_model)"
echo "AlphaZero latest: iteration $alphazero_latest_iteration ($alphazero_latest_model)"
echo "Cross-eval:       guided i$guided_fight_iteration vs AlphaZero iterations $milestones"
echo "Self-eval:        guided iteration 0 onward, every $interval_iterations iterations, $self_eval_games games per pair"
echo "Direct fight:     $fight_games games"
echo "Search:           MaxN, $simulations simulations, Dirichlet noise enabled"
echo "GPU / threads:    $gpu / $threads"
echo "Output:           $output_root"
if [[ "$rank_available" != true ]]; then
    echo "Rank eval:        unavailable (both checkpoint families require nn_use_rank_head=true)"
fi

fight_summaries=()
pool_summaries=()
for compare_iteration in $milestones; do
    [[ "$compare_iteration" =~ ^[1-9][0-9]*$ ]] || die "invalid milestone: $compare_iteration"
    alphazero_compare_model="$alphazero_model_dir/weight_iter_$((compare_iteration * alphazero_steps)).pt"
    if [[ ! -f "$alphazero_compare_model" ]]; then
        echo "Pending AlphaZero i$compare_iteration: checkpoint is missing"
        continue
    fi
    fight_output="$output_root/maxn_guided_i${guided_fight_iteration}_vs_alphazero_i${compare_iteration}"
    extend_arena_games "$fight_output/arena.json" "$fight_games"
    python3 tools/multiplayer-eval.py model-fight connect3x3 \
        "$guided_fight_model" "$alphazero_compare_model" \
        --conf-file-a "$config" \
        --conf-file-b "$alphazero_config" \
        --names "guided_i${guided_fight_iteration}" "alphazero_i${compare_iteration}" \
        --games "$fight_games" \
        --output "$fight_output" \
        --search-type maxn \
        --num-simulations "$simulations" \
        --noise \
        --seed "$seed" \
        -g "$gpu" \
        --num_threads "$threads" \
        --resume
    fight_summaries+=("$fight_output/fight_summary.csv")

    pool_output="$output_root/maxn_certified_pool_vs_alphazero_i${compare_iteration}"
    pool_arena="$pool_output/arena.json"
    python3 tools/guided-pool-eval.py create \
        "$run_dir" "$alphazero_compare_model" "$alphazero_config" "alphazero_i${compare_iteration}" \
        "$pool_arena" --config "$config" --games "$fight_games" \
        --simulations "$simulations" --seed "$seed" --noise
    python3 tools/multiplayer-eval.py "$pool_arena" "$pool_output" \
        -g "$gpu" --num_threads "$threads" --resume
    python3 tools/guided-pool-eval.py summarize \
        "$pool_output/games.jsonl" "alphazero_i${compare_iteration}" "$pool_output/pool_fight_summary.csv"
    pool_summaries+=("$pool_output/pool_fight_summary.csv")
done

python3 tools/multiplayer-eval.py self-eval connect3x3 "$training_dir" \
    --conf-file "$config" \
    --interval "$interval_checkpoints" \
    --games "$self_eval_games" \
    --start-index 0 \
    --output "$self_output" \
    --search-type maxn \
    --num-simulations "$simulations" \
    --noise \
    --seed "$seed" \
    -g "$gpu" \
    --num_threads "$threads" \
    --resume

python3 - "$self_output/elo.csv" "$training_steps" "${fight_summaries[@]}" <<'PY'
import csv
import sys
from pathlib import Path

self_path = Path(sys.argv[1])
training_steps = int(sys.argv[2])
fight_paths = [Path(value) for value in sys.argv[3:]]
rows = list(csv.DictReader(self_path.open()))

print("\n========== Connect3x3 strength summary ==========")
print("Guided vs multiplayer AlphaZero:")
for fight_path in fight_paths:
    fight = next(csv.DictReader(fight_path.open()))
    print(
        f"  {fight['model_a']} vs {fight['model_b']} | "
        f"wins={fight['model_a_wins']} losses={fight['model_b_wins']} "
        f"draws={fight['draws']} score={float(fight['model_a_score']):.1%} "
        f"errors={fight['errors']}"
    )
print("\nSequential self-eval:")
for row in rows:
    print(
        f"  i{int(row['P2']) // training_steps:>3} -> i{int(row['P1']) // training_steps:<3} | "
        f"new wins={row['P1 Wins']:>4} old wins={row['P2 Wins']:>4} "
        f"draws={row['Draw']:>4} score={float(row['WinRate']):.1%} "
        f"Elo={row['P1 Elo']}"
    )

if rows:
    scores = [float(row["WinRate"]) for row in rows]
    improved = sum(score > 0.5 for score in scores)
    print(f"\nNewer checkpoint scored above 50% in {improved}/{len(scores)} adjacent comparisons.")
for fight_path in fight_paths:
    print(f"CSV: {fight_path}")
print(f"CSV: {self_path}")
PY
