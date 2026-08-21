#!/usr/bin/env bash
set -euo pipefail

# Evaluate a final Strong-JPSRO checkpoint against regularly sampled history.
# Results are resumable and use Rank search with lambda=0.75 by default.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${RUN_DIR:-runs/blokus10_jpsro_strong_300_w4b32}"
training_dir="$run_dir/training"
config_file="$training_dir/training.cfg"
final_iteration="${FINAL_ITERATION:-500}"
history_start="${HISTORY_START:-0}"
history_end="${HISTORY_END:-$((final_iteration - 50))}"
history_step="${HISTORY_STEP:-50}"
final_steps="$((final_iteration * 500))"
output_root="${OUTPUT_DIR:-runs/blokus10_jpsro_l075_i${final_iteration}_stability}"
games="${GAMES:-700}"
gpu="${GPU:-0}"
threads="${THREADS:-3}"
rank_weight="${RANK_WEIGHT:-0.75}"
reverse="${REVERSE:-false}"

[[ -f "$config_file" ]] || { echo "error: missing config: $config_file" >&2; exit 2; }
[[ -f "$training_dir/model/weight_iter_${final_steps}.pt" ]] || {
  echo "error: missing final model: $training_dir/model/weight_iter_${final_steps}.pt" >&2
  exit 2
}
mkdir -p "$output_root"

for iteration in $(seq "$history_start" "$history_step" "$history_end"); do
  earlier="$((iteration * 500))"
  earlier_model="$training_dir/model/weight_iter_${earlier}.pt"
  [[ -f "$earlier_model" ]] || { echo "error: missing model: $earlier_model" >&2; exit 2; }
  echo "[Strong JPSRO i${iteration} vs i${final_iteration}]"
  python3 tools/multiplayer-eval.py model-fight blokus10 \
    "$earlier_model" \
    "$training_dir/model/weight_iter_${final_steps}.pt" \
    --conf-file-a "$config_file" \
    --conf-file-b "$config_file" \
    --names "iter_${earlier}" "iter_${final_steps}" \
    --games "$games" \
    --output "$output_root/${earlier}_vs_${final_steps}" \
    --search-type rank \
    --num-simulations 50 \
    --noise \
    --seed "$((20260820 + earlier))" \
    -g "$gpu" \
    --num_threads "$threads" \
    --conf-str "actor_rank_utility_weight=${rank_weight}" \
    --resume
done

plot_args=()
[[ "$reverse" == true || "$reverse" == 1 ]] && plot_args+=(--reverse)
python3 tools/plot-blokus10-final-vs-history.py \
  "$output_root" \
  --final-iteration "$final_iteration" \
  --history-start "$history_start" \
  --history-end "$history_end" \
  --history-step "$history_step" \
  --games "$games" \
  --label "JPSRO + Rank" \
  "${plot_args[@]}"
