#!/usr/bin/env bash
set -euo pipefail

cd /workspace

training_dir=/workspace/blokus10_maxn_01
config_file="$training_dir/blokus10_maxn_01.cfg"
final_iteration="${FINAL_ITERATION:-300}"
history_start="${HISTORY_START:-0}"
history_end="${HISTORY_END:-$((final_iteration - 20))}"
history_step="${HISTORY_STEP:-20}"
final_steps=$((final_iteration * 500))
output_root="/workspace/runs/blokus10_maz_i${final_iteration}_stability"
games="${GAMES:-700}"
gpu="${GPU:-0}"
threads="${THREADS:-1}"
reverse="${REVERSE:-false}"

mkdir -p "$output_root"
if [[ -n "${EARLIER_STEPS:-}" ]]; then
  earlier_steps="$EARLIER_STEPS"
else
  earlier_steps="$(seq $((history_start * 500)) $((history_step * 500)) $((history_end * 500)))"
fi
for earlier in $earlier_steps; do
  echo "[MAZ i$((earlier / 500)) vs i${final_iteration}]"
  python3 tools/multiplayer-eval.py model-fight blokus10 \
    "$training_dir/model/weight_iter_${earlier}.pt" \
    "$training_dir/model/weight_iter_${final_steps}.pt" \
    --conf-file-a "$config_file" \
    --conf-file-b "$config_file" \
    --names "iter_${earlier}" "iter_${final_steps}" \
    --games "$games" \
    --output "$output_root/${earlier}_vs_${final_steps}" \
    --search-type maxn \
    --num-simulations 50 \
    --noise \
    --seed "$((20260819 + earlier))" \
    -g "$gpu" \
    --num_threads "$threads" \
    --resume
done

plot_args=()
[[ "$reverse" == true || "$reverse" == 1 ]] && plot_args+=(--reverse)
python3 tools/plot-blokus10-final-vs-history.py \
  "$output_root" --final-iteration "$final_iteration" \
  --history-start "$history_start" --history-end "$history_end" \
  --history-step "$history_step" --games "$games" \
  --label "Multiplayer AlphaZero" \
  "${plot_args[@]}"
