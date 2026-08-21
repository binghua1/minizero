#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

training_arg="${1:-blokus10_maxn_01}"
if [[ "$training_arg" = /* ]]; then
    training_dir="$training_arg"
else
    training_dir="$repo_root/$training_arg"
fi
training_dir="$(realpath "$training_dir")"

game_type="${GAME_TYPE:-blokus10}"
model_dir="${MODEL_DIR:-$training_dir/model}"
checkpoint_scale="${CHECKPOINT_SCALE:-1}"
executable="${EXECUTABLE:-$repo_root/build/$game_type/minizero_$game_type}"

checkpoint_text="${CHECKPOINTS:-50000 80000 120000 200000 250000}"
read -r -a checkpoints <<< "$checkpoint_text"
games_per_pair="${GAMES_PER_PAIR:-700}"
gpu_list="${EVAL_GPU:-12}"
threads_per_gpu="${EVAL_THREADS_PER_GPU:-1}"
simulations="${EVAL_SIMULATIONS:-50}"
search_type="${EVAL_SEARCH_TYPE:-maxn}"
rank_weight="${EVAL_RANK_WEIGHT:-}"
noise="${EVAL_NOISE:-true}"
seed="${EVAL_SEED:-20260813}"
bootstrap_samples="${BOOTSTRAP_SAMPLES:-5000}"
dry_run="${DRY_RUN:-false}"
config_file="${CONFIG_FILE:-$training_dir/$(basename "$training_dir").cfg}"
output_override="${OUTPUT_DIR:-}"

if (( ${#checkpoints[@]} < 3 )); then
    echo "error: CHECKPOINTS must contain at least three iterations" >&2
    exit 2
fi
if [[ ! -f "$config_file" ]]; then
    echo "error: config not found: $config_file (set CONFIG_FILE explicitly)" >&2
    exit 2
fi
if [[ ! "$checkpoint_scale" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: CHECKPOINT_SCALE must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$executable" ]]; then
    echo "error: engine not found: $executable" >&2
    exit 2
fi
case "$noise" in
    true|1|yes) noise_flag="--noise"; noise_label="noise" ;;
    false|0|no) noise_flag="--no-noise"; noise_label="no-noise" ;;
    *) echo "error: EVAL_NOISE must be true or false" >&2; exit 2 ;;
esac
case "$search_type" in
    maxn|paranoid) search_label="$search_type" ;;
    rank)
        if [[ -z "$rank_weight" ]]; then
            echo "error: EVAL_RANK_WEIGHT is required when EVAL_SEARCH_TYPE=rank" >&2
            exit 2
        fi
        search_label="rank_l${rank_weight//./p}"
        ;;
    *) echo "error: EVAL_SEARCH_TYPE must be maxn, paranoid, or rank" >&2; exit 2 ;;
esac
output_dir="${output_override:-$training_dir/evaluation/checkpoint_cycles_${search_label}_n${simulations}_${noise_label}_g${games_per_pair}}"

mapfile -t checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort -n -u)
for checkpoint in "${checkpoints[@]}"; do
    model="$model_dir/weight_iter_$((checkpoint * checkpoint_scale)).pt"
    if [[ ! -f "$model" ]]; then
        echo "error: checkpoint not found: $model" >&2
        exit 2
    fi
done

mkdir -p "$output_dir"
printf 'Cross-play checkpoints: %s\n' "${checkpoints[*]}"
printf 'Games: %s per pair; GPU: %s; simulations: %s; search: %s; mode: %s\n' \
    "$games_per_pair" "$gpu_list" "$simulations" "$search_label" "$noise_label"
printf 'Output: %s\n' "$output_dir"

pair_index=0
for ((i = 0; i < ${#checkpoints[@]}; i++)); do
    for ((j = i + 1; j < ${#checkpoints[@]}; j++)); do
        first="${checkpoints[$i]}"
        second="${checkpoints[$j]}"
        pair_seed="$((seed + 1009 * pair_index))"
        pair_output="$output_dir/${first}_vs_${second}"
        command=(
            python3 tools/multiplayer-eval.py model-fight "$game_type"
            "$model_dir/weight_iter_$((first * checkpoint_scale)).pt"
            "$model_dir/weight_iter_$((second * checkpoint_scale)).pt"
            --conf-file-a "$config_file"
            --conf-file-b "$config_file"
            --names "iter_${first}" "iter_${second}"
            --games "$games_per_pair"
            --output "$pair_output"
            --search-type "$search_type"
            --num-simulations "$simulations"
            --executable "$executable"
            "$noise_flag"
            --seed "$pair_seed"
            -g "$gpu_list"
            --num_threads "$threads_per_gpu"
            --resume
        )
        if [[ "$search_type" == rank ]]; then
            command+=(--conf-str "actor_rank_utility_weight=$rank_weight")
        fi
        if [[ "$dry_run" = true || "$dry_run" = 1 ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${command[@]}"
            printf '\n'
        else
            printf '\n[%s vs %s]\n' "$first" "$second"
            "${command[@]}"
        fi
        pair_index="$((pair_index + 1))"
    done
done

if [[ "$dry_run" = true || "$dry_run" = 1 ]]; then
    exit 0
fi

python3 tools/analyze-checkpoint-crossplay.py "$output_dir" \
    --checkpoints "${checkpoints[@]}" \
    --bootstrap-samples "$bootstrap_samples" \
    --seed "$seed"
