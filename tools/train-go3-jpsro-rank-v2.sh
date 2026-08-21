#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Go3 JPSRO+Rank v2.  Relative to the original run, keep the training budget
# and mixture fixed while making admission stochastic and slightly less brittle
# across the three player roles.
run_dir="${1:-runs/go3_jpsro_rank_l075_v2_i200}"
source_config="${SOURCE_CONFIG:-go3_jpsro_rank_l075.cfg}"
gpu="${GPU:-0}"
selfplay_workers="${GUIDED_SELFPLAY_WORKERS:-4}"
selfplay_gpus="${GUIDED_SELFPLAY_GPU:-}"

if [[ -z "$selfplay_gpus" ]]; then
    selfplay_gpus=""
    for ((worker = 0; worker < selfplay_workers; ++worker)); do
        selfplay_gpus+="$gpu"
    done
fi

[[ "$gpu" =~ ^[0-9]$ ]] || {
    echo "error: GPU must be one GPU index, for example GPU=0" >&2
    exit 2
}
[[ "$selfplay_gpus" =~ ^[0-9]+$ ]] || {
    echo "error: GUIDED_SELFPLAY_GPU must be a GPU-index string such as 0000" >&2
    exit 2
}
(( ${#selfplay_gpus} == selfplay_workers )) || {
    echo "error: GUIDED_SELFPLAY_GPU must contain one index per self-play worker" >&2
    exit 2
}
[[ -f "$source_config" ]] || {
    echo "error: source config not found: $source_config" >&2
    exit 2
}

command=(
    env
    GUIDED_NUM_PLAYERS=3
    GUIDED_TOTAL_ITERATIONS="${GUIDED_TOTAL_ITERATIONS:-200}"
    GUIDED_BOOTSTRAP_ITERATIONS="${GUIDED_BOOTSTRAP_ITERATIONS:-25}"
    GUIDED_META_INTERVAL="${GUIDED_META_INTERVAL:-25}"
    GUIDED_GAMES_PER_ITERATION="${GUIDED_GAMES_PER_ITERATION:-2000}"
    GUIDED_TRAINING_STEPS="${GUIDED_TRAINING_STEPS:-500}"
    GUIDED_LEARNER_BATCH="${GUIDED_LEARNER_BATCH:-1024}"
    GUIDED_CURRENT_RATIO="${GUIDED_CURRENT_RATIO:-0.50}"
    GUIDED_HARD_RATIO="${GUIDED_HARD_RATIO:-0.15}"
    GUIDED_CCE_RATIO="${GUIDED_CCE_RATIO:-0.05}"
    GUIDED_HISTORY_RATIO="${GUIDED_HISTORY_RATIO:-0.30}"
    GUIDED_POPULATION_SIZE="${GUIDED_POPULATION_SIZE:-8}"
    GUIDED_CURRENT_SEAT_MIN=1
    GUIDED_CURRENT_SEAT_MAX=2
    GUIDED_HARD_TEMPERATURE="${GUIDED_HARD_TEMPERATURE:-0.20}"
    GUIDED_HARD_CONFIDENCE="${GUIDED_HARD_CONFIDENCE:-1.0}"
    GUIDED_EVAL_GAMES="${GUIDED_EVAL_GAMES:-20}"
    GUIDED_DEVIATION_GAMES="${GUIDED_DEVIATION_GAMES:-30}"
    GUIDED_EVAL_NOISE="${GUIDED_EVAL_NOISE:-true}"
    GUIDED_ADMISSION_GAIN="${GUIDED_ADMISSION_GAIN:-0.005}"
    GUIDED_ADMISSION_CONFIDENCE="${GUIDED_ADMISSION_CONFIDENCE:-1.5}"
    GUIDED_GENERALIST_ADMISSION=true
    GUIDED_MAX_SEAT_REGRESSION="${GUIDED_MAX_SEAT_REGRESSION:-0.10}"
    GUIDED_RANK_UTILITY_WEIGHT="${GUIDED_RANK_UTILITY_WEIGHT:-0.75}"
    GUIDED_SIMULATIONS="${GUIDED_SIMULATIONS:-50}"
    GUIDED_SEED="${GUIDED_SEED:-0}"
    GUIDED_GPU="$gpu"
    GUIDED_SELFPLAY_GPU="$selfplay_gpus"
    GUIDED_SELFPLAY_WORKERS="$selfplay_workers"
    GUIDED_SELFPLAY_BATCH="${GUIDED_SELFPLAY_BATCH:-32}"
    GUIDED_CPU_THREADS="${GUIDED_CPU_THREADS:-4}"
    GUIDED_EVAL_GPU="${GUIDED_EVAL_GPU:-$gpu}"
    GUIDED_EVAL_ACTORS="${GUIDED_EVAL_ACTORS:-1}"
    GUIDED_EVAL_BATCH="${GUIDED_EVAL_BATCH:-32}"
    GUIDED_EVAL_THREADS="${GUIDED_EVAL_THREADS:-3}"
    GUIDED_PORT="${GUIDED_PORT:-12039}"
    GUIDED_PERSISTENT_WORKERS=true
    tools/multiplayer-guided-pool.sh
    go3
    "$run_dir"
    "$source_config"
)

echo "Go3 JPSRO+Rank v2 training"
echo "run: $run_dir"
echo "GPU: $gpu; self-play GPUs: $selfplay_gpus; workers: $selfplay_workers"
echo "budget: ${GUIDED_TOTAL_ITERATIONS:-200} iterations x ${GUIDED_GAMES_PER_ITERATION:-2000} games x ${GUIDED_TRAINING_STEPS:-500} learner steps"
echo "mixture: ${GUIDED_CURRENT_RATIO:-0.50} current + ${GUIDED_HARD_RATIO:-0.15} hard + ${GUIDED_CCE_RATIO:-0.05} CCE + ${GUIDED_HISTORY_RATIO:-0.30} history"
echo "admission: noise=${GUIDED_EVAL_NOISE:-true}, max seat regression=${GUIDED_MAX_SEAT_REGRESSION:-0.10}"
echo "rank utility weight: ${GUIDED_RANK_UTILITY_WEIGHT:-0.75}"

if [[ "${DRY_RUN:-false}" == true || "${DRY_RUN:-false}" == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
