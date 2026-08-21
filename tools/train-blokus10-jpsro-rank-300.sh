#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${1:-runs/blokus10_jpsro_rank_300_s0}"
source_config="${SOURCE_CONFIG:-blokus10_balanced_population_rank_nofilm_l0p75_n50/blokus10_balanced_population_rank_nofilm_l0p75_n50.cfg}"
train_gpus="${TRAIN_GPUS:-12}"
train_gpus="${train_gpus//,/}"
selfplay_gpus="${GUIDED_SELFPLAY_GPU:-$train_gpus}"
selfplay_gpus="${selfplay_gpus//,/}"
selfplay_workers="${GUIDED_SELFPLAY_WORKERS:-${#selfplay_gpus}}"
eval_gpu="${GUIDED_EVAL_GPU:-${train_gpus:0:1}}"
port="${GUIDED_PORT:-11034}"

[[ "$train_gpus" =~ ^[0-9]+$ ]] || {
    echo "error: TRAIN_GPUS must be a GPU string such as 12 or 0123" >&2
    exit 2
}
[[ "$selfplay_gpus" =~ ^[0-9]+$ ]] || {
    echo "error: GUIDED_SELFPLAY_GPU must be a GPU string" >&2
    exit 2
}
(( ${#selfplay_gpus} == selfplay_workers )) || {
    echo "error: one self-play GPU index is required per worker" >&2
    exit 2
}
[[ -f "$source_config" ]] || {
    echo "error: source config not found: $source_config" >&2
    exit 2
}

# Performance-oriented JPSRO-guided training.  The final CURRENT network is
# the primary deployment target; the restricted-population CCE remains a
# secondary output/certificate.
command=(
    env
    GUIDED_TOTAL_ITERATIONS="${GUIDED_TOTAL_ITERATIONS:-300}"
    GUIDED_BOOTSTRAP_ITERATIONS="${GUIDED_BOOTSTRAP_ITERATIONS:-50}"
    GUIDED_META_INTERVAL="${GUIDED_META_INTERVAL:-25}"
    GUIDED_GAMES_PER_ITERATION="${GUIDED_GAMES_PER_ITERATION:-2000}"
    GUIDED_TRAINING_STEPS="${GUIDED_TRAINING_STEPS:-500}"
    GUIDED_LEARNER_BATCH="${GUIDED_LEARNER_BATCH:-1024}"
    GUIDED_CURRENT_RATIO="${GUIDED_CURRENT_RATIO:-0.60}"
    GUIDED_HARD_RATIO="${GUIDED_HARD_RATIO:-0.20}"
    GUIDED_CCE_RATIO="${GUIDED_CCE_RATIO:-0.05}"
    GUIDED_HISTORY_RATIO="${GUIDED_HISTORY_RATIO:-0.15}"
    GUIDED_POPULATION_SIZE="${GUIDED_POPULATION_SIZE:-8}"
    GUIDED_CURRENT_SEAT_MIN="${GUIDED_CURRENT_SEAT_MIN:-1}"
    GUIDED_CURRENT_SEAT_MAX="${GUIDED_CURRENT_SEAT_MAX:-3}"
    GUIDED_HARD_TEMPERATURE="${GUIDED_HARD_TEMPERATURE:-0.20}"
    GUIDED_HARD_CONFIDENCE="${GUIDED_HARD_CONFIDENCE:-1.0}"
    GUIDED_EVAL_GAMES="${GUIDED_EVAL_GAMES:-20}"
    GUIDED_DEVIATION_GAMES="${GUIDED_DEVIATION_GAMES:-100}"
    GUIDED_EVAL_NOISE="${GUIDED_EVAL_NOISE:-true}"
    GUIDED_ADMISSION_GAIN="${GUIDED_ADMISSION_GAIN:-0.01}"
    GUIDED_ADMISSION_CONFIDENCE="${GUIDED_ADMISSION_CONFIDENCE:-2.0}"
    GUIDED_GENERALIST_ADMISSION="${GUIDED_GENERALIST_ADMISSION:-true}"
    GUIDED_MAX_SEAT_REGRESSION="${GUIDED_MAX_SEAT_REGRESSION:-0.03}"
    GUIDED_SIMULATIONS="${GUIDED_SIMULATIONS:-50}"
    GUIDED_SEED="${GUIDED_SEED:-0}"
    GUIDED_GPU="$train_gpus"
    GUIDED_SELFPLAY_GPU="$selfplay_gpus"
    GUIDED_SELFPLAY_WORKERS="$selfplay_workers"
    GUIDED_SELFPLAY_BATCH="${GUIDED_SELFPLAY_BATCH:-64}"
    GUIDED_CPU_THREADS="${GUIDED_CPU_THREADS:-4}"
    GUIDED_EVAL_GPU="$eval_gpu"
    GUIDED_EVAL_BATCH="${GUIDED_EVAL_BATCH:-64}"
    GUIDED_EVAL_THREADS="${GUIDED_EVAL_THREADS:-4}"
    GUIDED_PORT="$port"
    GUIDED_PERSISTENT_WORKERS="${GUIDED_PERSISTENT_WORKERS:-true}"
    tools/multiplayer-guided-pool.sh
    blokus10
    "$run_dir"
    "$source_config"
)

echo "Blokus10 JPSRO-guided + Rank training"
echo "run: $run_dir"
echo "training GPUs: $train_gpus; self-play GPUs: $selfplay_gpus; evaluation GPU: $eval_gpu"
echo "budget: 300 iterations x 2000 games x 500 learner steps (unless overridden)"
echo "mixture: ${GUIDED_CURRENT_RATIO:-0.60} current + ${GUIDED_HARD_RATIO:-0.20} hard + ${GUIDED_CCE_RATIO:-0.05} CCE + ${GUIDED_HISTORY_RATIO:-0.15} history"
echo "rank utility weight: ${GUIDED_RANK_UTILITY_WEIGHT:-source config}"

if [[ "${DRY_RUN:-false}" == true || "${DRY_RUN:-false}" == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
