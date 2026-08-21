#!/usr/bin/env bash
set -euo pipefail

# Matched checkpoint cross-play for before/after stability evidence.
# MaxN is deliberately shared by all methods to isolate training stability.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${GPU:-12}"
threads="${THREADS:-1}"
games="${GAMES_PER_PAIR:-700}"
checkpoints="${CHECKPOINTS:-30000 60000 90000 120000 150000}"
methods=" ${METHODS:-vanilla pool_only pool_rank strong_jpsro} "
dry_run="${DRY_RUN:-false}"
out_root="runs/blokus10_common_stability"

run_one() {
  local label=$1 train_dir=$2 cfg=$3
  echo "[$label] $train_dir"
  CHECKPOINTS="$checkpoints" GAMES_PER_PAIR="$games" EVAL_GPU="$gpu" \
  EVAL_THREADS_PER_GPU="$threads" EVAL_SIMULATIONS=50 EVAL_NOISE=true \
  CONFIG_FILE="$cfg" OUTPUT_DIR="$out_root/$label" DRY_RUN="$dry_run" \
    tools/evaluate-checkpoint-cycles.sh "$train_dir"
}

if [[ "$methods" == *" vanilla "* ]]; then
  run_one vanilla blokus10_maxn_01 blokus10_maxn_01/blokus10_maxn_01.cfg
fi
if [[ "$methods" == *" pool_only "* ]]; then
  run_one pool_only blokus10_balanced_population_maxn_nofilm_n50 \
    blokus10_balanced_population_maxn_nofilm_n50/blokus10_balanced_population_maxn_nofilm_n50.cfg
fi
if [[ "$methods" == *" pool_rank "* ]]; then
  run_one pool_rank blokus10_balanced_population_rank_nofilm_l0p75_n50 \
    blokus10_balanced_population_rank_nofilm_l0p75_n50/blokus10_balanced_population_rank_nofilm_l0p75_n50.cfg
fi
if [[ "$methods" == *" strong_jpsro "* ]]; then
  run_one strong_jpsro runs/blokus10_jpsro_strong_300_w4b32/training \
    runs/blokus10_jpsro_strong_300_w4b32/training/training.cfg
fi

if [[ "$dry_run" != true && "$dry_run" != 1 ]]; then
  python3 tools/plot-blokus10-stability-comparison.py "$out_root"
fi

echo "Done. Combined results: $out_root"
