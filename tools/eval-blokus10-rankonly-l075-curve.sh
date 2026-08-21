#!/usr/bin/env bash
set -euo pipefail

# Evaluate the existing no-pool Rank-only checkpoints against the fixed
# Multiplayer AlphaZero i100 (50k-step) anchor. Re-running resumes safely.
# Run inside the MiniZero container from /workspace.

gpu="${GPU:-0}"
threads="${THREADS:-1}"
mode="${MODE:-resume}"
train_dir="blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50"
# The adaptive-training config contains legacy rank_adaptive-only keys that
# the current evaluation console does not accept. The TorchScript checkpoint
# carries the trained network; use the compatible fixed-rank evaluation config
# here and explicitly override population=false/search=lambda below.
cfg="blokus10_balanced_population_rank_nofilm_l0p75_n50/blokus10_balanced_population_rank_nofilm_l0p75_n50.cfg"

for step in $(seq 0 10000 150000); do
  echo "[$(date --iso-8601=seconds)] Rank-only step ${step}, search lambda=.75"
  TRAIN_DIR="$train_dir" \
  CANDIDATE_CFG="$cfg" \
  MODEL="$step" \
  BASELINE_MODEL=50000 \
  SEARCHES=rank \
  BASELINES=nohead \
  RANK_UTILITY_WEIGHT=0.75 \
  OUTPUT_TAG=rankonly_curve_l075 \
  SIMULATIONS=50 \
  GAMES_PER_SEATING=50 \
  GPU="$gpu" \
  THREADS="$threads" \
  MODE="$mode" \
  local_tools/eval_blokus10_balanced_population_rank.sh
done

echo "Rank-only curve results: ${train_dir}/evaluation/*_rank_rankonly_curve_l075_vs_nohead50000_*"
