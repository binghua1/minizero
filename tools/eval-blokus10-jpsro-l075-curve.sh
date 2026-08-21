#!/usr/bin/env bash
set -euo pipefail

# Evaluate the completed Strong JPSRO checkpoints with rank-search lambda=.75
# against the fixed Multiplayer AlphaZero i100 (50k-step) anchor.
# Run inside the MiniZero container from /workspace. Re-running resumes safely.

gpu="${GPU:-1}"
threads="${THREADS:-1}"
mode="${MODE:-resume}"
train_dir="runs/blokus10_jpsro_strong_300_w4b32/training"
cfg="${train_dir}/training.cfg"

for step in $(seq 0 10000 150000); do
  echo "[$(date --iso-8601=seconds)] Strong JPSRO step ${step}, lambda=.75"
  TRAIN_DIR="$train_dir" \
  CANDIDATE_CFG="$cfg" \
  MODEL="$step" \
  BASELINE_MODEL=50000 \
  SEARCHES=rank \
  BASELINES=nohead \
  RANK_UTILITY_WEIGHT=0.75 \
  OUTPUT_TAG=curve_l075 \
  SIMULATIONS=50 \
  GAMES_PER_SEATING=50 \
  GPU="$gpu" \
  THREADS="$threads" \
  MODE="$mode" \
  local_tools/eval_blokus10_balanced_population_rank.sh
done

python3 tools/plot-blokus10-final-l025-winrate.py

echo "Curve results: ${train_dir}/evaluation/*_rank_curve_l075_vs_nohead50000_*"
echo "Updated plot: runs/blokus10_i100_anchor_winrate/final_l025/winrate_vs_alphazero_i100_final_l025.png"
