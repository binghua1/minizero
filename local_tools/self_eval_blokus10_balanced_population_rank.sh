#!/usr/bin/env bash
set -euo pipefail

# Run inside the container from /workspace.
# Usage: ./local_tools/self_eval_blokus10_balanced_population_rank.sh

TRAIN_DIR=${TRAIN_DIR:-blokus10_balanced_population_rank_l0p75_n50}
CFG=${CFG:-$TRAIN_DIR/blokus10_balanced_population_rank_l0p75_n50.cfg}
START=${START:-0}
INTERVAL=${INTERVAL:-10}
GAMES=${GAMES:-200}
GPU=${GPU:-0}
NUM_THREADS=${NUM_THREADS:-1}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
EXECUTABLE=${EXECUTABLE:-build/blokus10/minizero_blokus10}
MODE=${MODE:-resume}

python3 tools/multiplayer-eval.py self-eval \
  blokus10 "$TRAIN_DIR" \
  --conf-file "$CFG" \
  --interval "$INTERVAL" \
  --games "$GAMES" \
  --start-index "$START" \
  --output "$TRAIN_DIR/self_eval_rank_l0p75" \
  --search-type rank \
  --num-simulations 50 \
  --noise \
  --executable "$EXECUTABLE" \
  -g "$GPU" \
  --num_threads "$NUM_THREADS" \
  --omp-num-threads "$OMP_NUM_THREADS" \
  -conf_str 'zero_use_population=false:actor_rank_utility_weight=0.75' \
  "--$MODE"
