#!/usr/bin/env bash
set -euo pipefail

# Run inside the container from /workspace.
# Usage: ./local_tools/train_blokus10_balanced_population_rank.sh 500

END_ITER=${1:-500}
CFG=${CFG:-blokus10_balanced_population_rank_l0p75_n50.cfg}
TRAIN_DIR=${TRAIN_DIR:-blokus10_balanced_population_rank_l0p75_n50}
PORT=${PORT:-10042}
SP_GPU=${SP_GPU:-0000}
OP_GPU=${OP_GPU:-0}
CPU_THREAD_PER_GPU=${CPU_THREAD_PER_GPU:-16}
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ ! -f "$CFG" ]]; then
  echo "Missing cfg: $CFG"
  exit 1
fi

if [[ ! -x build/blokus10/minizero_blokus10 ]]; then
  echo "Missing build/blokus10/minizero_blokus10"
  echo "Build first: ./scripts/build.sh blokus10"
  exit 1
fi

for key in actor_rank_utility_weight learner_rank_loss_scale zero_population_balance_seats nn_use_rank_head; do
  if ! grep -q "$key" minizero/config/configuration.cpp; then
    echo "Current source does not support $key; switch to feat/multiplayer-balanced-population-rank and rebuild."
    exit 1
  fi
done

tools/quick-run.sh train blokus10 "$CFG" "$END_ITER" \
  -n "$TRAIN_DIR" \
  -c "$CPU_THREAD_PER_GPU" \
  -b "$BATCH_SIZE" \
  --sp_gpu "$SP_GPU" \
  --op_gpu "$OP_GPU" \
  -p "$PORT"
