#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

start=${SWEEP_START:-20}
end=${SWEEP_END:-500}
interval=${SWEEP_INTERVAL:-20}
steps_per_iteration=${STEPS_PER_ITERATION:-500}
baseline_iteration=${BASELINE_ITERATION:-50}
games_per_seating=${GAMES_PER_SEATING:-50}
simulations=${SIMULATIONS:-50}
gpu=${GPU:-0123}
# The evaluator creates THREADS workers per listed GPU.  With GPU=0123 and
# THREADS=1 this gives four workers total, one on each GPU.
workers_per_gpu=${THREADS_PER_GPU:-1}
mode=${MODE:-overwrite}
methods=" ${SWEEP_METHODS:-old poolonly jpsro vanilla} "

evaluator=local_tools/eval_blokus10_balanced_population_rank.sh

die() { echo "error: $*" >&2; exit 1; }

[[ -x "$evaluator" ]] || die "missing evaluator: $evaluator"
[[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$interval" =~ ^[1-9][0-9]*$ ]] \
  || die "SWEEP_START, SWEEP_END, and SWEEP_INTERVAL must be integers"
(( start <= end )) || die "SWEEP_START must not exceed SWEEP_END"

baseline_step=$((baseline_iteration * steps_per_iteration))

run_sweep() {
  local label=$1
  local train_dir=$2
  local config=$3
  local search=$4

  echo
  echo "========== $label =========="
  echo "candidate: i${start}..i${end}, interval=${interval}"
  echo "baseline: multiplayer AlphaZero i${baseline_iteration} (step ${baseline_step})"
  echo "search: candidate=${search}, baseline=maxn"

  local iteration step
  for ((iteration=start; iteration<=end; iteration+=interval)); do
    step=$((iteration * steps_per_iteration))
    echo
    echo "[$label] i${iteration} (step ${step}) vs AlphaZero i${baseline_iteration}"
    TRAIN_DIR="$train_dir" \
    CANDIDATE_CFG="$config" \
    MODEL="$step" \
    BASELINE_MODEL="$baseline_step" \
    SEARCHES="$search" \
    BASELINES=nohead \
    SIMULATIONS="$simulations" \
    GAMES_PER_SEATING="$games_per_seating" \
    GPU="$gpu" \
    THREADS="$workers_per_gpu" \
    MODE="$mode" \
      "$evaluator"
  done
}

# This checkpoint family has a trained rank head, so evaluate it with Rank
# Utility search while the fixed AlphaZero baseline continues to use MaxN.
if [[ "$methods" == *" old "* ]]; then
  run_sweep \
    "Old Opponent Pool + Rank" \
    "blokus10_balanced_population_rank_nofilm_l0p75_n50" \
    "blokus10_balanced_population_rank_nofilm_l0p75_n50/blokus10_balanced_population_rank_nofilm_l0p75_n50.cfg" \
    "rank"
fi

# Clean population-only ablation: historical opponents with MaxN and no rank
# head.  This isolates the contribution of the opponent pool from Rank search.
if [[ "$methods" == *" poolonly "* ]]; then
  run_sweep \
    "Opponent Pool Only + MaxN" \
    "blokus10_balanced_population_maxn_nofilm_n50" \
    "blokus10_balanced_population_maxn_nofilm_n50/blokus10_balanced_population_maxn_nofilm_n50.cfg" \
    "maxn"
fi

# The existing Guided/JPSRO run has no rank head; evaluate it with MaxN.
if [[ "$methods" == *" jpsro "* ]]; then
  run_sweep \
    "JPSRO Guided Pool + MaxN" \
    "runs/blokus10_guided_pool_500_s0/training" \
    "runs/blokus10_guided_pool_500_s0/training/training.cfg" \
    "maxn"
fi

# Plain multiplayer AlphaZero learning curve against the same fixed i100
# AlphaZero anchor.  This is the no-opponent-pool MaxN reference curve.
if [[ "$methods" == *" vanilla "* ]]; then
  run_sweep \
    "Vanilla AlphaZero + MaxN" \
    "blokus10_maxn_01" \
    "blokus10_maxn_01/blokus10_maxn_01.cfg" \
    "maxn"
fi

echo
echo "All Blokus10 sweeps completed."
PLOT_END=${PLOT_END:-300} python3 tools/plot-blokus10-anchor-winrate.py
