#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Reproducible launcher for the stochastic, higher-throughput Go3 run.
# Method-level mixture and training budget remain identical to v1; admission
# uses stochastic evaluation with more samples, and self-play uses batch 64.
export GUIDED_EVAL_NOISE="${GUIDED_EVAL_NOISE:-true}"
export GUIDED_EVAL_GAMES="${GUIDED_EVAL_GAMES:-100}"
export GUIDED_DEVIATION_GAMES="${GUIDED_DEVIATION_GAMES:-100}"
export GUIDED_MAX_SEAT_REGRESSION="${GUIDED_MAX_SEAT_REGRESSION:-0.05}"
export GUIDED_SELFPLAY_BATCH="${GUIDED_SELFPLAY_BATCH:-64}"
export GUIDED_PORT="${GUIDED_PORT:-12042}"

run_dir="${1:-runs/go3_jpsro_rank_l075_v3_b64_i200}"
exec tools/train-go3-jpsro-rank-v2.sh "$run_dir"
