#!/usr/bin/env bash
set -euo pipefail

# Resume-safe wrapper for every evaluation required by report_final.md.
# Completed results are reused; missing results are launched.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${GPU:-0}"
threads="${THREADS:-3}"
dry_run="${DRY_RUN:-false}"
confirm="${CONFIRM:-true}"

jpsro_run="${JPSRO_RUN:-runs/blokus10_jpsro_strong_300_w4b32}"
jpsro_step="${JPSRO_MODEL:-150000}"
jpsro_rank_weight="${JPSRO_RANK_WEIGHT:-0.25}"
jpsro_screen_games_per_seating="${JPSRO_SCREEN_GAMES_PER_SEATING:-50}"
jpsro_confirm_games="${JPSRO_CONFIRM_GAMES:-2002}"

echo "Blokus10 report evaluation wrapper"
echo "GPU=$gpu THREADS=$threads CONFIRM=$confirm DRY_RUN=$dry_run"
echo

echo "[1/4] A/B/D learning curve and matched i300 ablations"
GPU="$gpu" \
THREADS_PER_GPU="$threads" \
MODE=resume \
DRY_RUN="$dry_run" \
  tools/eval-blokus10-poolonly-ablation-300.sh

echo
echo "[2/4] Vanilla checkpoint cross-play and regression analysis"
# This is report evidence for non-monotonicity/regression, not a promise that
# a rock-paper-scissors cycle must exist.
EVAL_GPU="$gpu" \
EVAL_THREADS_PER_GPU="$threads" \
DRY_RUN="$dry_run" \
  tools/evaluate-checkpoint-cycles.sh blokus10_maxn_01

echo
echo "[3/4] Rank sensitivity, equal-compute scaling, held-out suite, and confirmation"
GPU="$gpu" \
THREADS="$threads" \
MODE=resume \
CONFIRM="$confirm" \
DRY_RUN="$dry_run" \
  tools/eval-blokus10-final-evidence-suite.sh

echo
echo "[4/4] Strong JPSRO final comparison"
jpsro_training="$jpsro_run/training"
jpsro_checkpoint="$jpsro_training/model/weight_iter_${jpsro_step}.pt"
if [[ ! -f "$jpsro_checkpoint" ]]; then
  echo "SKIP: strong JPSRO final checkpoint is not ready: $jpsro_checkpoint"
  echo "Rerun this wrapper after i300; completed steps 1-3 will resume without duplication."
  exit 0
fi

if [[ "$dry_run" == true || "$dry_run" == 1 ]]; then
  echo "DRY RUN: JPSRO rank(lambda=$jpsro_rank_weight) vs nohead, poolonly, poolrank"
  exit 0
fi

TRAIN_DIR="$jpsro_training" \
CANDIDATE_CFG="$jpsro_training/training.cfg" \
MODEL="$jpsro_step" \
BASELINE_MODEL=150000 \
SEARCHES=rank \
BASELINES="nohead poolonly poolrank" \
RANK_UTILITY_WEIGHT="$jpsro_rank_weight" \
OUTPUT_TAG=report_final \
SIMULATIONS=50 \
GAMES_PER_SEATING="$jpsro_screen_games_per_seating" \
GPU="$gpu" \
THREADS="$threads" \
MODE=resume \
  local_tools/eval_blokus10_balanced_population_rank.sh

if [[ "$confirm" == true || "$confirm" == 1 ]]; then
  TRAIN_DIR="$jpsro_training" \
  CANDIDATE_CFG="$jpsro_training/training.cfg" \
  MODEL="$jpsro_step" \
  BASELINE_MODEL=150000 \
  SEARCHES=rank \
  BASELINES="nohead poolonly poolrank" \
  RANK_UTILITY_WEIGHT="$jpsro_rank_weight" \
  OUTPUT_TAG=report_confirm \
  SIMULATIONS=50 \
  NUM_GAMES="$jpsro_confirm_games" \
  GPU="$gpu" \
  THREADS="$threads" \
  MODE=resume \
    local_tools/eval_blokus10_balanced_population_rank.sh
fi

python3 tools/summarize_blokus_evaluation.py \
  "$jpsro_training/evaluation" \
  --output "$jpsro_training/evaluation/report_summary"

echo
echo "Report evaluation wrapper complete."
echo "Main deck: $repo_root/report_final.md"
echo "Pool+Rank summary: $repo_root/blokus10_balanced_population_rank_nofilm_l0p75_n50/evaluation/final_evidence_summary"
echo "JPSRO summary: $repo_root/$jpsro_training/evaluation/report_summary"
