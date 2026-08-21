#!/usr/bin/env bash
set -euo pipefail

# Final report evaluation for the validation-selected rank weight (lambda=.25)
# and the completed strong JPSRO network.  Every matchup is resume-safe.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${GPU:-12}"
threads="${THREADS:-1}"
mode="${MODE:-resume}"
games_per_seating="${GAMES_PER_SEATING:-50}"
rank_weight="${RANK_UTILITY_WEIGHT:-0.25}"
run_curves="${RUN_CURVES:-true}"
run_final="${RUN_FINAL:-true}"
curve_methods=" ${CURVE_METHODS:-poolrank jpsro} "
final_methods=" ${FINAL_METHODS:-poolrank jpsro} "
confirm="${CONFIRM:-true}"
dry_run="${DRY_RUN:-false}"

baseline_step=50000
final_step=150000
evaluator="local_tools/eval_blokus10_balanced_population_rank.sh"
poolrank_dir="blokus10_balanced_population_rank_nofilm_l0p75_n50"
jpsro_dir="runs/blokus10_jpsro_strong_300_w4b32/training"

run_eval() {
  local train_dir=$1 cfg=$2 step=$3 baselines=$4 tag=$5 num_games=${6:-}
  if [[ "$dry_run" == true || "$dry_run" == 1 ]]; then
    echo "DRY RUN: $train_dir step=$step vs [$baselines] tag=$tag games=${num_games:-$((games_per_seating * 14))}"
    return
  fi
  TRAIN_DIR="$train_dir" CANDIDATE_CFG="$cfg" MODEL="$step" \
  BASELINE_MODEL="${BASELINE_MODEL:-$final_step}" SEARCHES=rank \
  BASELINES="$baselines" RANK_UTILITY_WEIGHT="$rank_weight" \
  BASELINE_RANK_UTILITY_WEIGHT="$rank_weight" OUTPUT_TAG="$tag" \
  SIMULATIONS=50 GAMES_PER_SEATING="$games_per_seating" NUM_GAMES="$num_games" \
  GPU="$gpu" THREADS="$threads" MODE="$mode" "$evaluator"
}

[[ -f "$jpsro_dir/model/weight_iter_${final_step}.pt" ]] || {
  echo "error: strong JPSRO i300 checkpoint is missing" >&2
  exit 2
}

echo "Final lambda=.25 report evaluation"
echo "GPU=$gpu THREADS=$threads MODE=$mode"

if [[ "$run_curves" == true || "$run_curves" == 1 ]]; then
  echo "[1/2] Complete i0-i300 curves against fixed Vanilla i100"
  for iteration in $(seq 0 20 300); do
    step=$((iteration * 500))
    if [[ "$curve_methods" == *" poolrank "* ]]; then
      BASELINE_MODEL="$baseline_step" run_eval "$poolrank_dir" \
        "$poolrank_dir/$poolrank_dir.cfg" "$step" nohead curve_l025
    fi
    if [[ "$curve_methods" == *" jpsro "* ]]; then
      BASELINE_MODEL="$baseline_step" run_eval "$jpsro_dir" \
        "$jpsro_dir/training.cfg" "$step" nohead curve_l025
    fi
  done
fi

if [[ "$run_final" == true || "$run_final" == 1 ]]; then
  echo "[2/2] Locked final/held-out comparisons"
  if [[ "$confirm" == true || "$confirm" == 1 ]]; then
    # Go directly to confirmatory sample sizes for the headline pairs; do not
    # spend another 700-game screen on comparisons already chosen in advance.
    if [[ "$final_methods" == *" poolrank "* ]]; then
      run_eval "$poolrank_dir" "$poolrank_dir/$poolrank_dir.cfg" "$final_step" \
        "adaptivetrained oldpopulation" final_l025_heldout
      run_eval "$poolrank_dir" "$poolrank_dir/$poolrank_dir.cfg" "$final_step" \
        "nohead poolonly" final_l025_confirm 2002
    fi
    if [[ "$final_methods" == *" jpsro "* ]]; then
      run_eval "$jpsro_dir" "$jpsro_dir/training.cfg" "$final_step" \
        "nohead poolonly poolrank" final_l025_jpsro_confirm 2002
    fi
  else
    if [[ "$final_methods" == *" poolrank "* ]]; then
      run_eval "$poolrank_dir" "$poolrank_dir/$poolrank_dir.cfg" "$final_step" \
        "nohead poolonly adaptivetrained oldpopulation" final_l025_heldout
    fi
    if [[ "$final_methods" == *" jpsro "* ]]; then
      run_eval "$jpsro_dir" "$jpsro_dir/training.cfg" "$final_step" \
        "nohead poolonly poolrank" final_l025_jpsro
    fi
  fi
fi

if [[ "$dry_run" != true && "$dry_run" != 1 && ( "$run_curves" == true || "$run_curves" == 1 ) ]]; then
  python3 tools/plot-blokus10-final-l025-winrate.py
fi

echo "Done. Curve output: runs/blokus10_i100_anchor_winrate/final_l025"
