#!/usr/bin/env bash
set -euo pipefail

# Final evidence suite for Rank-Aware Balanced Population Training.
#
# Design:
#   1. Rank-weight sensitivity on a validation opponent (all points reported).
#   2. Equal-compute MCTS scaling against the primary Vanilla baseline.
#   3. A held-out multi-opponent suite at the predeclared n=50, lambda=0.75.
#   4. MaxN-vs-Rank inference ablation on the same trained network.
#   5. Optional 2,002-game confirmation of the held-out suite.
#
# This script never gives the candidate more MCTS simulations than its
# reference inside a matchup, and it uses resume-safe output directories.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${GPU:-0}"
threads="${THREADS:-3}"
mode="${MODE:-resume}"

candidate_dir="${CANDIDATE_DIR:-blokus10_balanced_population_rank_nofilm_l0p75_n50}"
candidate_cfg="${CANDIDATE_CFG:-${candidate_dir}/$(basename "$candidate_dir").cfg}"
candidate_model="${MODEL:-150000}"
baseline_model="${BASELINE_MODEL:-150000}"

screen_games_per_seating="${SCREEN_GAMES_PER_SEATING:-50}"
confirm_games="${CONFIRM_GAMES:-2002}"
confirm="${CONFIRM:-false}"
dry_run="${DRY_RUN:-false}"

lambda_grid="${LAMBDA_GRID:-0 0.25 0.50 0.75 1.0}"
simulation_grid="${SIMULATION_GRID:-25 50 100 200}"
locked_lambda="${LOCKED_LAMBDA:-0.75}"
locked_simulations="${LOCKED_SIMULATIONS:-50}"

# rankclass is used only as a validation/sensitivity opponent.  The headline
# held-out suite does not use it for hyperparameter selection.
validation_baseline="${VALIDATION_BASELINE:-rankclass}"
test_baselines="${TEST_BASELINES:-nohead poolonly adaptivetrained oldpopulation}"
search_ablation_baselines="${SEARCH_ABLATION_BASELINES:-nohead poolonly}"
confirm_baselines="${CONFIRM_BASELINES:-nohead poolonly}"

evaluator="local_tools/eval_blokus10_balanced_population_rank.sh"

die() {
  echo "error: $*" >&2
  exit 2
}

tokenize_decimal() {
  printf '%s' "$1" | sed 's/[.]/p/g'
}

run_eval() {
  local searches=$1
  local baselines=$2
  local simulations=$3
  local rank_weight=$4
  local output_tag=$5
  local games_per_seating=$6
  local num_games=${7:-}

  if [[ "$dry_run" == true || "$dry_run" == 1 ]]; then
    echo "DRY RUN: search=$searches baselines=[$baselines] n=$simulations lambda=$rank_weight tag=$output_tag games=${num_games:-$((games_per_seating * 14))}"
    return
  fi

  TRAIN_DIR="$candidate_dir" \
  CANDIDATE_CFG="$candidate_cfg" \
  MODEL="$candidate_model" \
  BASELINE_MODEL="$baseline_model" \
  SEARCHES="$searches" \
  BASELINES="$baselines" \
  SIMULATIONS="$simulations" \
  RANK_UTILITY_WEIGHT="$rank_weight" \
  OUTPUT_TAG="$output_tag" \
  GAMES_PER_SEATING="$games_per_seating" \
  NUM_GAMES="$num_games" \
  GPU="$gpu" \
  THREADS="$threads" \
  MODE="$mode" \
    "$evaluator"
}

[[ -x "$evaluator" ]] || die "missing evaluator: $evaluator"
[[ -f "$candidate_cfg" ]] || die "missing candidate config: $candidate_cfg"
[[ -f "$candidate_dir/model/weight_iter_${candidate_model}.pt" ]] \
  || die "missing candidate checkpoint: $candidate_model"

echo "Blokus10 final evidence suite"
echo "candidate: $candidate_dir @ $candidate_model"
echo "matched baseline step: $baseline_model"
echo "GPU=$gpu THREADS=$threads MODE=$mode"
echo "screen: $((screen_games_per_seating * 14)) games/comparison"
echo "locked setting: lambda=$locked_lambda simulations=$locked_simulations"
echo "held-out baselines: $test_baselines"
echo

echo "[1/5] Rank-weight sensitivity on validation baseline: $validation_baseline"
for lambda in $lambda_grid; do
  lambda_token="$(tokenize_decimal "$lambda")"
  # The engine intentionally rejects rank search with zero rank utility.
  # lambda=0 is exactly the same-network MaxN control: the rank head makes no
  # contribution to tree search.
  if [[ "$lambda" == 0 || "$lambda" == 0.0 || "$lambda" == 0.00 ]]; then
    run_eval maxn "$validation_baseline" "$locked_simulations" 0 \
      "lambda_${lambda_token}_no_rank_utility" "$screen_games_per_seating"
  else
    run_eval rank "$validation_baseline" "$locked_simulations" "$lambda" \
      "lambda_${lambda_token}" "$screen_games_per_seating"
  fi
done

echo "[2/5] Equal-compute MCTS scaling against Vanilla"
for simulations in $simulation_grid; do
  # n=50 at lambda=0.75 is also the predeclared headline configuration;
  # tagging keeps the sensitivity analysis separate from confirmatory output.
  run_eval rank nohead "$simulations" "$locked_lambda" \
    "scaling_lambda_$(tokenize_decimal "$locked_lambda")" \
    "$screen_games_per_seating"
done

echo "[3/5] Held-out multi-opponent suite at the locked setting"
run_eval rank "$test_baselines" "$locked_simulations" "$locked_lambda" \
  "heldout" "$screen_games_per_seating"

echo "[4/5] Search ablation on the same trained network"
run_eval maxn "$search_ablation_baselines" "$locked_simulations" 0 \
  "search_ablation" "$screen_games_per_seating"

if [[ "$confirm" == true || "$confirm" == 1 ]]; then
  echo "[5/5] Confirmatory primary pairs: ${confirm_games} games/comparison"
  run_eval rank "$confirm_baselines" "$locked_simulations" "$locked_lambda" \
    "confirm" "$screen_games_per_seating" "$confirm_games"
else
  echo "[5/5] Confirmation skipped (set CONFIRM=true to run it)"
fi

if [[ "$dry_run" == true || "$dry_run" == 1 ]]; then
  echo
  echo "DRY RUN complete; no evaluations were launched."
  exit 0
fi

summary_dir="$candidate_dir/evaluation/final_evidence_summary"
python3 tools/summarize_blokus_evaluation.py \
  "$candidate_dir/evaluation" \
  --output "$summary_dir"

echo
echo "All requested evaluations completed."
echo "Raw results: $repo_root/$candidate_dir/evaluation"
echo "Summary CSV: $repo_root/$summary_dir/evaluation_summary_with_ci.csv"
echo "Summary report: $repo_root/$summary_dir/evaluation_report.md"
echo
echo "Primary report claims must use the heldout/confirm rows."
echo "Lambda and simulation sweeps are sensitivity analyses; report every point."
