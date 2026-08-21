#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${GPU:-12}"
threads="${THREADS_PER_GPU:-1}"
mode="${MODE:-resume}"
simulations="${SIMULATIONS:-50}"
games_per_seating="${GAMES_PER_SEATING:-50}"

pool_dir="blokus10_balanced_population_maxn_nofilm_n50"
pool_cfg="$pool_dir/$pool_dir.cfg"
rank_dir="blokus10_balanced_population_rank_nofilm_l0p75_n50"
rank_cfg="$rank_dir/$rank_dir.cfg"
evaluator="local_tools/eval_blokus10_balanced_population_rank.sh"

for required in \
    "$pool_dir/model/weight_iter_150000.pt" \
    "$rank_dir/model/weight_iter_150000.pt" \
    "blokus10_maxn_01/model/weight_iter_150000.pt" \
    "$pool_cfg" "$rank_cfg" "$evaluator"; do
    [[ -e "$required" ]] || {
        echo "error: missing required file: $required" >&2
        exit 2
    }
done

echo "Blokus10 300-iteration pool-only ablation"
echo "GPU=$gpu, threads/GPU=$threads, simulations=$simulations"
echo "Each comparison uses $((games_per_seating * 14)) seat-balanced games."
echo
echo "1/3 Pool-only learning curve: i0-i300 vs fixed AlphaZero i100 anchor"
echo "2/3 Matched final: Pool-only i300 vs Vanilla i300"
echo "3/3 Matched final: Pool+Rank i300 vs Pool-only i300"

if [[ "${DRY_RUN:-false}" == true || "${DRY_RUN:-false}" == 1 ]]; then
    echo "DRY RUN: no evaluations were launched"
    exit 0
fi

# Generate the missing fourth curve and rewrite the shared CSV/figures with
# all complete methods.  The sweep script resumes completed pair directories.
SWEEP_START=0 \
SWEEP_END=300 \
SWEEP_INTERVAL=20 \
STEPS_PER_ITERATION=500 \
BASELINE_ITERATION=100 \
GAMES_PER_SEATING="$games_per_seating" \
SIMULATIONS="$simulations" \
GPU="$gpu" \
THREADS_PER_GPU="$threads" \
MODE="$mode" \
SWEEP_METHODS=poolonly \
    tools/eval-blokus10-i50-two-sweeps.sh

# A vs B at the same 300-iteration budget: the pure opponent-pool effect.
TRAIN_DIR="$pool_dir" \
CANDIDATE_CFG="$pool_cfg" \
MODEL=150000 \
BASELINE_MODEL=150000 \
SEARCHES=maxn \
BASELINES=nohead \
SIMULATIONS="$simulations" \
GAMES_PER_SEATING="$games_per_seating" \
GPU="$gpu" \
THREADS="$threads" \
MODE="$mode" \
    "$evaluator"

# B vs D at the same 300-iteration budget: the incremental Rank effect once
# historical-opponent training is already present.
TRAIN_DIR="$rank_dir" \
CANDIDATE_CFG="$rank_cfg" \
MODEL=150000 \
BASELINE_MODEL=150000 \
SEARCHES=rank \
BASELINES=poolonly \
SIMULATIONS="$simulations" \
GAMES_PER_SEATING="$games_per_seating" \
GPU="$gpu" \
THREADS="$threads" \
MODE="$mode" \
    "$evaluator"

# Rebuild after every evaluation so the shared artifact is always current.
PLOT_END=300 python3 tools/plot-blokus10-anchor-winrate.py

echo
echo "Shared learning-curve results: $repo_root/runs/blokus10_i100_anchor_winrate"
echo "Matched A-vs-B result: $repo_root/$pool_dir/evaluation/balancedrank150000_maxn_vs_nohead150000_maxn_n${simulations}_noise_$((games_per_seating * 14))"
echo "Matched D-vs-B result: $repo_root/$rank_dir/evaluation/balancedrank150000_rank_vs_poolonly150000_maxn_n${simulations}_noise_$((games_per_seating * 14))"
