#!/usr/bin/env bash
set -euo pipefail

# JPSRO i300 (150k learner steps) against selected earlier JPSRO checkpoints.
# Run inside the MiniZero container from /workspace. Every fight resumes safely.

cd /workspace

training_dir=/workspace/runs/blokus10_jpsro_strong_300_w4b32/training
config_file="$training_dir/training.cfg"
output_root=/workspace/runs/blokus10_jpsro_l075_i300_stability
games="${GAMES:-700}"
gpu="${GPU:-0}"
threads="${THREADS:-1}"

mkdir -p "$output_root"

for earlier in ${EARLIER_STEPS:-80000 120000}; do
    python3 tools/multiplayer-eval.py model-fight blokus10 \
        "$training_dir/model/weight_iter_${earlier}.pt" \
        "$training_dir/model/weight_iter_150000.pt" \
        --conf-file-a "$config_file" \
        --conf-file-b "$config_file" \
        --names "iter_${earlier}" iter_150000 \
        --games "$games" \
        --output "$output_root/${earlier}_vs_150000" \
        --search-type rank \
        --num-simulations 50 \
        --noise \
        --seed "$((20260817 + earlier))" \
        -g "$gpu" \
        --num_threads "$threads" \
        --conf-str actor_rank_utility_weight=0.75 \
        --resume
done

echo "JPSRO i300-vs-history results: $output_root"
