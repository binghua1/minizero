#!/usr/bin/env bash
set -e

DIR=blokus10_maxn_01_head_rank_250000_rank_utility
MODEL=250000
SIM=50
GAMES=50
GPU=0
THREADS=2
LMAX=0.75
TAU=0.20
K=12

./scripts/build.sh blokus10

python3 tools/multiplayer-eval.py auto \
  blokus10 $DIR \
  --model $MODEL \
  --search-types maxn rank_adaptive \
  --rank-weight $LMAX \
  --rank-adaptive-gap-threshold $TAU \
  --rank-adaptive-gap-scale $K \
  --num-simulations $SIM \
  --noise \
  --games-per-seating $GAMES \
  --executable build/blokus10/minizero_blokus10 \
  --output $DIR/evaluation/rank_head_maxn_vs_rank_adaptive_lmax0p75_tau0p20_k12_n50_noise_700 \
  --gpu $GPU \
  --num_threads $THREADS \
  --overwrite
