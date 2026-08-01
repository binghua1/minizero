#!/usr/bin/env bash
set -e

CFG=blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50.cfg

tools/quick-run.sh train blokus10 $CFG 500 \
  -n blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50 \
  --sp_gpu 00000 \
  -p 10031

  # -c 12 \