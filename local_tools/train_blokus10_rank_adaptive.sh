#!/usr/bin/env bash
set -e

CFG=blokus10_maxn_01_head_rank_250000_rank_utility/blokus10_maxn_01_head_rank_250000_rank_utility.cfg

tools/quick-run.sh train blokus10 $CFG 500 \
  -n blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50 \
  -c 16 \
  -b 32 \
  --sp_gpu 3333 \
  --op_gpu 3 \
  -p 10031 \
  -conf_str 'actor_multiplayer_search_type=rank_adaptive:actor_rank_utility_weight=0.75:actor_rank_adaptive_gap_threshold=0.20:actor_rank_adaptive_gap_scale=12:actor_num_simulation=50:learner_rank_loss_scale=1:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0' \
  --sp_progress
