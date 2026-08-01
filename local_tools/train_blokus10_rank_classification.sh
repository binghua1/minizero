#!/usr/bin/env bash
set -e

CFG=${CFG:-blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50.cfg}
END_ITER=${1:-500}
TRAIN_DIR=${TRAIN_DIR:-blokus10_maxn_rank_classification_n50}
PORT=${PORT:-10031}
SP_GPU=${SP_GPU:-0000}
OP_GPU=${OP_GPU:-0}

# Keep self-play identical to the MaxN rank-regression experiment. Rank utility
# search is evaluated after training so the head representation is the only
# changed training variable.
tools/quick-run.sh train blokus10 "$CFG" "$END_ITER" \
  -n "$TRAIN_DIR" \
  -c 16 \
  -b 32 \
  --sp_gpu "$SP_GPU" \
  --op_gpu "$OP_GPU" \
  -p "$PORT" \
  -conf_str 'actor_multiplayer_search_type=maxn:actor_rank_utility_weight=0:actor_num_simulation=50:learner_rank_loss_scale=1:zero_disable_resign_ratio=1:zero_actor_intermediate_sequence_length=0' \
  # --sp_progress
