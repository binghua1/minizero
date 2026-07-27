#!/usr/bin/env bash

set -euo pipefail

cd /workspace

output="blokus10_maxn_01/checkpoint_sweep/reference_250000_step_1000_n50_stochastic"
log="${output}.run.log"
gpu="${GPU:-0}"
threads="${NUM_THREADS:-1}"

mkdir -p "${output}"

# A killed arena leaves a PID lock behind. Only remove locks whose process no
# longer exists in this container; an active lock means another sweep is live.
while IFS= read -r -d '' lock; do
    lock_pid="$(tr -d '[:space:]' < "${lock}")"
    if [[ "${lock_pid}" =~ ^[0-9]+$ ]] && kill -0 "${lock_pid}" 2>/dev/null; then
        echo "active arena lock: ${lock} (PID ${lock_pid})" >&2
        exit 1
    fi
    unlink "${lock}"
done < <(find "${output}" -mindepth 2 -maxdepth 2 -type f -name arena.lock -print0)

exec >>"${log}" 2>&1

echo
echo "[$(date --iso-8601=seconds)] resuming stochastic checkpoint sweep on GPU ${gpu} with ${threads} worker(s)"

exec python3 tools/multiplayer-eval.py checkpoint-sweep \
    blokus10 \
    blokus10_maxn_01 \
    --reference 250000 \
    --step 1000 \
    --games 100 \
    --search-type maxn \
    --num-simulations 50 \
    --noise \
    --conf-str=actor_select_action_by_count=false:actor_select_action_by_softmax_count=true:actor_use_random_rotation_features=false \
    --seed 20260724 \
    --output "${output}" \
    --resume \
    -g "${gpu}" \
    --num_threads "${threads}"
