#!/bin/bash
set -euo pipefail

game=${1:-tictacmo}
end_iteration=${2:-8}
config_file=${3:-config/cdaz/tictacmo_smoke.cfg}
output_root=${4:-experiments/cdaz_smoke}
eval_games=${5:-60}
baseline_port=${BASELINE_PORT:-19001}
cdaz_port=${CDAZ_PORT:-19002}

if [[ ${game} != tictacmo ]]; then
    echo "This smoke script currently ships a paired TicTacMo config; pass a game-specific config for ${game}." >&2
fi

mkdir -p "${output_root}"

tools/quick-run.sh train "${game}" "${config_file}" "${end_iteration}" \
    -p "${baseline_port}" \
    -n "${output_root}/baseline" \
    -conf_str "actor_policy_target_type=visit:learner_policy_reference_loss_scale=0"

tools/quick-run.sh train "${game}" "${config_file}" "${end_iteration}" \
    -p "${cdaz_port}" \
    -n "${output_root}/cdaz" \
    -conf_str "actor_policy_target_type=certified_deviation:learner_policy_reference_loss_scale=0.05"

python3 tools/multiplayer-eval.py self-eval "${game}" "${output_root}/baseline" \
    --conf-file "${output_root}/baseline/baseline.cfg" \
    --interval 1 --games "${eval_games}" --no-noise \
    --output "${output_root}/baseline/self_eval"

python3 tools/multiplayer-eval.py self-eval "${game}" "${output_root}/cdaz" \
    --conf-file "${output_root}/cdaz/cdaz.cfg" \
    --interval 1 --games "${eval_games}" --no-noise \
    --output "${output_root}/cdaz/self_eval"

echo "Paired runs completed under ${output_root}."
