#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 GAME BASELINE_DIR END_ITER [OUTPUT_ROOT] [EVAL_INTERVAL] [EVAL_GAMES]" >&2
    echo "Example: $0 tictacmo /baseline/tictacmo_maxn_smoke_02 30 experiments/tictacmo_matched 10 200" >&2
    exit 1
}

[[ $# -ge 3 ]] || usage

game=$1
baseline_dir=$(readlink -f "$2")
end_iteration=$3
output_root=${4:-experiments/${game}_cdaz_matched}
eval_interval=${5:-10}
eval_games=${6:-200}
fixed_port=${FIXED_PORT:-19101}
cdaz_port=${CDAZ_PORT:-19102}

[[ -d ${baseline_dir} ]] || { echo "Baseline directory not found: ${baseline_dir}" >&2; exit 1; }
baseline_config=$(find "${baseline_dir}" -maxdepth 1 -type f -name '*.cfg' -print -quit)
[[ -n ${baseline_config} ]] || { echo "No baseline config found in ${baseline_dir}." >&2; exit 1; }
[[ -f ${baseline_dir}/sgf/${end_iteration}.sgf ]] || {
    echo "Baseline is not complete through iteration ${end_iteration}: missing sgf/${end_iteration}.sgf" >&2
    exit 1
}

initial_model_prefix=${baseline_dir}/model/weight_iter_0
[[ -f ${initial_model_prefix}.pt && -f ${initial_model_prefix}.pkl ]] || {
    echo "Baseline must contain weight_iter_0.pt and weight_iter_0.pkl." >&2
    exit 1
}

mkdir -p "${output_root}"

echo "Reuse completed current baseline: ${baseline_dir}"
echo "Train Fixed-MPAZ with the exact baseline config and initial checkpoint."
tools/quick-run.sh train "${game}" "${baseline_config}" "${end_iteration}" \
    -p "${fixed_port}" \
    -n "${output_root}/fixed_mpaz" \
    --initial_model_prefix "${initial_model_prefix}" \
    -conf_str "actor_policy_target_type=visit:learner_policy_reference_loss_scale=0"
cmp "${initial_model_prefix}.pt" "${output_root}/fixed_mpaz/model/weight_iter_0.pt"
cmp "${initial_model_prefix}.pkl" "${output_root}/fixed_mpaz/model/weight_iter_0.pkl"

echo "Train CD-AZ with the exact baseline config and initial checkpoint."
tools/quick-run.sh train "${game}" "${baseline_config}" "${end_iteration}" \
    -p "${cdaz_port}" \
    -n "${output_root}/cdaz" \
    --initial_model_prefix "${initial_model_prefix}" \
    -conf_str "actor_policy_target_type=certified_deviation:learner_policy_reference_loss_scale=0.05"
cmp "${initial_model_prefix}.pt" "${output_root}/cdaz/model/weight_iter_0.pt"
cmp "${initial_model_prefix}.pkl" "${output_root}/cdaz/model/weight_iter_0.pkl"

executable=build/${game}/minizero_${game}
fixed_config=${output_root}/fixed_mpaz/fixed_mpaz.cfg
cdaz_config=${output_root}/cdaz/cdaz.cfg

echo "Evaluate current, fixed, and CD-AZ checkpoints with the same executable and arena settings."
python3 tools/multiplayer-eval.py self-eval "${game}" "${baseline_dir}" \
    --conf-file "${baseline_config}" --interval "${eval_interval}" --games "${eval_games}" \
    --no-noise --resume --executable "${executable}" --output "${output_root}/current_mpaz_self_eval"
python3 tools/multiplayer-eval.py self-eval "${game}" "${output_root}/fixed_mpaz" \
    --conf-file "${fixed_config}" --interval "${eval_interval}" --games "${eval_games}" \
    --no-noise --resume --executable "${executable}" --output "${output_root}/fixed_mpaz/self_eval"
python3 tools/multiplayer-eval.py self-eval "${game}" "${output_root}/cdaz" \
    --conf-file "${cdaz_config}" --interval "${eval_interval}" --games "${eval_games}" \
    --no-noise --resume --executable "${executable}" --output "${output_root}/cdaz/self_eval"

echo "Matched experiment completed under ${output_root}."
