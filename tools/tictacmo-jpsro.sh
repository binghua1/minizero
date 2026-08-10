#!/bin/bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage: tools/tictacmo-jpsro.sh RUN_DIR CONFIG.cfg

Train one from-scratch, fixed-budget TicTacMo JPSRO run. The default generation
boundaries are 5,15,30. Override them and other settings with JPSRO_* variables.
Re-running the same command resumes an interrupted run.
EOF
}

die() { echo "error: $*" >&2; exit 1; }

latest_model() {
    local model
    model=$(find "$training_dir/model" -maxdepth 1 -name 'weight_iter_*.pt' -print 2>/dev/null | sort -V | tail -n 1)
    [[ -n "$model" ]] || die "no checkpoint in $training_dir/model"
    readlink -f "$model"
}

completed_iterations() {
    local count
    if [[ ! -d "$training_dir/model" ]]; then
        echo 0
        return
    fi
    count=$(find "$training_dir/model" -maxdepth 1 -name 'weight_iter_*.pt' -print 2>/dev/null | wc -l)
    echo $((count > 0 ? count - 1 : 0))
}

policy_registered() {
    local policy_id=$1
    [[ -f "$meta_dir/policies.json" ]] && grep -q "\"policy_id\": \"$policy_id\"" "$meta_dir/policies.json"
}

train_to() {
    local end_iteration=$1
    local profile_file=${2:-}
    local replay_start=${3:-1}
    if (( $(completed_iterations) >= end_iteration )); then
        echo "training already reached iteration $end_iteration"
        return
    fi

    local conf="$common_conf:zero_use_jpsro=false"
    if [[ -n "$profile_file" ]]; then
        conf="$common_conf:zero_use_jpsro=true:zero_jpsro_profile_file=$profile_file:zero_jpsro_replay_start_iteration=$replay_start"
    fi
    local args=(tools/quick-run.sh train tictacmo "$config" "$end_iteration"
                -n "$training_dir" -g "$gpu" -p "$port"
                -b "$selfplay_batch" -c "$cpu_threads" -conf_str "$conf")
    [[ "$sp_gpu" != "$gpu" ]] && args+=(--sp_gpu "$sp_gpu")
    [[ -d "$training_dir" ]] && args+=(--continue-training)
    "${args[@]}"
}

evaluate_profiles() {
    local output_dir=$1
    local candidate=${2:-}
    local manifest="$output_dir/arena.json"
    mkdir -p "$output_dir"
    if [[ ! -f "$manifest" ]]; then
        local args=(python3 tools/jpsro.py make-eval "$meta_dir" "$manifest"
                    --game tictacmo --conf-file "$config" --executable "$executable"
                    --games-per-profile "$eval_games" --min-games "$eval_games"
                    --num-simulations "$simulations")
        [[ -n "$candidate" ]] && args+=(--candidate "$candidate")
        [[ "$eval_noise" == true ]] && args+=(--noise)
        "${args[@]}"
    fi
    [[ -f "$manifest" ]] || return
    python3 tools/multiplayer-eval.py "$manifest" "$output_dir" \
        -g "$gpu" --num_threads "$eval_threads" --resume
    python3 tools/jpsro.py ingest "$meta_dir" "$manifest" "$output_dir/games.jsonl"
}

freeze_policy() {
    local generation=$1
    local policy_id="p$generation"
    local model="$frozen_dir/$policy_id.pt"
    [[ -f "$model" ]] || cp "$(latest_model)" "$model"
    if ! policy_registered "$policy_id"; then
        python3 tools/jpsro.py add-policy "$meta_dir" "$policy_id" "$model" --generation "$generation"
    fi
}

solve_meta() {
    local generation=$1
    python3 tools/jpsro.py solve "$meta_dir" --min-games "$eval_games" --tolerance "$tolerance"
    cp "$meta_dir/meta_strategy.json" "$meta_dir/meta_strategy_g${generation}.json"
}

[[ $# -eq 2 ]] || { usage; exit 2; }
run_dir=$(readlink -m "$1")
source_config=$(readlink -f "$2")
[[ -f "$source_config" ]] || die "config not found: $2"
case "$run_dir" in
    "$repo_root"/*) ;;
    *) die "RUN_DIR must be inside $repo_root" ;;
esac
[[ "$run_dir" != *[,:[:space:]]* ]] || die "RUN_DIR cannot contain spaces, commas, or colons"

boundaries_string=${JPSRO_BOUNDARIES:-5,15,30}
IFS=, read -ra boundaries <<< "$boundaries_string"
(( ${#boundaries[@]} >= 2 )) || die "JPSRO_BOUNDARIES needs bootstrap and at least one oracle boundary"
previous=0
for boundary in "${boundaries[@]}"; do
    [[ "$boundary" =~ ^[1-9][0-9]*$ ]] || die "invalid boundary: $boundary"
    (( boundary > previous )) || die "boundaries must be strictly increasing"
    previous=$boundary
done

gpu=${JPSRO_GPU:-0}
sp_gpu=${JPSRO_SP_GPU:-$gpu}
games=${JPSRO_GAMES_PER_ITERATION:-2000}
training_steps=${JPSRO_TRAINING_STEPS:-500}
learner_batch=${JPSRO_LEARNER_BATCH:-1024}
selfplay_batch=${JPSRO_SELFPLAY_BATCH:-32}
cpu_threads=${JPSRO_CPU_THREADS:-4}
eval_games=${JPSRO_EVAL_GAMES:-8}
eval_threads=${JPSRO_EVAL_THREADS:-2}
eval_noise=${JPSRO_EVAL_NOISE:-true}
simulations=${JPSRO_SIMULATIONS:-50}
seed=${JPSRO_SEED:-0}
tolerance=${JPSRO_TOLERANCE:-0.01}
port=${JPSRO_PORT:-10021}

config="$run_dir/tictacmo.cfg"
training_dir="$run_dir/training"
meta_dir="$run_dir/meta"
frozen_dir="$run_dir/frozen"
executable="$repo_root/build/tictacmo/minizero_tictacmo"
settings_file="$run_dir/budgeted_jpsro.settings"
settings="boundaries=$boundaries_string games=$games steps=$training_steps learner_batch=$learner_batch selfplay_batch=$selfplay_batch cpu_threads=$cpu_threads eval_games=$eval_games eval_noise=$eval_noise simulations=$simulations seed=$seed"
common_conf="zero_use_population=false:zero_disable_resign_ratio=1:zero_num_games_per_iteration=$games:learner_training_step=$training_steps:learner_batch_size=$learner_batch:actor_num_simulation=$simulations:program_auto_seed=false:program_seed=$seed"

mkdir -p "$run_dir" "$frozen_dir"
if [[ -f "$settings_file" ]]; then
    [[ "$(<"$settings_file")" == "$settings" ]] || die "resume settings differ from $settings_file"
else
    echo "$settings" > "$settings_file"
fi
[[ -f "$config" ]] || cp "$source_config" "$config"

# Generation 0 is the in-run bootstrap policy, not an external AlphaZero baseline.
bootstrap_end=${boundaries[0]}
train_to "$bootstrap_end"
if [[ ! -f "$run_dir/.generation_0_complete" ]]; then
    [[ -d "$meta_dir" ]] || python3 tools/jpsro.py init "$meta_dir" --players 3 --shared-pool
    freeze_policy 0
    evaluate_profiles "$run_dir/eval_g0_full"
    solve_meta 0
    touch "$run_dir/.generation_0_complete"
fi

for ((generation = 1; generation < ${#boundaries[@]}; ++generation)); do
    marker="$run_dir/.generation_${generation}_complete"
    [[ -f "$marker" ]] && continue
    start_iteration=$((boundaries[generation - 1] + 1))
    end_iteration=${boundaries[generation]}
    profile_file="$run_dir/oracle_g${generation}.tsv"
    python3 tools/jpsro.py oracle-plan "$meta_dir" "$profile_file" --responders all
    train_to "$end_iteration" "$profile_file" "$start_iteration"

    freeze_policy "$generation"
    evaluate_profiles "$run_dir/eval_g${generation}_deviation" "p$generation"
    python3 tools/jpsro.py deviation-gap "$meta_dir" "p$generation" | tee "$run_dir/deviation_g${generation}.json"
    evaluate_profiles "$run_dir/eval_g${generation}_full"
    solve_meta "$generation"
    touch "$marker"
done

echo "training complete: $training_dir"
echo "latest single model: $(latest_model)"
echo "population CCE: $meta_dir/meta_strategy.json"
