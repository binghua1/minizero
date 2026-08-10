#!/bin/bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage: tools/tictacmo-jpsro.sh RUN_DIR CONFIG.cfg

Train one from-scratch Adaptive JPSRO-guided AlphaZero run. Normal all-seat
self-play preserves AlphaZero data efficiency; the remaining workers train
against CCE-selected joint opponents. Candidates are tested every oracle
interval and enter only the player pools where their lower-confidence gain is
positive. Re-running the same command resumes an interrupted run.
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
    if (( $(completed_iterations) >= end_iteration )); then
        echo "training already reached iteration $end_iteration"
        return
    fi

    local conf="$common_conf:zero_use_jpsro=false"
    if [[ -n "$profile_file" ]]; then
        conf="$common_conf:zero_use_jpsro=true:zero_jpsro_profile_file=$profile_file"
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
    if python3 -c 'import json,sys; raise SystemExit("batched_evaluation" not in json.load(open(sys.argv[1])))' "$manifest"; then
        python3 tools/jpsro-batched-eval.py "$manifest" "$output_dir" \
            -g "$eval_gpu" --batch-size "$eval_batch" \
            --cpu-threads "$eval_threads" --resume
    else
        # Compatibility for manifests created by the earlier console-arena controller.
        python3 tools/multiplayer-eval.py "$manifest" "$output_dir" \
            -g "$eval_gpu" --num_threads "$eval_threads" --resume
    fi
    python3 tools/jpsro.py ingest "$meta_dir" "$manifest" "$output_dir/games.jsonl"
}

freeze_policy() {
    local policy_id=$1
    local generation=$2
    local model="$frozen_dir/$policy_id.pt"
    [[ -f "$model" ]] || cp "$(latest_model)" "$model"
    if ! policy_registered "$policy_id"; then
        python3 tools/jpsro.py add-policy "$meta_dir" "$policy_id" "$model" --generation "$generation"
    fi
}

solve_meta() {
    local label=$1
    python3 tools/jpsro.py solve "$meta_dir" --min-games "$eval_games" --tolerance "$tolerance"
    cp "$meta_dir/meta_strategy.json" "$meta_dir/meta_strategy_${label}.json"
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

gpu=${JPSRO_GPU:-0}
bootstrap_iterations=${JPSRO_BOOTSTRAP_ITERATIONS:-5}
oracle_interval=${JPSRO_ORACLE_INTERVAL:-5}
total_iterations=${JPSRO_TOTAL_ITERATIONS:-30}
games=${JPSRO_GAMES_PER_ITERATION:-2000}
training_steps=${JPSRO_TRAINING_STEPS:-500}
learner_batch=${JPSRO_LEARNER_BATCH:-1024}
selfplay_workers=${JPSRO_SELFPLAY_WORKERS:-4}
selfplay_batch=${JPSRO_SELFPLAY_BATCH:-64}
selfplay_ratio=${JPSRO_SELFPLAY_RATIO:-0.7}
cpu_threads=${JPSRO_CPU_THREADS:-4}
eval_games=${JPSRO_EVAL_GAMES:-20}
eval_batch=${JPSRO_EVAL_BATCH:-64}
eval_threads=${JPSRO_EVAL_THREADS:-4}
eval_noise=${JPSRO_EVAL_NOISE:-true}
simulations=${JPSRO_SIMULATIONS:-50}
seed=${JPSRO_SEED:-0}
tolerance=${JPSRO_TOLERANCE:-0.01}
admission_gain=${JPSRO_ADMISSION_GAIN:-0.02}
admission_confidence=${JPSRO_ADMISSION_CONFIDENCE:-2.0}
port=${JPSRO_PORT:-10021}

for value in "$bootstrap_iterations" "$oracle_interval" "$total_iterations" "$selfplay_workers" "$selfplay_batch" "$eval_batch" "$eval_threads"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "iteration and self-play parallelism settings must be positive integers"
done
(( bootstrap_iterations < total_iterations )) || die "bootstrap iterations must be below total iterations"
python3 -c 'import sys; value=float(sys.argv[1]); assert 0 <= value <= 1' "$selfplay_ratio" || die "JPSRO_SELFPLAY_RATIO must be between 0 and 1"
worker_gpu=${JPSRO_SELFPLAY_GPU:-$gpu}
[[ "$worker_gpu" =~ ^[0-9]$ ]] || die "JPSRO_SELFPLAY_GPU must be one GPU index"
eval_gpu=${JPSRO_EVAL_GPU:-$gpu}
[[ "$eval_gpu" =~ ^[0-9]$ ]] || die "JPSRO_EVAL_GPU must be one GPU index"
sp_gpu=""
for ((worker = 0; worker < selfplay_workers; ++worker)); do sp_gpu+="$worker_gpu"; done

config="$run_dir/tictacmo.cfg"
training_dir="$run_dir/training"
meta_dir="$run_dir/meta"
frozen_dir="$run_dir/frozen"
executable="$repo_root/build/tictacmo/minizero_tictacmo"
settings_file="$run_dir/adaptive_jpsro.settings"
settings="version=2 bootstrap=$bootstrap_iterations interval=$oracle_interval total=$total_iterations games=$games steps=$training_steps learner_batch=$learner_batch selfplay_workers=$selfplay_workers selfplay_batch=$selfplay_batch selfplay_ratio=$selfplay_ratio cpu_threads=$cpu_threads eval_games=$eval_games eval_noise=$eval_noise simulations=$simulations seed=$seed admission_gain=$admission_gain admission_confidence=$admission_confidence"
common_conf="zero_use_population=false:zero_jpsro_selfplay_ratio=$selfplay_ratio:zero_jpsro_num_workers=$selfplay_workers:zero_disable_resign_ratio=1:zero_num_games_per_iteration=$games:learner_training_step=$training_steps:learner_batch_size=$learner_batch:actor_num_simulation=$simulations:program_auto_seed=false:program_seed=$seed"

mkdir -p "$run_dir" "$frozen_dir"
[[ ! -f "$run_dir/budgeted_jpsro.settings" ]] || die "old fixed-JPSRO run detected; use a new RUN_DIR for the adaptive method"
if [[ -f "$settings_file" ]]; then
    [[ "$(<"$settings_file")" == "$settings" ]] || die "resume settings differ from $settings_file"
else
    echo "$settings" > "$settings_file"
fi
[[ -f "$config" ]] || cp "$source_config" "$config"

# p0 is a short in-run bootstrap, never an external AlphaZero warm start.
train_to "$bootstrap_iterations"
if [[ ! -f "$run_dir/.generation_0_complete" ]]; then
    [[ -d "$meta_dir" ]] || python3 tools/jpsro.py init "$meta_dir" --players 3 --shared-pool
    freeze_policy p0 0
    evaluate_profiles "$run_dir/eval_g0_full"
    solve_meta g0
    touch "$run_dir/.generation_0_complete"
fi

end_iteration=$bootstrap_iterations
while (( end_iteration < total_iterations )); do
    end_iteration=$((end_iteration + oracle_interval))
    (( end_iteration > total_iterations )) && end_iteration=$total_iterations
    marker="$run_dir/.candidate_${end_iteration}_complete"
    [[ -f "$marker" ]] && continue
    candidate="p$end_iteration"
    profile_file="$run_dir/oracle_i${end_iteration}.tsv"
    python3 tools/jpsro.py oracle-plan "$meta_dir" "$profile_file" --responders all
    train_to "$end_iteration" "$profile_file"

    freeze_policy "$candidate" "$end_iteration"
    evaluate_profiles "$run_dir/eval_i${end_iteration}_deviation" "$candidate"
    decision="$run_dir/admission_i${end_iteration}.json"
    python3 tools/jpsro.py admit-candidate "$meta_dir" "$candidate" \
        --min-gain "$admission_gain" --confidence "$admission_confidence" --output "$decision"
    if [[ $(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["accepted"]).lower())' "$decision") == true ]]; then
        evaluate_profiles "$run_dir/eval_i${end_iteration}_full"
        solve_meta "i$end_iteration"
    else
        rm -f "$frozen_dir/$candidate.pt"
        echo "candidate $candidate rejected; continuing with the certified pool"
    fi
    touch "$marker"
done

echo "training complete: $training_dir"
echo "latest single model: $(latest_model)"
echo "population CCE: $meta_dir/meta_strategy.json"
