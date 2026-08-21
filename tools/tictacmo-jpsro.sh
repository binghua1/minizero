#!/bin/bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage:
  tools/tictacmo-jpsro.sh RUN_DIR CONFIG.cfg
  tools/multiplayer-guided-pool.sh GAME RUN_DIR CONFIG.cfg

Train one from-scratch JPSRO-guided opponent-pool run. A single CURRENT model
learns from role-balanced hard, CCE and historical joint profiles, plus a small
amount of all-CURRENT self-play. Candidates enter the certified frozen pool only
when their lower-confidence deviation gain is positive. TicTacMo/Connect3x3
default to three players and Blokus to four; other games set GUIDED_NUM_PLAYERS.
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

stage_complete() {
    python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(sys.argv[2] not in d.get("completed_stages", []))' "$state_file" "$1"
}

mark_stage_complete() {
    python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {"version":1,"completed_stages":[]}; s=sys.argv[2]; d["completed_stages"] += [] if s in d["completed_stages"] else [s]; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,indent=2)+"\n"); t.replace(p)' "$state_file" "$1"
}

migrate_legacy_markers() {
    local marker stage
    if [[ -f "$run_dir/.generation_0_complete" ]]; then
        mark_stage_complete g0
        rm -f "$run_dir/.generation_0_complete"
    fi
    for marker in "$run_dir"/.candidate_*_complete; do
        [[ -f "$marker" ]] || continue
        stage=${marker##*/.candidate_}
        stage="i${stage%_complete}"
        mark_stage_complete "$stage"
        rm -f "$marker"
    done
}

migrate_legacy_output_layout() {
    local source destination name
    for source in "$run_dir"/guided_after_*.tsv "$run_dir"/guided_after_*.tsv.json "$run_dir"/admission_i*.json; do
        [[ -f "$source" ]] || continue
        name=${source##*/}
        destination="$guided_dir/$name"
        [[ ! -e "$destination" ]] || die "cannot migrate legacy output; destination exists: $destination"
        mv "$source" "$destination"
    done
    for source in "$run_dir"/eval_*; do
        [[ -d "$source" ]] || continue
        name=${source##*/}
        destination="$evaluations_dir/$name"
        [[ ! -e "$destination" ]] || die "cannot migrate legacy evaluation; destination exists: $destination"
        mv "$source" "$destination"
    done
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
    local args=(tools/quick-run.sh train "$game" "$config" "$end_iteration"
                -n "$training_dir" -g "$gpu" -p "$port"
                -b "$selfplay_batch" -c "$cpu_threads" -conf_str "$conf")
    [[ "$sp_gpu" != "$gpu" ]] && args+=(--sp_gpu "$sp_gpu")
    [[ -d "$training_dir" ]] && args+=(--continue-training)
    "${args[@]}"
}

evaluate_profiles() {
    local output_dir=$1
    local candidate=${2:-}
    local requested_games=${3:-$eval_games}
    local manifest="$output_dir/arena.json"
    mkdir -p "$output_dir"
    if [[ ! -f "$manifest" ]]; then
        local args=(python3 tools/jpsro.py make-eval "$meta_dir" "$manifest"
                    --game "$game" --conf-file "$config" --executable "$executable"
                    --games-per-profile "$requested_games" --min-games "$requested_games"
                    --num-simulations "$simulations")
        [[ -n "$candidate" ]] && args+=(--candidate "$candidate")
        [[ "$eval_noise" == true ]] && args+=(--noise)
        "${args[@]}"
    fi
    # No manifest means every requested payoff is already present. This is a
    # successful no-op (often candidate deviation games already completed the
    # newly admitted restricted game), not an error under `set -e`.
    [[ -f "$manifest" ]] || return 0
    if python3 -c 'import json,sys; raise SystemExit("batched_evaluation" not in json.load(open(sys.argv[1])))' "$manifest"; then
        python3 tools/jpsro-batched-eval.py "$manifest" "$output_dir" \
            -g "$eval_gpu" --batch-size "$eval_batch" \
            --cpu-threads "$eval_threads" --actors "$eval_actors" --resume
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
    local checkpoint_iteration=${3:-$generation}
    local model="$frozen_dir/$policy_id.pt"
    local source_model="$training_dir/model/weight_iter_$((checkpoint_iteration * training_steps)).pt"
    [[ -f "$source_model" ]] || die "missing checkpoint for $policy_id: $source_model"
    [[ -f "$model" ]] || cp "$source_model" "$model"
    if ! policy_registered "$policy_id"; then
        python3 tools/jpsro.py add-policy "$meta_dir" "$policy_id" "$model" --generation "$generation"
    fi
}

process_generation_zero() {
    stage_complete g0 && return
    [[ -d "$meta_dir" ]] || python3 tools/jpsro.py init "$meta_dir" --players "$num_players" --shared-pool
    freeze_policy p0 0 "$bootstrap_iterations"
    evaluate_profiles "$evaluations_dir/eval_g0_full"
    solve_meta g0
    python3 tools/jpsro.py guided-plan "$meta_dir" "$guided_dir/guided_after_${bootstrap_iterations}.tsv" \
        --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
        --temperature "$hard_temperature" --confidence "$hard_confidence" \
        --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
    mark_stage_complete g0
}

process_candidate() {
    local end_iteration=$1
    local candidate="p$end_iteration"
    local next_profile="$guided_dir/guided_after_${end_iteration}.tsv"
    local decision="$guided_dir/admission_i${end_iteration}.json"
    local stage="i$end_iteration"
    stage_complete "$stage" && return

    freeze_policy "$candidate" "$end_iteration"
    evaluate_profiles "$evaluations_dir/eval_i${end_iteration}_deviation" "$candidate" "$deviation_games"
    local admission_args=(python3 tools/jpsro.py admit-candidate "$meta_dir" "$candidate"
                          --min-gain "$admission_gain" --confidence "$admission_confidence"
                          --output "$decision")
    if [[ "$generalist_admission" == true ]]; then
        admission_args+=(--max-regression "$max_seat_regression" --promote-all-players)
    fi
    "${admission_args[@]}"
    if [[ $(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["accepted"]).lower())' "$decision") == true ]]; then
        if [[ "$hard_population_cap" == true ]]; then
            python3 tools/jpsro.py hard-prune "$meta_dir" \
                --maximum "$population_size" --protect "$candidate" \
                --output "$guided_dir/hard_prune_i${end_iteration}.json"
        fi
        evaluate_profiles "$evaluations_dir/eval_i${end_iteration}_full"
        solve_meta "i$end_iteration" "$candidate"
        python3 tools/jpsro.py guided-plan "$meta_dir" "$next_profile" --candidate "$candidate" \
            --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
            --temperature "$hard_temperature" --confidence "$hard_confidence" \
            --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
    else
        rm -f "$frozen_dir/$candidate.pt"
        # The rejected candidate has already been removed from the registry.
        # Rebuild the plan from the certified pool so no deleted model path can
        # leak into the next training segment.
        python3 tools/jpsro.py guided-plan "$meta_dir" "$next_profile" --candidate "$candidate" \
            --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
            --temperature "$hard_temperature" --confidence "$hard_confidence" \
            --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
        echo "candidate $candidate rejected; continuing with the certified pool"
    fi
    mark_stage_complete "$stage"
}

solve_meta() {
    local label=$1
    local protect=${2:-}
    python3 tools/jpsro.py solve "$meta_dir" --min-games "$eval_games" --tolerance "$tolerance"
    local prune_args=(python3 tools/jpsro.py prune "$meta_dir" --maximum "$population_size")
    [[ -n "$protect" ]] && prune_args+=(--protect "$protect")
    "${prune_args[@]}"
    cp "$meta_dir/meta_strategy.json" "$meta_dir/meta_strategy_${label}.json"
}

if [[ $# -eq 2 ]]; then
    game=tictacmo
elif [[ $# -eq 3 ]]; then
    game=${1,,}
    shift
else
    usage
    exit 2
fi
[[ "$game" =~ ^[a-z0-9_]+$ ]] || die "invalid GAME name"
run_dir=$(readlink -m "$1")
source_config=$(readlink -f "$2")
[[ -f "$source_config" ]] || die "config not found: $2"
if [[ -n ${GUIDED_NUM_PLAYERS:-} ]]; then
    num_players=$GUIDED_NUM_PLAYERS
elif [[ "$game" == tictacmo || "$game" == connect3x3 ]]; then
    num_players=3
elif [[ "$game" == blokus* ]]; then
    num_players=4
else
    die "set GUIDED_NUM_PLAYERS for $game"
fi
[[ "$num_players" =~ ^[3-6]$ ]] || die "GUIDED_NUM_PLAYERS must be between 3 and 6"
current_seat_min=${GUIDED_CURRENT_SEAT_MIN:-1}
current_seat_max=${GUIDED_CURRENT_SEAT_MAX:-$((num_players - 1))}
[[ "$current_seat_min" =~ ^[1-9][0-9]*$ && "$current_seat_max" =~ ^[1-9][0-9]*$ ]] || die "guided seat limits must be positive integers"
(( current_seat_min <= current_seat_max && current_seat_max < num_players )) || die "guided seat limits must satisfy 1 <= min <= max < players"
case "$run_dir" in
    "$repo_root"/*) ;;
    *) die "RUN_DIR must be inside $repo_root" ;;
esac
[[ "$run_dir" != *[,:[:space:]]* ]] || die "RUN_DIR cannot contain spaces, commas, or colons"

gpu=${GUIDED_GPU:-${JPSRO_GPU:-0}}
bootstrap_iterations=${GUIDED_BOOTSTRAP_ITERATIONS:-${JPSRO_BOOTSTRAP_ITERATIONS:-10}}
oracle_interval=${GUIDED_META_INTERVAL:-${JPSRO_ORACLE_INTERVAL:-10}}
total_iterations=${GUIDED_TOTAL_ITERATIONS:-${JPSRO_TOTAL_ITERATIONS:-30}}
games=${GUIDED_GAMES_PER_ITERATION:-${JPSRO_GAMES_PER_ITERATION:-2000}}
training_steps=${GUIDED_TRAINING_STEPS:-${JPSRO_TRAINING_STEPS:-500}}
learner_batch=${GUIDED_LEARNER_BATCH:-${JPSRO_LEARNER_BATCH:-1024}}
selfplay_workers=${GUIDED_SELFPLAY_WORKERS:-${JPSRO_SELFPLAY_WORKERS:-4}}
selfplay_batch=${GUIDED_SELFPLAY_BATCH:-${JPSRO_SELFPLAY_BATCH:-64}}
selfplay_ratio=${GUIDED_CURRENT_RATIO:-0.30}
hard_ratio=${GUIDED_HARD_RATIO:-0.30}
cce_ratio=${GUIDED_CCE_RATIO:-0.20}
history_ratio=${GUIDED_HISTORY_RATIO:-0.20}
population_size=${GUIDED_POPULATION_SIZE:-8}
hard_temperature=${GUIDED_HARD_TEMPERATURE:-0.20}
hard_confidence=${GUIDED_HARD_CONFIDENCE:-1.0}
cpu_threads=${GUIDED_CPU_THREADS:-${JPSRO_CPU_THREADS:-4}}
eval_games=${GUIDED_EVAL_GAMES:-${JPSRO_EVAL_GAMES:-20}}
deviation_games=${GUIDED_DEVIATION_GAMES:-100}
eval_batch=${GUIDED_EVAL_BATCH:-${JPSRO_EVAL_BATCH:-64}}
eval_threads=${GUIDED_EVAL_THREADS:-${JPSRO_EVAL_THREADS:-4}}
eval_actors=${GUIDED_EVAL_ACTORS:-${JPSRO_EVAL_ACTORS:-1}}
eval_noise=${GUIDED_EVAL_NOISE:-${JPSRO_EVAL_NOISE:-false}}
simulations=${GUIDED_SIMULATIONS:-${JPSRO_SIMULATIONS:-50}}
rank_utility_weight=${GUIDED_RANK_UTILITY_WEIGHT:-}
seed=${GUIDED_SEED:-${JPSRO_SEED:-0}}
tolerance=${GUIDED_TOLERANCE:-${JPSRO_TOLERANCE:-0.01}}
admission_gain=${GUIDED_ADMISSION_GAIN:-${JPSRO_ADMISSION_GAIN:-0.02}}
admission_confidence=${GUIDED_ADMISSION_CONFIDENCE:-${JPSRO_ADMISSION_CONFIDENCE:-2.0}}
generalist_admission=${GUIDED_GENERALIST_ADMISSION:-true}
max_seat_regression=${GUIDED_MAX_SEAT_REGRESSION:-0.05}
port=${GUIDED_PORT:-${JPSRO_PORT:-10021}}
persistent_workers=${GUIDED_PERSISTENT_WORKERS:-true}
hard_population_cap=${GUIDED_HARD_POPULATION_CAP:-false}

for value in "$bootstrap_iterations" "$oracle_interval" "$total_iterations" "$games" "$training_steps" "$learner_batch" "$selfplay_workers" "$selfplay_batch" "$cpu_threads" "$eval_games" "$deviation_games" "$eval_batch" "$eval_threads" "$eval_actors" "$simulations" "$population_size"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "iteration and self-play parallelism settings must be positive integers"
done
(( bootstrap_iterations < total_iterations )) || die "bootstrap iterations must be below total iterations"
[[ "$persistent_workers" == true || "$persistent_workers" == false ]] || die "GUIDED_PERSISTENT_WORKERS must be true or false"
[[ "$hard_population_cap" == true || "$hard_population_cap" == false ]] || die "GUIDED_HARD_POPULATION_CAP must be true or false"
[[ "$generalist_admission" == true || "$generalist_admission" == false ]] || die "GUIDED_GENERALIST_ADMISSION must be true or false"
if [[ "$persistent_workers" == true ]]; then
    (( (total_iterations - bootstrap_iterations) % oracle_interval == 0 )) || \
        die "persistent mode requires total iterations to equal bootstrap + N * meta interval"
fi
python3 - "$selfplay_ratio" "$hard_ratio" "$cce_ratio" "$history_ratio" <<'PY' || die "guided ratios must be non-negative and sum to 1"
import sys
values = [float(value) for value in sys.argv[1:]]
assert all(value >= 0 for value in values)
assert abs(sum(values) - 1.0) <= 1e-6
PY
python3 - "$hard_temperature" "$hard_confidence" <<'PY' || die "hard temperature must be positive and confidence non-negative"
import sys
assert float(sys.argv[1]) > 0
assert float(sys.argv[2]) >= 0
PY
python3 - "$max_seat_regression" <<'PY' || die "GUIDED_MAX_SEAT_REGRESSION must be non-negative"
import sys
assert float(sys.argv[1]) >= 0
PY
if [[ -n "$rank_utility_weight" ]]; then
    python3 - "$rank_utility_weight" <<'PY' || die "GUIDED_RANK_UTILITY_WEIGHT must be between 0 and 1"
import sys
value = float(sys.argv[1])
assert 0.0 <= value <= 1.0
PY
fi
[[ "$gpu" =~ ^[0-9]+$ ]] || die "GUIDED_GPU must be a GPU-index string such as 0 or 0123"
sp_gpu=${GUIDED_SELFPLAY_GPU:-${JPSRO_SELFPLAY_GPU:-}}
if [[ -z "$sp_gpu" ]]; then
    sp_gpu=$gpu
    if (( ${#gpu} == 1 && selfplay_workers > 1 )); then
        for ((worker = 1; worker < selfplay_workers; ++worker)); do sp_gpu+="$gpu"; done
    fi
fi
[[ "$sp_gpu" =~ ^[0-9]+$ ]] || die "GUIDED_SELFPLAY_GPU must be a GPU-index string"
(( ${#sp_gpu} == selfplay_workers )) || die "GUIDED_SELFPLAY_WORKERS must equal the number of GUIDED_SELFPLAY_GPU indices"
eval_gpu=${GUIDED_EVAL_GPU:-${JPSRO_EVAL_GPU:-${gpu:0:1}}}
[[ "$eval_gpu" =~ ^[0-9]$ ]] || die "GUIDED_EVAL_GPU must be one GPU index"

config="$run_dir/$game.cfg"
training_dir="$run_dir/training"
meta_dir="$run_dir/meta"
frozen_dir="$run_dir/frozen"
guided_dir="$run_dir/guided"
evaluations_dir="$run_dir/evaluations"
executable="$repo_root/build/$game/minizero_$game"
settings_file="$run_dir/guided_pool.settings"
state_file="$run_dir/controller_state.json"
settings="version=2 game=$game players=$num_players seats=$current_seat_min-$current_seat_max bootstrap=$bootstrap_iterations interval=$oracle_interval total=$total_iterations games=$games steps=$training_steps learner_batch=$learner_batch selfplay_workers=$selfplay_workers selfplay_batch=$selfplay_batch current_ratio=$selfplay_ratio hard_ratio=$hard_ratio cce_ratio=$cce_ratio history_ratio=$history_ratio population_size=$population_size hard_temperature=$hard_temperature hard_confidence=$hard_confidence cpu_threads=$cpu_threads eval_games=$eval_games deviation_games=$deviation_games eval_noise=$eval_noise simulations=$simulations seed=$seed admission_gain=$admission_gain admission_confidence=$admission_confidence generalist_admission=$generalist_admission max_seat_regression=$max_seat_regression"
[[ -n "$rank_utility_weight" ]] && settings+=" rank_utility_weight=$rank_utility_weight"
common_conf="zero_use_population=false:zero_jpsro_selfplay_ratio=$selfplay_ratio:zero_jpsro_num_workers=$selfplay_workers:zero_population_current_seat_min=$current_seat_min:zero_population_current_seat_max=$current_seat_max:zero_population_balance_seats=true:zero_disable_resign_ratio=1:zero_num_games_per_iteration=$games:learner_training_step=$training_steps:learner_batch_size=$learner_batch:actor_num_simulation=$simulations:program_auto_seed=false:program_seed=$seed"
[[ -n "$rank_utility_weight" ]] && common_conf+=":actor_rank_utility_weight=$rank_utility_weight"

mkdir -p "$run_dir" "$frozen_dir" "$guided_dir" "$evaluations_dir"
migrate_legacy_output_layout
if [[ -f "$settings_file" ]]; then
    saved_settings=$(<"$settings_file")
    saved_total=$(sed -nE 's/.*(^| )total=([0-9]+)( |$).*/\2/p' <<<"$saved_settings")
    saved_fixed=$(sed -E 's/(^| )total=[0-9]+( |$)/ /' <<<"$saved_settings")
    current_fixed=$(sed -E 's/(^| )total=[0-9]+( |$)/ /' <<<"$settings")
    [[ -n "$saved_total" && "$saved_fixed" == "$current_fixed" ]] || die "resume settings other than total iterations differ from $settings_file"
    (( total_iterations >= saved_total )) || die "GUIDED_TOTAL_ITERATIONS cannot decrease below saved total $saved_total"
    if (( total_iterations > saved_total )); then
        echo "$settings" > "$settings_file"
        echo "extending guided run from $saved_total to $total_iterations iterations"
    fi
else
    echo "$settings" > "$settings_file"
fi
[[ -f "$config" ]] || cp "$source_config" "$config"
migrate_legacy_markers

run_segmented_training() {
    train_to "$bootstrap_iterations"
    process_generation_zero
    local end_iteration=$bootstrap_iterations
    while (( end_iteration < total_iterations )); do
        local previous_iteration=$end_iteration
        end_iteration=$((end_iteration + oracle_interval))
        (( end_iteration > total_iterations )) && end_iteration=$total_iterations
        if ! stage_complete "i$end_iteration"; then
            local profile_file="$guided_dir/guided_after_${previous_iteration}.tsv"
            [[ -f "$profile_file" ]] || die "missing guided plan: $profile_file"
            train_to "$end_iteration" "$profile_file"
            process_candidate "$end_iteration"
        fi
    done
}

wait_for_meta_request() {
    local expected=$1
    local training_pid=$2
    while true; do
        if [[ -f "$guided_dir/meta_request" && $(<"$guided_dir/meta_request") == "$expected" ]]; then
            return
        fi
        if ! kill -0 "$training_pid" 2>/dev/null; then
            wait "$training_pid" || true
            die "training stopped before meta iteration $expected"
        fi
        sleep 1
    done
}

signal_meta_complete() {
    local iteration=$1
    local temporary="$guided_dir/meta_response.tmp"
    echo "$iteration" > "$temporary"
    mv "$temporary" "$guided_dir/meta_response"
}

run_persistent_training() {
    local completed initial_profile="" latest_boundary
    completed=$(completed_iterations)
    if (( completed >= bootstrap_iterations )); then
        latest_boundary=$((bootstrap_iterations + (completed - bootstrap_iterations) / oracle_interval * oracle_interval))
        if (( latest_boundary == bootstrap_iterations )); then
            process_generation_zero
        else
            process_candidate "$latest_boundary"
        fi
        initial_profile="$guided_dir/guided_after_${latest_boundary}.tsv"
        [[ -f "$initial_profile" ]] || die "missing resume profile: $initial_profile"
    fi

    rm -f "$guided_dir/meta_request" "$guided_dir/meta_response" "$guided_dir/meta_response.tmp"
    local conf="$common_conf:zero_use_jpsro=true:zero_jpsro_profile_file=$initial_profile:zero_jpsro_sync_directory=$guided_dir:zero_jpsro_sync_first_iteration=$bootstrap_iterations:zero_jpsro_sync_interval=$oracle_interval"
    local args=(tools/quick-run.sh train "$game" "$config" "$total_iterations"
                -n "$training_dir" -g "$gpu" -p "$port"
                -b "$selfplay_batch" -c "$cpu_threads" -conf_str "$conf")
    [[ "$sp_gpu" != "$gpu" ]] && args+=(--sp_gpu "$sp_gpu")
    [[ -d "$training_dir" ]] && args+=(--continue-training)
    "${args[@]}" &
    local training_pid=$!
    trap 'kill "$training_pid" 2>/dev/null || true' EXIT INT TERM

    local boundary
    if (( completed < bootstrap_iterations )); then
        boundary=$bootstrap_iterations
    else
        boundary=$((bootstrap_iterations + (completed - bootstrap_iterations) / oracle_interval * oracle_interval + oracle_interval))
    fi
    while (( boundary <= total_iterations )); do
        wait_for_meta_request "$boundary" "$training_pid"
        if (( boundary == bootstrap_iterations )); then
            process_generation_zero
        else
            process_candidate "$boundary"
        fi
        signal_meta_complete "$boundary"
        boundary=$((boundary + oracle_interval))
    done
    wait "$training_pid"
    trap - EXIT INT TERM
}

if [[ "$persistent_workers" == true ]]; then
    run_persistent_training
else
    run_segmented_training
fi

echo "training complete: $training_dir"
echo "latest single model: $(latest_model)"
echo "certified population CCE: $meta_dir/meta_strategy.json"
echo "last guided distribution: $guided_dir/guided_after_${total_iterations}.tsv.json"
