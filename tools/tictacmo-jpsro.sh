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
    local manifest="$output_dir/arena.json"
    mkdir -p "$output_dir"
    if [[ ! -f "$manifest" ]]; then
        local args=(python3 tools/jpsro.py make-eval "$meta_dir" "$manifest"
                    --game "$game" --conf-file "$config" --executable "$executable"
                    --games-per-profile "$eval_games" --min-games "$eval_games"
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
selfplay_ratio=${GUIDED_CURRENT_RATIO:-0.15}
hard_ratio=${GUIDED_HARD_RATIO:-0.60}
cce_ratio=${GUIDED_CCE_RATIO:-0.10}
history_ratio=${GUIDED_HISTORY_RATIO:-0.15}
population_size=${GUIDED_POPULATION_SIZE:-8}
hard_temperature=${GUIDED_HARD_TEMPERATURE:-0.20}
hard_confidence=${GUIDED_HARD_CONFIDENCE:-1.0}
cpu_threads=${GUIDED_CPU_THREADS:-${JPSRO_CPU_THREADS:-4}}
eval_games=${GUIDED_EVAL_GAMES:-${JPSRO_EVAL_GAMES:-20}}
eval_batch=${GUIDED_EVAL_BATCH:-${JPSRO_EVAL_BATCH:-64}}
eval_threads=${GUIDED_EVAL_THREADS:-${JPSRO_EVAL_THREADS:-4}}
eval_noise=${GUIDED_EVAL_NOISE:-${JPSRO_EVAL_NOISE:-true}}
simulations=${GUIDED_SIMULATIONS:-${JPSRO_SIMULATIONS:-50}}
seed=${GUIDED_SEED:-${JPSRO_SEED:-0}}
tolerance=${GUIDED_TOLERANCE:-${JPSRO_TOLERANCE:-0.01}}
admission_gain=${GUIDED_ADMISSION_GAIN:-${JPSRO_ADMISSION_GAIN:-0.02}}
admission_confidence=${GUIDED_ADMISSION_CONFIDENCE:-${JPSRO_ADMISSION_CONFIDENCE:-2.0}}
port=${GUIDED_PORT:-${JPSRO_PORT:-10021}}

for value in "$bootstrap_iterations" "$oracle_interval" "$total_iterations" "$games" "$training_steps" "$learner_batch" "$selfplay_workers" "$selfplay_batch" "$cpu_threads" "$eval_games" "$eval_batch" "$eval_threads" "$simulations" "$population_size"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "iteration and self-play parallelism settings must be positive integers"
done
(( bootstrap_iterations < total_iterations )) || die "bootstrap iterations must be below total iterations"
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
executable="$repo_root/build/$game/minizero_$game"
settings_file="$run_dir/guided_pool.settings"
state_file="$run_dir/controller_state.json"
settings="version=1 game=$game players=$num_players seats=$current_seat_min-$current_seat_max bootstrap=$bootstrap_iterations interval=$oracle_interval total=$total_iterations games=$games steps=$training_steps learner_batch=$learner_batch selfplay_workers=$selfplay_workers selfplay_batch=$selfplay_batch current_ratio=$selfplay_ratio hard_ratio=$hard_ratio cce_ratio=$cce_ratio history_ratio=$history_ratio population_size=$population_size hard_temperature=$hard_temperature hard_confidence=$hard_confidence cpu_threads=$cpu_threads eval_games=$eval_games eval_noise=$eval_noise simulations=$simulations seed=$seed admission_gain=$admission_gain admission_confidence=$admission_confidence"
common_conf="zero_use_population=false:zero_jpsro_selfplay_ratio=$selfplay_ratio:zero_jpsro_num_workers=$selfplay_workers:zero_population_current_seat_min=$current_seat_min:zero_population_current_seat_max=$current_seat_max:zero_population_balance_seats=true:zero_disable_resign_ratio=1:zero_num_games_per_iteration=$games:learner_training_step=$training_steps:learner_batch_size=$learner_batch:actor_num_simulation=$simulations:program_auto_seed=false:program_seed=$seed"

mkdir -p "$run_dir" "$frozen_dir"
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

# p0 is the first checkpoint of this same from-scratch run, not an external warm start.
train_to "$bootstrap_iterations"
if ! stage_complete g0; then
    [[ -d "$meta_dir" ]] || python3 tools/jpsro.py init "$meta_dir" --players "$num_players" --shared-pool
    freeze_policy p0 0
    evaluate_profiles "$run_dir/eval_g0_full"
    solve_meta g0
    python3 tools/jpsro.py guided-plan "$meta_dir" "$run_dir/guided_after_${bootstrap_iterations}.tsv" \
        --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
        --temperature "$hard_temperature" --confidence "$hard_confidence" \
        --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
    mark_stage_complete g0
fi

end_iteration=$bootstrap_iterations
while (( end_iteration < total_iterations )); do
    previous_iteration=$end_iteration
    end_iteration=$((end_iteration + oracle_interval))
    (( end_iteration > total_iterations )) && end_iteration=$total_iterations
    stage="i$end_iteration"
    stage_complete "$stage" && continue
    candidate="p$end_iteration"
    profile_file="$run_dir/guided_after_${previous_iteration}.tsv"
    [[ -f "$profile_file" ]] || die "missing guided plan: $profile_file"
    train_to "$end_iteration" "$profile_file"

    freeze_policy "$candidate" "$end_iteration"
    evaluate_profiles "$run_dir/eval_i${end_iteration}_deviation" "$candidate"
    next_profile="$run_dir/guided_after_${end_iteration}.tsv"
    python3 tools/jpsro.py guided-plan "$meta_dir" "$next_profile" --candidate "$candidate" \
        --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
        --temperature "$hard_temperature" --confidence "$hard_confidence" \
        --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
    decision="$run_dir/admission_i${end_iteration}.json"
    python3 tools/jpsro.py admit-candidate "$meta_dir" "$candidate" \
        --min-gain "$admission_gain" --confidence "$admission_confidence" --output "$decision"
    if [[ $(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["accepted"]).lower())' "$decision") == true ]]; then
        evaluate_profiles "$run_dir/eval_i${end_iteration}_full"
        solve_meta "i$end_iteration" "$candidate"
        python3 tools/jpsro.py guided-plan "$meta_dir" "$next_profile" --candidate "$candidate" \
            --hard-ratio "$hard_ratio" --cce-ratio "$cce_ratio" --history-ratio "$history_ratio" \
            --temperature "$hard_temperature" --confidence "$hard_confidence" \
            --current-seat-min "$current_seat_min" --current-seat-max "$current_seat_max"
    else
        rm -f "$frozen_dir/$candidate.pt"
        echo "candidate $candidate rejected; continuing with the certified pool"
    fi
    mark_stage_complete "$stage"
done

echo "training complete: $training_dir"
echo "latest single model: $(latest_model)"
echo "certified population CCE: $meta_dir/meta_strategy.json"
echo "last guided distribution: $run_dir/guided_after_${total_iterations}.tsv.json"
