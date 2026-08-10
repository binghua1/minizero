#!/bin/bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage:
  tools/tictacmo-jpsro.sh baseline RUN_DIR CONFIG.cfg
  tools/tictacmo-jpsro.sh reuse    RUN_DIR EXISTING_TRAIN_DIR CONFIG.cfg
  tools/tictacmo-jpsro.sh oracle   RUN_DIR
  tools/tictacmo-jpsro.sh evaluate RUN_DIR
  tools/tictacmo-jpsro.sh all      RUN_DIR CONFIG.cfg

The baseline/all path gives p0 and p1 the same seeded random initialization.
The reuse path keeps an existing p0 and only trains p1 from scratch.
Override defaults with JPSRO_* environment variables.
EOF
}

die() { echo "error: $*" >&2; exit 1; }

latest_model() {
    local training_dir=$1
    local model
    model=$(find "$training_dir/model" -maxdepth 1 -name 'weight_iter_*.pt' -print 2>/dev/null | sort -V | tail -n 1)
    [[ -n "$model" ]] || die "no checkpoint in $training_dir/model"
    readlink -f "$model"
}

model_at_iteration() {
    local training_dir=$1
    local iteration=$2
    local model="$training_dir/model/weight_iter_$iteration.pt"
    [[ -f "$model" ]] || die "checkpoint not found: $model"
    readlink -f "$model"
}

run_arena() {
    local manifest=$1
    local output=$2
    python3 tools/multiplayer-eval.py "$manifest" "$output" \
        -g "$gpu" --num_threads "$eval_threads"
}

train_baseline() {
    local source_config=$1
    [[ -f "$source_config" ]] || die "config not found: $source_config"
    [[ ! -e "$run_dir/baseline_p0" ]] || die "$run_dir/baseline_p0 already exists"
    mkdir -p "$run_dir"
    source_config=$(readlink -f "$source_config")
    [[ "$source_config" == "$config" ]] || cp "$source_config" "$config"

    tools/quick-run.sh train tictacmo "$config" "$iterations" \
        -n "$run_dir/baseline_p0" -g "$gpu" -p "$baseline_port" \
        -b "$selfplay_batch" -c "$cpu_threads" -conf_str \
        "$common_conf:zero_use_jpsro=false"
}

reuse_baseline() {
    local training_dir=$1
    local source_config=$2
    [[ -d "$training_dir/model" ]] || die "model directory not found: $training_dir/model"
    [[ -f "$source_config" ]] || die "config not found: $source_config"
    [[ ! -e "$config" && ! -e "$run_dir/frozen" ]] || die "$run_dir was already prepared"
    mkdir -p "$run_dir/frozen"
    training_dir=$(readlink -f "$training_dir")
    source_config=$(readlink -f "$source_config")
    cp "$source_config" "$config"
    cp "$(model_at_iteration "$training_dir" "$mid_iteration")" \
        "$run_dir/frozen/base_mid.pt"
    cp "$(latest_model "$training_dir")" "$run_dir/frozen/base_final.pt"
    echo "reused baseline checkpoints from $training_dir"
}

train_oracle() {
    [[ -f "$config" ]] || die "run baseline first; missing $config"
    [[ ! -e "$meta_dir" ]] || die "$meta_dir already exists; oracle stage was already started"
    [[ ! -e "$run_dir/oracle_p1" ]] || die "$run_dir/oracle_p1 already exists"
    [[ -x "$executable" ]] || die "missing executable: $executable"

    mkdir -p "$run_dir/frozen" "$run_dir/eval_g0"
    if [[ ! -f "$run_dir/frozen/base_mid.pt" || ! -f "$run_dir/frozen/base_final.pt" ]]; then
        cp "$(model_at_iteration "$run_dir/baseline_p0" "$mid_iteration")" \
            "$run_dir/frozen/base_mid.pt"
        cp "$(latest_model "$run_dir/baseline_p0")" "$run_dir/frozen/base_final.pt"
    fi

    python3 tools/jpsro.py init "$meta_dir" --players 3 --shared-pool
    python3 tools/jpsro.py add-policy "$meta_dir" b0 "$run_dir/frozen/base_mid.pt" --generation 0
    python3 tools/jpsro.py add-policy "$meta_dir" b1 "$run_dir/frozen/base_final.pt" --generation 0
    python3 tools/jpsro.py make-eval "$meta_dir" "$run_dir/eval_g0/arena.json" \
        --game tictacmo --conf-file "$config" --executable "$executable" \
        --games-per-profile "$eval_games" --num-simulations "$simulations" --noise
    run_arena "$run_dir/eval_g0/arena.json" "$run_dir/eval_g0"
    python3 tools/jpsro.py ingest "$meta_dir" "$run_dir/eval_g0/arena.json" \
        "$run_dir/eval_g0/games.jsonl"
    python3 tools/jpsro.py solve "$meta_dir" --min-games "$eval_games" --tolerance "$tolerance"
    python3 tools/jpsro.py oracle-plan "$meta_dir" "$run_dir/oracle_g1.tsv" --responders all

    # No checkpoint is copied into oracle_p1: this is intentionally from scratch.
    tools/quick-run.sh train tictacmo "$config" "$iterations" \
        -n "$run_dir/oracle_p1" -g "$gpu" -p "$oracle_port" \
        -b "$selfplay_batch" -c "$cpu_threads" -conf_str \
        "$common_conf:zero_use_jpsro=true:zero_jpsro_profile_file=$run_dir/oracle_g1.tsv"
}

evaluate_oracle() {
    [[ -d "$meta_dir" ]] || die "run oracle first; missing $meta_dir"
    [[ ! -e "$run_dir/frozen/p1.pt" ]] || die "$run_dir/frozen/p1.pt already exists; evaluation was already started"
    mkdir -p "$run_dir/frozen" "$run_dir/deviation_g1" "$run_dir/full_g1"
    cp "$(latest_model "$run_dir/oracle_p1")" "$run_dir/frozen/p1.pt"

    python3 tools/jpsro.py add-policy "$meta_dir" p1 "$run_dir/frozen/p1.pt" --generation 1
    python3 tools/jpsro.py make-eval "$meta_dir" "$run_dir/deviation_g1/arena.json" \
        --candidate p1 --game tictacmo --conf-file "$config" --executable "$executable" \
        --games-per-profile "$eval_games" --num-simulations "$simulations" --noise
    run_arena "$run_dir/deviation_g1/arena.json" "$run_dir/deviation_g1"
    python3 tools/jpsro.py ingest "$meta_dir" "$run_dir/deviation_g1/arena.json" \
        "$run_dir/deviation_g1/games.jsonl"
    python3 tools/jpsro.py deviation-gap "$meta_dir" p1

    python3 tools/jpsro.py make-eval "$meta_dir" "$run_dir/full_g1/arena.json" \
        --game tictacmo --conf-file "$config" --executable "$executable" \
        --games-per-profile "$eval_games" --min-games "$eval_games" \
        --num-simulations "$simulations" --noise
    run_arena "$run_dir/full_g1/arena.json" "$run_dir/full_g1"
    python3 tools/jpsro.py ingest "$meta_dir" "$run_dir/full_g1/arena.json" \
        "$run_dir/full_g1/games.jsonl"
    python3 tools/jpsro.py solve "$meta_dir" --min-games "$eval_games" --tolerance "$tolerance"
    echo "final CCE: $meta_dir/meta_strategy.json"
}

[[ $# -ge 2 ]] || { usage; exit 2; }
stage=$1
run_dir=$(readlink -m "$2")
config="$run_dir/tictacmo.cfg"
meta_dir="$run_dir/meta"
executable="$repo_root/build/tictacmo/minizero_tictacmo"

case "$run_dir" in
    "$repo_root"/*) ;;
    *) die "RUN_DIR must be inside $repo_root so workers can read its files" ;;
esac
[[ "$run_dir" != *[,:[:space:]]* ]] || die "RUN_DIR cannot contain spaces, commas, or colons"

gpu=${JPSRO_GPU:-0}
iterations=${JPSRO_ITERATIONS:-30}
mid_iteration=${JPSRO_BASELINE_MID_ITERATION:-$((iterations / 2))}
games=${JPSRO_GAMES_PER_ITERATION:-2000}
training_steps=${JPSRO_TRAINING_STEPS:-500}
learner_batch=${JPSRO_LEARNER_BATCH:-1024}
selfplay_batch=${JPSRO_SELFPLAY_BATCH:-32}
cpu_threads=${JPSRO_CPU_THREADS:-4}
eval_games=${JPSRO_EVAL_GAMES:-8}
eval_threads=${JPSRO_EVAL_THREADS:-2}
simulations=${JPSRO_SIMULATIONS:-50}
seed=${JPSRO_SEED:-0}
tolerance=${JPSRO_TOLERANCE:-0.01}
baseline_port=${JPSRO_BASELINE_PORT:-10021}
oracle_port=${JPSRO_ORACLE_PORT:-10022}
common_conf="zero_use_population=false:zero_num_games_per_iteration=$games:learner_training_step=$training_steps:learner_batch_size=$learner_batch:actor_num_simulation=$simulations:program_auto_seed=false:program_seed=$seed"
(( mid_iteration > 0 && mid_iteration < iterations )) || \
    die "JPSRO_BASELINE_MID_ITERATION must be between 1 and JPSRO_ITERATIONS-1"

case "$stage" in
    baseline)
        [[ $# -eq 3 ]] || die "baseline requires CONFIG.cfg"
        train_baseline "$3"
        ;;
    reuse)
        [[ $# -eq 4 ]] || die "reuse requires EXISTING_TRAIN_DIR and CONFIG.cfg"
        reuse_baseline "$3" "$4"
        ;;
    oracle)
        [[ $# -eq 2 ]] || die "oracle only needs RUN_DIR"
        train_oracle
        ;;
    evaluate)
        [[ $# -eq 2 ]] || die "evaluate only needs RUN_DIR"
        evaluate_oracle
        ;;
    all)
        [[ $# -eq 3 ]] || die "all requires CONFIG.cfg"
        train_baseline "$3"
        train_oracle
        evaluate_oracle
        ;;
    *) usage; exit 2 ;;
esac
