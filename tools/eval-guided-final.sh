#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage:
  tools/eval-guided-final.sh GAME {self|fight|all} [RUN_DIR] [BASELINE_DIR]

Supported defaults:
  connect3x3: runs/connect3x3_guided_generalist_300_s0 vs connect3x3_maxn_smoke_01
  blokus10:   runs/blokus10_guided_pool_500_s0       vs blokus10_maxn_01

Optional environment variables:
  EVAL_GPU=0                 GPU list (also accepts 0123)
  EVAL_THREADS=4             parallel arena workers
  EVAL_SIMULATIONS=50        MCTS simulations
  EVAL_SEED=0                deterministic program seed sequence
  EVAL_NOISE=true            true: Dirichlet noise; false: no noise
  SELF_EVAL_GAMES=120        games for each consecutive checkpoint pair
  SELF_EVAL_INTERVAL=10      compare checkpoints ten training iterations apart
  SELF_EVAL_START=0          first checkpoint index
  FIGHT_GAMES=600            total seat-balanced model-fight games
  GUIDED_ITERATION=300       guided checkpoint to evaluate
  BASELINE_ITERATION=300     baseline checkpoint to evaluate
  BASELINE_NAME=alphazero    label used in output files
EOF
}

die() { echo "error: $*" >&2; exit 1; }

game=${1:-}
mode=${2:-}
[[ "$mode" == self || "$mode" == fight || "$mode" == all ]] || {
    usage
    exit 2
}

case "$game" in
    connect3x3)
        default_run=runs/connect3x3_guided_generalist_300_s0
        default_baseline=connect3x3_maxn_smoke_01
        default_iteration=300
        default_fight_games=600
        ;;
    blokus10)
        default_run=runs/blokus10_guided_pool_500_s0
        default_baseline=blokus10_maxn_01
        default_iteration=500
        default_fight_games=700
        ;;
    *) die "unsupported game: $game" ;;
esac

run_dir=$(readlink -m "${3:-$default_run}")
baseline_dir=$(readlink -m "${4:-$default_baseline}")
training_dir="$run_dir/training"
guided_config="$training_dir/training.cfg"
settings="$run_dir/guided_pool.settings"

gpu=${EVAL_GPU:-0}
threads=${EVAL_THREADS:-4}
simulations=${EVAL_SIMULATIONS:-50}
seed=${EVAL_SEED:-0}
noise=${EVAL_NOISE:-true}
self_games=${SELF_EVAL_GAMES:-120}
self_interval=${SELF_EVAL_INTERVAL:-10}
self_start=${SELF_EVAL_START:-0}
fight_games=${FIGHT_GAMES:-$default_fight_games}
guided_iteration=${GUIDED_ITERATION:-$default_iteration}
baseline_iteration=${BASELINE_ITERATION:-$guided_iteration}
baseline_label=${BASELINE_NAME:-alphazero}

[[ -d "$training_dir/model" ]] || die "guided model directory not found: $training_dir/model"
[[ -f "$guided_config" ]] || die "guided config not found: $guided_config"
[[ -x "build/$game/minizero_$game" ]] || die "engine is not built: build/$game/minizero_$game"

training_steps=500
if [[ -f "$settings" ]]; then
    saved_steps=$(sed -nE 's/.*(^| )steps=([0-9]+)( |$).*/\2/p' "$settings")
    [[ -z "$saved_steps" ]] || training_steps=$saved_steps
fi

noise_flag=--noise
[[ "$noise" == true ]] || noise_flag=--no-noise
common=(--search-type maxn --num-simulations "$simulations" "$noise_flag"
        --seed "$seed" -g "$gpu" --num_threads "$threads" --resume)

if [[ "$mode" == self || "$mode" == all ]]; then
    self_output="$run_dir/evaluations/final_self_eval_maxn_every_${self_interval}i"
    echo "Self-eval: $game, $run_dir"
    echo "Output:    $self_output"
    python3 tools/multiplayer-eval.py self-eval "$game" "$training_dir" \
        --conf-file "$guided_config" \
        --interval "$self_interval" \
        --games "$self_games" \
        --start-index "$self_start" \
        --output "$self_output" \
        "${common[@]}"
fi

if [[ "$mode" == fight || "$mode" == all ]]; then
    baseline_config=$(find "$baseline_dir" -maxdepth 1 -type f -name '*.cfg' -print | sort | tail -n 1)
    [[ -n "$baseline_config" ]] || die "baseline config not found in $baseline_dir"
    baseline_steps=$(sed -nE 's/^[[:space:]]*learner_training_step=([0-9]+).*/\1/p' "$baseline_config" | head -n 1)
    [[ -n "$baseline_steps" ]] || baseline_steps=500

    guided_model="$training_dir/model/weight_iter_$((guided_iteration * training_steps)).pt"
    baseline_model="$baseline_dir/model/weight_iter_$((baseline_iteration * baseline_steps)).pt"
    [[ -f "$guided_model" ]] || die "guided checkpoint not found: $guided_model"
    [[ -f "$baseline_model" ]] || die "baseline checkpoint not found: $baseline_model"

    guided_name="guided_i${guided_iteration}"
    baseline_name="${baseline_label}_i${baseline_iteration}"
    fight_output="$run_dir/evaluations/final_strength/maxn_${guided_name}_vs_${baseline_name}"
    echo "Model fight: $guided_name vs $baseline_name, $fight_games total seat-balanced games"
    echo "Output:      $fight_output"
    python3 tools/multiplayer-eval.py model-fight "$game" \
        "$guided_model" "$baseline_model" \
        --conf-file-a "$guided_config" \
        --conf-file-b "$baseline_config" \
        --names "$guided_name" "$baseline_name" \
        --games "$fight_games" \
        --output "$fight_output" \
        "${common[@]}"
fi
