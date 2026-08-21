#!/usr/bin/env bash
set -euo pipefail

# Run all held-out 700-game matchups inside the container from /workspace.
# By default each baseline is tested against both the complete Rank Utility
# method and the same trained network using plain MaxN.

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_dir="${repo_dir}/${TRAIN_DIR:-blokus10_balanced_population_rank_l0p75_n50}"
candidate_iter="${MODEL:-latest}"
baseline_iter="${BASELINE_MODEL:-250000}"
simulations="${SIMULATIONS:-50}"
games_per_seating="${GAMES_PER_SEATING:-50}"
num_games="${NUM_GAMES:-}"
gpu="${GPU:-0}"
threads="${THREADS:-3}"
mode="${MODE:-overwrite}"
searches="${SEARCHES:-rank maxn}"
baselines="${BASELINES:-nohead oldrank rankclass adaptivetrained}"
rank_utility_weight="${RANK_UTILITY_WEIGHT:-0.75}"
output_tag="${OUTPUT_TAG:-}"
executable="${EXECUTABLE:-${repo_dir}/build/blokus10/minizero_blokus10}"

latest_checkpoint_iter() {
  find "$1" -maxdepth 1 -type f -name 'weight_iter_*.pt' -printf '%f\n' \
    | sed -E 's/^weight_iter_([0-9]+)\.pt$/\1/' | sort -n | tail -1
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1"
    exit 1
  fi
}

if [[ "$candidate_iter" == "latest" ]]; then
  candidate_iter="$(latest_checkpoint_iter "${train_dir}/model")"
fi
if [[ -z "$candidate_iter" ]]; then
  echo "No candidate checkpoint found under ${train_dir}/model"
  exit 1
fi
if [[ "$mode" != "overwrite" && "$mode" != "resume" ]]; then
  echo "MODE must be overwrite or resume"
  exit 1
fi
if ! [[ "$rank_utility_weight" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "RANK_UTILITY_WEIGHT must be between 0 and 1"
  exit 1
fi
if [[ -n "$output_tag" && ! "$output_tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "OUTPUT_TAG may contain only letters, digits, underscores, and hyphens"
  exit 1
fi
if [[ -n "$num_games" && ( ! "$num_games" =~ ^[1-9][0-9]*$ ) ]]; then
  echo "NUM_GAMES must be a positive integer"
  exit 1
fi
if [[ ! -x "$executable" ]]; then
  echo "Missing executable: $executable"
  exit 1
fi

candidate_model="${train_dir}/model/weight_iter_${candidate_iter}.pt"
candidate_cfg="${CANDIDATE_CFG:-${train_dir}/$(basename "${train_dir}").cfg}"
default_baseline_cfg="${repo_dir}/blokus10_maxn_01/blokus10_maxn_01.cfg"

declare -A baseline_models=(
  [nohead]="${repo_dir}/blokus10_maxn_01/model/weight_iter_${baseline_iter}.pt"
  [poolonly]="${repo_dir}/blokus10_balanced_population_maxn_nofilm_n50/model/weight_iter_${baseline_iter}.pt"
  [oldrank]="${repo_dir}/blokus10_maxn_01_head_rank_250000_rank_utility/model/weight_iter_${baseline_iter}.pt"
  [rankclass]="${repo_dir}/blokus10_maxn_rank_classification_n50/model/weight_iter_${baseline_iter}.pt"
  [adaptivetrained]="${repo_dir}/blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50/model/weight_iter_${baseline_iter}.pt"
  [oldpopulation]="${repo_dir}/blokus10_behavior_population_n50/model/weight_iter_${baseline_iter}.pt"
  [filmrank]="${repo_dir}/blokus10_balanced_population_rank_l0p75_n50/model/weight_iter_${baseline_iter}.pt"
  [poolrank]="${repo_dir}/blokus10_balanced_population_rank_nofilm_l0p75_n50/model/weight_iter_${baseline_iter}.pt"
)

declare -A baseline_cfgs=(
  [nohead]="${default_baseline_cfg}"
  [poolonly]="${repo_dir}/blokus10_balanced_population_maxn_nofilm_n50/blokus10_balanced_population_maxn_nofilm_n50.cfg"
  [oldrank]="${default_baseline_cfg}"
  [rankclass]="${default_baseline_cfg}"
  [adaptivetrained]="${default_baseline_cfg}"
  [oldpopulation]="${repo_dir}/blokus10_behavior_population_n50/blokus10_behavior_population_n50.cfg"
  [filmrank]="${repo_dir}/blokus10_balanced_population_rank_l0p75_n50/blokus10_balanced_population_rank_l0p75_n50.cfg"
  [poolrank]="${repo_dir}/blokus10_balanced_population_rank_nofilm_l0p75_n50/blokus10_balanced_population_rank_nofilm_l0p75_n50.cfg"
)

need_file "$candidate_model"
need_file "$candidate_cfg"
for baseline in $baselines; do
  if [[ -z "${baseline_models[$baseline]+x}" ]]; then
    echo "Unknown baseline: $baseline"
    echo "Supported baselines: nohead poolonly oldrank rankclass adaptivetrained oldpopulation filmrank poolrank"
    exit 1
  fi
  need_file "${baseline_models[$baseline]}"
  need_file "${baseline_cfgs[$baseline]}"
done

make_manifest() {
  local output=$1
  local candidate_search=$2
  local baseline_name=$3
  local baseline_model=$4
  local baseline_search=$5
  local baseline_rank_weight=$6
  mkdir -p "$output"

  python3 - "$repo_dir" "$executable" "$candidate_model" "$candidate_cfg" \
    "$candidate_search" "$baseline_name" "$baseline_model" "$baseline_cfg" \
    "$baseline_search" "$simulations" "$games_per_seating" "$num_games" \
    "$rank_utility_weight" "$baseline_rank_weight" > "$output/arena.json" <<'PY'
import json
import sys

(repo, executable, candidate_model, candidate_cfg, candidate_search,
 baseline_name, baseline_model, baseline_cfg, baseline_search, simulations,
 games_per_seating, num_games, rank_utility_weight,
 baseline_rank_weight) = sys.argv[1:]

common = {
    "program_seed": "{seed}",
    "program_auto_seed": "false",
    "actor_num_simulation": simulations,
    "actor_use_dirichlet_noise": "true",
    "actor_use_random_rotation_features": "false",
    "actor_select_action_by_count": "true",
    "actor_select_action_by_softmax_count": "false",
    "actor_use_gumbel": "false",
    "actor_use_gumbel_noise": "false",
    "actor_mcts_value_rescale": "false",
    "zero_disable_resign_ratio": "1",
    "zero_actor_intermediate_sequence_length": "0",
    "zero_use_population": "false",
}

def agent(name, model, cfg, search, rank_weight):
    values = dict(common)
    values.update({
        "nn_file_name": model,
        "actor_multiplayer_search_type": search,
        "actor_rank_utility_weight": rank_weight if search == "rank" else "0",
    })
    conf = ":".join(f"{key}={value}" for key, value in values.items())
    return {
        "name": name,
        "cwd": repo,
        "env": {"OMP_NUM_THREADS": "1"},
        "command": [executable, "-mode", "console", "-conf_file", cfg, "-conf_str", conf],
    }

candidate_name = f"candidate_{candidate_search}"
manifest = {
    "game": "blokus10",
    "players": ["b", "w", "r", "g"],
    "agents": [
        agent(baseline_name, baseline_model, baseline_cfg, baseline_search, baseline_rank_weight),
        agent(candidate_name, candidate_model, candidate_cfg, candidate_search, rank_utility_weight),
    ],
    "lineups": [
        [baseline_name, baseline_name, baseline_name, candidate_name],
        [baseline_name, baseline_name, candidate_name, candidate_name],
        [baseline_name, candidate_name, candidate_name, candidate_name],
    ],
    "seat_mode": "all_permutations",
    "max_moves": 160,
    "terminal_passes": 4,
    "pass_mode": "elimination",
    "command_timeout": 300,
    "seed": 0,
}
if num_games:
    manifest["num_games"] = int(num_games)
else:
    manifest["games_per_seating"] = int(games_per_seating)
json.dump(manifest, sys.stdout, indent=2)
sys.stdout.write("\n")
PY
}

eval_root="${train_dir}/evaluation"
mkdir -p "$eval_root"

echo "candidate=${candidate_model}"
if [[ -n "$num_games" ]]; then
  total_games="$num_games"
else
  total_games=$((games_per_seating * 14))
fi
echo "simulations=${simulations}, noise=true, games=${total_games}"
echo "GPU=${gpu}, THREADS=${threads}, searches=${searches}, baselines=${baselines}"
echo "rank utility weight=${rank_utility_weight}, output tag=${output_tag:-none}"

for search in $searches; do
  if [[ "$search" != "rank" && "$search" != "maxn" ]]; then
    echo "SEARCHES only supports rank and maxn"
    exit 1
  fi
  if [[ "$search" == "rank" && "$rank_utility_weight" =~ ^0([.]0+)?$ ]]; then
    echo "Rank search requires RANK_UTILITY_WEIGHT > 0; use SEARCHES=maxn for the lambda=0 control."
    exit 1
  fi
  for baseline in $baselines; do
    baseline_search=maxn
    baseline_rank_weight=0
    if [[ "$baseline" == poolrank || "$baseline" == filmrank ]]; then
      baseline_search=rank
      baseline_rank_weight="${BASELINE_RANK_UTILITY_WEIGHT:-0.75}"
    fi
    output="${eval_root}/balancedrank${candidate_iter}_${search}${output_tag:+_${output_tag}}_vs_${baseline}${baseline_iter}_${baseline_search}_n${simulations}_noise_${total_games}"
    baseline_cfg="${baseline_cfgs[$baseline]}"
    make_manifest "$output" "$search" "${baseline}_${baseline_search}" \
      "${baseline_models[$baseline]}" "$baseline_search" "$baseline_rank_weight"
    echo
    echo "${search} candidate vs ${baseline} ${baseline_search}"
    echo "result: ${output}"
    python3 "${repo_dir}/tools/multiplayer-eval.py" \
      "${output}/arena.json" "$output" \
      --gpu "$gpu" --num_threads "$threads" "--$mode" \
      2>&1 | tee "${output}/run.log"
  done
done

echo "Done. Results are under ${eval_root}."
