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
gpu="${GPU:-0}"
threads="${THREADS:-3}"
mode="${MODE:-overwrite}"
searches="${SEARCHES:-rank maxn}"
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
if [[ ! -x "$executable" ]]; then
  echo "Missing executable: $executable"
  exit 1
fi

candidate_model="${train_dir}/model/weight_iter_${candidate_iter}.pt"
candidate_cfg="${train_dir}/blokus10_balanced_population_rank_l0p75_n50.cfg"
baseline_cfg="${repo_dir}/blokus10_maxn_01/blokus10_maxn_01.cfg"

declare -A baseline_models=(
  [nohead]="${repo_dir}/blokus10_maxn_01/model/weight_iter_${baseline_iter}.pt"
  [oldrank]="${repo_dir}/blokus10_maxn_01_head_rank_250000_rank_utility/model/weight_iter_${baseline_iter}.pt"
  [rankclass]="${repo_dir}/blokus10_maxn_rank_classification_n50/model/weight_iter_${baseline_iter}.pt"
  [adaptivetrained]="${repo_dir}/blokus10_rank_adaptive_lmax0p75_tau0p20_k12_n50/model/weight_iter_${baseline_iter}.pt"
)

need_file "$candidate_model"
need_file "$candidate_cfg"
need_file "$baseline_cfg"
for baseline in nohead oldrank rankclass adaptivetrained; do
  need_file "${baseline_models[$baseline]}"
done

make_manifest() {
  local output=$1
  local candidate_search=$2
  local baseline_name=$3
  local baseline_model=$4
  mkdir -p "$output"

  python3 - "$repo_dir" "$executable" "$candidate_model" "$candidate_cfg" \
    "$candidate_search" "$baseline_name" "$baseline_model" "$baseline_cfg" \
    "$simulations" "$games_per_seating" > "$output/arena.json" <<'PY'
import json
import sys

(repo, executable, candidate_model, candidate_cfg, candidate_search,
 baseline_name, baseline_model, baseline_cfg, simulations, games_per_seating) = sys.argv[1:]

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

def agent(name, model, cfg, search):
    values = dict(common)
    values.update({
        "nn_file_name": model,
        "actor_multiplayer_search_type": search,
        "actor_rank_utility_weight": "0.75" if search == "rank" else "0",
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
        agent(baseline_name, baseline_model, baseline_cfg, "maxn"),
        agent(candidate_name, candidate_model, candidate_cfg, candidate_search),
    ],
    "lineups": [
        [baseline_name, baseline_name, baseline_name, candidate_name],
        [baseline_name, baseline_name, candidate_name, candidate_name],
        [baseline_name, candidate_name, candidate_name, candidate_name],
    ],
    "seat_mode": "all_permutations",
    "games_per_seating": int(games_per_seating),
    "max_moves": 160,
    "terminal_passes": 4,
    "pass_mode": "elimination",
    "command_timeout": 300,
    "seed": 0,
}
json.dump(manifest, sys.stdout, indent=2)
sys.stdout.write("\n")
PY
}

eval_root="${train_dir}/evaluation"
mkdir -p "$eval_root"

echo "candidate=${candidate_model}"
echo "simulations=${simulations}, noise=true, games=700 when GAMES_PER_SEATING=50"
echo "GPU=${gpu}, THREADS=${threads}, searches=${searches}"

for search in $searches; do
  if [[ "$search" != "rank" && "$search" != "maxn" ]]; then
    echo "SEARCHES only supports rank and maxn"
    exit 1
  fi
  for baseline in nohead oldrank rankclass adaptivetrained; do
    output="${eval_root}/balancedrank${candidate_iter}_${search}_vs_${baseline}${baseline_iter}_maxn_n${simulations}_noise_700"
    make_manifest "$output" "$search" "${baseline}_maxn" "${baseline_models[$baseline]}"
    echo
    echo "${search} candidate vs ${baseline} MaxN"
    echo "result: ${output}"
    python3 "${repo_dir}/tools/multiplayer-eval.py" \
      "${output}/arena.json" "$output" \
      --gpu "$gpu" --num_threads "$threads" "--$mode" \
      2>&1 | tee "${output}/run.log"
  done
done

echo "Done. Results are under ${eval_root}."
