#!/usr/bin/env bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"

anchor_dir=$(readlink -m "${ANCHOR_DIR:-connect3x3_maxn_smoke_01}")
anchor_iteration=${ANCHOR_ITERATION:-150}
anchor_name=${ANCHOR_NAME:-multiplayer AlphaZero}
games=${FIGHT_GAMES:-600}
gpu=${EVAL_GPU:-0}
threads=${EVAL_THREADS:-4}
simulations=${EVAL_SIMULATIONS:-50}
seed=${EVAL_SEED:-0}
noise=${EVAL_NOISE:-true}
start=${SWEEP_START:-10}
end=${SWEEP_END:-300}
interval=${SWEEP_INTERVAL:-10}
output_root=$(readlink -m "${OUTPUT_DIR:-runs/connect3x3_anchor_i${anchor_iteration}_sweep}")

die() { echo "error: $*" >&2; exit 1; }

anchor_config=${ANCHOR_CFG:-$(find "$anchor_dir" -maxdepth 1 -type f -name '*.cfg' -print | sort | tail -n 1)}
[[ -n "$anchor_config" ]] || die "anchor config not found in $anchor_dir"
anchor_steps=$(sed -nE 's/^[[:space:]]*learner_training_step=([0-9]+).*/\1/p' "$anchor_config" | head -n 1)
[[ -n "$anchor_steps" ]] || anchor_steps=500
anchor_model=${ANCHOR_MODEL:-$anchor_dir/model/weight_iter_$((anchor_iteration * anchor_steps)).pt}
[[ -f "$anchor_model" ]] || die "anchor checkpoint not found: $anchor_model"
[[ -x build/connect3x3/minizero_connect3x3 ]] || die "connect3x3 engine is not built"

noise_flag=--noise
[[ "$noise" == true ]] || noise_flag=--no-noise
mkdir -p "$output_root"

runs=(
    "old:runs/connect3x3_guided_pool_100_s0"
    "new:runs/connect3x3_guided_generalist_300_s0"
)

for spec in "${runs[@]}"; do
    label=${spec%%:*}
    run_dir=${spec#*:}
    config="$run_dir/training/training.cfg"
    [[ -f "$config" ]] || die "training config not found: $config"
    for ((iteration=start; iteration<=end; iteration+=interval)); do
        model="$run_dir/training/model/weight_iter_$((iteration * 500)).pt"
        [[ -f "$model" ]] || die "$label checkpoint not found: $model"
        output="$output_root/${label}_i${iteration}_vs_anchor_i${anchor_iteration}"
        echo "[$label] i$iteration vs $anchor_name i$anchor_iteration ($games games)"
        python3 tools/multiplayer-eval.py model-fight connect3x3 \
            "$model" "$anchor_model" \
            --conf-file-a "$config" \
            --conf-file-b "$anchor_config" \
            --names "${label}_i${iteration}" "anchor_i${anchor_iteration}" \
            --games "$games" \
            --output "$output" \
            --search-type maxn \
            --num-simulations "$simulations" \
            "$noise_flag" \
            --seed "$seed" \
            -g "$gpu" \
            --num_threads "$threads" \
            --resume
    done
done

python3 - "$output_root" "$anchor_iteration" "$anchor_name" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
anchor_iteration = int(sys.argv[2])
anchor_name = sys.argv[3]
rows = []
for path in sorted(root.glob(f"*_vs_anchor_i{anchor_iteration}/fight_summary.csv")):
    row = next(csv.DictReader(path.open()))
    valid = int(row["valid_games"])
    wins = int(row["model_a_wins"])
    draws = int(row["draws"])
    score = (wins + 0.5 * draws) / valid
    half_width = 1.96 * math.sqrt(score * (1.0 - score) / valid)
    # Match tools/multiplayer-eval.py::updated_elo exactly, with the common
    # AlphaZero anchor fixed at Elo 0.
    if score >= 1.0:
        elo = 1000.0
    elif score <= 0.0:
        elo = -1000.0
    else:
        elo = max(-1000.0, min(1000.0, 400.0 * math.log10(score / (1.0 - score))))
    label, iteration = row["model_a"].rsplit("_i", 1)
    rows.append({
        "version": label,
        "iteration": int(iteration),
        "wins": wins,
        "losses": int(row["model_b_wins"]),
        "draws": draws,
        "errors": int(row["errors"]),
        "valid_games": valid,
        "score": score,
        "ci95_half_width": half_width,
        "elo_vs_anchor": elo,
    })

rows.sort(key=lambda row: (row["iteration"], row["version"]))
output = root / "anchor_sweep_summary.csv"
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== Connect3x3 anchor sweep ==========")
print(f"Common anchor: {anchor_name} i{anchor_iteration}")
print("iteration | old score (Elo) | new score (Elo)")
by_iteration = {}
for row in rows:
    by_iteration.setdefault(row["iteration"], {})[row["version"]] = row
for iteration, values in sorted(by_iteration.items()):
    def cell(label):
        row = values.get(label)
        if row is None:
            return "missing"
        return (f'{100*row["score"]:.1f}% ± {100*row["ci95_half_width"]:.1f}% '
                f'({row["elo_vs_anchor"]:+.1f})')
    print(f"i{iteration:>3}      | {cell('old'):>24} | {cell('new'):>24}")
print(f"CSV: {output}")
PY
