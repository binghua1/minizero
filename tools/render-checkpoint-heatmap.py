#!/usr/bin/env python3
"""Render a clean checkpoint heatmap from an existing pair_summary.csv."""

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--game", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-scale", type=int, default=1)
    parser.add_argument("--checkpoint-prefix", default="i")
    parser.add_argument("--output-stem", default="crossplay_heatmap")
    args = parser.parse_args()

    root = args.evaluation_dir.resolve()
    rows = list(csv.DictReader((root / "pair_summary.csv").open()))
    if not rows:
        raise SystemExit(f"empty pair summary: {root / 'pair_summary.csv'}")

    checkpoints = sorted(
        {int(row["row_checkpoint"]) for row in rows}
        | {int(row["column_checkpoint"]) for row in rows}
    )
    index = {checkpoint: i for i, checkpoint in enumerate(checkpoints)}
    matrix = [[0.5 for _ in checkpoints] for _ in checkpoints]
    margins = [[0.0 for _ in checkpoints] for _ in checkpoints]
    for row in rows:
        a = int(row["row_checkpoint"])
        b = int(row["column_checkpoint"])
        score = float(row["row_model_score"])
        low = float(row["row_score_ci95_low"])
        high = float(row["row_score_ci95_high"])
        matrix[index[a]][index[b]] = score
        matrix[index[b]][index[a]] = 1.0 - score
        margin = 0.5 * (high - low)
        margins[index[a]][index[b]] = margin
        margins[index[b]][index[a]] = margin

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    size = max(6.8, 1.08 * len(checkpoints))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(
        matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    )
    labels = [
        f"{args.checkpoint_prefix}{checkpoint // args.checkpoint_scale}"
        for checkpoint in checkpoints
    ]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Column checkpoint")
    axis.set_ylabel("Row checkpoint")
    games = rows[0].get("scheduled_games") or rows[0].get("valid_games") or "?"
    axis.set_title(f"{args.game} — {args.method} ({games} games/pair)")

    for i in range(len(checkpoints)):
        for j in range(len(checkpoints)):
            if i == j:
                label = "—"
                color = "black"
            else:
                score = matrix[i][j]
                label = f"{100.0 * score:.1f}% ± {100.0 * margins[i][j]:.1f}%"
                color = "white" if score < 0.25 or score > 0.75 else "black"
            axis.text(j, i, label, ha="center", va="center", color=color, fontsize=8)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-model score (W + 0.5D)")
    figure.tight_layout()
    png = root / f"{args.output_stem}.png"
    pdf = root / f"{args.output_stem}.pdf"
    figure.savefig(png, dpi=200)
    figure.savefig(pdf)
    plt.close(figure)
    print(png)


if __name__ == "__main__":
    main()
