#!/usr/bin/env python3
"""Plot detailed paper figures from paper-multiplayer-summary.py outputs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def errorbar_panel(ax, x, rows, value_key, low_key, high_key, null, ylabel, percent=False):
    scale = 100.0 if percent else 1.0
    values = [scale * float(row[value_key]) for row in rows]
    lows = [scale * float(row[low_key]) for row in rows]
    highs = [scale * float(row[high_key]) for row in rows]
    ax.errorbar(
        x,
        values,
        yerr=[[value - low for value, low in zip(values, lows)],
              [high - value for value, high in zip(values, highs)]],
        color="#1769aa",
        ecolor="#4c78a8",
        marker="o",
        markersize=5,
        linewidth=1.6,
        capsize=3,
    )
    ax.axhline(scale * null, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        label = f"{value:.1f}%" if percent else f"{value:+.3f}"
        ax.annotate(label, (index, value), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)


def plot_overview(summary_dir, rows):
    complete = [row for row in rows if row["complete"] == "true"]
    labels = [row["pair"].replace("_vs_", " vs ") for row in complete]
    x = np.arange(len(complete))
    fig, axes = plt.subplots(4, 1, figsize=(16, 17), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.0, 1.0, 1.0]})

    errorbar_panel(
        axes[0], x, complete,
        "target_first_place_share", "target_first_place_ci_low", "target_first_place_ci_high",
        0.5, "Newer-model first-place credit (%)", percent=True,
    )
    axes[0].set_ylim(25, 75)
    axes[0].set_title("A. Game-level first-place credit with 95% stratified bootstrap CI", loc="left", weight="bold")

    wins = np.array([int(row["target_model_wins"]) for row in complete], dtype=float)
    draws = np.array([int(row["shared_model_top_draws"]) for row in complete], dtype=float)
    losses = np.array([int(row["target_model_losses"]) for row in complete], dtype=float)
    totals = wins + draws + losses
    win_pct, draw_pct, loss_pct = 100 * wins / totals, 100 * draws / totals, 100 * losses / totals
    axes[1].bar(x, win_pct, color="#2e7d32", label="newer top only (win)")
    axes[1].bar(x, draw_pct, bottom=win_pct, color="#f9a825", label="both models tied at top")
    axes[1].bar(x, loss_pct, bottom=win_pct + draw_pct, color="#c62828", label="older top only (loss)")
    axes[1].axhline(50, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Games (%)")
    axes[1].set_title("B. Model-level top outcome (one observation per physical game)", loc="left", weight="bold")
    axes[1].legend(ncol=3, loc="upper center", fontsize=9)
    for index, (w, d, loss) in enumerate(zip(wins.astype(int), draws.astype(int), losses.astype(int))):
        axes[1].text(index, 3, f"{w}/{d}/{loss}", rotation=90, ha="center", va="bottom", fontsize=7, color="white")

    errorbar_panel(
        axes[2], x, complete,
        "return_advantage", "return_advantage_ci_low", "return_advantage_ci_high",
        0.0, "Newer − older normalized return",
    )
    axes[2].set_title("C. Within-game mean return advantage", loc="left", weight="bold")

    errorbar_panel(
        axes[3], x, complete,
        "rank_advantage", "rank_advantage_ci_low", "rank_advantage_ci_high",
        0.0, "Rank advantage (positive = newer better)",
    )
    axes[3].set_title("D. Within-game mean-rank advantage", loc="left", weight="bold")
    axes[3].set_xticks(x, labels, rotation=50, ha="right")
    axes[3].set_xlabel("Checkpoint comparison (newer vs older)")

    fig.suptitle(
        "Blokus checkpoint self-evaluation: paper-oriented game-level analysis\n"
        "Balanced AAAB/AABB/ABBB lineups and absolute-seat assignments",
        fontsize=17,
        weight="bold",
    )
    fig.text(0.5, 0.002,
             "Numbers in panel B are newer wins / shared-top draws / newer losses. "
             "A tied first place splits one unit of credit among tied seats in panel A.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    save_figure(fig, summary_dir, "paper_detailed_overview")


def annotated_heatmap(ax, matrix, row_labels, column_labels, title, colorbar_label, fmt, center_zero=True):
    finite = matrix[np.isfinite(matrix)]
    limit = max(abs(finite.min()), abs(finite.max())) if finite.size else 1.0
    if center_zero:
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", weight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(column, row, fmt.format(value), ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.58 * limit else "black")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    return image


def plot_compositions(summary_dir, pair_rows, composition_rows):
    complete_pairs = [row["pair"] for row in pair_rows if row["complete"] == "true"]
    by_key = {(row["pair"], int(row["target_seats"])): row for row in composition_rows}
    compositions = [1, 2, 3]
    excess = np.full((len(complete_pairs), 3), np.nan)
    returns = np.full((len(complete_pairs), 3), np.nan)
    for row_index, pair in enumerate(complete_pairs):
        for column, count in enumerate(compositions):
            row = by_key.get((pair, count))
            if row:
                excess[row_index, column] = 100 * float(row["excess_first_place_share"])
                returns[row_index, column] = float(row["return_advantage"])
    row_labels = [pair.replace("_vs_", " vs ") for pair in complete_pairs]
    column_labels = ["1 newer / 3 older", "2 newer / 2 older", "3 newer / 1 older"]
    fig, axes = plt.subplots(1, 2, figsize=(17, max(9, len(complete_pairs) * 0.55)), constrained_layout=True)
    annotated_heatmap(
        axes[0], excess, row_labels, column_labels,
        "A. First-place share above equal-model composition baseline",
        "Excess first-place share (percentage points)", "{:+.1f}",
    )
    annotated_heatmap(
        axes[1], returns, row_labels, column_labels,
        "B. Within-game return advantage by lineup composition",
        "Newer − older normalized return", "{:+.3f}",
    )
    axes[0].set_ylabel("Checkpoint comparison")
    fig.suptitle(
        "Composition robustness: does the conclusion change across AAAB, AABB, and ABBB?",
        fontsize=16,
        weight="bold",
    )
    save_figure(fig, summary_dir, "paper_composition_breakdown")


def plot_seats(summary_dir, pair_rows, seat_rows):
    complete = [row for row in pair_rows if row["complete"] == "true"]
    pairs = [row["pair"] for row in complete]
    target_for_pair = {row["pair"]: row["target"] for row in complete}
    reference_for_pair = {row["pair"]: row["reference"] for row in complete}
    by_key = {(row["pair"], row["agent"], int(row["seat_index"])): row for row in seat_rows}
    matrices = [np.full((len(pairs), 4), np.nan) for _ in range(4)]
    for row_index, pair in enumerate(pairs):
        for seat in range(4):
            target = by_key.get((pair, target_for_pair[pair], seat))
            reference = by_key.get((pair, reference_for_pair[pair], seat))
            if target:
                matrices[0][row_index, seat] = 100 * (float(target["seat_top1_rate"]) - 0.25)
                matrices[2][row_index, seat] = float(target["mean_return"])
            if reference:
                matrices[1][row_index, seat] = 100 * (float(reference["seat_top1_rate"]) - 0.25)
                matrices[3][row_index, seat] = float(reference["mean_return"])
    row_labels = [pair.replace("_vs_", " vs ") for pair in pairs]
    seat_labels = ["B / P1", "W / P2", "R / P3", "G / P4"]
    fig, axes = plt.subplots(2, 2, figsize=(17, max(15, len(pairs) * 0.95)), constrained_layout=True)
    annotated_heatmap(axes[0, 0], matrices[0], row_labels, seat_labels,
                      "A. Newer model: seat top-1 deviation from 25%",
                      "Percentage points above/below 25%", "{:+.1f}")
    annotated_heatmap(axes[0, 1], matrices[1], row_labels, seat_labels,
                      "B. Older model: seat top-1 deviation from 25%",
                      "Percentage points above/below 25%", "{:+.1f}")
    annotated_heatmap(axes[1, 0], matrices[2], row_labels, seat_labels,
                      "C. Newer model: mean normalized return by absolute seat",
                      "Mean normalized return", "{:+.3f}")
    annotated_heatmap(axes[1, 1], matrices[3], row_labels, seat_labels,
                      "D. Older model: mean normalized return by absolute seat",
                      "Mean normalized return", "{:+.3f}")
    axes[0, 0].set_ylabel("Checkpoint comparison")
    axes[1, 0].set_ylabel("Checkpoint comparison")
    fig.suptitle(
        "Absolute-seat diagnostics (appendix figure; not the primary model win rate)",
        fontsize=16,
        weight="bold",
    )
    save_figure(fig, summary_dir, "paper_seat_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_dir", type=Path)
    args = parser.parse_args()
    summary_dir = args.summary_dir.resolve()
    expected_outputs = [
        summary_dir / f"paper_{name}.{extension}"
        for name in ("detailed_overview", "composition_breakdown", "seat_diagnostics")
        for extension in ("png", "pdf")
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing figures: {existing[0]}")
    pair_rows = read_csv(summary_dir / "pair_summary.csv")
    composition_rows = read_csv(summary_dir / "composition_summary.csv")
    seat_rows = read_csv(summary_dir / "seat_diagnostics.csv")
    plot_overview(summary_dir, pair_rows)
    plot_compositions(summary_dir, pair_rows, composition_rows)
    plot_seats(summary_dir, pair_rows, seat_rows)
    for path in expected_outputs:
        print(path)


if __name__ == "__main__":
    main()
