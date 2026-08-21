#!/usr/bin/env python3
"""Summarize Blokus multiplayer evaluation directories with 95% CIs.

The primary model-level rate is W + 0.5 D per physical game.  Its interval is
a percentile bootstrap stratified by the exact candidate-seat signature.  This
preserves the balanced seat schedule and avoids treating seats in one game as
independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from statistics import NormalDist


PLAYERS = ["B", "W", "R", "G"]
CASE_ORDER = {"1v3": 0, "2v2": 1, "3v1": 2, "all": 3}
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260807


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_ci(records: list[dict], samples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for score values, stratified by seat signature."""
    strata: dict[tuple[bool, ...], list[float]] = {}
    for record in records:
        strata.setdefault(record["signature"], []).append(record["score"])
    rng = random.Random(seed)
    estimates = []
    total = len(records)
    for _ in range(samples):
        total_score = 0.0
        for values in strata.values():
            total_score += sum(rng.choices(values, k=len(values)))
        estimates.append(total_score / total)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def discover_runs(evaluation_dir: Path) -> list[tuple[str, Path, dict]]:
    """Return (label, games_path, arena) for direct and nested result dirs."""
    runs = []
    for directory in sorted(path for path in evaluation_dir.iterdir() if path.is_dir()):
        direct = directory / "games.jsonl"
        if direct.exists():
            runs.append((directory.name, direct, read_json(directory / "arena.json")))
            continue
        nested = sorted(directory.glob("*/games.jsonl"))
        for games_path in nested:
            arena_path = games_path.parent / "arena.json"
            if arena_path.exists():
                runs.append((f"{directory.name}/{games_path.parent.name}", games_path, read_json(arena_path)))
    return runs


def candidate_and_reference(arena: dict) -> tuple[str, str]:
    names = [agent["name"] for agent in arena.get("agents", [])]
    candidates = [name for name in names if "candidate" in name]
    if candidates:
        candidate = candidates[0]
        reference = next(name for name in names if name != candidate)
        return candidate, reference
    self_eval = arena.get("self_eval", {})
    newer = self_eval.get("newer_iteration")
    if newer is not None and f"iter_{newer}" in names:
        candidate = f"iter_{newer}"
        reference = next(name for name in names if name != candidate)
        return candidate, reference
    if len(names) != 2:
        raise ValueError(f"cannot identify two models from agents={names}")
    return names[0], names[1]


def load_records(games_path: Path, candidate: str, reference: str) -> tuple[list[dict], int]:
    records = []
    errors = 0
    latest = {}
    with games_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            # Resume mode appends a replacement record for a failed game ID.
            # Only the last record for each physical game is authoritative.
            latest[raw["game_id"]] = (line_number, raw)
    for line_number, raw in latest.values():
        if raw.get("error"):
            errors += 1
            continue
        seating = raw["seating"]
        returns = [float(value) for value in raw["returns"]]
        if len(seating) != len(returns):
            raise ValueError(f"seat/return mismatch in {games_path}:{line_number}")
        if set(seating) != {candidate, reference}:
            raise ValueError(f"unexpected agents in {games_path}:{line_number}: {set(seating)}")
        top = max(returns)
        top_seats = [index for index, value in enumerate(returns) if value == top]
        top_models = {seating[index] for index in top_seats}
        if top_models == {candidate}:
            outcome, score = "W", 1.0
        elif top_models == {reference}:
            outcome, score = "L", 0.0
        else:
            outcome, score = "D", 0.5
        candidate_seats = sum(agent == candidate for agent in seating)
        records.append({
            "seating": seating,
            "returns": returns,
            "candidate_seats": candidate_seats,
            "case": f"{candidate_seats}v{len(seating) - candidate_seats}",
            "signature": tuple(agent == candidate for agent in seating),
            "outcome": outcome,
            "score": score,
        })
    return records, errors


def stats(records: list[dict], samples: int, seed: int) -> dict:
    wins = sum(record["outcome"] == "W" for record in records)
    draws = sum(record["outcome"] == "D" for record in records)
    losses = sum(record["outcome"] == "L" for record in records)
    games = len(records)
    score = (wins + 0.5 * draws) / games if games else float("nan")
    low, high = bootstrap_ci(records, samples, seed) if records else (float("nan"), float("nan"))
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate_pct": 100 * score if games else float("nan"),
        "ci_low_pct": 100 * low,
        "ci_high_pct": 100 * high,
    }


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    output = (args.output or evaluation_dir / "organized_with_ci").resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    composition_rows = []
    seat_rows = []
    report_sections = []
    inventory_rows = []

    discovered = discover_runs(evaluation_dir)
    discovered_paths = {games_path.parent for _, games_path, _ in discovered}
    for directory in sorted(path for path in evaluation_dir.iterdir() if path.is_dir()):
        if directory.resolve() == output:
            continue
        if directory in discovered_paths or any(path.parent == directory for path in discovered_paths):
            continue
        if (directory / "games.jsonl").exists() or list(directory.glob("*/games.jsonl")):
            continue
        inventory_rows.append({"run": directory.name, "status": "empty_or_missing_arena", "games": 0})

    for index, (label, games_path, arena) in enumerate(discovered):
        candidate, reference = candidate_and_reference(arena)
        records, errors = load_records(games_path, candidate, reference)
        inventory_rows.append({"run": label, "status": "complete" if records else "empty", "games": len(records), "errors": errors})
        if not records:
            continue
        seed = args.seed + index * 1000
        overall = stats(records, args.bootstrap_samples, seed)
        summary_rows.append({
            "run": label,
            "candidate": candidate,
            "reference": reference,
            "games": overall["games"],
            "candidate_wins": overall["wins"],
            "draws": overall["draws"],
            "candidate_losses": overall["losses"],
            "candidate_win_rate_pct": fmt(overall["win_rate_pct"], 4),
            "ci95_low_pct": fmt(overall["ci_low_pct"], 4),
            "ci95_high_pct": fmt(overall["ci_high_pct"], 4),
            "errors": errors,
        })

        section = [
            f"## `{label}`",
            "",
            f"Candidate: `{candidate}`; reference: `{reference}`.  Valid games: **{len(records)}**; errors: **{errors}**.",
            "",
            "### 模型勝率（每場遊戲）",
            "",
            "| 配置 | 場數 | 勝 | 和 | 負 | 勝率（W+0.5D） | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---|"]
        for case in ["1v3", "2v2", "3v1", "all"]:
            subset = records if case == "all" else [record for record in records if record["case"] == case]
            if not subset:
                continue
            result = stats(subset, args.bootstrap_samples, seed + CASE_ORDER[case] + 1)
            composition_rows.append({
                "run": label,
                "candidate": candidate,
                "reference": reference,
                "case": case,
                "games": result["games"],
                "candidate_wins": result["wins"],
                "draws": result["draws"],
                "candidate_losses": result["losses"],
                "candidate_win_rate_pct": fmt(result["win_rate_pct"], 4),
                "ci95_low_pct": fmt(result["ci_low_pct"], 4),
                "ci95_high_pct": fmt(result["ci_high_pct"], 4),
                "errors": errors,
            })
            section.append(
                f"| {case} | {result['games']} | {result['wins']} | {result['draws']} | {result['losses']} | {result['win_rate_pct']:.2f}% | {result['ci_low_pct']:.2f}%–{result['ci_high_pct']:.2f}% |")

        section += ["", "### 座位勝率（該座位成為唯一勝者）", "", "| 模型 | 座位 | 場數 | 勝 | 和 | 負 | 座位勝率 | 95% CI（Wilson） |", "|---|---|---:|---:|---:|---:|---:|---|"]
        for model in [candidate, reference]:
            for seat_index, seat in enumerate(PLAYERS):
                model_records = [record for record in records if record["seating"][seat_index] == model]
                if not model_records:
                    continue
                wins = 0
                draws = 0
                for record in model_records:
                    top = max(record["returns"])
                    top_seats = [i for i, value in enumerate(record["returns"]) if value == top]
                    top_models = {record["seating"][i] for i in top_seats}
                    if len(top_seats) == 1 and top_models == {model} and seat_index in top_seats:
                        wins += 1
                    elif record["outcome"] == "D":
                        draws += 1
                losses = len(model_records) - wins - draws
                low, high = wilson_ci(wins, len(model_records))
                seat_rows.append({
                    "run": label,
                    "model": model,
                    "seat": seat,
                    "games": len(model_records),
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "seat_unique_win_rate_pct": fmt(100 * wins / len(model_records), 4),
                    "ci95_low_pct": fmt(100 * low, 4),
                    "ci95_high_pct": fmt(100 * high, 4),
                })
                section.append(f"| {model} | {seat} | {len(model_records)} | {wins} | {draws} | {losses} | {100*wins/len(model_records):.2f}% | {100*low:.2f}%–{100*high:.2f}% |")
        report_sections.append("\n".join(section))

    summary_rows.sort(key=lambda row: row["run"])
    composition_rows.sort(key=lambda row: (row["run"], CASE_ORDER[row["case"]]))
    seat_rows.sort(key=lambda row: (row["run"], row["model"], PLAYERS.index(row["seat"])))
    inventory_rows.sort(key=lambda row: row["run"])
    write_csv(output / "evaluation_summary_with_ci.csv", summary_rows)
    write_csv(output / "evaluation_composition_with_ci.csv", composition_rows)
    write_csv(output / "evaluation_seat_with_ci.csv", seat_rows)
    write_csv(output / "evaluation_inventory.csv", inventory_rows)

    overall_lines = [
        "# Blokus evaluation 結果（含 95% CI）", "",
        f"來源：`{evaluation_dir}`。本報告整理 {len(summary_rows)} 個有有效對局的 evaluation run。",
        "",
        "## 口徑",
        "",
        "- 主要勝率是每場遊戲的模型分數：唯一勝者 = 1、模型間共同最高分 = 0.5、對手唯一勝者 = 0；因此為 `(W + 0.5×D) / 場數`。",
        f"- 主要 95% CI 是依照精確 candidate 座位配置分層的 {args.bootstrap_samples:,}-次 percentile bootstrap；同一場遊戲的多個座位不會被當成獨立樣本。",
        "- 座位表的勝率是該座位成為唯一勝者的比例，CI 採 95% Wilson interval；它是座位診斷，不是模型整體勝率。",
        "- `evaluation_inventory.csv` 會列出空的或缺少可用結果的目錄。",
        "",
        "## 總表",
        "",
        "| Evaluation | Candidate | Reference | 場數 | 勝 | 和 | 負 | 勝率 | 95% CI | Errors |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        overall_lines.append(f"| `{row['run']}` | `{row['candidate']}` | `{row['reference']}` | {row['games']} | {row['candidate_wins']} | {row['draws']} | {row['candidate_losses']} | {row['candidate_win_rate_pct']}% | {row['ci95_low_pct']}%–{row['ci95_high_pct']}% | {row['errors']} |")
    overall_lines += ["", "## 各 evaluation 詳細表", ""] + report_sections
    (output / "evaluation_report.md").write_text("\n".join(overall_lines) + "\n")
    print(f"created {output}")
    print(f"runs={len(summary_rows)} composition_rows={len(composition_rows)} seat_rows={len(seat_rows)}")


if __name__ == "__main__":
    main()
