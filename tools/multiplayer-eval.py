#!/usr/bin/env python3

"""Run seat-balanced matches between multiple MiniZero console engines."""

import argparse
import csv
import itertools
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PLAYER_CODES = ("b", "w", "r", "g", "y", "p")
DEFAULT_MAX_MOVES = {"tictacmo": 15, "connect3x3": 42, "go3": 400, "blokus": 400, "blokus10": 160, "blokus15": 220}
DEFAULT_EVAL_OVERRIDES = {
    "actor_use_gumbel": "false",
    "actor_use_gumbel_noise": "false",
    "actor_use_dirichlet_noise": "false",
    "actor_use_random_rotation_features": "false",
    "actor_select_action_by_count": "true",
    "actor_select_action_by_softmax_count": "false",
    "actor_mcts_value_rescale": "false",
    "zero_disable_resign_ratio": "1",
    "zero_actor_intermediate_sequence_length": "0",
}
MULTIPLAYER_SELF_EVAL_OVERRIDES = {
    "actor_use_gumbel": "false",
    "actor_use_gumbel_noise": "false",
    "actor_mcts_value_rescale": "false",
    "zero_disable_resign_ratio": "1",
    "zero_actor_intermediate_sequence_length": "0",
}


def default_terminal_passes(game):
    if game in ("blokus", "blokus10", "blokus15"):
        return 4
    if game == "go3":
        return 3
    return 1


def default_pass_mode(game):
    if game in ("blokus", "blokus10", "blokus15"):
        return "elimination"
    if game == "go3":
        return "consecutive"
    return "terminal"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    command: object
    cwd: str = None
    env: dict = None


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    lineup_id: int
    seating_id: int
    repeat: int
    seating: tuple


@dataclass(frozen=True)
class SeatingTask:
    task_id: int
    games: tuple


class Engine:
    def __init__(self, agent, context, timeout, stderr_path=None):
        self.agent = agent
        self.timeout = timeout
        self.command = format_command(agent.command, context)
        self.stderr_file = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
        env = os.environ.copy()
        if context["gpu"] != "" and "CUDA_VISIBLE_DEVICES" not in (agent.env or {}):
            env["CUDA_VISIBLE_DEVICES"] = context["gpu"]
        for key, value in (agent.env or {}).items():
            env[key] = str(value).format_map(context)
        self.proc = subprocess.Popen(
            self.command,
            cwd=agent.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
        )
        self.stdout_lines = queue.Queue()
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self.stdout_lines.put(line.rstrip("\r\n"))
        self.stdout_lines.put(None)

    def command_response(self, command):
        if self.proc.poll() is not None:
            raise RuntimeError(f'{self.agent.name} exited before command "{command}"')
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

        lines = []
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{self.agent.name} timed out after {self.timeout}s: "{command}"')
            try:
                line = self.stdout_lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f'{self.agent.name} timed out after {self.timeout}s: "{command}"') from exc
            if line is None:
                raise RuntimeError(f'{self.agent.name} closed stdout after command "{command}"')
            if line == "":
                break
            lines.append(line)

        if not lines:
            raise RuntimeError(f'{self.agent.name} returned an empty response to "{command}"')
        first = lines[0]
        if not first.startswith(("=", "?")):
            raise RuntimeError(f'{self.agent.name} returned an invalid response to "{command}": {first}')
        payload = "\n".join([first[1:].strip()] + lines[1:]).strip()
        if first.startswith("?"):
            raise RuntimeError(payload or f'{self.agent.name} rejected command "{command}"')
        return payload

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if self.stderr_file is not subprocess.DEVNULL:
            self.stderr_file.close()


def format_command(command, context):
    if isinstance(command, str):
        return shlex.split(command.format_map(context))
    if isinstance(command, list) and command:
        return [str(part).format_map(context) for part in command]
    raise ValueError("agent command must be a non-empty string or list")


def load_manifest(path):
    manifest_path = Path(path).resolve()
    with manifest_path.open() as stream:
        data = json.load(stream)

    players = [str(code).lower() for code in data.get("players", ["b", "w", "r"])]
    if not 2 <= len(players) <= len(PLAYER_CODES):
        raise ValueError("players must contain between 2 and 6 entries")
    if players != list(PLAYER_CODES[:len(players)]):
        raise ValueError(f"players must be {list(PLAYER_CODES[:len(players)])}")

    agents = {}
    for item in data.get("agents", []):
        name = item.get("name", "").strip()
        if not name or name in agents:
            raise ValueError(f"agent names must be non-empty and unique: {name!r}")
        cwd = item.get("cwd")
        if cwd and not os.path.isabs(cwd):
            cwd = str((manifest_path.parent / cwd).resolve())
        agents[name] = AgentSpec(name, item.get("command"), cwd, item.get("env", {}))
    if len(agents) < 1:
        raise ValueError("manifest must define at least one agent")

    lineups = data.get("lineups")
    if lineups is None:
        if len(agents) < len(players):
            raise ValueError("fewer agents than seats; provide explicit lineups to repeat agents")
        lineups = [list(lineup) for lineup in itertools.combinations(agents, len(players))]
    for lineup in lineups:
        if len(lineup) != len(players):
            raise ValueError(f"each lineup must contain exactly {len(players)} agent names")
        unknown = [name for name in lineup if name not in agents]
        if unknown:
            raise ValueError(f"unknown agents in lineup: {unknown}")

    config = {
        "game": str(data.get("game", "multiplayer")),
        "players": players,
        "agents": agents,
        "lineups": lineups,
        "seat_mode": data.get("seat_mode", "all_permutations"),
        "games_per_seating": int(data.get("games_per_seating", 1)),
        "num_games": int(data["num_games"]) if data.get("num_games") is not None else None,
        "max_moves": int(data.get("max_moves", 2048)),
        "terminal_passes": int(data.get("terminal_passes", 1)),
        "pass_mode": str(data.get("pass_mode", default_pass_mode(str(data.get("game", "multiplayer"))))),
        "command_timeout": float(data.get("command_timeout", 300)),
        "seed": int(data.get("seed", 0)),
        "share_agent_engines": bool(data.get("share_agent_engines", False)),
    }
    if config["seat_mode"] not in ("fixed", "cyclic", "all_permutations"):
        raise ValueError("seat_mode must be fixed, cyclic, or all_permutations")
    if config["pass_mode"] not in ("terminal", "consecutive", "elimination"):
        raise ValueError("pass_mode must be terminal, consecutive, or elimination")
    if (config["games_per_seating"] < 1 or
            (config["num_games"] is not None and config["num_games"] < 1) or
            config["max_moves"] < 1 or config["terminal_passes"] < 1 or config["command_timeout"] <= 0):
        raise ValueError("game counts, max_moves, and command_timeout must be positive")
    return config


def unique_seatings(lineup, mode):
    if mode == "fixed":
        candidates = [tuple(lineup)]
    elif mode == "cyclic":
        candidates = [tuple(lineup[offset:] + lineup[:offset]) for offset in range(len(lineup))]
    else:
        candidates = itertools.permutations(lineup)
    return list(dict.fromkeys(candidates))


def balanced_remainder_indices(seatings, remainder):
    if remainder == 0:
        return set()

    features = []
    for _, _, seating in seatings:
        item = []
        for seat, agent in enumerate(seating):
            item.extend(((agent, seat), (agent, None)))
        features.append(item)
    totals = {}
    for item in features:
        for feature in item:
            totals[feature] = totals.get(feature, 0) + 1
    expected = {feature: count * remainder / len(seatings) for feature, count in totals.items()}

    combination_count = math.comb(len(seatings), remainder)
    if combination_count <= 100000:
        candidates = itertools.combinations(range(len(seatings)), remainder)
    else:
        candidates = (tuple(range(remainder)),)
    best_score = None
    best_indices = None
    for indices in candidates:
        counts = {}
        for index in indices:
            for feature in features[index]:
                counts[feature] = counts.get(feature, 0) + 1
        score = sum((counts.get(feature, 0) - target) ** 2 for feature, target in expected.items())
        if best_score is None or score < best_score:
            best_score = score
            best_indices = indices
    return set(best_indices)


def create_schedule(config):
    tasks = []
    game_id = 0
    seatings = []
    for lineup_id, lineup in enumerate(config["lineups"]):
        for seating_id, seating in enumerate(unique_seatings(list(lineup), config["seat_mode"])):
            seatings.append((lineup_id, seating_id, seating))

    if config.get("num_games") is None:
        game_counts = [config["games_per_seating"]] * len(seatings)
    else:
        games_per_seating, remainder = divmod(config["num_games"], len(seatings))
        extra_games = balanced_remainder_indices(seatings, remainder)
        game_counts = [games_per_seating + int(index in extra_games) for index in range(len(seatings))]

    for task_id, ((lineup_id, seating_id, seating), game_count) in enumerate(zip(seatings, game_counts)):
        games = []
        for repeat in range(game_count):
            games.append(GameSpec(game_id, lineup_id, seating_id, repeat, seating))
            game_id += 1
        if games:
            tasks.append(SeatingTask(task_id, tuple(games)))
    return tasks


def parse_returns(game_string, num_players):
    match = re.search(r"RE\[([^]]*)\]", game_string)
    if not match:
        raise RuntimeError("terminal game_string has no RE property")
    values = [float(token) for token in match.group(1).split(",")]
    if num_players == 2 and len(values) == 1:
        values.append(-values[0])
    if len(values) != num_players:
        raise RuntimeError(f"expected {num_players} returns, got {values}")
    return values


def result_info(returns):
    best = max(returns)
    winners = [index for index, value in enumerate(returns) if value == best]
    return (winners[0], False) if len(winners) == 1 else (None, True)


def unique_engines(engines):
    return list(dict.fromkeys(engines))


def play_game(engines, players, max_moves, terminal_passes=1, pass_mode="terminal"):
    for engine in unique_engines(engines):
        engine.command_response("clear_board")

    moves = []
    turn = 0
    eliminated = set()
    consecutive_passes = 0
    for ply in range(max_moves + 1):
        action = engines[turn].command_response(f"genmove {players[turn]}").strip()
        if action.upper() == "PASS":
            consecutive_passes += 1
            if pass_mode == "terminal" or consecutive_passes >= terminal_passes:
                game_string = engines[turn].command_response("game_string")
                returns = parse_returns(game_string, len(players))
                winner_seat, draw = result_info(returns)
                moves.append({"player": players[turn], "action": action})
                return moves, returns, winner_seat, draw
            if pass_mode == "elimination":
                eliminated.add(turn)
                if len(eliminated) >= terminal_passes:
                    moves.append({"player": players[turn], "action": action})
                    game_string = engines[turn].command_response("game_string")
                    returns = parse_returns(game_string, len(players))
                    winner_seat, draw = result_info(returns)
                    return moves, returns, winner_seat, draw
        else:
            consecutive_passes = 0
        if action.lower() == "resign":
            raise RuntimeError("multiplayer resignation has no generic winner semantics; disable resignation")
        if ply == max_moves:
            raise RuntimeError(f"reached max_moves={max_moves} without a terminal PASS")

        moves.append({"player": players[turn], "action": action})
        acting_engine = engines[turn]
        for engine in unique_engines(engines):
            if engine is not acting_engine:
                engine.command_response(f"play {players[turn]} {action}")
        turn = (turn + 1) % len(players)
        while turn in eliminated:
            turn = (turn + 1) % len(players)

    raise AssertionError("unreachable")


def sgf_escape(value):
    return str(value).replace("\\", "\\\\").replace("]", "\\]").replace("\n", " ")


def write_sgf(path, game, game_name):
    seating = game["seating"]
    properties = [f"GM[{sgf_escape(game_name)}]", "EV[multiplayer-eval]", f"RE[{','.join(map(str, game['returns']))}]"]
    properties.extend(
        f"P{game['players'][index].upper()}N[{sgf_escape(agent)}]"
        for index, agent in enumerate(seating)
    )
    content = "(;" + "".join(properties)
    for move in game["moves"]:
        content += f";{move['player'].upper()}[{sgf_escape(move['action'])}]"
    path.write_text(content + ")\n")


def run_seating(task, config, output_dir, completed, result_queue, worker_id, gpu):
    pending_games = [game for game in task.games if game.game_id not in completed]
    if not pending_games:
        return
    seating = task.games[0].seating
    engines = []
    stderr_dir = output_dir / "engine_logs"
    stderr_dir.mkdir(exist_ok=True)
    try:
        engines_by_agent = {}
        for seat, agent_name in enumerate(seating):
            if config.get("share_agent_engines") and agent_name in engines_by_agent:
                engines.append(engines_by_agent[agent_name])
                continue
            context = {
                "agent": agent_name,
                "seat": config["players"][seat],
                "seat_index": seat,
                "seed": config["seed"] + task.task_id * len(seating) + seat,
                "task_id": task.task_id,
                "worker_id": worker_id,
                "gpu": gpu,
            }
            stderr_path = stderr_dir / f"task_{task.task_id:04d}_seat_{seat + 1}_{safe_name(agent_name)}.log"
            engine = Engine(config["agents"][agent_name], context, config["command_timeout"], stderr_path)
            engines.append(engine)
            engines_by_agent[agent_name] = engine

        for game_index, spec in enumerate(task.games):
            if spec.game_id in completed:
                # Engines stay alive for every repeat of one seating, so their
                # RNG streams also span those repeats. Replaying completed
                # games after a restart restores the exact RNG position before
                # continuing; otherwise --resume would duplicate early random
                # trajectories from the same seating.
                play_game(
                    engines,
                    config["players"],
                    config["max_moves"],
                    config.get("terminal_passes", 1),
                    config.get("pass_mode", "terminal"),
                )
                continue
            started = time.monotonic()
            record = {
                "game_id": spec.game_id,
                "lineup_id": spec.lineup_id,
                "seating_id": spec.seating_id,
                "repeat": spec.repeat,
                "seating": list(spec.seating),
                "players": config["players"],
                "moves": [],
                "returns": [],
                "winner_seat": None,
                "winner_agent": None,
                "draw": False,
                "error": None,
            }
            try:
                moves, returns, winner_seat, draw = play_game(
                    engines,
                    config["players"],
                    config["max_moves"],
                    config.get("terminal_passes", 1),
                    config.get("pass_mode", "terminal"),
                )
                record.update({
                    "moves": moves,
                    "returns": returns,
                    "winner_seat": winner_seat,
                    "winner_agent": None if winner_seat is None else spec.seating[winner_seat],
                    "draw": draw,
                })
                write_sgf(output_dir / "sgf" / f"game_{spec.game_id:06d}.sgf", record, config["game"])
            except Exception as exc:
                record["error"] = str(exc)
            record["duration_seconds"] = round(time.monotonic() - started, 6)
            result_queue.put(record)
            if record["error"]:
                abort_error = f"seating aborted after game {spec.game_id}: {record['error']}"
                for remaining in task.games[game_index + 1:]:
                    if remaining.game_id in completed:
                        continue
                    result_queue.put({
                        "game_id": remaining.game_id,
                        "lineup_id": remaining.lineup_id,
                        "seating_id": remaining.seating_id,
                        "repeat": remaining.repeat,
                        "seating": list(remaining.seating),
                        "players": config["players"],
                        "moves": [],
                        "returns": [],
                        "winner_seat": None,
                        "winner_agent": None,
                        "draw": False,
                        "error": abort_error,
                        "duration_seconds": 0.0,
                    })
                break
    except Exception as exc:
        for spec in pending_games:
            result_queue.put({
                "game_id": spec.game_id,
                "lineup_id": spec.lineup_id,
                "seating_id": spec.seating_id,
                "repeat": spec.repeat,
                "seating": list(spec.seating),
                "players": config["players"],
                "moves": [],
                "returns": [],
                "winner_seat": None,
                "winner_agent": None,
                "draw": False,
                "error": f"engine startup failed: {exc}",
                "duration_seconds": 0.0,
            })
    finally:
        for engine in unique_engines(engines):
            engine.close()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def worker(worker_id, gpu, tasks, config, output_dir, completed, result_queue):
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            run_seating(task, config, output_dir, completed, result_queue, worker_id, gpu)
        finally:
            tasks.task_done()


def load_existing_results(path):
    results = {}
    if not path.exists():
        return []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    results[record["game_id"]] = record
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return [results[game_id] for game_id in sorted(results)]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results, agent_names, players, output_dir):
    valid = [record for record in results if not record.get("error")]
    agent_stats = {name: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "return_sum": 0.0} for name in agent_names}
    seat_stats = {}
    seating_stats = {}
    for record in valid:
        seating_key = (record["lineup_id"], record["seating_id"], tuple(record["seating"]))
        per_seating = seating_stats.setdefault(seating_key, {
            "games": 0,
            "draws": 0,
            "wins": [0] * len(players),
            "return_sums": [0.0] * len(players),
        })
        per_seating["games"] += 1
        per_seating["draws"] += int(record["draw"])
        if record["winner_seat"] is not None:
            per_seating["wins"][record["winner_seat"]] += 1
        for seat, value in enumerate(record["returns"]):
            per_seating["return_sums"][seat] += value

        for seat, (agent, value) in enumerate(zip(record["seating"], record["returns"])):
            stats = agent_stats[agent]
            key = (agent, players[seat])
            per_seat = seat_stats.setdefault(key, {"games": 0, "wins": 0, "draws": 0, "losses": 0, "return_sum": 0.0})
            for target in (stats, per_seat):
                target["games"] += 1
                target["return_sum"] += value
                if record["draw"]:
                    target["draws"] += 1
                elif record["winner_seat"] == seat:
                    target["wins"] += 1
                else:
                    target["losses"] += 1

    def row(name, stats, seat=None):
        games = stats["games"]
        return {
            "agent": name,
            **({"seat": seat.upper()} if seat is not None else {}),
            "games": games,
            "wins": stats["wins"],
            "draws": stats["draws"],
            "losses": stats["losses"],
            "win_rate": stats["wins"] / games if games else 0.0,
            "draw_rate": stats["draws"] / games if games else 0.0,
            "avg_return": stats["return_sum"] / games if games else 0.0,
        }

    agent_rows = [row(name, agent_stats[name]) for name in agent_names]
    seat_rows = [row(name, stats, seat) for (name, seat), stats in sorted(seat_stats.items())]
    errors_by_seating = {}
    for record in results:
        if record.get("error"):
            key = (record["lineup_id"], record["seating_id"], tuple(record["seating"]))
            errors_by_seating[key] = errors_by_seating.get(key, 0) + 1
    seating_keys = sorted(set(seating_stats) | set(errors_by_seating))
    seating_rows = []
    for lineup_id, seating_id, seating in seating_keys:
        key = (lineup_id, seating_id, seating)
        stats = seating_stats.get(key, {
            "games": 0,
            "draws": 0,
            "wins": [0] * len(players),
            "return_sums": [0.0] * len(players),
        })
        games = stats["games"]
        seating_row = {
            "lineup_id": lineup_id,
            "seating_id": seating_id,
            "games": games,
            "errors": errors_by_seating.get(key, 0),
            "draw_rate": stats["draws"] / games if games else 0.0,
        }
        for seat, player in enumerate(players):
            prefix = player.upper()
            seating_row[f"{prefix}_agent"] = seating[seat]
            seating_row[f"{prefix}_wins"] = stats["wins"][seat]
            seating_row[f"{prefix}_win_rate"] = stats["wins"][seat] / games if games else 0.0
            seating_row[f"{prefix}_avg_return"] = stats["return_sums"][seat] / games if games else 0.0
        seating_rows.append(seating_row)
    metrics = ["agent", "games", "wins", "draws", "losses", "win_rate", "draw_rate", "avg_return"]
    write_csv(output_dir / "agent_summary.csv", metrics, agent_rows)
    write_csv(output_dir / "seat_summary.csv", ["agent", "seat"] + metrics[1:], seat_rows)
    seating_fields = ["lineup_id", "seating_id", "games", "errors", "draw_rate"]
    for player in players:
        prefix = player.upper()
        seating_fields.extend([f"{prefix}_agent", f"{prefix}_wins", f"{prefix}_win_rate", f"{prefix}_avg_return"])
    write_csv(output_dir / "seating_summary.csv", seating_fields, seating_rows)
    write_csv(output_dir / "errors.csv", ["game_id", "seating", "error"], [
        {"game_id": record["game_id"], "seating": "/".join(record["seating"]), "error": record["error"]}
        for record in results if record.get("error")
    ])
    write_arena_plots(output_dir, agent_rows, seat_rows)
    return agent_rows, len(valid), len(results) - len(valid)


def write_arena_plots(output_dir, agent_rows, seat_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / "plot_warning.txt").write_text("matplotlib unavailable; summary PNG files were not generated\n")
        return

    if agent_rows:
        names = [row["agent"] for row in agent_rows]
        win_rates = [float(row["win_rate"]) for row in agent_rows]
        avg_returns = [float(row["avg_return"]) for row in agent_rows]
        figure, axes = plt.subplots(1, 2, figsize=(max(8, 1.2 * len(names)), 4))
        axes[0].bar(names, win_rates)
        axes[0].set_ylabel("win rate")
        axes[0].set_ylim(0, max(1.0, max(win_rates) * 1.15 if win_rates else 1.0))
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(names, avg_returns)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("average return")
        axes[1].tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(output_dir / "agent_summary.png")
        plt.close(figure)

    if seat_rows:
        agents = list(dict.fromkeys(row["agent"] for row in seat_rows))
        seats = list(dict.fromkeys(row["seat"] for row in seat_rows))
        values = {(row["agent"], row["seat"]): float(row["avg_return"]) for row in seat_rows}
        width = 0.8 / max(1, len(seats))
        x_positions = list(range(len(agents)))
        figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(agents)), 4))
        for offset, seat in enumerate(seats):
            shift = (offset - (len(seats) - 1) / 2) * width
            axis.bar(
                [x + shift for x in x_positions],
                [values.get((agent, seat), 0.0) for agent in agents],
                width=width,
                label=seat,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(agents, rotation=30)
        axis.set_ylabel("average return by seat")
        axis.legend(title="seat")
        figure.tight_layout()
        figure.savefig(output_dir / "seat_summary.png")
        plt.close(figure)

    write_mcts_sweep_outputs(output_dir, agent_rows, plt)


def write_mcts_sweep_outputs(output_dir, agent_rows, plt):
    pattern = re.compile(r"^alg_uct_(maxn|paranoid)_s(\d+)$")
    rows = []
    model_rows = []
    for row in agent_rows:
        match = pattern.fullmatch(row["agent"])
        if match:
            rows.append({
                "search": match.group(1),
                "simulations": int(match.group(2)),
                "agent": row["agent"],
                "games": row["games"],
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "win_rate": row["win_rate"],
                "draw_rate": row["draw_rate"],
                "avg_return": row["avg_return"],
            })
        elif not row["agent"].startswith("alg_"):
            model_rows.append(row)
    if not rows:
        return

    rows.sort(key=lambda item: (item["search"], item["simulations"]))
    write_csv(
        output_dir / "mcts_sweep_summary.csv",
        ["search", "simulations", "agent", "games", "wins", "draws", "losses", "win_rate", "draw_rate", "avg_return"],
        rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = {
        "maxn": {"label": "UCT MaxN", "marker": "o"},
        "paranoid": {"label": "UCT Paranoid", "marker": "s"},
    }
    for search, style in styles.items():
        series = [row for row in rows if row["search"] == search]
        if not series:
            continue
        x = [row["simulations"] for row in series]
        axes[0].plot(x, [float(row["win_rate"]) for row in series], marker=style["marker"], label=style["label"])
        axes[1].plot(x, [float(row["avg_return"]) for row in series], marker=style["marker"], label=style["label"])

    if model_rows:
        model = model_rows[0]
        axes[0].axhline(float(model["win_rate"]), linestyle="--", color="black", linewidth=1, label=f"{model['agent']} reference")
        axes[1].axhline(float(model["avg_return"]), linestyle="--", color="black", linewidth=1, label=f"{model['agent']} reference")

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("pure MCTS simulations per move")
        axis.grid(True, which="both", linestyle=":", linewidth=0.6)
        axis.legend()
    axes[0].set_ylabel("win rate")
    axes[0].set_ylim(0, max(1.0, axes[0].get_ylim()[1]))
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("average centered return")
    figure.tight_layout()
    figure.savefig(output_dir / "mcts_sweep.png")
    plt.close(figure)


def prepare_output(output_dir, resume, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "games.jsonl"
    if results_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"{results_path} exists; use --resume or --overwrite")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    (output_dir / "sgf").mkdir(exist_ok=True)
    return results_path


def parse_gpu_list(value):
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    if "," in value:
        devices = [device.strip() for device in value.split(",") if device.strip()]
    elif value.isdigit():
        devices = list(value)
    else:
        devices = value.split()
    if not devices:
        raise ValueError("GPU list is empty")
    return devices


def find_latest_model(training_dir):
    models = list((training_dir / "model").glob("weight_iter_*.pt"))
    if not models:
        raise FileNotFoundError(f"no weight_iter_*.pt found in {training_dir / 'model'}")

    def model_key(path):
        match = re.fullmatch(r"weight_iter_(\d+)\.pt", path.name)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime_ns)

    return max(models, key=model_key)


def resolve_auto_model(training_dir, value):
    if value is None:
        return find_latest_model(training_dir)
    if value.isdigit():
        value = f"weight_iter_{value}.pt"
    path = Path(value)
    candidates = [path] if path.is_absolute() else [training_dir / "model" / path, training_dir / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"model not found: {value}")


def resolve_auto_config(training_dir, value):
    if value is not None:
        path = Path(value)
        candidates = [path] if path.is_absolute() else [training_dir / path, path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"config not found: {value}")
    configs = list(training_dir.glob("*.cfg"))
    if not configs:
        raise FileNotFoundError(f"no *.cfg found in {training_dir}")
    return max(configs, key=lambda path: path.stat().st_mtime_ns).resolve()


def read_config_value(path, key):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\s]+)")
    with path.open() as stream:
        for line in stream:
            match = pattern.match(line)
            if match:
                return match.group(1)
    return None


def create_balanced_lineups(agent_names, num_players):
    lineups = list(itertools.combinations_with_replacement(agent_names, num_players))
    mixed = [list(lineup) for lineup in lineups if len(set(lineup)) > 1]
    return mixed or [list(lineups[0])]


def create_auto_manifest(args, repo_root, models, configs, executable):
    overrides = dict(DEFAULT_EVAL_OVERRIDES)
    if args.noise:
        overrides["actor_use_dirichlet_noise"] = "true"
    if args.num_simulations is not None:
        overrides["actor_num_simulation"] = str(args.num_simulations)

    agents = []
    for search_type in args.search_types:
        agent_overrides = {
            "nn_file_name": str(models[search_type]),
            "actor_multiplayer_search_type": search_type,
            "program_seed": "{seed}",
            "program_auto_seed": "false",
            **overrides,
        }
        if search_type == "rank":
            if args.rank_weight is not None:
                agent_overrides["actor_rank_utility_weight"] = str(args.rank_weight)
        else:
            agent_overrides["actor_rank_utility_weight"] = "0"
        conf_str = ":".join(f"{key}={value}" for key, value in agent_overrides.items())
        agents.append({
            "name": search_type,
            "cwd": str(repo_root),
            "env": {"OMP_NUM_THREADS": str(args.omp_num_threads)},
            "command": [
                str(executable), "-mode", "console",
                "-conf_file", str(configs[search_type]),
                "-conf_str", conf_str,
            ],
        })

    return {
        "game": args.game,
        "players": list(PLAYER_CODES[:args.num_players]),
        "agents": agents,
        "lineups": create_balanced_lineups(args.search_types, args.num_players),
        "seat_mode": "all_permutations",
        "games_per_seating": args.games_per_seating,
        "max_moves": args.max_moves if args.max_moves is not None else DEFAULT_MAX_MOVES.get(args.game, 2048),
        "terminal_passes": default_terminal_passes(args.game),
        "pass_mode": default_pass_mode(args.game),
        "command_timeout": args.command_timeout,
        "seed": args.seed,
    }


def create_model_console_agent(name, model, config, executable, repo_root, search_type, args):
    overrides = dict(DEFAULT_EVAL_OVERRIDES)
    if args.noise:
        overrides["actor_use_dirichlet_noise"] = "true"
    if args.num_simulations is not None:
        overrides["actor_num_simulation"] = str(args.num_simulations)
    overrides.update(parse_conf_overrides(getattr(args, "conf_str", "")))
    overrides.update({
        "nn_file_name": str(model),
        "actor_multiplayer_search_type": search_type,
        "program_seed": "{seed}",
        "program_auto_seed": "false",
    })
    conf_str = ":".join(f"{key}={value}" for key, value in overrides.items())
    return {
        "name": name,
        "cwd": str(repo_root),
        "env": {"OMP_NUM_THREADS": str(args.omp_num_threads)},
        "command": [
            str(executable), "-mode", "console",
            "-conf_file", str(config),
            "-conf_str", conf_str,
        ],
    }


def create_blokus_baseline_agent(policy, repo_root, args, simulations=None):
    name = f"alg_{policy}" if simulations is None else f"alg_{policy}_s{simulations}"
    command = [
        sys.executable,
        str(repo_root / "tools" / "blokus-baseline-agent.py"),
        "--policy", policy,
        "--seed", "{seed}",
    ]
    if policy == "rollout":
        command.extend([
            "--rollouts", str(args.rollouts),
            "--candidate-limit", str(args.candidate_limit),
            "--playout-policy", args.playout_policy,
            "--max-plies", str(args.rollout_max_plies),
        ])
    if policy in ("uct_maxn", "uct_paranoid"):
        command.extend([
            "--mcts-simulations", str(simulations if simulations is not None else args.mcts_simulations[0]),
            "--mcts-cpuct", str(args.mcts_cpuct),
            "--mcts-candidate-limit", str(args.mcts_candidate_limit),
            "--mcts-leaf-eval", args.mcts_leaf_eval,
            "--mcts-playout-policy", args.mcts_playout_policy,
            "--mcts-max-plies", str(args.mcts_max_plies),
        ])
    return {
        "name": name,
        "cwd": str(repo_root),
        "command": command,
    }


def create_pairwise_lineups(model_name, baseline_names, num_players):
    lineups = []
    for baseline in baseline_names:
        lineups.extend(create_balanced_lineups([model_name, baseline], num_players))
    return lineups


def create_blokus_baseline_manifest(args, repo_root, executable, config, model):
    model_name = args.model_name
    baseline_specs = []
    for policy in args.baselines:
        if policy in ("uct_maxn", "uct_paranoid"):
            baseline_specs.extend((policy, simulations) for simulations in args.mcts_simulations)
        else:
            baseline_specs.append((policy, None))
    baseline_names = [
        f"alg_{policy}" if simulations is None else f"alg_{policy}_s{simulations}"
        for policy, simulations in baseline_specs
    ]
    return {
        "game": "blokus",
        "players": list(PLAYER_CODES[:args.num_players]),
        "agents": [
            create_model_console_agent(model_name, model, config, executable, repo_root, args.search_type, args),
            *[create_blokus_baseline_agent(policy, repo_root, args, simulations) for policy, simulations in baseline_specs],
        ],
        # Pairwise lineups avoid an unreadable random/greedy/rollout/model
        # four-way soup.  Each baseline gets the same seat-balanced mixture:
        # 1 model vs 3 baseline, 2 vs 2, and 3 model vs 1 baseline, with all
        # unique seat permutations.
        "lineups": create_pairwise_lineups(model_name, baseline_names, args.num_players),
        "seat_mode": "all_permutations",
        "num_games": args.games,
        "max_moves": args.max_moves if args.max_moves is not None else DEFAULT_MAX_MOVES["blokus"],
        "terminal_passes": 4,
        "command_timeout": args.command_timeout,
        "seed": args.seed,
    }


def blokus_baseline_main(argv):
    parser = argparse.ArgumentParser(
        prog="multiplayer-eval.py blokus-baseline",
        description="Compare a Blokus neural model against pure algorithmic baselines.",
    )
    parser.add_argument("training_dir", help="training folder containing model/ and a config")
    parser.add_argument("--model", help="model path, file name, or iteration number (default: latest iteration)")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--conf-file", help="evaluation config (default: newest *.cfg in the training folder)")
    parser.add_argument("--executable", help="engine executable (default: build/blokus/minizero_blokus)")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--search-type", choices=("maxn", "paranoid"), default="maxn")
    parser.add_argument("--baselines", nargs="+", default=["random", "greedy", "greedy_mobility", "rollout"],
                        choices=("random", "greedy", "greedy_mobility", "rollout", "uct_maxn", "uct_paranoid"))
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--num-simulations", type=int, help="override actor_num_simulation for the model agent")
    parser.add_argument("--noise", action="store_true", help="enable Dirichlet noise for the model agent")
    parser.add_argument("-conf_str", "--conf-str", dest="conf_str", default="")
    parser.add_argument("--games", type=int, default=120, help="total games across all generated seatings")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-num-threads", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--playout-policy", choices=("random", "greedy", "greedy_mobility"), default="random")
    parser.add_argument("--rollout-max-plies", type=int, default=400)
    parser.add_argument("--mcts-simulations", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600],
                        help="simulation sweep for uct_maxn/uct_paranoid baselines")
    parser.add_argument("--mcts-cpuct", type=float, default=1.0)
    parser.add_argument("--mcts-candidate-limit", type=int, default=64,
                        help="legal action candidates considered at each pure-MCTS node")
    parser.add_argument("--mcts-leaf-eval", choices=("zero", "rollout"), default="zero",
                        help="zero matches a DumbNet-style uninformed MCTS; rollout uses terminal playouts at leaves")
    parser.add_argument("--mcts-playout-policy", choices=("random", "greedy", "greedy_mobility"), default="random")
    parser.add_argument("--mcts-max-plies", type=int, default=400)
    parser.add_argument("-g", "--gpu", help="GPU list, for example 0123 or 0,1,2,3")
    parser.add_argument("--num_threads", "--num-threads", "--threads", dest="num_threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only generate arena.json")
    args = parser.parse_args(argv)

    if args.num_players != 4:
        parser.error("Blokus baseline mode currently expects --num-players 4")
    if len(set(args.baselines)) != len(args.baselines):
        parser.error("--baselines must not contain duplicates")
    positive = (
        args.games, args.command_timeout, args.omp_num_threads, args.num_threads,
        args.rollouts, args.candidate_limit, args.rollout_max_plies,
        args.mcts_candidate_limit, args.mcts_max_plies,
    )
    if (any(value <= 0 for value in positive) or any(value <= 0 for value in args.mcts_simulations) or
            args.mcts_cpuct < 0 or (args.num_simulations is not None and args.num_simulations <= 0)):
        parser.error("game, timeout, thread, simulation, and rollout counts must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    training_dir = Path(args.training_dir).resolve()
    if not training_dir.is_dir():
        parser.error(f"training directory not found: {training_dir}")
    model = resolve_auto_model(training_dir, args.model)
    config = resolve_auto_config(training_dir, args.conf_file)
    executable = Path(args.executable).resolve() if args.executable else repo_root / "build" / "blokus" / "minizero_blokus"
    if not executable.is_file():
        parser.error(f"engine executable not found: {executable}; build it before evaluation")
    baseline_label = "_".join(
        f"{policy}_{'-'.join(map(str, args.mcts_simulations))}" if policy in ("uct_maxn", "uct_paranoid") else policy
        for policy in args.baselines
    )
    simulation_label = f"_n{args.num_simulations}" if args.num_simulations is not None else ""
    noise_label = "_noise" if args.noise else ""
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else training_dir / "evaluation" / f"{model.stem}_{args.search_type}_vs_{baseline_label}{simulation_label}{noise_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "arena.json"
    manifest = create_blokus_baseline_manifest(args, repo_root, executable.resolve(), config, model)
    results_path = output_dir / "games.jsonl"
    if results_path.exists() and not (args.resume or args.overwrite or args.dry_run):
        raise FileExistsError(f"{results_path} exists; use --resume or --overwrite")
    if args.resume and manifest_path.exists():
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        if previous_manifest != manifest:
            raise ValueError(f"generated settings differ from the existing resume manifest: {manifest_path}")
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"generated arena manifest: {manifest_path}", flush=True)
    print(f"model: {model}", flush=True)
    print(f"config: {config}", flush=True)
    print(f"baselines: {', '.join(args.baselines)}", flush=True)
    if any(policy in ("uct_maxn", "uct_paranoid") for policy in args.baselines):
        print(f"MCTS simulation sweep: {', '.join(map(str, args.mcts_simulations))}", flush=True)

    if args.dry_run:
        return
    run(argparse.Namespace(
        manifest=str(manifest_path),
        output=str(output_dir),
        gpu=args.gpu,
        num_threads=args.num_threads,
        games_per_seating=None,
        num_games=None,
        max_moves=None,
        resume=args.resume,
        overwrite=args.overwrite,
    ))


def auto_main(argv):
    parser = argparse.ArgumentParser(
        prog="multiplayer-eval.py auto",
        description="Automatically create and run a seat-balanced multiplayer search arena.",
    )
    parser.add_argument("game", help="MiniZero game type, for example tictacmo")
    parser.add_argument("training_dir", help="training folder containing a config and model/ weights")
    parser.add_argument("--model", help="model path, file name, or iteration number (default: latest iteration)")
    parser.add_argument("--maxn-model", help="model used by the MaxN agent (default: --model)")
    parser.add_argument("--paranoid-model", help="model used by the Paranoid agent (default: --model)")
    parser.add_argument("--rank-model", help="model used by the Rank Utility agent (default: --model)")
    parser.add_argument("--conf-file", help="config path (default: newest *.cfg in the training folder)")
    parser.add_argument("--maxn-conf-file", help="config used by the MaxN agent")
    parser.add_argument("--paranoid-conf-file", help="config used by the Paranoid agent")
    parser.add_argument("--rank-conf-file", help="config used by the Rank Utility agent")
    parser.add_argument("--executable", help="engine executable (default: build/GAME/minizero_GAME)")
    parser.add_argument("--output", help="output directory (default: TRAINING_DIR/evaluation/MODEL_SEARCHES)")
    parser.add_argument("--search-types", nargs="+", default=["maxn", "paranoid"], choices=("maxn", "paranoid", "rank"))
    parser.add_argument("--rank-weight", type=float, help="rank contribution for the Rank Utility agent; must be in (0, 1]")
    parser.add_argument("--num-players", type=int)
    parser.add_argument("--num-simulations", type=int, help="override actor_num_simulation")
    parser.add_argument("--noise", action="store_true", help="enable Dirichlet noise for varied reproducible games")
    parser.add_argument("--games-per-seating", type=int, default=1)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-num-threads", type=int, default=2)
    parser.add_argument("-g", "--gpu", help="GPU list, for example 0123 or 0,1,2,3")
    parser.add_argument("--num_threads", "--num-threads", "--threads", dest="num_threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only generate arena.json")
    args = parser.parse_args(argv)
    args.game = args.game.lower()
    if args.num_players is None:
        args.num_players = 4 if args.game in ("blokus", "blokus10", "blokus15") else 3

    if not 2 <= args.num_players <= len(PLAYER_CODES):
        parser.error(f"--num-players must be between 2 and {len(PLAYER_CODES)}")
    if len(set(args.search_types)) != len(args.search_types):
        parser.error("--search-types must not contain duplicates")
    positive = (args.games_per_seating, args.command_timeout, args.omp_num_threads, args.num_threads)
    if any(value <= 0 for value in positive) or (args.num_simulations is not None and args.num_simulations <= 0):
        parser.error("game, timeout, thread, and simulation counts must be positive")
    if args.rank_weight is not None and not 0.0 < args.rank_weight <= 1.0:
        parser.error("--rank-weight must be in (0, 1]")

    repo_root = Path(__file__).resolve().parents[1]
    training_dir = Path(args.training_dir).resolve()
    if not training_dir.is_dir():
        parser.error(f"training directory not found: {training_dir}")
    common_model = resolve_auto_model(training_dir, args.model)
    common_config = resolve_auto_config(training_dir, args.conf_file)
    models = {}
    configs = {}
    for search_type in args.search_types:
        model_argument = getattr(args, f"{search_type}_model")
        models[search_type] = resolve_auto_model(training_dir, model_argument) if model_argument else common_model
        config_argument = getattr(args, f"{search_type}_conf_file")
        if config_argument:
            configs[search_type] = resolve_auto_config(training_dir, config_argument)
        elif model_argument and models[search_type].parent.name == "model":
            configs[search_type] = resolve_auto_config(models[search_type].parent.parent, None)
        else:
            configs[search_type] = common_config
    executable = Path(args.executable).resolve() if args.executable else repo_root / "build" / args.game / f"minizero_{args.game}"
    if not executable.is_file():
        parser.error(f"engine executable not found: {executable}; build it before evaluation")

    search_label = "_vs_".join(args.search_types)
    config_simulations = {
        value for value in (read_config_value(configs[search_type], "actor_num_simulation") for search_type in args.search_types)
        if value is not None
    }
    if args.num_simulations is None and len(config_simulations) > 1:
        parser.error("agent configs use different actor_num_simulation values; set --num-simulations explicitly")
    effective_simulations = args.num_simulations or (next(iter(config_simulations)) if config_simulations else None)
    simulation_label = f"_n{effective_simulations}" if effective_simulations else ""
    effective_rank_weight = args.rank_weight
    if effective_rank_weight is None and "rank" in args.search_types:
        config_weight = read_config_value(configs["rank"], "actor_rank_utility_weight")
        effective_rank_weight = float(config_weight) if config_weight is not None else 0.0
    if "rank" in args.search_types and effective_rank_weight <= 0.0:
        parser.error("Rank Utility evaluation requires --rank-weight or a positive actor_rank_utility_weight in its config")
    rank_label = f"_lambda{effective_rank_weight:g}" if effective_rank_weight is not None else ""
    noise_label = "_noise" if args.noise else ""
    if len(set(models.values())) == 1:
        arena_label = f"{next(iter(models.values())).stem}_{search_label}"
    else:
        model_labels = []
        for search_type in args.search_types:
            model = models[search_type]
            run_name = model.parent.parent.name if model.parent.name == "model" else model.parent.name
            model_labels.append(f"{search_type}_{safe_name(run_name)}_{model.stem}")
        arena_label = "_vs_".join(model_labels)
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else training_dir / "evaluation" / f"{arena_label}{simulation_label}{rank_label}{noise_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "arena.json"
    manifest = create_auto_manifest(args, repo_root, models, configs, executable.resolve())
    results_path = output_dir / "games.jsonl"
    if results_path.exists() and not (args.resume or args.overwrite or args.dry_run):
        raise FileExistsError(f"{results_path} exists; use --resume or --overwrite")
    if args.resume and manifest_path.exists():
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        if previous_manifest != manifest:
            raise ValueError(f"generated settings differ from the existing resume manifest: {manifest_path}")
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"generated arena manifest: {manifest_path}", flush=True)
    for search_type in args.search_types:
        print(f"{search_type} model: {models[search_type]}", flush=True)
        print(f"{search_type} config: {configs[search_type]}", flush=True)

    if args.dry_run:
        return
    run(argparse.Namespace(
        manifest=str(manifest_path),
        output=str(output_dir),
        gpu=args.gpu,
        num_threads=args.num_threads,
        games_per_seating=None,
        num_games=None,
        max_moves=None,
        resume=args.resume,
        overwrite=args.overwrite,
    ))


def model_iteration(path):
    match = re.fullmatch(r"weight_iter_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"invalid checkpoint file name: {path.name}")
    return int(match.group(1))


def find_checkpoints(training_dir):
    checkpoints = []
    for path in (training_dir / "model").glob("weight_iter_*.pt"):
        try:
            iteration = model_iteration(path)
        except ValueError:
            continue
        checkpoints.append((iteration, path.resolve()))
    if not checkpoints:
        raise FileNotFoundError(f"no weight_iter_*.pt found in {training_dir / 'model'}")
    return [path for _, path in sorted(checkpoints)]


def parse_conf_overrides(conf_str):
    overrides = {}
    if not conf_str:
        return overrides
    for item in conf_str.split(":"):
        if not item or "=" not in item:
            raise ValueError(f"invalid configuration override: {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"invalid configuration override: {item!r}")
        overrides[key] = value
    return overrides


def create_checkpoint_agent(name, model, config, executable, repo_root, search_type, args):
    # Match quick-run self-eval by preserving evaluation choices from the config.
    # Only force settings required by the currently supported multiplayer path.
    overrides = dict(MULTIPLAYER_SELF_EVAL_OVERRIDES)
    overrides.update(parse_conf_overrides(args.conf_str))
    if args.noise is not None:
        overrides["actor_use_dirichlet_noise"] = str(args.noise).lower()
    if args.num_simulations is not None:
        overrides["actor_num_simulation"] = str(args.num_simulations)
    overrides.update({
        "nn_file_name": str(model),
        "actor_multiplayer_search_type": search_type,
        "program_seed": "{seed}",
        "program_auto_seed": "false",
    })
    conf_str = ":".join(f"{key}={value}" for key, value in overrides.items())
    return {
        "name": name,
        "cwd": str(repo_root),
        "env": {"OMP_NUM_THREADS": str(args.omp_num_threads)},
        "command": [
            str(executable), "-mode", "console",
            "-conf_file", str(config),
            "-conf_str", conf_str,
        ],
    }


def create_checkpoint_manifest(args, repo_root, executable, config, older, newer):
    older_iteration = model_iteration(older)
    newer_iteration = model_iteration(newer)
    older_name = f"iter_{older_iteration}"
    newer_name = f"iter_{newer_iteration}"
    agent_names = [older_name, newer_name]
    return {
        "game": args.game,
        "players": list(PLAYER_CODES[:args.num_players]),
        "agents": [
            create_checkpoint_agent(older_name, older, config, executable, repo_root, args.search_type, args),
            create_checkpoint_agent(newer_name, newer, config, executable, repo_root, args.search_type, args),
        ],
        "lineups": create_balanced_lineups(agent_names, args.num_players),
        "seat_mode": "all_permutations",
        "num_games": args.games,
        "max_moves": args.max_moves if args.max_moves is not None else DEFAULT_MAX_MOVES.get(args.game, 2048),
        "terminal_passes": default_terminal_passes(args.game),
        "pass_mode": default_pass_mode(args.game),
        "command_timeout": args.command_timeout,
        "seed": args.seed,
        "self_eval": {
            "older_iteration": older_iteration,
            "newer_iteration": newer_iteration,
            "search_type": args.search_type,
        },
    }


def summarize_checkpoint_pair(results_path, older_name, newer_name):
    newer_wins = older_wins = draws = errors = 0
    with results_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                errors += 1
            else:
                returns = record.get("returns", [])
                seating = record.get("seating", [])
                if not returns or len(returns) != len(seating):
                    raise ValueError(f"invalid returns/seating in game {record.get('game_id')}")
                best_return = max(returns)
                top_models = {
                    seating[seat]
                    for seat, value in enumerate(returns)
                    if value == best_return
                }
                unexpected = top_models - {older_name, newer_name}
                if unexpected:
                    raise ValueError(f"unexpected top agent(s): {sorted(unexpected)}")
                if top_models == {newer_name}:
                    newer_wins += 1
                elif top_models == {older_name}:
                    older_wins += 1
                else:
                    # This is a model-level draw only when both checkpoints own
                    # at least one top seat. Multiple tied seats belonging to the
                    # same checkpoint are still a win for that checkpoint.
                    draws += 1
    valid = newer_wins + older_wins + draws
    win_rate = (newer_wins + 0.5 * draws) / valid if valid else float("nan")
    return newer_wins, older_wins, draws, errors, valid, win_rate


def updated_elo(opponent_elo, score):
    if score >= 1:
        return opponent_elo + 1000
    if score <= 0:
        return opponent_elo - 1000
    difference = 400 * math.log10(score / (1 - score))
    return opponent_elo + max(-1000, min(1000, difference))


def write_self_eval_plot(path, ratings):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(path.parent / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; elo.png was not generated", file=sys.stderr)
        return
    iterations = sorted(ratings)
    figure, axis = plt.subplots()
    axis.plot(iterations, [ratings[iteration] for iteration in iterations], marker="o", label="model")
    axis.set_xlabel("iteration")
    axis.set_ylabel("elo rating")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def self_eval_main(argv):
    parser = argparse.ArgumentParser(
        prog="multiplayer-eval.py self-eval",
        description="Evaluate consecutive checkpoint pairs with seat-balanced multiplayer games.",
    )
    parser.add_argument("game", help="MiniZero multiplayer game type")
    parser.add_argument("training_dir", help="training folder containing model/ and a config")
    parser.add_argument("--conf-file", help="evaluation config (default: newest *.cfg in the training folder)")
    parser.add_argument("--interval", type=int, default=10, help="checkpoint index interval, matching quick-run self-eval")
    parser.add_argument("--games", type=int, default=100, help="total games for each checkpoint pair")
    parser.add_argument("-s", "--start-index", type=int, default=0)
    parser.add_argument("-d", "--output", help="result directory (default: TRAINING_DIR/self_eval)")
    parser.add_argument("--search-type", choices=("maxn", "paranoid"), help="default: value from config, or maxn")
    parser.add_argument("--num-players", type=int)
    parser.add_argument("--num-simulations", type=int)
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--noise", dest="noise", action="store_true", help="enable Dirichlet noise")
    noise_group.add_argument("--no-noise", dest="noise", action="store_false", help="disable Dirichlet noise")
    parser.set_defaults(noise=None)
    parser.add_argument("-conf_str", "--conf-str", dest="conf_str", default="")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-num-threads", type=int, default=2)
    parser.add_argument("--executable", help="engine executable (default: build/GAME/minizero_GAME)")
    parser.add_argument("-g", "--gpu", help="GPU list, for example 0123 or 0,1,2,3")
    parser.add_argument("--num_threads", "--num-threads", "--threads", dest="num_threads", type=int, default=1)
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument("--resume", action="store_true", help="continue existing pairs using their saved arena manifests")
    resume_mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only generate pair arena manifests")
    args = parser.parse_args(argv)
    args.game = args.game.lower()
    if args.num_players is None:
        args.num_players = 4 if args.game in ("blokus", "blokus10", "blokus15") else 3

    positive = (args.interval, args.games, args.num_players, args.command_timeout,
                args.omp_num_threads, args.num_threads)
    if any(value <= 0 for value in positive) or args.start_index < 0:
        parser.error("interval, game, player, timeout, and thread values must be positive; start index cannot be negative")
    if not 2 <= args.num_players <= len(PLAYER_CODES):
        parser.error(f"--num-players must be between 2 and {len(PLAYER_CODES)}")
    if args.num_simulations is not None and args.num_simulations <= 0:
        parser.error("--num-simulations must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    training_dir = Path(args.training_dir).resolve()
    if not training_dir.is_dir():
        parser.error(f"training directory not found: {training_dir}")
    config = resolve_auto_config(training_dir, args.conf_file)
    if args.search_type is None:
        args.search_type = read_config_value(config, "actor_multiplayer_search_type") or "maxn"
    executable = Path(args.executable).resolve() if args.executable else repo_root / "build" / args.game / f"minizero_{args.game}"
    if not executable.is_file():
        parser.error(f"engine executable not found: {executable}; build it before evaluation")
    checkpoints = find_checkpoints(training_dir)
    if args.start_index >= len(checkpoints):
        parser.error(f"--start-index {args.start_index} exceeds {len(checkpoints)} available checkpoints")
    pairs = [
        (checkpoints[index], checkpoints[index + args.interval])
        for index in range(args.start_index, len(checkpoints) - args.interval, args.interval)
    ]
    if not pairs:
        parser.error(f"not enough checkpoints for interval {args.interval} starting at index {args.start_index}")

    output_dir = Path(args.output).resolve() if args.output else training_dir / "self_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    ratings = {model_iteration(pairs[0][0]): 0.0}
    for older, newer in pairs:
        older_iteration = model_iteration(older)
        newer_iteration = model_iteration(newer)
        older_name = f"iter_{older_iteration}"
        newer_name = f"iter_{newer_iteration}"
        pair_dir = output_dir / f"{newer_iteration}_vs_{older_iteration}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = pair_dir / "arena.json"
        manifest = create_checkpoint_manifest(args, repo_root, executable.resolve(), config, older, newer)
        if manifest_path.exists() and not args.overwrite:
            with manifest_path.open() as stream:
                previous_manifest = json.load(stream)
            if previous_manifest != manifest:
                if not args.resume:
                    raise ValueError(
                        f"generated settings differ from existing self-eval pair: {manifest_path}; "
                        "use --resume to continue its saved settings or --overwrite to replace it"
                    )
                manifest = previous_manifest
                print(f"checkpoint pair {newer_iteration} vs {older_iteration}: continuing saved arena settings", flush=True)
        if not (args.resume and manifest_path.exists()):
            with manifest_path.open("w") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
        scheduled_games = manifest.get("num_games")
        game_description = f"{scheduled_games} total games" if scheduled_games is not None else "saved seating schedule"
        print(f"checkpoint pair {newer_iteration} vs {older_iteration}: {game_description}", flush=True)
        if args.dry_run:
            continue

        results_path = pair_dir / "games.jsonl"
        run(argparse.Namespace(
            manifest=str(manifest_path),
            output=str(pair_dir),
            gpu=args.gpu,
            num_threads=args.num_threads,
            games_per_seating=None,
            num_games=None,
            max_moves=None,
            resume=results_path.exists() and not args.overwrite,
            overwrite=args.overwrite,
        ))
        newer_wins, older_wins, draws, errors, valid, win_rate = summarize_checkpoint_pair(
            results_path, older_name, newer_name)
        older_elo = ratings.get(older_iteration, 0.0)
        newer_elo = updated_elo(older_elo, win_rate) if valid else float("nan")
        ratings[newer_iteration] = newer_elo
        pair_rows.append({
            "P1": newer_iteration,
            "P2": older_iteration,
            "P1 Wins": newer_wins,
            "P2 Wins": older_wins,
            "Draw": draws,
            "Errors": errors,
            "Total": valid,
            "WinRate": round(win_rate, 6) if valid else "",
            "P1 Elo": round(newer_elo, 3) if valid else "",
        })

    if args.dry_run:
        print(f"generated {len(pairs)} checkpoint-pair manifests under {output_dir}")
        return
    fields = ["P1", "P2", "P1 Wins", "P2 Wins", "Draw", "Errors", "Total", "WinRate", "P1 Elo"]
    write_csv(output_dir / "elo.csv", fields, pair_rows)
    write_self_eval_plot(output_dir / "elo.png", ratings)
    print(f"self-eval summary: {output_dir / 'elo.csv'}")


def checkpoint_sweep_main(argv):
    parser = argparse.ArgumentParser(
        prog="multiplayer-eval.py checkpoint-sweep",
        description="Evaluate one fixed reference model against regularly spaced earlier checkpoints.",
    )
    parser.add_argument("game", help="MiniZero multiplayer game type")
    parser.add_argument("training_dir", help="training folder containing model/ and a config")
    parser.add_argument("--reference", required=True,
                        help="fixed reference model path, file name, or iteration number")
    parser.add_argument("--step", type=int, default=1000,
                        help="training-step spacing between evaluated checkpoints")
    parser.add_argument("--start", type=int, default=0,
                        help="first training step to evaluate")
    parser.add_argument("--end", type=int,
                        help="last training step to evaluate (default: immediately before reference)")
    parser.add_argument("--games", type=int, default=200,
                        help="total seat-balanced games for each reference/checkpoint pair")
    parser.add_argument("--conf-file", help="evaluation config (default: newest *.cfg in the training folder)")
    parser.add_argument("-d", "--output", help="result directory")
    parser.add_argument("--search-type", choices=("maxn", "paranoid"),
                        help="default: value from config, or maxn")
    parser.add_argument("--num-players", type=int)
    parser.add_argument("--num-simulations", type=int)
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--noise", dest="noise", action="store_true", help="enable Dirichlet noise")
    noise_group.add_argument("--no-noise", dest="noise", action="store_false", help="disable Dirichlet noise")
    parser.set_defaults(noise=False)
    parser.add_argument(
        "-conf_str", "--conf-str",
        dest="conf_str",
        default=("actor_select_action_by_count=true:"
                 "actor_select_action_by_softmax_count=false:"
                 "actor_use_random_rotation_features=false"),
    )
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-num-threads", type=int, default=2)
    parser.add_argument("--executable", help="engine executable (default: build/GAME/minizero_GAME)")
    parser.add_argument("-g", "--gpu", help="GPU list, for example 0123 or 0,1,2,3")
    parser.add_argument("--num_threads", "--num-threads", "--threads",
                        dest="num_threads", type=int, default=1)
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument("--resume", action="store_true")
    resume_mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only generate pair arena manifests")
    args = parser.parse_args(argv)
    args.game = args.game.lower()
    if args.num_players is None:
        args.num_players = 4 if args.game in ("blokus", "blokus10", "blokus15") else 3

    positive = (args.step, args.games, args.num_players, args.command_timeout,
                args.omp_num_threads, args.num_threads)
    if any(value <= 0 for value in positive) or args.start < 0:
        parser.error("step, game, player, timeout, and thread values must be positive; start cannot be negative")
    if not 2 <= args.num_players <= len(PLAYER_CODES):
        parser.error(f"--num-players must be between 2 and {len(PLAYER_CODES)}")
    if args.num_simulations is not None and args.num_simulations <= 0:
        parser.error("--num-simulations must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    training_dir = Path(args.training_dir).resolve()
    if not training_dir.is_dir():
        parser.error(f"training directory not found: {training_dir}")
    config = resolve_auto_config(training_dir, args.conf_file)
    if args.search_type is None:
        args.search_type = read_config_value(config, "actor_multiplayer_search_type") or "maxn"
    executable = Path(args.executable).resolve() if args.executable else repo_root / "build" / args.game / f"minizero_{args.game}"
    if not executable.is_file():
        parser.error(f"engine executable not found: {executable}; build it before evaluation")

    reference = resolve_auto_model(training_dir, args.reference)
    reference_iteration = model_iteration(reference)
    end = reference_iteration - 1 if args.end is None else args.end
    if end >= reference_iteration:
        parser.error("--end must be earlier than the reference iteration")
    if end < args.start:
        parser.error("--end must be greater than or equal to --start")
    checkpoints = {
        model_iteration(path): path
        for path in find_checkpoints(training_dir)
    }
    requested_iterations = list(range(args.start, end + 1, args.step))
    missing = [iteration for iteration in requested_iterations if iteration not in checkpoints]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = " ..." if len(missing) > 10 else ""
        parser.error(f"missing requested checkpoints: {preview}{suffix}")

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else training_dir / "checkpoint_sweep" / f"reference_{reference_iteration}_step_{args.step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for older_iteration in requested_iterations:
        older = checkpoints[older_iteration]
        older_name = f"iter_{older_iteration}"
        reference_name = f"iter_{reference_iteration}"
        pair_dir = output_dir / f"{reference_iteration}_vs_{older_iteration}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = pair_dir / "arena.json"
        manifest = create_checkpoint_manifest(args, repo_root, executable.resolve(), config, older, reference)
        manifest["checkpoint_sweep"] = {
            "reference_iteration": reference_iteration,
            "comparison_iteration": older_iteration,
            "step": args.step,
        }
        manifest["share_agent_engines"] = True
        if manifest_path.exists() and not args.overwrite:
            with manifest_path.open() as stream:
                previous_manifest = json.load(stream)
            if previous_manifest != manifest:
                if not args.resume:
                    raise ValueError(
                        f"generated settings differ from existing checkpoint sweep pair: {manifest_path}; "
                        "use --resume to continue its saved settings or --overwrite to replace it"
                    )
                manifest = previous_manifest
                print(f"reference {reference_iteration} vs {older_iteration}: continuing saved settings", flush=True)
        if not (args.resume and manifest_path.exists()):
            with manifest_path.open("w") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
        print(f"reference {reference_iteration} vs {older_iteration}: {manifest['num_games']} total games", flush=True)
        if args.dry_run:
            continue

        results_path = pair_dir / "games.jsonl"
        run(argparse.Namespace(
            manifest=str(manifest_path),
            output=str(pair_dir),
            gpu=args.gpu,
            num_threads=args.num_threads,
            games_per_seating=None,
            num_games=None,
            max_moves=None,
            resume=results_path.exists() and not args.overwrite,
            overwrite=args.overwrite,
        ))
        reference_wins, older_wins, draws, errors, valid, score = summarize_checkpoint_pair(
            results_path, older_name, reference_name)
        rows.append({
            "reference": reference_iteration,
            "comparison": older_iteration,
            "reference_wins": reference_wins,
            "comparison_wins": older_wins,
            "draws": draws,
            "errors": errors,
            "valid_games": valid,
            "reference_score": round(score, 6) if valid else "",
        })
        write_csv(
            output_dir / "sweep.csv",
            ["reference", "comparison", "reference_wins", "comparison_wins",
             "draws", "errors", "valid_games", "reference_score"],
            rows,
        )

    if args.dry_run:
        print(f"generated {len(requested_iterations)} checkpoint-pair manifests under {output_dir}")
        return
    print(f"checkpoint sweep summary: {output_dir / 'sweep.csv'}")


def run(args):
    config = load_manifest(args.manifest)
    if args.games_per_seating is not None:
        config["games_per_seating"] = args.games_per_seating
        config["num_games"] = None
    if getattr(args, "num_games", None) is not None:
        config["num_games"] = args.num_games
    if args.max_moves is not None:
        config["max_moves"] = args.max_moves
    output_dir = Path(args.output).resolve()
    results_path = prepare_output(output_dir, args.resume, args.overwrite)
    existing = load_existing_results(results_path) if args.resume else []
    completed = {record["game_id"] for record in existing if not record.get("error")}
    schedule = create_schedule(config)

    lock_path = output_dir / "arena.lock"
    try:
        with lock_path.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"arena appears to be running: {lock_path}") from exc

    result_queue = queue.Queue()
    task_queue = queue.Queue()
    for task in schedule:
        if any(game.game_id not in completed for game in task.games):
            task_queue.put(task)

    new_results = []
    mode = "a" if args.resume else "w"
    total_pending = sum(1 for task in schedule for game in task.games if game.game_id not in completed)
    try:
        with results_path.open(mode) as stream:
            gpus = parse_gpu_list(args.gpu)
            worker_devices = [gpu for _ in range(args.num_threads) for gpu in gpus] if gpus else [""] * args.num_threads
            worker_devices = worker_devices[:max(1, min(len(worker_devices), task_queue.qsize() or 1))]
            threads = [
                threading.Thread(
                    target=worker,
                    args=(worker_id, gpu, task_queue, config, output_dir, completed, result_queue),
                    daemon=True,
                )
                for worker_id, gpu in enumerate(worker_devices)
            ]
            if total_pending:
                assignment = ", ".join(
                    f"worker {worker_id}={'GPU ' + gpu if gpu else 'inherited device environment'}"
                    for worker_id, gpu in enumerate(worker_devices)
                )
                print(f"arena workers: {assignment}", flush=True)
            for thread in threads:
                thread.start()

            received = 0
            while received < total_pending:
                record = result_queue.get()
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                new_results.append(record)
                received += 1
                status = f"ERROR: {record['error']}" if record.get("error") else f"RE={record['returns']}"
                print(f"game {record['game_id'] + 1}: {'/'.join(record['seating'])}: {status}", flush=True)
            for thread in threads:
                thread.join()
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    latest_results = {record["game_id"]: record for record in existing}
    latest_results.update({record["game_id"]: record for record in new_results})
    all_results = [latest_results[game_id] for game_id in sorted(latest_results)]
    agent_rows, valid, errors = summarize(all_results, config["agents"], config["players"], output_dir)
    print(f"completed {valid} valid games with {errors} errors")
    for stats in agent_rows:
        print(
            f"{stats['agent']}: games={stats['games']} wins={stats['wins']} draws={stats['draws']} "
            f"losses={stats['losses']} win_rate={stats['win_rate']:.4f} avg_return={stats['avg_return']:.4f}"
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        auto_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "blokus-baseline":
        blokus_baseline_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "self-eval":
        self_eval_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "checkpoint-sweep":
        checkpoint_sweep_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON arena manifest")
    parser.add_argument("output", help="output directory")
    parser.add_argument("-g", "--gpu", help="GPU list, matching other MiniZero tools (for example 0123 or 0,1,2,3)")
    parser.add_argument(
        "--num_threads", "--num-threads", "--threads",
        dest="num_threads",
        type=int,
        default=1,
        help="parallel seating workers per GPU (or total workers when -g is omitted)",
    )
    parser.add_argument("--games-per-seating", type=int, help="override manifest value")
    parser.add_argument("--num-games", type=int, help="override with an exact total game count")
    parser.add_argument("--max-moves", type=int, help="override manifest value")
    parser.add_argument("--resume", action="store_true", help="skip game IDs already present in games.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="replace games.jsonl and summaries")
    args = parser.parse_args()
    if (args.num_threads < 1 or
            (args.games_per_seating is not None and args.games_per_seating < 1) or
            (args.num_games is not None and args.num_games < 1)):
        parser.error("thread and game counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
