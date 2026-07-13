#!/usr/bin/env python3

"""Run seat-balanced matches between multiple MiniZero console engines."""

import argparse
import csv
import itertools
import json
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
        "max_moves": int(data.get("max_moves", 2048)),
        "command_timeout": float(data.get("command_timeout", 300)),
        "seed": int(data.get("seed", 0)),
    }
    if config["seat_mode"] not in ("fixed", "cyclic", "all_permutations"):
        raise ValueError("seat_mode must be fixed, cyclic, or all_permutations")
    if config["games_per_seating"] < 1 or config["max_moves"] < 1 or config["command_timeout"] <= 0:
        raise ValueError("games_per_seating, max_moves, and command_timeout must be positive")
    return config


def unique_seatings(lineup, mode):
    if mode == "fixed":
        candidates = [tuple(lineup)]
    elif mode == "cyclic":
        candidates = [tuple(lineup[offset:] + lineup[:offset]) for offset in range(len(lineup))]
    else:
        candidates = itertools.permutations(lineup)
    return list(dict.fromkeys(candidates))


def create_schedule(config):
    tasks = []
    game_id = 0
    task_id = 0
    for lineup_id, lineup in enumerate(config["lineups"]):
        for seating_id, seating in enumerate(unique_seatings(list(lineup), config["seat_mode"])):
            games = []
            for repeat in range(config["games_per_seating"]):
                games.append(GameSpec(game_id, lineup_id, seating_id, repeat, seating))
                game_id += 1
            tasks.append(SeatingTask(task_id, tuple(games)))
            task_id += 1
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


def play_game(engines, players, max_moves):
    for engine in engines:
        engine.command_response("clear_board")

    moves = []
    turn = 0
    for ply in range(max_moves + 1):
        action = engines[turn].command_response(f"genmove {players[turn]}").strip()
        if action.upper() == "PASS":
            game_string = engines[turn].command_response("game_string")
            returns = parse_returns(game_string, len(players))
            winner_seat, draw = result_info(returns)
            return moves, returns, winner_seat, draw
        if action.lower() == "resign":
            raise RuntimeError("multiplayer resignation has no generic winner semantics; disable resignation")
        if ply == max_moves:
            raise RuntimeError(f"reached max_moves={max_moves} without a terminal PASS")

        moves.append({"player": players[turn], "action": action})
        for seat, engine in enumerate(engines):
            if seat != turn:
                engine.command_response(f"play {players[turn]} {action}")
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
    games = [game for game in task.games if game.game_id not in completed]
    if not games:
        return
    seating = games[0].seating
    engines = []
    stderr_dir = output_dir / "engine_logs"
    stderr_dir.mkdir(exist_ok=True)
    try:
        for seat, agent_name in enumerate(seating):
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
            engines.append(Engine(config["agents"][agent_name], context, config["command_timeout"], stderr_path))

        for game_index, spec in enumerate(games):
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
                moves, returns, winner_seat, draw = play_game(engines, config["players"], config["max_moves"])
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
                for remaining in games[game_index + 1:]:
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
        for spec in games:
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
        for engine in engines:
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
    return agent_rows, len(valid), len(results) - len(valid)


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
        "max_moves": args.max_moves if args.max_moves is not None else (15 if args.game == "tictacmo" else 2048),
        "command_timeout": args.command_timeout,
        "seed": args.seed,
    }


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
    parser.add_argument("--conf-file", help="config path (default: newest *.cfg in the training folder)")
    parser.add_argument("--maxn-conf-file", help="config used by the MaxN agent")
    parser.add_argument("--paranoid-conf-file", help="config used by the Paranoid agent")
    parser.add_argument("--executable", help="engine executable (default: build/GAME/minizero_GAME)")
    parser.add_argument("--output", help="output directory (default: TRAINING_DIR/evaluation/MODEL_SEARCHES)")
    parser.add_argument("--search-types", nargs="+", default=["maxn", "paranoid"], choices=("maxn", "paranoid"))
    parser.add_argument("--num-players", type=int, default=3)
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

    if not 2 <= args.num_players <= len(PLAYER_CODES):
        parser.error(f"--num-players must be between 2 and {len(PLAYER_CODES)}")
    if len(set(args.search_types)) != len(args.search_types):
        parser.error("--search-types must not contain duplicates")
    positive = (args.games_per_seating, args.command_timeout, args.omp_num_threads, args.num_threads)
    if any(value <= 0 for value in positive) or (args.num_simulations is not None and args.num_simulations <= 0):
        parser.error("game, timeout, thread, and simulation counts must be positive")

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
        else training_dir / "evaluation" / f"{arena_label}{simulation_label}{noise_label}"
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
        max_moves=None,
        resume=args.resume,
        overwrite=args.overwrite,
    ))


def run(args):
    config = load_manifest(args.manifest)
    if args.games_per_seating is not None:
        config["games_per_seating"] = args.games_per_seating
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
    parser.add_argument("--max-moves", type=int, help="override manifest value")
    parser.add_argument("--resume", action="store_true", help="skip game IDs already present in games.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="replace games.jsonl and summaries")
    args = parser.parse_args()
    if args.num_threads < 1 or (args.games_per_seating is not None and args.games_per_seating < 1):
        parser.error("--num_threads and --games-per-seating must be positive")
    run(args)


if __name__ == "__main__":
    main()
