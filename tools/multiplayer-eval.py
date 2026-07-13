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
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PLAYER_CODES = ("b", "w", "r", "g", "y", "p")


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


def run_seating(task, config, output_dir, completed, result_queue):
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


def worker(tasks, config, output_dir, completed, result_queue):
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            run_seating(task, config, output_dir, completed, result_queue)
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
            num_threads = max(1, min(args.threads, task_queue.qsize() or 1))
            threads = [
                threading.Thread(target=worker, args=(task_queue, config, output_dir, completed, result_queue), daemon=True)
                for _ in range(num_threads)
            ]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON arena manifest")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--threads", type=int, default=1, help="parallel seatings (each launches one engine per seat)")
    parser.add_argument("--games-per-seating", type=int, help="override manifest value")
    parser.add_argument("--max-moves", type=int, help="override manifest value")
    parser.add_argument("--resume", action="store_true", help="skip game IDs already present in games.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="replace games.jsonl and summaries")
    args = parser.parse_args()
    if args.threads < 1 or (args.games_per_seating is not None and args.games_per_seating < 1):
        parser.error("--threads and --games-per-seating must be positive")
    run(args)


if __name__ == "__main__":
    main()
