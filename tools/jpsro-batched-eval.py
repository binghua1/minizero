#!/usr/bin/env python3
"""Evaluate fixed JPSRO profiles with MiniZero's batched self-play actor."""

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    batch = data.get("batched_evaluation")
    if not batch:
        raise ValueError("manifest has no batched_evaluation section")
    if data.get("seat_mode") != "fixed":
        raise ValueError("batched JPSRO evaluation requires seat_mode=fixed")

    profiles = data.get("jpsro_profiles", data.get("lineups", []))
    players = data.get("players", [])
    if not profiles or not players:
        raise ValueError("manifest has no profiles or players")
    if any(len(profile) != len(players) for profile in profiles):
        raise ValueError("every profile must specify one policy per player")

    models = {
        agent["name"]: str(Path(agent["model_path"]).resolve())
        for agent in data.get("agents", [])
        if agent.get("model_path")
    }
    missing = sorted({policy for profile in profiles for policy in profile} - models.keys())
    if missing:
        raise ValueError(f"manifest has no model_path for policies: {missing}")
    for policy, model in models.items():
        if not Path(model).is_file():
            raise ValueError(f"model for {policy} does not exist: {model}")

    executable = Path(batch["executable"]).resolve()
    config_file = Path(batch["config_file"]).resolve()
    if not executable.is_file() or not config_file.is_file():
        raise ValueError("batched evaluator executable or config file does not exist")
    games_per_profile = int(data.get("games_per_seating", 1))
    if games_per_profile < 1:
        raise ValueError("games_per_seating must be positive")

    return {
        "path": path,
        "game": str(data.get("game", "multiplayer")),
        "players": list(players),
        "profiles": [list(profile) for profile in profiles],
        "models": models,
        "games_per_profile": games_per_profile,
        "seed": int(data.get("seed", 0)),
        "timeout": float(data.get("command_timeout", 300)),
        "executable": executable,
        "config_file": config_file,
        "cwd": Path(batch.get("cwd", executable.parent)).resolve(),
        "simulations": batch.get("num_simulations"),
        "search_type": str(batch.get("search_type", "maxn")),
        "noise": bool(batch.get("noise", False)),
    }


def parse_selfplay(line, num_players):
    fields = line.rstrip("\r\n").split(" ", 5)
    if len(fields) != 6 or fields[0] != "SelfPlay" or fields[1] != "true":
        return None
    returns = [float(value) for value in fields[4].split(",")]
    if len(returns) != num_players:
        raise RuntimeError(f"expected {num_players} returns, got {returns}")
    record = fields[5]
    if not record.endswith(" #"):
        raise RuntimeError("batched actor returned a malformed game record")
    return returns, record[:-2]


def result_info(returns):
    best = max(returns)
    winners = [index for index, value in enumerate(returns) if value == best]
    return (winners[0], False) if len(winners) == 1 else (None, True)


def configuration(manifest, profile, parallel_games, cpu_threads, seed):
    values = {
        "nn_file_name": manifest["models"][profile[0]],
        "program_auto_seed": "false",
        "program_seed": seed,
        "program_quiet": "true",
        "actor_multiplayer_search_type": manifest["search_type"],
        "actor_use_dirichlet_noise": str(manifest["noise"]).lower(),
        "actor_use_gumbel": "false",
        "actor_use_gumbel_noise": "false",
        "actor_use_random_rotation_features": str(manifest["noise"]).lower(),
        "actor_select_action_by_count": "true",
        "actor_select_action_by_softmax_count": "false",
        "actor_mcts_value_rescale": "false",
        "zero_disable_resign_ratio": 1,
        "zero_actor_intermediate_sequence_length": 0,
        "zero_num_parallel_games": parallel_games,
        "zero_num_threads": cpu_threads,
        "zero_use_population": "false",
        "zero_use_jpsro": "true",
        "zero_jpsro_eval_only": "true",
        "zero_jpsro_profile_file": manifest["path"],
    }
    if manifest["simulations"] is not None:
        values["actor_num_simulation"] = int(manifest["simulations"])
    return ":".join(f"{key}={value}" for key, value in values.items())


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=10)
    except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_profile(manifest, lineup_id, profile, count, batch_size, cpu_threads,
                gpu, stderr_path):
    parallel_games = min(count, batch_size)
    seed = manifest["seed"] + lineup_id * 1000003
    command = [
        str(manifest["executable"]), "-mode", "sp",
        "-conf_file", str(manifest["config_file"]),
        "-conf_str", configuration(manifest, profile, parallel_games, cpu_threads, seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output = queue.Queue()
    started = time.monotonic()
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command, cwd=manifest["cwd"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, bufsize=1,
        )

        def read_stdout():
            for line in process.stdout:
                output.put(line)
            output.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        games = []
        try:
            paths = ",".join(manifest["models"][policy] for policy in profile)
            process.stdin.write(
                f"load_profile eval_{lineup_id} {','.join(profile)} "
                f"{','.join('0' for _ in profile)} {paths}\nstart\n"
            )
            process.stdin.flush()
            while len(games) < count:
                try:
                    line = output.get(timeout=manifest["timeout"])
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"profile {lineup_id} produced no game for {manifest['timeout']} seconds"
                    ) from exc
                if line is None:
                    raise RuntimeError(
                        f"batched actor exited with code {process.poll()} after "
                        f"{len(games)}/{count} games; see {stderr_path}"
                    )
                parsed = parse_selfplay(line, len(profile))
                if parsed is not None:
                    games.append(parsed)
        finally:
            stop_process(process)
            reader.join(timeout=2)
    return games, time.monotonic() - started


def load_existing(path):
    records = []
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    return records


def run(args):
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sgf").mkdir(exist_ok=True)
    log_dir = output_dir / "engine_logs"
    log_dir.mkdir(exist_ok=True)
    games_path = output_dir / "games.jsonl"
    if args.overwrite:
        games_path.write_text("")
    elif games_path.exists() and not args.resume:
        raise ValueError(f"{games_path} exists; use --resume or --overwrite")

    existing = load_existing(games_path)
    completed = {int(game["game_id"]) for game in existing if not game.get("error")}
    lock_path = output_dir / "arena.lock"
    try:
        with lock_path.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"arena appears to be running: {lock_path}") from exc

    total = len(manifest["profiles"]) * manifest["games_per_profile"]
    try:
        with games_path.open("a") as stream:
            for lineup_id, profile in enumerate(manifest["profiles"]):
                first_id = lineup_id * manifest["games_per_profile"]
                specs = [
                    (first_id + repeat, repeat)
                    for repeat in range(manifest["games_per_profile"])
                    if first_id + repeat not in completed
                ]
                if not specs:
                    continue
                print(
                    f"profile {lineup_id + 1}/{len(manifest['profiles'])}: "
                    f"{'/'.join(profile)}, {len(specs)} games, batch={min(args.batch_size, len(specs))}",
                    flush=True,
                )
                games, elapsed = run_profile(
                    manifest, lineup_id, profile, len(specs), args.batch_size,
                    args.cpu_threads, args.gpu,
                    log_dir / f"profile_{lineup_id:04d}.log",
                )
                for (game_id, repeat), (returns, game_record) in zip(specs, games):
                    winner_seat, draw = result_info(returns)
                    record = {
                        "game_id": game_id,
                        "lineup_id": lineup_id,
                        "seating_id": 0,
                        "repeat": repeat,
                        "seating": profile,
                        "players": manifest["players"],
                        "moves": [],
                        "returns": returns,
                        "winner_seat": winner_seat,
                        "winner_agent": None if winner_seat is None else profile[winner_seat],
                        "draw": draw,
                        "error": None,
                        "duration_seconds": round(elapsed / len(games), 6),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    (output_dir / "sgf" / f"game_{game_id:06d}.sgf").write_text(game_record + "\n")
                stream.flush()
                print(f"  completed in {elapsed:.2f}s", flush=True)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    valid = {int(game["game_id"]) for game in load_existing(games_path) if not game.get("error")}
    print(f"completed {len(valid)}/{total} valid batched payoff games")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("-g", "--gpu", default="0", help="one CUDA device index")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="parallel games inside one batched actor")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="CPU worker threads supporting batched search")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU thread count must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
