#!/usr/bin/env python3

import argparse
import os
import queue
import subprocess
import threading


class Engine:
    def __init__(self, command):
        self.command = command
        self.proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self.command_response("quit")
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()

    def command_response(self, command):
        if self.proc.poll() is not None:
            raise RuntimeError(f"engine exited: {self.command}")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"engine closed stdout after command: {command}")
            line = line.rstrip("\n")
            if line == "":
                break
            lines.append(line)
        if not lines:
            raise RuntimeError(f"empty response from command: {command}")

        first = lines[0]
        ok = first.startswith("=")
        fail = first.startswith("?")
        if not ok and not fail:
            raise RuntimeError(f"bad GTP response to {command}: {first}")
        payload = first[1:].strip()
        if len(lines) > 1:
            payload = "\n".join([payload] + lines[1:]).strip()
        if fail:
            raise RuntimeError(payload or f"command failed: {command}")
        return payload


def gtp_quote(action):
    return action.replace("\\", "\\\\").replace("]", "\\]")


def write_sgf(path, result, moves):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fout:
        fout.write(f"(;GM[shogi66]RE[{result:.6f}]")
        for color, action in moves:
            tag = "B" if color == "b" else "W"
            fout.write(f";{tag}[{gtp_quote(action)}]")
        fout.write(")\n")


def dat_line(game_id, result, alternate, err="0", err_msg=""):
    # dat columns are shaped like gogui-twogtp output enough for tools/eval.py:
    # 0 GAME, 1 RES_B, 2 RES_W, 3 RES_R, 4 ALT, 5 DUP, 11 ERR, 12 ERR_MSG
    fields = [
        str(game_id),
        f"{result:.6f}",
        f"{-result:.6f}",
        f"{result:.6f}",
        "1" if alternate else "0",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        err,
        err_msg,
    ]
    return "\t".join(fields)


def play_one_game(black_engine, white_engine, game_id, alternate, max_moves):
    engines = {"b": black_engine, "w": white_engine}
    for engine in set(engines.values()):
        engine.command_response("clear_board")

    moves = []
    color = "b"
    result = 0.0
    err = "0"
    err_msg = ""

    for _ in range(max_moves):
        engine = engines[color]
        opponent = engines["w" if color == "b" else "b"]
        try:
            action = engine.command_response(f"genmove {color}").strip()
        except Exception as exc:
            err = "1"
            err_msg = f"genmove failed: {exc}"
            break

        if action.upper() == "PASS":
            try:
                result = float(engine.command_response("final_score"))
            except Exception:
                result = 0.0
            break
        if action == "Resign":
            result = -1.0 if color == "b" else 1.0
            break

        moves.append((color, action))
        other_color = "w" if color == "b" else "b"
        try:
            opponent.command_response(f"play {color} {action}")
        except Exception as exc:
            err = "1"
            err_msg = f'opponent rejected move "{action}": {exc}'
            break
        color = other_color
    else:
        err = "1"
        err_msg = f"reached max moves {max_moves}; adjudicate draw"
        result = 0.0

    return result, moves, dat_line(game_id, result, alternate, err, err_msg)


def run_games_worker(worker_id, args, games, dat, write_lock):
    black = Engine(args.black)
    white = Engine(args.white)
    try:
        while True:
            try:
                game_id = games.get_nowait()
            except queue.Empty:
                return

            alternate = args.alternate and (game_id % 2 == 1)
            if alternate:
                black_engine, white_engine = white, black
            else:
                black_engine, white_engine = black, white

            try:
                result, moves, line = play_one_game(
                    black_engine, white_engine, game_id, alternate, args.max_moves
                )
            except Exception as exc:
                result = 0.0
                moves = []
                line = dat_line(game_id, result, alternate, "1", str(exc))

            write_sgf(f"{args.sgffile}-{game_id}.sgf", result, moves)
            with write_lock:
                dat.write(line + "\n")
                dat.flush()
                print(
                    f"thread {worker_id}: game {game_id + 1}/{args.games}: "
                    f"{'alternate ' if alternate else ''}RE={result:.6f}",
                    flush=True,
                )
            games.task_done()
    finally:
        black.close()
        white.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-black", required=True)
    parser.add_argument("-white", required=True)
    parser.add_argument("-games", type=int, required=True)
    parser.add_argument("-sgffile", required=True)
    parser.add_argument("-alternate", action="store_true")
    parser.add_argument("-threads", type=int, default=2)
    parser.add_argument("-max_moves", type=int, default=2048)
    args = parser.parse_args()

    dat_path = args.sgffile + ".dat"
    lock_path = args.sgffile + ".lock"
    os.makedirs(os.path.dirname(args.sgffile), exist_ok=True)
    with open(lock_path, "w") as fout:
        fout.write(str(os.getpid()))

    try:
        with open(dat_path, "w") as dat:
            dat.write(f"BlackCommand {args.black}\n")
            dat.write(f"WhiteCommand {args.white}\n")
            dat.write("ERR_MSG\n")
            dat.flush()

            games = queue.Queue()
            for game_id in range(args.games):
                games.put(game_id)

            write_lock = threading.Lock()
            num_threads = max(1, min(args.threads, args.games))
            threads = [
                threading.Thread(
                    target=run_games_worker,
                    args=(worker_id, args, games, dat, write_lock),
                    daemon=True,
                )
                for worker_id in range(num_threads)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
