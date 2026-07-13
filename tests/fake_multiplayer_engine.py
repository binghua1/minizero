#!/usr/bin/env python3

"""Tiny GTP-like console used by test_multiplayer_eval.py."""

import argparse
import os
import sys


def reply(payload=""):
    print(f"= {payload}\n", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--num-players", type=int, default=3)
args = parser.parse_args()
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", file=sys.stderr, flush=True)

moves = []
actions = {"b": "A1", "w": "B1", "r": "C1"}
for raw_command in sys.stdin:
    command = raw_command.strip().split()
    if not command:
        continue
    if command[0] == "quit":
        break
    if command[0] == "clear_board":
        moves = []
        reply()
    elif command[0] == "genmove":
        if len(moves) >= args.num_players:
            reply("PASS")
        else:
            player = command[1].lower()
            action = actions[player]
            moves.append((player, action))
            reply(action)
    elif command[0] == "play":
        moves.append((command[1].lower(), command[2]))
        reply()
    elif command[0] == "game_string":
        result = "1" if args.num_players == 2 else "1,-1,-1"
        reply(f"(;GM[fake]RE[{result}])")
    else:
        print(f"? unsupported command: {' '.join(command)}\n", flush=True)
