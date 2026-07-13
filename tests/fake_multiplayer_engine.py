#!/usr/bin/env python3

"""Tiny GTP-like console used by test_multiplayer_eval.py."""

import sys


def reply(payload=""):
    print(f"= {payload}\n", flush=True)


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
        if len(moves) >= 3:
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
        reply("(;GM[fake]RE[1,-1,-1])")
    else:
        print(f"? unsupported command: {' '.join(command)}\n", flush=True)
