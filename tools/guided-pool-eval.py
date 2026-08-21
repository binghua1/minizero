#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def allocate(weights, total):
    raw = [weight * total / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def agent(name, model, config, executable, repo, simulations, noise):
    conf = ":".join([
        "actor_use_gumbel=false",
        "actor_use_gumbel_noise=false",
        "actor_mcts_value_rescale=false",
        "zero_disable_resign_ratio=1",
        "zero_actor_intermediate_sequence_length=0",
        f"actor_use_dirichlet_noise={'true' if noise else 'false'}",
        f"actor_num_simulation={simulations}",
        f"nn_file_name={model}",
        "actor_multiplayer_search_type=maxn",
        "program_seed={seed}",
        "program_auto_seed=false",
    ])
    return {
        "name": name,
        "cwd": str(repo),
        "env": {"OMP_NUM_THREADS": "2"},
        "command": [str(executable), "-mode", "console", "-conf_file", str(config), "-conf_str", conf],
    }


def create(args):
    output = Path(args.output).resolve()
    if output.exists():
        print(f"using existing pool arena: {output}")
        return
    run = Path(args.run_dir).resolve()
    repo = Path(__file__).resolve().parents[1]
    meta = json.loads((run / "meta/meta_strategy.json").read_text())
    profiles = [(tuple(row["profile"]), float(row["probability"])) for row in meta["distribution"]]
    if not profiles or abs(sum(weight for _, weight in profiles) - 1.0) > 1e-5:
        raise SystemExit("invalid or empty meta strategy")

    masks = []
    for count in (1, 2):
        masks.extend(itertools.combinations(range(3), count))
    cases = [(profile, seats, probability / len(masks)) for profile, probability in profiles for seats in masks]
    counts = allocate([item[2] for item in cases], args.games)
    alpha_name = args.alpha_name
    lineups = []
    for (profile, seats, _), count in zip(cases, counts):
        lineup = list(profile)
        for seat in seats:
            lineup[seat] = alpha_name
        lineups.extend([lineup] * count)

    policy_ids = sorted({policy for profile, _ in profiles for policy in profile})
    config = Path(args.config).resolve()
    executable = repo / "build/connect3x3/minizero_connect3x3"
    agents = [
        agent(policy, run / f"frozen/{policy}.pt", config, executable, repo, args.simulations, args.noise)
        for policy in policy_ids
    ]
    agents.append(agent(alpha_name, Path(args.alpha_model).resolve(), Path(args.alpha_config).resolve(),
                        executable, repo, args.simulations, args.noise))
    missing = [item["command"][-1].split("nn_file_name=", 1)[-1].split(":", 1)[0]
               for item in agents if not Path(item["command"][-1].split("nn_file_name=", 1)[-1].split(":", 1)[0]).is_file()]
    if missing:
        raise SystemExit(f"missing pool model(s): {missing}")

    manifest = {
        "game": "connect3x3",
        "players": ["b", "w", "r"],
        "agents": agents,
        "lineups": lineups,
        "seat_mode": "fixed",
        "num_games": args.games,
        "max_moves": 42,
        "terminal_passes": 1,
        "pass_mode": "terminal",
        "command_timeout": 300,
        "seed": args.seed,
        "share_agent_engines": True,
        "guided_pool_eval": {
            "alpha_agent": alpha_name,
            "meta_strategy": meta,
            "replacement_seat_counts": [1, 2],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    print(f"created pool arena with {len(lineups)} games: {output}")


def summarize(args):
    games = Path(args.games_file)
    output = Path(args.output)
    pool_wins = alpha_wins = draws = errors = 0
    with games.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                errors += 1
                continue
            best = max(row["returns"])
            top = {name for name, value in zip(row["seating"], row["returns"]) if value == best}
            alpha_top = args.alpha_name in top
            pool_top = bool(top - {args.alpha_name})
            if alpha_top and pool_top:
                draws += 1
            elif alpha_top:
                alpha_wins += 1
            else:
                pool_wins += 1
    valid = pool_wins + alpha_wins + draws
    score = (pool_wins + 0.5 * draws) / valid if valid else float("nan")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "guided_pool_wins", "alphazero_wins", "draws", "errors", "valid_games", "guided_pool_score"
        ])
        writer.writeheader()
        writer.writerow({
            "guided_pool_wins": pool_wins, "alphazero_wins": alpha_wins, "draws": draws,
            "errors": errors, "valid_games": valid, "guided_pool_score": round(score, 6),
        })
    print(f"pool vs {args.alpha_name}: wins={pool_wins} losses={alpha_wins} draws={draws} score={score:.1%}")


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
make = sub.add_parser("create")
make.add_argument("run_dir")
make.add_argument("alpha_model")
make.add_argument("alpha_config")
make.add_argument("alpha_name")
make.add_argument("output")
make.add_argument("--config", required=True)
make.add_argument("--games", type=int, default=600)
make.add_argument("--simulations", type=int, default=50)
make.add_argument("--seed", type=int, default=0)
make.add_argument("--noise", action="store_true")
make.set_defaults(func=create)
summary = sub.add_parser("summarize")
summary.add_argument("games_file")
summary.add_argument("alpha_name")
summary.add_argument("output")
summary.set_defaults(func=summarize)
arguments = parser.parse_args()
arguments.func(arguments)
