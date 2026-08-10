#!/usr/bin/env python3
"""Manage a MiniZero JPSRO restricted game and AlphaZero oracle profiles."""

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_meta_solver = importlib.import_module("minizero.jpsro.meta_solver")
_state = importlib.import_module("minizero.jpsro.state")
_workflow = importlib.import_module("minizero.jpsro.workflow")
solve_cce = _meta_solver.solve_cce
JPSROState = _state.JPSROState
atomic_json = _workflow.atomic_json
candidate_deviation_gains = _workflow.candidate_deviation_gains
create_evaluation_manifest = _workflow.create_evaluation_manifest
ingest_evaluation = _workflow.ingest_evaluation
required_deviation_profiles = _workflow.required_deviation_profiles
write_oracle_plan = _workflow.write_oracle_plan


def parse_players(value, num_players):
    if value in (None, "all"):
        return list(range(num_players))
    result = [int(item) for item in value.split(",")]
    if any(item < 0 or item >= num_players for item in result):
        raise ValueError(f"players must be between 0 and {num_players - 1}")
    return result


def command_init(args):
    state = JPSROState.create(args.run_dir, args.players, args.shared_pool,
                              args.utility_min, args.utility_max)
    print(f"initialized {state.root} ({state.num_players} players, shared_pool={state.shared_pool})")


def command_add_policy(args):
    state = JPSROState.load(args.run_dir)
    players = parse_players(args.for_players, state.num_players)
    state.add_policy(args.policy_id, args.model, players, args.generation)
    print(f"registered {args.policy_id} for players {players}")


def selected_profiles(state, args):
    if args.candidate:
        return required_deviation_profiles(state, args.candidate)
    return state.payoffs.missing(state.policy_sets(), args.min_games)


def command_make_eval(args):
    state = JPSROState.load(args.run_dir)
    profiles = selected_profiles(state, args)
    if args.max_profiles is not None:
        profiles = profiles[:args.max_profiles]
    if not profiles:
        print("all requested profiles already have enough games")
        return
    manifest = create_evaluation_manifest(
        state, profiles, args.game, args.conf_file, args.executable,
        args.repo_root, args.games_per_profile, args.num_simulations,
        args.search_type, args.noise, args.seed, args.command_timeout,
    )
    atomic_json(args.output, manifest)
    print(f"wrote {len(profiles)} ordered profiles to {Path(args.output).resolve()}")


def command_ingest(args):
    state = JPSROState.load(args.run_dir)
    accepted, errors = ingest_evaluation(state, args.manifest, args.games)
    print(f"ingested {accepted} games; skipped {errors} errors")


def command_solve(args):
    state = JPSROState.load(args.run_dir)
    policy_sets = state.policy_sets()
    missing = state.payoffs.missing(policy_sets, args.min_games)
    if missing:
        raise ValueError(f"cannot certify CCE: {len(missing)} restricted profiles need evaluation")
    result = solve_cce(
        policy_sets, state.payoffs.indexed_means(policy_sets, args.min_games),
        args.iterations, args.tolerance, args.eta,
        state.config["utility_min"], state.config["utility_max"],
    )
    output = state.write_meta_strategy(result)
    print(f"empirical CCE gap={output['gap']:.6g} after {output['iterations']} iterations")
    if output.get("witness"):
        print(f"largest deviation: player {output['witness']['player']} -> {output['witness']['policy']}")


def command_oracle_plan(args):
    state = JPSROState.load(args.run_dir)
    responders = parse_players(args.responders, state.num_players)
    count = write_oracle_plan(state, args.output, responders, args.response_id)
    print(f"wrote {count} one-responder profiles to {Path(args.output).resolve()}")


def command_deviation_gap(args):
    state = JPSROState.load(args.run_dir)
    gains = candidate_deviation_gains(state, args.candidate)
    print(json.dumps({
        "candidate": args.candidate,
        "max_gain": max(item["gain"] for item in gains),
        "per_player": gains,
    }, indent=2))


def command_status(args):
    state = JPSROState.load(args.run_dir)
    policy_sets = state.policy_sets()
    missing = state.payoffs.missing(policy_sets, args.min_games)
    summary = {
        "run_dir": str(state.root),
        "players": state.num_players,
        "shared_pool": state.shared_pool,
        "policies_per_player": [len(items) for items in policy_sets],
        "restricted_profiles": __import__("math").prod(len(items) for items in policy_sets),
        "evaluated_profiles": len(state.payoffs.entries),
        "missing_profiles": len(missing),
        "has_meta_strategy": (state.root / "meta_strategy.json").is_file(),
    }
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    item = subparsers.add_parser("init", help="create an empty JPSRO run")
    item.add_argument("run_dir")
    item.add_argument("--players", type=int, required=True)
    item.add_argument("--shared-pool", action="store_true")
    item.add_argument("--utility-min", type=float, default=-1.0)
    item.add_argument("--utility-max", type=float, default=1.0)
    item.set_defaults(func=command_init)

    item = subparsers.add_parser("add-policy", help="register a frozen policy")
    item.add_argument("run_dir")
    item.add_argument("policy_id")
    item.add_argument("model")
    item.add_argument("--for-players", default="all", help="comma-separated zero-based seats or all")
    item.add_argument("--generation", type=int)
    item.set_defaults(func=command_add_policy)

    item = subparsers.add_parser("make-eval", help="write an arena manifest for missing payoffs")
    item.add_argument("run_dir")
    item.add_argument("output")
    item.add_argument("--game", required=True)
    item.add_argument("--conf-file", required=True)
    item.add_argument("--executable", required=True)
    item.add_argument("--repo-root", default=str(REPO_ROOT))
    item.add_argument("--games-per-profile", type=int, default=4)
    item.add_argument("--min-games", type=int, default=1)
    item.add_argument("--max-profiles", type=int)
    item.add_argument("--candidate", help="only evaluate unilateral profiles for this registered policy")
    item.add_argument("--num-simulations", type=int)
    item.add_argument("--search-type", choices=("maxn", "paranoid", "rank"), default="maxn")
    item.add_argument("--noise", action="store_true")
    item.add_argument("--seed", type=int, default=0)
    item.add_argument("--command-timeout", type=float, default=300)
    item.set_defaults(func=command_make_eval)

    item = subparsers.add_parser("ingest", help="add arena games to the payoff table")
    item.add_argument("run_dir")
    item.add_argument("manifest")
    item.add_argument("games")
    item.set_defaults(func=command_ingest)

    item = subparsers.add_parser("solve", help="solve and certify the complete restricted game")
    item.add_argument("run_dir")
    item.add_argument("--min-games", type=int, default=1)
    item.add_argument("--iterations", type=int, default=10000)
    item.add_argument("--tolerance", type=float, default=0.01)
    item.add_argument("--eta", type=float)
    item.set_defaults(func=command_solve)

    item = subparsers.add_parser("oracle-plan", help="write CCE-opponent profiles for AlphaZero training")
    item.add_argument("run_dir")
    item.add_argument("output")
    item.add_argument("--responders", default="all", help="comma-separated zero-based seats or all")
    item.add_argument("--response-id", default="CURRENT")
    item.set_defaults(func=command_oracle_plan)

    item = subparsers.add_parser("deviation-gap", help="measure a candidate oracle's gain against the previous CCE")
    item.add_argument("run_dir")
    item.add_argument("candidate")
    item.set_defaults(func=command_deviation_gap)

    item = subparsers.add_parser("status", help="summarize a JPSRO run")
    item.add_argument("run_dir")
    item.add_argument("--min-games", type=int, default=1)
    item.set_defaults(func=command_status)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
