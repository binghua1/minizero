#!/usr/bin/env python3

import csv
import importlib.util
import json
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA = REPO_ROOT / "tools" / "multiplayer-eval.py"
FAKE_ENGINE = REPO_ROOT / "tests" / "fake_multiplayer_engine.py"


def load_arena_module():
    spec = importlib.util.spec_from_file_location("multiplayer_eval_under_test", ARENA)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiplayerEvalTest(unittest.TestCase):
    def test_resume_replays_completed_games_to_restore_engine_rng_state(self):
        arena = load_arena_module()
        calls = []

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        def fake_play_game(*args, **kwargs):
            calls.append(len(calls) + 1)
            value = float(calls[-1])
            return [], [value, -value], 0, False

        original_engine = arena.Engine
        original_play_game = arena.play_game
        arena.Engine = FakeEngine
        arena.play_game = fake_play_game
        try:
            task = arena.SeatingTask(0, (
                arena.GameSpec(0, 0, 0, 0, ("new", "old")),
                arena.GameSpec(1, 0, 0, 1, ("new", "old")),
            ))
            config = {
                "agents": {"new": {}, "old": {}},
                "players": ["b", "w"],
                "command_timeout": 1,
                "max_moves": 10,
                "game": "fake",
                "seed": 0,
                "share_agent_engines": True,
            }
            results = queue.Queue()
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir)
                (output / "sgf").mkdir()
                arena.run_seating(task, config, output, {0}, results, 0, "")
            resumed = results.get_nowait()
        finally:
            arena.Engine = original_engine
            arena.play_game = original_play_game

        self.assertEqual(calls, [1, 2])
        self.assertEqual(resumed["game_id"], 1)
        self.assertEqual(resumed["returns"], [2.0, -2.0])

    def test_checkpoint_summary_treats_same_model_top_seats_as_a_win(self):
        arena = load_arena_module()
        records = [
            {
                "game_id": 0,
                "error": None,
                "returns": [1.0, 1.0, 0.0, -1.0],
                "seating": ["new", "new", "old", "old"],
            },
            {
                "game_id": 1,
                "error": None,
                "returns": [1.0, 1.0, 0.0, -1.0],
                "seating": ["new", "old", "new", "old"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir) / "games.jsonl"
            results.write_text("".join(json.dumps(record) + "\n" for record in records))
            new_wins, old_wins, draws, errors, valid, score = arena.summarize_checkpoint_pair(
                results, "old", "new"
            )
        self.assertEqual((new_wins, old_wins, draws, errors, valid), (1, 0, 1, 0, 2))
        self.assertEqual(score, 0.75)

    def test_model_fight_builds_both_compositions_and_all_seats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            model_a = temp / "jpsro.pt"
            model_b = temp / "alphazero.pt"
            config_a = temp / "jpsro.cfg"
            config_b = temp / "alphazero.cfg"
            for path in (model_a, model_b):
                path.touch()
            for path in (config_a, config_b):
                path.write_text("actor_num_simulation=50\n")
            output = temp / "fight"

            subprocess.run(
                [
                    sys.executable, str(ARENA), "model-fight", "tictacmo",
                    str(model_a), str(model_b),
                    "--conf-file-a", str(config_a),
                    "--conf-file-b", str(config_b),
                    "--names", "jpsro", "alphazero",
                    "--games", "12",
                    "--executable", str(FAKE_ENGINE),
                    "--output", str(output),
                    "--num_threads", "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output / "arena.json").read_text())
            self.assertEqual(manifest["lineups"], [
                ["jpsro", "jpsro", "alphazero"],
                ["jpsro", "alphazero", "alphazero"],
            ])
            self.assertEqual(manifest["seat_mode"], "all_permutations")
            self.assertEqual(manifest["num_games"], 12)
            self.assertEqual(len((output / "games.jsonl").read_text().splitlines()), 12)
            with (output / "fight_summary.csv").open() as stream:
                summary = next(csv.DictReader(stream))
            self.assertEqual(summary["model_a"], "jpsro")
            self.assertEqual(summary["model_b"], "alphazero")
            self.assertEqual(summary["valid_games"], "12")

    def test_blokus_elimination_passes_skip_players_and_end_after_four(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output = temp / "output"
            command = [sys.executable, str(FAKE_ENGINE), "--num-players", "4", "--pass-only"]
            manifest = {
                "game": "blokus",
                "players": ["b", "w", "r", "g"],
                "agents": [{"name": "passer", "command": command}],
                "lineups": [["passer"] * 4],
                "seat_mode": "fixed",
                "games_per_seating": 1,
                "max_moves": 10,
                "terminal_passes": 4,
            }
            manifest_path = temp / "arena.json"
            manifest_path.write_text(json.dumps(manifest))
            subprocess.run(
                [sys.executable, str(ARENA), str(manifest_path), str(output)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            game = json.loads((output / "games.jsonl").read_text())
            self.assertFalse(game["error"])
            self.assertEqual(game["returns"], [1.0, -1.0, -1.0, -1.0])
            self.assertEqual([move["player"] for move in game["moves"]], ["b", "w", "r", "g"])

    def test_self_eval_pairs_checkpoints_and_uses_total_game_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "connect3x3_run"
            model_dir = training / "model"
            model_dir.mkdir(parents=True)
            config = training / "run.cfg"
            config.write_text("actor_num_simulation=50\nactor_use_dirichlet_noise=true\n")
            for iteration in (0, 500, 1000):
                (model_dir / f"weight_iter_{iteration}.pt").touch()
            output = training / "self_eval"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "self-eval",
                    "connect3x3",
                    str(training),
                    "--conf-file",
                    str(config),
                    "--interval",
                    "1",
                    "--games",
                    "10",
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--num_threads",
                    "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            first_manifest = (output / "500_vs_0" / "arena.json").read_text()
            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "self-eval",
                    "connect3x3",
                    str(training),
                    "--conf-file",
                    str(config),
                    "--interval",
                    "1",
                    "--games",
                    "10",
                    "--no-noise",
                    "--resume",
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((output / "500_vs_0" / "arena.json").read_text(), first_manifest)

            pair_names = ("500_vs_0", "1000_vs_500")
            for pair_name in pair_names:
                pair_dir = output / pair_name
                manifest = json.loads((pair_dir / "arena.json").read_text())
                self.assertEqual(manifest["num_games"], 10)
                self.assertEqual(len((pair_dir / "games.jsonl").read_text().splitlines()), 10)
                self.assertEqual(len(manifest["agents"]), 2)
                self.assertEqual(len(manifest["lineups"]), 2)
                self.assertTrue(all(
                    "actor_use_dirichlet_noise=" not in agent["command"][-1]
                    for agent in manifest["agents"]
                ))

            with (output / "elo.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(row["P1"], row["P2"]) for row in rows], [("500", "0"), ("1000", "500")])
            self.assertTrue(all(int(row["Total"]) == 10 for row in rows))
            self.assertTrue(all(float(row["WinRate"]) == 0.5 for row in rows))

    def test_checkpoint_sweep_uses_fixed_reference_and_training_step_spacing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "connect3x3_run"
            model_dir = training / "model"
            model_dir.mkdir(parents=True)
            config = training / "run.cfg"
            config.write_text("actor_num_simulation=50\nactor_multiplayer_search_type=maxn\n")
            for iteration in (0, 500, 1000, 1500, 2000):
                (model_dir / f"weight_iter_{iteration}.pt").touch()
            output = training / "checkpoint_sweep"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "checkpoint-sweep",
                    "connect3x3",
                    str(training),
                    "--reference",
                    "2000",
                    "--step",
                    "1000",
                    "--games",
                    "10",
                    "--no-noise",
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--num_threads",
                    "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            pair_names = ("2000_vs_0", "2000_vs_1000")
            self.assertEqual(
                sorted(path.name for path in output.iterdir() if path.is_dir()),
                list(pair_names),
            )
            for pair_name in pair_names:
                pair_dir = output / pair_name
                manifest = json.loads((pair_dir / "arena.json").read_text())
                self.assertEqual(manifest["num_games"], 10)
                self.assertEqual(manifest["checkpoint_sweep"]["reference_iteration"], 2000)
                self.assertTrue(manifest["share_agent_engines"])
                self.assertEqual(len((pair_dir / "games.jsonl").read_text().splitlines()), 10)
                agents = {agent["name"]: agent for agent in manifest["agents"]}
                self.assertIn("weight_iter_2000.pt", agents["iter_2000"]["command"][-1])
                self.assertNotIn("actor_use_dirichlet_noise=true", agents["iter_2000"]["command"][-1])
                self.assertIn("actor_select_action_by_count=true", agents["iter_2000"]["command"][-1])
                # Six unique seatings, with one shared engine per checkpoint instead of per seat.
                self.assertEqual(len(list((pair_dir / "engine_logs").glob("*.log"))), 12)

            with (output / "sweep.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [(row["reference"], row["comparison"]) for row in rows],
                [("2000", "0"), ("2000", "1000")],
            )
            self.assertTrue(all(int(row["valid_games"]) == 10 for row in rows))

    def test_auto_mode_accepts_different_models_and_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            maxn_run = temp / "maxn_run"
            paranoid_run = temp / "paranoid_run"
            for run, iteration in ((maxn_run, 100), (paranoid_run, 200)):
                (run / "model").mkdir(parents=True)
                (run / "run.cfg").write_text("actor_num_simulation=50\n")
                (run / "model" / f"weight_iter_{iteration}.pt").touch()
            output = temp / "evaluation"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "auto",
                    "tictacmo",
                    str(maxn_run),
                    "--paranoid-model",
                    str(paranoid_run / "model" / "weight_iter_200.pt"),
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output / "arena.json").read_text())
            agents = {agent["name"]: agent for agent in manifest["agents"]}
            self.assertIn(str(maxn_run / "model" / "weight_iter_100.pt"), agents["maxn"]["command"][-1])
            self.assertIn(str(paranoid_run / "model" / "weight_iter_200.pt"), agents["paranoid"]["command"][-1])
            self.assertEqual(agents["maxn"]["command"][-3], str((maxn_run / "run.cfg").resolve()))
            self.assertEqual(agents["paranoid"]["command"][-3], str((paranoid_run / "run.cfg").resolve()))

    def test_auto_mode_uses_connect3x3_move_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "connect3x3_run"
            (training / "model").mkdir(parents=True)
            (training / "run.cfg").write_text("actor_num_simulation=50\n")
            (training / "model" / "weight_iter_1.pt").touch()
            output = temp / "evaluation"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "auto",
                    "connect3x3",
                    str(training),
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output / "arena.json").read_text())
            self.assertEqual(manifest["max_moves"], 42)

    def test_auto_mode_uses_blokus_player_and_pass_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "blokus_run"
            (training / "model").mkdir(parents=True)
            (training / "run.cfg").write_text("actor_num_simulation=50\n")
            (training / "model" / "weight_iter_1.pt").touch()
            output = temp / "evaluation"
            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "auto",
                    "blokus",
                    str(training),
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "arena.json").read_text())
            self.assertEqual(manifest["players"], ["b", "w", "r", "g"])
            self.assertEqual(manifest["max_moves"], 400)
            self.assertEqual(manifest["terminal_passes"], 4)
            self.assertTrue(all(len(lineup) == 4 for lineup in manifest["lineups"]))

    def test_auto_mode_generates_balanced_search_arena(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "tictacmo_run"
            model_dir = training / "model"
            model_dir.mkdir(parents=True)
            (training / "run.cfg").write_text("actor_num_simulation=50\n")
            (model_dir / "weight_iter_9.pt").touch()
            (model_dir / "weight_iter_100.pt").touch()
            output = temp / "evaluation"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "auto",
                    "tictacmo",
                    str(training),
                    "--executable",
                    str(FAKE_ENGINE),
                    "--output",
                    str(output),
                    "--num-simulations",
                    "200",
                    "--noise",
                    "--games-per-seating",
                    "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output / "arena.json").read_text())
            self.assertEqual([agent["name"] for agent in manifest["agents"]], ["maxn", "paranoid"])
            self.assertEqual(
                manifest["lineups"],
                [["maxn", "maxn", "paranoid"], ["maxn", "paranoid", "paranoid"]],
            )
            self.assertEqual(manifest["max_moves"], 15)
            for agent in manifest["agents"]:
                conf_str = agent["command"][-1]
                self.assertIn("weight_iter_100.pt", conf_str)
                self.assertIn("actor_num_simulation=200", conf_str)
                self.assertIn("actor_use_dirichlet_noise=true", conf_str)
                self.assertIn(f"actor_multiplayer_search_type={agent['name']}", conf_str)
            games = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
            self.assertEqual(len(games), 12)
            self.assertTrue(all(not game["error"] for game in games))

    def test_auto_mode_supports_fixed_rank_utility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "rank_run"
            (training / "model").mkdir(parents=True)
            (training / "run.cfg").write_text(
                "actor_num_simulation=50\nactor_rank_utility_weight=0.75\n"
            )
            (training / "model" / "weight_iter_100.pt").touch()
            output = temp / "evaluation"

            subprocess.run(
                [
                    sys.executable,
                    str(ARENA),
                    "auto",
                    "tictacmo",
                    str(training),
                    "--search-types", "maxn", "rank",
                    "--rank-weight", "0.75",
                    "--executable", str(FAKE_ENGINE),
                    "--output", str(output),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output / "arena.json").read_text())
            agents = {agent["name"]: agent for agent in manifest["agents"]}
            self.assertIn("actor_rank_utility_weight=0", agents["maxn"]["command"][-1])
            self.assertIn("actor_rank_utility_weight=0.75", agents["rank"]["command"][-1])

    def test_all_permutations_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            command = [sys.executable, str(FAKE_ENGINE)]
            manifest = {
                "game": "fake",
                "players": ["b", "w", "r"],
                "agents": [
                    {"name": "alpha", "command": command},
                    {"name": "beta", "command": command},
                    {"name": "gamma", "command": command},
                ],
                "lineups": [["alpha", "beta", "gamma"]],
                "seat_mode": "all_permutations",
                "games_per_seating": 1,
                "max_moves": 3,
                "command_timeout": 5,
            }
            manifest_path = temp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = temp / "output"

            subprocess.run(
                [sys.executable, str(ARENA), str(manifest_path), str(output), "--num_threads", "2"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            games = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
            self.assertEqual(len(games), 6)
            self.assertTrue(all(game["returns"] == [1.0, -1.0, -1.0] for game in games))
            self.assertEqual({tuple(game["seating"]) for game in games}, {
                tuple(seating)
                for seating in __import__("itertools").permutations(("alpha", "beta", "gamma"))
            })

            with (output / "agent_summary.csv").open() as stream:
                summaries = {row["agent"]: row for row in csv.DictReader(stream)}
            for name in ("alpha", "beta", "gamma"):
                self.assertEqual(int(summaries[name]["games"]), 6)
                self.assertEqual(int(summaries[name]["wins"]), 2)
                self.assertEqual(int(summaries[name]["losses"]), 4)
                self.assertAlmostEqual(float(summaries[name]["win_rate"]), 1 / 3)
                self.assertAlmostEqual(float(summaries[name]["avg_return"]), -1 / 3)

            with (output / "seat_summary.csv").open() as stream:
                seat_summaries = list(csv.DictReader(stream))
            self.assertEqual(len(seat_summaries), 9)
            self.assertTrue(all(int(row["games"]) == 2 for row in seat_summaries))
            with (output / "seating_summary.csv").open() as stream:
                seating_summaries = list(csv.DictReader(stream))
            self.assertEqual(len(seating_summaries), 6)
            self.assertTrue(all(int(row["B_wins"]) == 1 for row in seating_summaries))
            self.assertEqual(len(list((output / "sgf").glob("*.sgf"))), 6)
            self.assertEqual((output / "errors.csv").read_text().strip(), "game_id,seating,error")

            subprocess.run(
                [sys.executable, str(ARENA), str(manifest_path), str(output), "--resume"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(len((output / "games.jsonl").read_text().splitlines()), 6)

    def test_legacy_two_player_scalar_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            command = [sys.executable, str(FAKE_ENGINE), "--num-players", "2"]
            manifest = {
                "game": "fake2",
                "players": ["b", "w"],
                "agents": [
                    {"name": "old", "command": command},
                    {"name": "new", "command": command},
                ],
                "seat_mode": "all_permutations",
                "games_per_seating": 1,
                "max_moves": 2,
                "command_timeout": 5,
            }
            manifest_path = temp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = temp / "output"
            subprocess.run(
                [sys.executable, str(ARENA), str(manifest_path), str(output), "-g", "01", "--num_threads", "1"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            games = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
            self.assertEqual(len(games), 2)
            self.assertTrue(all(game["returns"] == [1.0, -1.0] for game in games))
            engine_logs = [path.read_text() for path in (output / "engine_logs").glob("*.log")]
            self.assertTrue(any("CUDA_VISIBLE_DEVICES=0" in log for log in engine_logs))
            self.assertTrue(any("CUDA_VISIBLE_DEVICES=1" in log for log in engine_logs))
            with (output / "agent_summary.csv").open() as stream:
                summaries = list(csv.DictReader(stream))
            self.assertTrue(all(int(row["wins"]) == 1 and int(row["losses"]) == 1 for row in summaries))


if __name__ == "__main__":
    unittest.main()
