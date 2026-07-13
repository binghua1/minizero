#!/usr/bin/env python3

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA = REPO_ROOT / "tools" / "multiplayer-eval.py"
FAKE_ENGINE = REPO_ROOT / "tests" / "fake_multiplayer_engine.py"


class MultiplayerEvalTest(unittest.TestCase):
    def test_self_eval_pairs_checkpoints_and_uses_total_game_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training = temp / "connect3x3_run"
            model_dir = training / "model"
            model_dir.mkdir(parents=True)
            config = training / "run.cfg"
            config.write_text("actor_num_simulation=50\n")
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

            pair_names = ("500_vs_0", "1000_vs_500")
            for pair_name in pair_names:
                pair_dir = output / pair_name
                manifest = json.loads((pair_dir / "arena.json").read_text())
                self.assertEqual(manifest["num_games"], 10)
                self.assertEqual(len((pair_dir / "games.jsonl").read_text().splitlines()), 10)
                self.assertEqual(len(manifest["agents"]), 2)
                self.assertEqual(len(manifest["lineups"]), 2)

            with (output / "elo.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(row["P1"], row["P2"]) for row in rows], [("500", "0"), ("1000", "500")])
            self.assertTrue(all(int(row["Total"]) == 10 for row in rows))
            self.assertTrue(all(float(row["WinRate"]) == 0.5 for row in rows))

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
