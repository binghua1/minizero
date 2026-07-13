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
                [sys.executable, str(ARENA), str(manifest_path), str(output), "--threads", "2"],
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


if __name__ == "__main__":
    unittest.main()
