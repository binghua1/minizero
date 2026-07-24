import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "paper-multiplayer-summary.py"


class PaperMultiplayerSummaryTest(unittest.TestCase):
    def test_repeated_agent_is_credited_once_per_physical_game(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pair = root / "1_vs_0"
            pair.mkdir()
            (pair / "arena.json").write_text(json.dumps({
                "num_games": 1,
                "self_eval": {"newer_iteration": 1, "older_iteration": 0},
            }))
            game = {
                "error": None,
                "players": ["b", "w", "r", "g"],
                "seating": ["iter_0", "iter_1", "iter_1", "iter_1"],
                "returns": [-0.1, 0.2, 0.0, -0.1],
            }
            (pair / "games.jsonl").write_text(json.dumps(game) + "\n")

            subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--bootstrap-samples", "100"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with (root / "paper_summary_v1" / "pair_summary.csv").open() as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(int(row["valid_games"]), 1)
            self.assertEqual(int(row["target_seat_exposures"]), 3)
            self.assertEqual(float(row["target_first_place_credit"]), 1.0)
            self.assertEqual(int(row["target_model_wins"]), 1)
            self.assertEqual(int(row["target_model_losses"]), 0)


if __name__ == "__main__":
    unittest.main()
