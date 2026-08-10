import json
import math
import tempfile
import unittest
from pathlib import Path

from minizero.jpsro.meta_solver import cce_gap, solve_cce
from minizero.jpsro.state import JPSROState, PayoffTable
from minizero.jpsro.workflow import (
    candidate_deviation_gains, create_evaluation_manifest, create_oracle_profiles, ingest_evaluation,
    write_oracle_plan,
)


class JPSROTest(unittest.TestCase):
    def test_evaluation_manifest_reuses_duplicate_policy_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(Path(directory) / "state", 3, shared_pool=True)
            model = Path(directory) / "p0.pt"
            model.touch()
            state.add_policy("p0", model)
            manifest = create_evaluation_manifest(
                state, [("p0", "p0", "p0")], "tictacmo",
                Path(directory) / "game.cfg", Path(directory) / "game",
                directory,
            )
            self.assertTrue(manifest["share_agent_engines"])

    def test_payoff_online_statistics_round_trip(self):
        table = PayoffTable(3)
        table.add(("a", "b", "c"), (1, 2, 3))
        table.add(("a", "b", "c"), (3, 2, 1))
        self.assertEqual(table.get(("a", "b", "c"))["mean"], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(table.stderr(("a", "b", "c"))[0], 1.0)

    def test_matching_pennies_cce(self):
        policies = [["h", "t"], ["h", "t"]]
        payoffs = {}
        for a in range(2):
            for b in range(2):
                first = 1.0 if a == b else -1.0
                payoffs[(a, b)] = (first, -first)
        result = solve_cce(policies, payoffs, iterations=20000, tolerance=0.02)
        self.assertLessEqual(result["gap"], 0.02)
        marginal = [0.0, 0.0]
        for profile, probability in result["distribution"].items():
            marginal[profile[0]] += probability
        self.assertAlmostEqual(marginal[0], 0.5, delta=0.08)

    def test_shared_pool_oracle_rotates_one_responder(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 3, shared_pool=True)
            model = Path(directory) / "p0.pt"
            model.touch()
            state.add_policy("p0", model)
            state._write_json("meta_strategy.json", {
                "distribution": [{"profile": ["p0", "p0", "p0"], "probability": 1.0}]
            })
            rows = create_oracle_profiles(state)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(sum(mask) == 1 for _, _, mask in rows))
            plan = Path(directory) / "oracle.tsv"
            write_oracle_plan(state, plan)
            self.assertEqual(plan.read_text().count("CURRENT"), 6)  # ID and path per row.

    def test_fixed_manifest_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(Path(directory) / "state", 2, shared_pool=True)
            model = Path(directory) / "p.pt"
            model.touch()
            state.add_policy("p", model)
            manifest = Path(directory) / "arena.json"
            manifest.write_text(json.dumps({"jpsro_profiles": [["p", "p"]]}))
            games = Path(directory) / "games.jsonl"
            games.write_text(json.dumps({
                "game_id": 0, "lineup_id": 0, "seating": ["p", "p"],
                "returns": [1, -1], "error": None
            }) + "\n")
            accepted, errors = ingest_evaluation(state, manifest, games)
            self.assertEqual((accepted, errors), (1, 0))
            accepted, errors = ingest_evaluation(state, manifest, games)
            self.assertEqual((accepted, errors), (0, 0))
            self.assertEqual(state.payoffs.get(("p", "p"))["mean"], [1.0, -1.0])

    def test_candidate_deviation_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 2, shared_pool=True)
            for policy_id in ("p0", "p1"):
                model = Path(directory) / f"{policy_id}.pt"
                model.touch()
                state.add_policy(policy_id, model)
            state._write_json("meta_strategy.json", {
                "distribution": [{"profile": ["p0", "p0"], "probability": 1.0}]
            })
            for _ in range(2):
                state.payoffs.add(("p0", "p0"), (0.0, 0.0))
                state.payoffs.add(("p1", "p0"), (0.25, -0.25))
                state.payoffs.add(("p0", "p1"), (-0.5, 0.5))
            gains = candidate_deviation_gains(state, "p1")
            self.assertEqual([item["gain"] for item in gains], [0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
