import json
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

from minizero.jpsro.meta_solver import cce_gap, solve_cce
from minizero.jpsro.state import JPSROState, PayoffTable
from minizero.jpsro.workflow import (
    admit_candidate, candidate_deviation_gains, create_evaluation_manifest,
    create_guided_profiles, create_oracle_profiles, hard_prune_population,
    ingest_evaluation, prune_population,
    write_guided_plan, write_oracle_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCHED_EVAL = REPO_ROOT / "tools" / "jpsro-batched-eval.py"


def load_batched_eval_module():
    spec = importlib.util.spec_from_file_location("jpsro_batched_eval_under_test", BATCHED_EVAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JPSROTest(unittest.TestCase):
    def test_batched_eval_parses_multiplayer_selfplay_result(self):
        evaluator = load_batched_eval_module()
        parsed = evaluator.parse_selfplay(
            "SelfPlay true 0 7 1,-1,-1 (;GM[tictacmo]RE[1,-1,-1]) #\n", 3
        )
        self.assertEqual(parsed, ([1.0, -1.0, -1.0],
                                  "(;GM[tictacmo]RE[1,-1,-1])"))

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
            self.assertEqual(manifest["agents"][0]["model_path"], str(model.resolve()))
            self.assertEqual(manifest["batched_evaluation"]["config_file"],
                             str((Path(directory) / "game.cfg").resolve()))

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

    def test_cce_output_compacts_to_certified_pure_support(self):
        policies = [["good", "bad"], ["good", "bad"]]
        payoffs = {
            (0, 0): (1.0, 1.0), (0, 1): (1.0, -1.0),
            (1, 0): (-1.0, 1.0), (1, 1): (-1.0, -1.0),
        }
        result = solve_cce(policies, payoffs, iterations=5000, tolerance=0.01)
        self.assertEqual(result["support_size"], 1)
        self.assertLessEqual(result["gap"], 0.01)

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

    def test_guided_plan_supports_one_and_two_current_seats(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 3, shared_pool=True)
            model = Path(directory) / "p0.pt"
            model.touch()
            state.add_policy("p0", model)
            state._write_json("meta_strategy.json", {
                "distribution": [{"profile": ["p0", "p0", "p0"], "probability": 1.0}]
            })
            rows = create_guided_profiles(
                state, hard_ratio=0.6, cce_ratio=0.1, history_ratio=0.15,
                current_seat_min=1, current_seat_max=2,
            )
            self.assertEqual({sum(row["mask"]) for row in rows}, {1, 2})
            self.assertAlmostEqual(sum(row["weight"] for row in rows), 0.85)
            plan = Path(directory) / "guided.tsv"
            write_guided_plan(state, plan, current_seat_min=1, current_seat_max=2)
            self.assertTrue(plan.is_file())
            self.assertTrue(plan.with_suffix(".tsv.json").is_file())

    def test_guided_hard_sampling_prefers_low_candidate_return(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 2, shared_pool=True)
            for policy_id in ("easy", "hard", "candidate"):
                model = Path(directory) / f"{policy_id}.pt"
                model.touch()
                state.add_policy(policy_id, model)
            state._write_json("meta_strategy.json", {"distribution": [
                {"profile": ["easy", "easy"], "probability": 0.5},
                {"profile": ["hard", "hard"], "probability": 0.5},
            ]})
            for _ in range(4):
                state.payoffs.add(("candidate", "easy"), (1.0, -1.0))
                state.payoffs.add(("candidate", "hard"), (-1.0, 1.0))
                state.payoffs.add(("easy", "candidate"), (-1.0, 1.0))
                state.payoffs.add(("hard", "candidate"), (1.0, -1.0))
            rows = create_guided_profiles(
                state, candidate_id="candidate", hard_ratio=1.0,
                cce_ratio=0.0, history_ratio=0.0, temperature=0.2,
                current_seat_min=1, current_seat_max=1,
            )
            weights = {tuple(row["profile"]): row["weight"] for row in rows}
            self.assertGreater(weights[("CURRENT", "hard")],
                               weights[("CURRENT", "easy")])

            # A rejected candidate may still score hardness from its retained
            # payoff observations, but must never appear as a frozen opponent.
            state.remove_policy("candidate", keep_payoffs=True)
            rows = create_guided_profiles(
                state, candidate_id="candidate", hard_ratio=1.0,
                cce_ratio=0.0, history_ratio=0.0, temperature=0.2,
                current_seat_min=1, current_seat_max=1,
            )
            self.assertTrue(rows)
            self.assertTrue(all("candidate" not in row["profile"] for row in rows))

    def test_guided_hard_sampling_centers_asymmetric_seat_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 2, shared_pool=True)
            for policy_id in ("easy", "hard", "candidate"):
                model = Path(directory) / f"{policy_id}.pt"
                model.touch()
                state.add_policy(policy_id, model)
            state._write_json("meta_strategy.json", {"distribution": [
                {"profile": ["easy", "easy"], "probability": 0.5},
                {"profile": ["hard", "hard"], "probability": 0.5},
            ]})
            # Both seats lose the same 0.4 return against hard rather than
            # easy.  Seat 1 has a lower absolute baseline, which must not make
            # every seat-1 response look harder than every seat-0 response.
            for _ in range(4):
                state.payoffs.add(("candidate", "easy"), (0.8, 0.0))
                state.payoffs.add(("candidate", "hard"), (0.4, 0.0))
                state.payoffs.add(("easy", "candidate"), (0.0, -0.2))
                state.payoffs.add(("hard", "candidate"), (0.0, -0.6))
            rows = create_guided_profiles(
                state, candidate_id="candidate", hard_ratio=1.0,
                cce_ratio=0.0, history_ratio=0.0, temperature=0.2,
                confidence=0.0, current_seat_min=1, current_seat_max=1,
            )
            mass_by_current_seat = [0.0, 0.0]
            for row in rows:
                current_seat = row["mask"].index(True)
                mass_by_current_seat[current_seat] += row["weight"]
            self.assertAlmostEqual(mass_by_current_seat[0], 0.5)
            self.assertAlmostEqual(mass_by_current_seat[1], 0.5)

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

            decision = admit_candidate(state, "p1", min_gain=0.3, confidence=0.0)
            self.assertTrue(decision["accepted"])
            self.assertEqual(decision["eligible_players"], [1])
            self.assertEqual(state.policy_sets(), [["p0"], ["p0", "p1"]])

    def test_generalist_admission_rejects_regression_and_promotes_all_seats(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 2, shared_pool=True)
            for policy_id in ("p0", "regresses", "generalist"):
                model = Path(directory) / f"{policy_id}.pt"
                model.touch()
                state.add_policy(policy_id, model)
            state._write_json("meta_strategy.json", {
                "distribution": [{"profile": ["p0", "p0"], "probability": 1.0}]
            })
            for _ in range(2):
                state.payoffs.add(("p0", "p0"), (0.0, 0.0))
                state.payoffs.add(("regresses", "p0"), (0.4, -0.4))
                state.payoffs.add(("p0", "regresses"), (0.1, -0.1))
                state.payoffs.add(("generalist", "p0"), (0.4, -0.4))
                state.payoffs.add(("p0", "generalist"), (0.0, 0.0))

            rejected = admit_candidate(
                state, "regresses", min_gain=0.02, confidence=0.0,
                max_regression=0.05, promote_all_players=True,
            )
            self.assertFalse(rejected["accepted"])
            self.assertFalse(rejected["regression_check_passed"])
            self.assertNotIn("regresses", state.policies)

            accepted = admit_candidate(
                state, "generalist", min_gain=0.02, confidence=0.0,
                max_regression=0.05, promote_all_players=True,
            )
            self.assertTrue(accepted["accepted"])
            self.assertEqual(accepted["admitted_players"], [0, 1])
            self.assertEqual(state.policies["generalist"].players, (0, 1))

    def test_population_prune_preserves_cce_support(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 3, shared_pool=True)
            for generation in range(4):
                model = Path(directory) / f"p{generation}.pt"
                model.touch()
                state.add_policy(f"p{generation}", model, generation=generation)
            state._write_json("meta_strategy.json", {
                "distribution": [{"profile": ["p0", "p1", "p0"], "probability": 1.0}]
            })
            removed = prune_population(state, 3)
            self.assertEqual(set(state.policies), {"p0", "p1", "p3"})
            self.assertEqual(removed, ["p2"])

    def test_hard_population_prune_uses_marginal_mass_and_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JPSROState.create(directory, 2, shared_pool=True)
            for generation in range(4):
                model = Path(directory) / f"p{generation}.pt"
                model.touch()
                state.add_policy(f"p{generation}", model, generation=generation)
            state.payoffs.add(("p0", "p1"), (0.1, -0.1), "archived")
            state._write_json("meta_strategy.json", {
                "distribution": [
                    {"profile": ["p0", "p0"], "probability": 0.7},
                    {"profile": ["p2", "p2"], "probability": 0.3},
                ]
            })
            result = hard_prune_population(state, 2, protected=["p3"])
            self.assertEqual(set(state.policies), {"p0", "p3"})
            self.assertEqual(
                {row["policy"] for row in result["removed"]}, {"p1", "p2"}
            )
            self.assertIsNotNone(state.payoffs.get(("p0", "p1")))


if __name__ == "__main__":
    unittest.main()
