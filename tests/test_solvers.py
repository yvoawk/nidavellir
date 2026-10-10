"""Compare exact catalogue optimization with an independent exhaustive oracle."""

import itertools
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

from nidavellir.solvers import solve


def exhaustive(candidates, positive_count):
    best = None
    for chosen in itertools.product((False, True), repeat=len(candidates)):
        coverage = cost = 0
        for included, candidate in zip(chosen, candidates):
            if included:
                coverage |= candidate["coverage"]
                cost += candidate["cost"]
        if coverage == (1 << positive_count) - 1 and (best is None or cost < best):
            best = cost
    return best


ILASP = shutil.which("ILASP") or str(Path(sys.executable).parent / "ILASP")


class SolverTests(unittest.TestCase):
    def test_random_catalogues_match_exhaustive_enumeration(self):
        randomizer = random.Random(81011)
        for trial in range(35):
            positive_count = randomizer.randrange(5)
            candidates = [{"id": f"rule:{i}", "cost": randomizer.randrange(1, 8),
                           "coverage": randomizer.randrange(1 << positive_count)}
                          for i in range(randomizer.randrange(8))]
            for strategy in ("usc,oll", "bb,lin"):
                with self.subTest(trial=trial, strategy=strategy):
                    result = solve(candidates, positive_count, internal_opt_strategy=strategy)
                    expected = exhaustive(candidates, positive_count)
                    self.assertEqual(result["status"], "unsatisfiable" if expected is None else "complete")
                    self.assertEqual(result["optimum_cost"], expected)
                    self.assertEqual(result["stats"]["optimization_strategy"], strategy)
                    json.dumps(result)  # Runner can serialize all artifacts/statistics.

    def test_equal_cost_candidates_count_separately_and_may_overlap(self):
        candidates = [{"id": "first", "cost": 2, "coverage": 0b011},
                      {"id": "second", "cost": 2, "coverage": 0b110},
                      {"id": "expensive", "cost": 5, "coverage": 0b111}]
        result = solve(candidates, 3)
        self.assertEqual(result["selected"], ["first", "second"])
        self.assertEqual(result["optimum_cost"], 4)

    def test_timeout_never_claims_optimality(self):
        result = solve([{"id": "a", "cost": 1, "coverage": 1}], 1, timeout=1e-12)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["optimum_cost"])
        self.assertEqual(result["cost"], 1)
        self.assertEqual(result["selected"], ["a"])
        self.assertEqual(result["stats"]["incumbent_source"], "greedy_feasible_cover")

    def test_timeout_distinguishes_empty_cover_from_no_feasible_cover(self):
        for count, cost in ((0, 0), (1, None)):
            with self.subTest(positive_count=count):
                result = solve([], count, timeout=1e-12)
                self.assertEqual(result["status"], "timeout")
                self.assertEqual(result["cost"], cost)
                self.assertIsNone(result["optimum_cost"])
                self.assertEqual(result["selected"], [])

    def test_input_contract_rejects_ambiguous_catalogues(self):
        candidate = {"id": "a", "cost": 1, "coverage": 1}
        for catalogue in [[candidate, candidate], [{**candidate, "cost": 0}],
                          [{**candidate, "coverage": 2}], [{**candidate, "coverage": -1}]]:
            with self.subTest(catalogue=catalogue), self.assertRaises(ValueError):
                solve(catalogue, 1)
        with self.assertRaises(ValueError):
            solve([], 0, backend="ilasp", verify_ilasp=True)
        with self.assertRaisesRegex(ValueError, "internal_opt_strategy"):
            solve([], 0, internal_opt_strategy="unsupported")

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_ilasp_agrees_on_multirule_empty_and_unsatisfiable_tasks(self):
        for candidates, count in [([], 0), ([], 1),
                ([{"id": "a arbitrary id", "cost": 2, "coverage": 3},
                  {"id": "b", "cost": 2, "coverage": 6},
                  {"id": "c", "cost": 5, "coverage": 7}], 3)]:
            with self.subTest(candidates=candidates, positive_count=count):
                result = solve(candidates, count, verify_ilasp=True, ilasp_binary=ILASP, timeout=10)
                self.assertTrue(result["verification"]["matches"], result)
                direct = solve(candidates, count, backend="ilasp", ilasp_binary=ILASP, timeout=10)
                self.assertEqual(direct["status"], result["status"])
                self.assertEqual(direct["optimum_cost"], exhaustive(candidates, count))

    def test_unavailable_verifier_does_not_erase_internal_proof(self):
        result = solve([{"id": "a", "cost": 1, "coverage": 1}], 1,
                       verify_ilasp=True, ilasp_binary="/no/such/ILASP")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["optimum_cost"], 1)
        self.assertEqual(result["verification"]["status"], "error")
        self.assertIsNone(result["verification"]["matches"])

    def test_partial_ilasp_output_is_not_a_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ILASP"
            fake.write_text(f"#!{sys.executable}\nprint('pick(0).')\n")
            fake.chmod(0o700)
            result = solve([{"id": "a", "cost": 1, "coverage": 1}], 1,
                           backend="ilasp", ilasp_binary=fake, timeout=2)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["optimum_cost"])

    def test_ilasp_timeout_preserves_logs_but_no_optimum(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ILASP"
            fake.write_text("#!/bin/sh\nprintf 'pick(0).\\n'\nsleep 10\n")
            fake.chmod(0o700)
            result = solve([{"id": "a", "cost": 1, "coverage": 1}], 1,
                           backend="ilasp", ilasp_binary=fake, timeout=1, enumerate_optima=True)
            self.assertEqual(result["status"], "timeout")
            self.assertIsNone(result["optimum_cost"])
            self.assertIn("pick(0).", result["stdout"])
            self.assertEqual(result["cost"], 1)
            self.assertEqual(result["selected"], ["a"])
            self.assertEqual(result["stats"]["incumbent_source"], "greedy_feasible_cover")
            self.assertEqual(result["enumeration"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
