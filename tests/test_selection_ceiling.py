"""Feasible ceilings prune selection without using a solver's optimum proof."""

from pathlib import Path
import shutil
import sys
import unittest

from nidavellir.solvers import _retain_timeout_cover, set_cover_program, solve


ILASP = shutil.which("ILASP") or str(Path(sys.executable).parent / "ILASP")


class SelectionCeilingTests(unittest.TestCase):
    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_singleton_reduction_keeps_original_indices_and_all_optimal_ties(self):
        candidates = [{"id": "partial", "cost": 4, "coverage": 1},
                      {"id": "expensive", "cost": 7, "coverage": 3},
                      {"id": "best-left", "cost": 6, "coverage": 3},
                      {"id": "best-right", "cost": 6, "coverage": 3}]
        for backend in ("internal", "ilasp"):
            result = solve(candidates, 2, backend=backend, verify_ilasp=backend == "internal",
                           ilasp_binary=ILASP, timeout=5, enumerate_optima=True)
            self.assertEqual(result["status"], "complete", result)
            self.assertEqual(result["optimum_cost"], 6)
            self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["best-left"], ["best-right"]])
            selected = result["verification"] if backend == "internal" else result
            proof = selected["stats"]["singleton_reduction"]
            self.assertTrue(proof["applied"])
            self.assertEqual(proof["minimum_rule_cost"], 4)
            self.assertEqual(proof["feasible_upper_bound"], 6)
            self.assertEqual(proof["retained_original_indices"], [2, 3])
            self.assertEqual(proof["retained_candidate_ids"], ["best-left", "best-right"])
            self.assertEqual(proof["eliminated_candidate_count"], 2)
            self.assertNotIn("covers(0,", selected["program"])
            self.assertNotIn("covers(1,", selected["program"])
            self.assertIn(selected["selected"], [["best-left"], ["best-right"]])

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_equality_uses_general_selection_and_keeps_multirule_optima(self):
        candidates = [{"id": "first", "cost": 2, "coverage": 1},
                      {"id": "second", "cost": 2, "coverage": 2},
                      {"id": "joint", "cost": 4, "coverage": 3}]
        result = solve(candidates, 2, verify_ilasp=True, ilasp_binary=ILASP,
                       timeout=5, enumerate_optima=True)
        self.assertEqual(result["optimum_cost"], 4)
        self.assertTrue(result["verification"]["matches"])
        proof = result["verification"]["stats"]["singleton_reduction"]
        self.assertFalse(proof["applied"])
        self.assertEqual(2 * proof["minimum_rule_cost"], proof["feasible_upper_bound"])
        self.assertEqual(proof["retained_candidate_count"], 3)
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["first", "second"], ["joint"]])

    def test_internal_uses_feasible_upper_bound_without_claiming_its_optimality(self):
        candidates = [{"id": "a", "cost": 3, "coverage": 0b001111},
                      {"id": "b", "cost": 3, "coverage": 0b010011},
                      {"id": "c", "cost": 3, "coverage": 0b101100},
                      {"id": "d", "cost": 2, "coverage": 0b010000},
                      {"id": "e", "cost": 2, "coverage": 0b100000}]
        result = solve(candidates, 6, timeout=5)
        self.assertEqual(result["stats"]["feasible_cover_cost"], 7)
        self.assertTrue(result["stats"]["upper_bound_applied"])
        self.assertEqual(result["optimum_cost"], 6)
        self.assertIn(":- #sum { Cost,C : selected(C), candidate(C,Cost) } > 7.", result["program"])

    def test_nonoptimal_greedy_timeout_cover_never_becomes_an_optimum(self):
        candidates = [{"id": "a", "cost": 3, "coverage": 0b001111},
                      {"id": "b", "cost": 3, "coverage": 0b010011},
                      {"id": "c", "cost": 3, "coverage": 0b101100},
                      {"id": "d", "cost": 2, "coverage": 0b010000},
                      {"id": "e", "cost": 2, "coverage": 0b100000}]
        result = solve(candidates, 6, timeout=1e-12, enumerate_optima=True)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["cost"], 7)
        self.assertEqual(result["selected"], ["a", "d", "e"])
        self.assertIsNone(result["optimum_cost"])
        self.assertFalse(result["enumeration"]["complete"])
        # A better solver incumbent must survive the same timeout path.
        result.update(cost=6, selected=["b", "c"])
        result["stats"]["incumbent_source"] = "solver_model"
        _retain_timeout_cover(result, candidates, 6)
        self.assertEqual(result["cost"], 6)
        self.assertEqual(result["selected"], ["b", "c"])
        self.assertEqual(result["stats"]["incumbent_source"], "solver_model")

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_empty_and_impossible_catalogues_do_not_use_singleton_reduction(self):
        for candidates, count, optimum in [
            ([{"id": "irrelevant", "cost": 1, "coverage": 0}], 0, 0),
            ([{"id": "partial", "cost": 4, "coverage": 1}], 2, None),
        ]:
            result = solve(candidates, count, verify_ilasp=True, ilasp_binary=ILASP, timeout=5)
            self.assertEqual(result["optimum_cost"], optimum)
            self.assertTrue(result["verification"]["matches"], result)
            self.assertFalse(result["verification"]["stats"]["singleton_reduction"]["applied"])
            self.assertEqual(result["stats"]["upper_bound_applied"], optimum is not None)
        self.assertNotIn(":- #sum", set_cover_program([], 1))


if __name__ == "__main__":
    unittest.main()
