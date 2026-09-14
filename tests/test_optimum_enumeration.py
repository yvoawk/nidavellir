"""All-optima selection and variant products checked against exhaustive sets."""

from itertools import combinations
from pathlib import Path
import random
import shutil
import sys
import unittest

from nidavellir.solvers import (
    enumerate_optimal_covers, expand_optimal_programs, optimal_cover_program, solve,
)


ILASP = shutil.which("ILASP") or str(Path(sys.executable).parent / "ILASP")


def exhaustive_optima(candidates, positive_count):
    best = None
    hypotheses = set()
    for size in range(len(candidates) + 1):
        for chosen in combinations(candidates, size):
            mask, cost = 0, 0
            for candidate in chosen:
                mask |= candidate["coverage"]
                cost += candidate["cost"]
            if mask != (1 << positive_count) - 1:
                continue
            if best is None or cost < best:
                best, hypotheses = cost, set()
            if cost == best:
                hypotheses.add(tuple(sorted(candidate["id"] for candidate in chosen)))
    return best, hypotheses


class OptimumEnumerationTests(unittest.TestCase):
    def test_random_catalogues_match_every_exhaustive_optimum(self):
        rng = random.Random(101109)
        for trial in range(30):
            count = rng.randrange(5)
            candidates = [{"id": f"c{i}", "cost": rng.randrange(1, 6),
                           "coverage": rng.randrange(1 << count)}
                          for i in range(rng.randrange(8))]
            optimum, expected = exhaustive_optima(candidates, count)
            for backend in ("auto", "clingo"):
                with self.subTest(trial=trial, backend=backend):
                    result = solve(candidates, count, enumerate_optima=True, enumeration_backend=backend)
                    self.assertEqual(result["optimum_cost"], optimum)
                    self.assertTrue(result["enumeration"]["complete"])
                    actual = {tuple(h) for h in result["enumeration"]["semantic_hypotheses"]}
                    self.assertEqual(actual, expected)

    def test_singleton_fast_path_returns_all_tied_covers(self):
        candidates = [{"id": "left", "cost": 3, "coverage": 3},
                      {"id": "right", "cost": 3, "coverage": 3},
                      {"id": "partial", "cost": 2, "coverage": 1}]
        result = solve(candidates, 2, enumerate_optima=True)
        self.assertEqual(result["enumeration"]["backend"], "singleton_proof")
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["left"], ["right"]])

    def test_singleton_inequality_is_strict_and_multirule_ties_remain(self):
        candidates = [{"id": "first", "cost": 1, "coverage": 1},
                      {"id": "second", "cost": 1, "coverage": 2},
                      {"id": "joint", "cost": 2, "coverage": 3}]
        result = solve(candidates, 2, enumerate_optima=True)
        self.assertEqual(result["enumeration"]["backend"], "clingo_guarded_sum")
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["first", "second"], ["joint"]])
        program = optimal_cover_program(candidates, 2, 2)
        self.assertNotIn("#minimize", program)
        self.assertNotIn("total_cost", program)
        self.assertIn("} < 2.", program)
        self.assertIn("} > 2.", program)

    def test_overlapping_multirule_optima_and_single_rule_tie(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 0b011},
                      {"id": "b", "cost": 2, "coverage": 0b110},
                      {"id": "c", "cost": 2, "coverage": 0b101},
                      {"id": "d", "cost": 4, "coverage": 0b111}]
        result = solve(candidates, 3, enumerate_optima=True)
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["a", "b"], ["a", "c"], ["b", "c"], ["d"]])

    def test_enumeration_limit_does_not_erase_selection_optimum(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 1}, {"id": "b", "cost": 2, "coverage": 1}]
        for backend in ("auto", "clingo"):
            result = solve(candidates, 1, enumerate_optima=True, enumeration_backend=backend, max_optima=1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["optimum_cost"], 2)
            self.assertEqual(result["enumeration"]["status"], "limit")
            self.assertFalse(result["enumeration"]["complete"])
            self.assertEqual(result["enumeration"]["hypothesis_count"], 1)
            complete = solve(candidates, 1, enumerate_optima=True, enumeration_backend=backend, max_optima=2)
            self.assertTrue(complete["enumeration"]["complete"])

    def test_enumeration_timeout_is_separate_from_cost_proof(self):
        result = solve([{"id": "a", "cost": 2, "coverage": 1}], 1,
                       enumerate_optima=True, enumeration_timeout=1e-12)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["optimum_cost"], 2)
        self.assertEqual(result["enumeration"]["status"], "timeout")
        self.assertFalse(result["enumeration"]["complete"])

    def test_no_positive_has_one_empty_optimum_and_unsat_has_none(self):
        result = solve([], 0, enumerate_optima=True, expand_variants=True)
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [[]])
        self.assertEqual(result["variant_expansion"]["programs"], [{"selected": [], "rules": [], "cost": 0}])
        self.assertTrue(result["variant_expansion"]["all_optimal_programs_complete"])
        unsat = solve([], 1, enumerate_optima=True, expand_variants=True)
        self.assertEqual(unsat["status"], "unsatisfiable")
        self.assertEqual(unsat["enumeration"]["semantic_hypotheses"], [])
        self.assertEqual(unsat["variant_expansion"]["programs"], [])

    def test_variant_product_expands_all_six_programs(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 0b011,
                       "variants": ["a :- p.", "a :- q."]},
                      {"id": "b", "cost": 2, "coverage": 0b110,
                       "variants": ["b :- r.", "b :- s.", "b :- t."]}]
        result = solve(candidates, 3, enumerate_optima=True, expand_variants=True)
        expansion = result["variant_expansion"]
        self.assertTrue(expansion["all_optimal_programs_complete"])
        self.assertEqual(expansion["program_count"], 6)
        self.assertTrue(all(p["cost"] == 4 and len(p["rules"]) == 2 for p in expansion["programs"]))
        limited = solve(candidates, 3, enumerate_optima=True, expand_variants=True, max_programs=5)
        self.assertEqual(limited["status"], "complete")
        self.assertEqual(limited["variant_expansion"]["status"], "limit")
        self.assertFalse(limited["variant_expansion"]["all_optimal_programs_complete"])
        exact_limit = solve(candidates, 3, enumerate_optima=True, expand_variants=True, max_programs=6)
        self.assertTrue(exact_limit["variant_expansion"]["complete"])

    def test_variant_product_quotients_alpha_duplicates(self):
        candidates = [{"id": "a", "cost": 4, "coverage": 1,
                       "variants": ["a(V11) :- p(V10,V7,V11).", "a(V6) :- p(V5,V4,V6)."],
                       "alpha_variants": [["a(V11) :- p(V10,V7,V11).", "a(V6) :- p(V5,V4,V6)."],
                                          ["a(V11) :- q(V10,V7,V11)."]]}]
        result = solve(candidates, 1, enumerate_optima=True, expand_variants=True)
        expansion = result["variant_expansion"]
        self.assertTrue(expansion["all_optimal_programs_complete"])
        self.assertEqual(expansion["program_count"], 2)
        self.assertEqual({p["rules"][0] for p in expansion["programs"]},
                         {"a(V11) :- p(V10,V7,V11).", "a(V11) :- q(V10,V7,V11)."})

    def test_expansion_of_partial_semantic_list_does_not_claim_all_optima(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 1, "rule": "a :- p."},
                      {"id": "b", "cost": 2, "coverage": 1, "rule": "b :- q."}]
        result = solve(candidates, 1, enumerate_optima=True, expand_variants=True, max_optima=1)
        self.assertTrue(result["variant_expansion"]["complete"])
        self.assertFalse(result["variant_expansion"]["semantic_enumeration_complete"])
        self.assertFalse(result["variant_expansion"]["all_optimal_programs_complete"])

    def test_missing_or_contradictory_variants_are_explicit_errors(self):
        missing = solve([{"id": "a", "cost": 2, "coverage": 1}], 1,
                        enumerate_optima=True, expand_variants=True)
        self.assertEqual(missing["status"], "complete")
        self.assertEqual(missing["variant_expansion"]["status"], "error")
        with self.assertRaises(ValueError):
            expand_optimal_programs([{"id": "a", "cost": 1, "rule": "same."},
                                     {"id": "b", "cost": 1, "rule": "same."}], [["a", "b"]])

    def test_invalid_limits_and_inconsistent_cost_are_not_accepted(self):
        for kwargs in ({"max_optima": 0}, {"max_programs": True}, {"enumeration_timeout": -1},
                       {"enumeration_backend": "unknown"}, {"expand_variants": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve([], 0, **kwargs)
        invalid = enumerate_optimal_covers([{"id": "a", "cost": 2, "coverage": 1}], 1, 1)
        self.assertEqual(invalid["status"], "error")
        self.assertFalse(invalid["complete"])

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP unavailable")
    def test_real_ilasp_verification_is_preserved_with_enumeration(self):
        candidates = [{"id": "a", "cost": 1, "coverage": 1},
                      {"id": "b", "cost": 1, "coverage": 2},
                      {"id": "c", "cost": 2, "coverage": 3}]
        result = solve(candidates, 2, timeout=10, verify_ilasp=True, enumerate_optima=True)
        self.assertTrue(result["verification"]["matches"], result)
        self.assertEqual(result["enumeration"]["semantic_hypotheses"], [["a", "b"], ["c"]])
        direct = solve(candidates, 2, backend="ilasp", timeout=10, enumerate_optima=True)
        self.assertEqual(direct["optimum_cost"], result["optimum_cost"])
        self.assertEqual(direct["enumeration"]["semantic_hypotheses"], result["enumeration"]["semantic_hypotheses"])


if __name__ == "__main__":
    unittest.main()
