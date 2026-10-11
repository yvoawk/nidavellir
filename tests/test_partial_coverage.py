"""Lexicographic coverage, independent rule semantics, and public CLI proofs."""
import copy
from itertools import combinations, product
import json
from pathlib import Path
import random
import tempfile
import unittest

from nidavellir.solvers import solve, set_cover_program, selection_task
from nidavellir.verification import verify_rules
from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.search import compile_catalogue
from nidavellir.relational_compile import compile_catalogue as compile_relational
from nidavellir.relational_coverage import CoverageOracle
from tests import test_optimized_oracle as oracle_tests
from tests import test_cli_extended as cli_tests
from tests import test_target_execution as target_tests
from tests.test_solvers import ILASP


def exhaustive(candidates, count):
    best, winners = None, set()
    for included in product((False, True), repeat=len(candidates)):
        coverage = cost = 0
        selected = []
        for take, candidate in zip(included, candidates):
            if take:
                coverage |= candidate["coverage"]
                cost += candidate["cost"]
                selected.append(candidate["id"])
        score = (count - coverage.bit_count(), cost)
        if best is None or score < best:
            best, winners = score, set()
        if score == best:
            winners.add(tuple(sorted(selected)))
    return best, winners


class PartialCoverageTests(unittest.TestCase):
    task = oracle_tests.OptimizedOracleContractTests.task
    invoke = cli_tests.ExtendedCLITests.invoke
    inputs = target_tests.TargetExecutionTests.inputs

    def sample(self, multi=False):
        return self.task("p(a). p(n). q(b). r(b).", ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
            ["answer(a)", "answer(b)", "answer(c)"] + (["other(a)", "other(c)"] if multi else []),
            ["answer(n)"] + (["other(n)"] if multi else []),
            heads=("answer(var__(1))", "other(var__(1))") if multi else ("answer(var__(1))",))

    def test_random_lexicographic_optima_and_all_ties_against_exhaustive_subsets(self):
        rng = random.Random(47119)
        for trial in range(30):
            count = rng.randrange(6)
            attainable = rng.randrange(count + 1)
            candidates = [{"id": str(i), "cost": rng.randrange(1, 20),
                           "coverage": rng.randrange(1 << attainable)} for i in range(rng.randrange(8))]
            before = copy.deepcopy(candidates)
            expected, winners = exhaustive(candidates, count)
            for strategy in ("usc,oll", "bb,lin"):
                with self.subTest(trial=trial, strategy=strategy):
                    result = solve(candidates, count, allow_partial_coverage=True,
                                   internal_opt_strategy=strategy, enumerate_optima=True)
                    self.assertEqual(result["status"], "complete", result)
                    self.assertEqual(tuple(result["optimum_objective"]), expected)
                    self.assertEqual(result["coverage"]["missed_positives"], expected[0])
                    self.assertEqual({tuple(h) for h in result["enumeration"]["semantic_hypotheses"]}, winners)
                    self.assertTrue(result["enumeration"]["complete"])
                    self.assertEqual(candidates, before)

    def test_coverage_has_priority_over_any_cost_saving(self):
        candidates = [{"id": "cheap", "cost": 1, "coverage": 1},
                      {"id": "expensive", "cost": 10000, "coverage": 4}]
        self.assertEqual(solve(candidates, 4)["status"], "unsatisfiable")
        result = solve(candidates, 4, allow_partial_coverage=True)
        self.assertEqual(result["optimum_objective"], [2, 10001])
        self.assertEqual(result["coverage"]["uncoverable_positive_indices"], [1, 3])
        self.assertEqual(result["coverage"]["positive_recall"], 0.5)
        self.assertEqual(result["stats"]["positive_count"], 4)

    def test_timeout_keeps_maximal_cover_but_never_claims_minimum_cost(self):
        result = solve([{"id": "a", "cost": 3, "coverage": 4}], 4,
                       allow_partial_coverage=True, timeout=1e-12)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["optimum_objective"])
        self.assertIsNone(result["optimum_cost"])
        self.assertTrue(result["coverage"]["maximum_coverage_proven"])
        self.assertFalse(result["coverage"]["cost_optimal"])
        self.assertEqual(result["coverage"]["covered_positives"], 1)

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_both_optimizers_agree_with_sparse_empty_and_zero_coverage(self):
        for candidates, count in [([], 0), ([], 3),
                ([{"id": "unused", "cost": 2, "coverage": 0}], 3),
                ([{"id": "a", "cost": 2, "coverage": 5},
                  {"id": "b", "cost": 2, "coverage": 20},
                  {"id": "c", "cost": 5, "coverage": 21}], 6)]:
            with self.subTest(count=count, candidates=candidates):
                expected, _ = exhaustive(candidates, count)
                result = solve(candidates, count, allow_partial_coverage=True,
                               verify_ilasp=True, ilasp_binary=ILASP, timeout=10)
                self.assertTrue(result["verification"]["matches"], result)
                direct = solve(candidates, count, backend="ilasp", allow_partial_coverage=True,
                               ilasp_binary=ILASP, timeout=10)
                self.assertEqual(tuple(direct["optimum_objective"]), expected, direct)
                self.assertEqual(result["program"], set_cover_program(candidates, count, allow_partial_coverage=True))
                self.assertEqual(direct["program"], selection_task(candidates, count, allow_partial_coverage=True))

    def test_compiler_backends_match_independent_legal_rule_predictions(self):
        task = self.task("edge(a,u). edge(a,v). edge(b,w). edge(n,z). left(u). right(v). left(w). right(w). left(z).",
            ["edge(var__(1),var__(2))", "left(var__(2))", "right(var__(2))"],
            ["answer(a)", "answer(b)", "answer(c)"], ["answer(n)"], max_body=3,
            bias=":- not body(edge(var__(1),var__(2))). :- not body(left(var__(2))). :- not body(right(var__(2))).")
        positives = sorted(task.positives)
        raw = []
        for head in task.heads:
            for size in range(task.max_body + 1):
                for body in combinations(range(len(task.literals)), size):
                    if not task.accepts(head, body):
                        continue
                    rules = [task.rule(head, body)]
                    actual = oracle_tests.model_of(task.background + "\n" + rules[0])
                    if actual & task.negatives:
                        continue
                    raw.append({"id": str(len(raw)), "cost": 1 + size,
                                "coverage": sum(1 << i for i, atom in enumerate(positives) if atom in actual)})
        expected, _ = exhaustive(raw, len(positives))
        self.assertEqual(expected, (2, 4))  # Shared witnesses, not separate atomic masks.
        indices, _ = build_optimized_indices(task)
        for candidates, stats in (compile_catalogue(task, indices),
                                  compile_relational(task, CoverageOracle(task))):
            self.assertTrue(stats["complete"])
            result = solve(candidates, len(positives), allow_partial_coverage=True)
            self.assertEqual(tuple(result["optimum_objective"]), expected)
            rules = [c["rule"] for c in candidates if c["id"] in result["selected"]]
            self.assertTrue(verify_rules(task, rules, allow_partial_coverage=True)["matched"])
            self.assertFalse(verify_rules(task, rules)["matched"])

    def test_cli_opt_in_default_strict_negatives_and_variant_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            _, _, strict = self.invoke(folder, "strict", inputs=inputs)
            self.assertEqual(strict["status"], "unsatisfiable")
            for backend in ("bitmask", "relational"):
                process, output, status = self.invoke(folder, backend, "--allow-partial-coverage",
                    "--compiler-backend", backend, "--enumerate-optima", "--expand-variants", inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["optimum_objective"], [2, 2])
                self.assertEqual(status["coverage"]["positive_recall"], 1 / 3)
                self.assertEqual(status["coverage"]["uncoverable_positives"], ["answer(a)", "answer(c)"])
                self.assertEqual(status["bound_strategy"], "off")
                verification = json.loads((output / "verification.json").read_text())
                self.assertTrue(verification["matched"])
                self.assertFalse(verification["hard_labels_matched"])
                self.assertEqual(verification["covered_negatives"], [])
                variants = json.loads((output / "variants.json").read_text())
                self.assertTrue(variants["all_optimal_programs_complete"])
                self.assertEqual(variants["program_count"], 2)

    def test_separate_targets_quantify_all_missing_labels_and_match_grouped(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample(multi=True))
            for mode in ("separate", "grouped"):
                process, output, status = self.invoke(folder, mode, "--target-mode", mode,
                    "--allow-partial-coverage", "--enumerate-optima", "--expand-variants", inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status["optimum_objective"], [4, 2])
                self.assertEqual(status["coverage"]["maximum_covered_positives"], 1)
                verification = json.loads((output / "verification.json").read_text())
                self.assertEqual(len(verification["missing_positives"]), 4)
                self.assertFalse(verification["hard_labels_matched"])

    def test_cli_compilation_timeout_has_no_coverage_or_cost_optimality_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            process, output, status = self.invoke(folder, "timeout", "--allow-partial-coverage",
                "--compiler-backend", "relational", "--compile-seconds", "1e-12", inputs=inputs)
            self.assertEqual(process.returncode, 3, process.stdout + process.stderr)
            self.assertEqual(status["status"], "incomplete")
            self.assertIsNone(status["optimal_cost"])
            self.assertNotIn("coverage", status)
            self.assertFalse((output / "learned.lp").exists())

    def test_cli_selection_timeout_keeps_coverage_proof_separate_from_cost_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            process, output, status = self.invoke(folder, "timeout", "--allow-partial-coverage",
                "--solver-seconds", "1e-12", inputs=inputs)
            self.assertEqual(process.returncode, 3, process.stdout + process.stderr)
            self.assertEqual(status["status"], "feasible")
            self.assertTrue(status["coverage"]["maximum_coverage_proven"])
            self.assertFalse(status["coverage"]["cost_optimal"])
            self.assertEqual(status["coverage"]["covered_positives"], 1)
            self.assertIsNone(status["optimal_cost"])
            self.assertIsNone(status["optimum_objective"])

    def test_cli_empty_and_fully_coverable_tasks_preserve_strict_optimum(self):
        for positives, expected_cost in [([], 0), (["answer(a)", "answer(b)"], 4)]:
            with self.subTest(positives=positives), tempfile.TemporaryDirectory() as folder:
                task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"],
                                 positives, ["answer(n)"])
                inputs = self.inputs(folder, task)
                for name, extra in [("strict", []), ("partial", ["--allow-partial-coverage"])]:
                    process, _, status = self.invoke(folder, name, *extra, inputs=inputs)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    self.assertEqual(status["optimal_cost"], expected_cost)
                    if extra:
                        self.assertEqual(status["optimum_objective"], [0, expected_cost])
                        self.assertEqual(status["coverage"]["positive_recall"], 1)

    def test_verifier_still_rejects_negatives_and_option_is_boolean(self):
        task = self.sample()
        result = verify_rules(task, ["answer(X) :- p(X)."], allow_partial_coverage=True)
        self.assertFalse(result["matched"])
        self.assertEqual(result["covered_negatives"], ["answer(n)"])
        with self.assertRaises(ValueError):
            solve([], 0, allow_partial_coverage="yes")


if __name__ == "__main__":
    unittest.main()
