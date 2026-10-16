"""Whole-program proofs checked against exhaustive legal rules and Clingo BK."""
from itertools import combinations
from pathlib import Path
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.bias_analysis import _HeadSolver
from nidavellir.program_proof import _ProgramSearch, prove_program_optimality
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import Task
from nidavellir.verification import verify_rules


class ProgramProofTests(unittest.TestCase):
    def task(self, background, literals, *, heads=("answer(var__(1))",), max_body=3,
             examples="positive(answer(a)). positive(answer(b)). negative(answer(c)).", constraints=""):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        bias = "\n".join([f"candidate_head({head})." for head in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n" + constraints
        for name, text in (("bk", background), ("examples", examples), ("bias", bias)):
            (root / (name + ".lp")).write_text(text)
        return Task.load(root / "bk.lp", root / "examples.lp", root / "bias.lp")

    def component(self, task, names, head=None):
        head = task.heads[0] if head is None else clingo.parse_term(head)
        body = [i for i, literal in enumerate(task.literals) if str(literal) in names or literal.name in names]
        return {"head": str(head), "body": body, "cost": 1 + len(body),
                "rule": task.rule(head, body), "coverage": 0}

    def exhaustive(self, task):
        """Independent finite optimum: legal bodies, Clingo, then subset-mask DP."""
        positives = sorted(task.positives)
        components = []
        for head in task.heads:
            for size in range(min(task.max_body, len(task.literals)) + 1):
                for body in combinations(range(len(task.literals)), size):
                    if not task.accepts(head, body):
                        continue
                    result = verify_rules(task, [task.rule(head, body)])
                    predictions = {clingo.parse_term(atom) for atom in result["predictions"]}
                    if task.negatives & predictions:
                        continue
                    coverage = sum(1 << i for i, atom in enumerate(positives) if atom in predictions)
                    if coverage:
                        components.append({"head": str(head), "body": list(body), "cost": size + 1,
                                           "rule": task.rule(head, body), "coverage": coverage})
        best = {0: (0, [])}
        for row in components:
            for covered, (cost, rows) in list(best.items()):
                new = covered | row["coverage"]
                candidate = cost + row["cost"]
                if new not in best or best[new][0] > candidate:
                    best[new] = candidate, rows + [row]
        return best.get((1 << len(positives)) - 1), components

    def assert_optimum(self, task, incumbent):
        expected, _ = self.exhaustive(task)
        self.assertIsNotNone(expected)
        cover, result = prove_program_optimality(task, incumbent, seconds=30)
        self.assertEqual(result["status"], "complete", result)
        self.assertTrue(result["optimality_proven"])
        self.assertFalse(result["catalogue_complete"])
        self.assertEqual(result["cost"], expected[0])
        self.assertEqual(result["proved_unsatisfiable_through_cost"], expected[0] - 1)
        self.assertTrue(verify_rules(task, [row["rule"] for row in cover])["matched"])
        json.dumps(result)
        return cover, result

    def test_cheaper_program_found_then_exhausted_unsat(self):
        task = self.task("p(a). p(b). q(a). r(b).", ["p(var__(1))", "q(var__(1))", "r(var__(1))"])
        _, report = self.assert_optimum(task, [self.component(task, ["q"]), self.component(task, ["r"])])
        self.assertEqual(report["initial_cost"], 4)
        self.assertEqual(report["cost"], 2)
        self.assertTrue(report["improvements"])

    def test_equal_cost_alternatives_do_not_require_catalogue(self):
        task = self.task("p(a). p(b). q(a). q(b).", ["p(var__(1))", "q(var__(1))"])
        cover, report = self.assert_optimum(task, [self.component(task, ["q"])])
        self.assertEqual(cover[0]["rule"], "answer(V1) :- q(V1).")
        self.assertEqual(report["improvements"], [])

    def test_overlapping_rules_and_lazy_full_positive_coverage(self):
        task = self.task("p(a). p(b). q(b). q(d). r(a). r(b). r(d). r(c).",
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         examples="positive(answer(a)). positive(answer(b)). positive(answer(d)). negative(answer(c)).")
        incumbent = [self.component(task, ["p"]), self.component(task, ["q"])]
        expected, _ = self.exhaustive(task)
        cover, result = prove_program_optimality(task, incumbent, seconds=30, initial_positives=1)
        self.assertEqual(result["cost"], expected[0])
        self.assertTrue(result["optimality_proven"])
        self.assertGreater(result["active_positive_labels"], 1)
        self.assertTrue(verify_rules(task, [row["rule"] for row in cover])["matched"])

    def test_strict_negative_witnesses_include_unselected_head_arguments(self):
        task = self.task("p(a,x). p(b,y). p(c,z). q(x). q(y).",
                         ["p(var__(7),var__(42))", "q(var__(42))"], heads=("answer(var__(7))",))
        _, report = self.assert_optimum(task, [self.component(task, ["p", "q"])])
        self.assertEqual(report["cost"], 3)
        self.assertGreater(report["negative_witnesses"], 0)

    def test_nonmonotone_auxiliary_bias_and_disjunctive_stable_models(self):
        constraints = """
        left | right.
        permitted :- left, body(p(var__(1))), not body(q(var__(1))).
        permitted :- right, body(q(var__(1))), body(r(var__(1))).
        :- not permitted.
        :- body(s(var__(1))), not body(r(var__(1))).
        """
        task = self.task("p(a). p(b). q(a). q(b). r(a). r(b). s(a). s(b).",
                         [f"{name}(var__(1))" for name in ("p", "q", "r", "s")], constraints=constraints)
        self.assert_optimum(task, [self.component(task, ["q", "r", "s"])])

    def test_weighted_aggregate_bias_with_negative_body_literals(self):
        task = self.task("p(a). p(b). q(a). q(b). r(a). r(b).",
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"], constraints="""
        permitted :- 2 <= #sum { 2,p: body(p(var__(1))); 1,q: body(q(var__(1))); 1,r: not body(r(var__(1))) }.
        :- not permitted.
        """)
        self.assert_optimum(task, [self.component(task, ["p", "q", "r"])])

    def test_constants_nested_terms_and_multiple_head_templates(self):
        task = self.task("p(a,box(x)). p(b,box(x)). q(b).", ["p(var__(9),box(x))", "q(var__(9))"],
                         heads=("answer(var__(9))", "answer(b)", "other(tag)", "other(var__(9))"),
                         examples="positive(answer(a)). positive(answer(b)). positive(other(tag)). negative(answer(c)). negative(other(a)).")
        incumbent = [self.component(task, ["p"], "answer(var__(9))"),
                     self.component(task, [], "other(tag)")]
        self.assert_optimum(task, incumbent)

    def test_ground_facts_cost_one_and_multiple_facts_can_improve(self):
        task = self.task("ready.", ["ready"], heads=("answer(a)", "answer(b)"), max_body=1,
                         examples="positive(answer(a)). positive(answer(b)). negative(answer(c)).")
        incumbent = [self.component(task, ["ready"], str(head)) for head in task.heads]
        _, report = self.assert_optimum(task, incumbent)
        self.assertEqual(report["cost"], 2)

    def test_empty_program_is_optimal_without_asp_search(self):
        task = self.task("p(a).", ["p(var__(1))"], examples="negative(answer(c)).")
        with patch("nidavellir.program_proof._ProgramSearch", side_effect=AssertionError("must not search")):
            cover, report = prove_program_optimality(task, [], seconds=30)
        self.assertEqual(cover, [])
        self.assertTrue(report["optimality_proven"])
        self.assertEqual(report["cost"], 0)

    def test_timeout_retains_verified_incumbent_without_unsat_claim(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"])
        incumbent = [self.component(task, ["p"])]
        with patch("nidavellir.program_proof._ProgramSearch.next", side_effect=ResourceLimit("interrupted solve")):
            cover, report = prove_program_optimality(task, incumbent, seconds=30)
        self.assertEqual(cover, incumbent)
        self.assertEqual(report["status"], "timeout")
        self.assertTrue(report["incumbent_revalidated"])
        self.assertFalse(report["optimality_proven"])
        self.assertIsNone(report["proved_unsatisfiable_through_cost"])
        _, zero = prove_program_optimality(task, incumbent, seconds=0)
        self.assertFalse(zero["incumbent_revalidated"])
        self.assertFalse(zero["optimality_proven"])

    def test_invalid_incumbent_or_failed_verification_is_not_fallback(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"])
        incumbent = [self.component(task, ["p"])]
        with self.assertRaisesRegex(ValueError, "verification"):
            prove_program_optimality(task, incumbent, verify=lambda _: {"matched": False})
        invalid = [dict(incumbent[0], cost=1)]
        with self.assertRaisesRegex(ValueError, "metadata"):
            prove_program_optimality(task, invalid)

    def test_shared_oracle_callback_is_restored(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"])
        previous = lambda: None
        oracle = CoverageOracle(task, previous)
        _, result = prove_program_optimality(task, [self.component(task, ["p"])], oracle=oracle)
        self.assertTrue(result["optimality_proven"])
        self.assertIs(oracle.check, previous)

    def test_unknown_solver_result_never_means_unsat(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"])
        checker = _HeadSolver(task, task.heads[0], [])
        search = _ProgramSearch(task, {task.heads[0]: checker}, 2, lambda: None)
        class UnknownHandle:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def wait(self, _): return True
            def get(self): return clingo.SolveResult(0)
        class UnknownControl:
            def solve(self, **_): return UnknownHandle()
        search.control = UnknownControl()
        with self.assertRaisesRegex(RuntimeError, "definite result"):
            search.next()

    def test_timeout_after_verified_improvement_retains_better_feasible_cover(self):
        task = self.task("p(a). p(b). q(a). r(b).", ["p(var__(1))", "q(var__(1))", "r(var__(1))"])
        verified_count = 0
        def verify(rules):
            nonlocal verified_count
            verified_count += 1
            return verify_rules(task, rules)
        def check():
            if verified_count > 1:
                raise ResourceLimit("deadline crossed during independent verification")
        incumbent = [self.component(task, ["q"]), self.component(task, ["r"])]
        cover, report = prove_program_optimality(task, incumbent, verify=verify, check=check)
        self.assertEqual(report["cost"], 2)
        self.assertEqual(report["status"], "timeout")
        self.assertFalse(report["optimality_proven"])
        self.assertTrue(verify_rules(task, [row["rule"] for row in cover])["matched"])

    def test_unlabelled_predictions_are_unconstrained(self):
        task = self.task("p(a). p(b). p(unlabelled).", ["p(var__(1))"])
        cover, _ = self.assert_optimum(task, [self.component(task, ["p"])])
        self.assertIn("answer(unlabelled)", verify_rules(task, [row["rule"] for row in cover])["predictions"])

    def test_random_tiny_tasks_match_exhaustive_optima(self):
        for seed in range(12):
            rng = random.Random(seed)
            facts = " ".join(f"{p}({a})." for p in ("p", "q", "r", "s") for a in ("a", "b", "c")
                             if rng.random() < .6)
            # Always provide a feasible two-rule incumbent; other predicates
            # may improve it, overlap, or require a negative-separating join.
            facts += " only_a(a). only_b(b)."
            task = self.task(facts, [f"{p}(var__(1))" for p in ("p", "q", "r", "s", "only_a", "only_b")],
                             max_body=2)
            with self.subTest(seed=seed):
                self.assert_optimum(task, [self.component(task, ["only_a"]), self.component(task, ["only_b"])])

    def test_ground_bias_replay_enumerates_exactly_all_legal_bodies(self):
        task = self.task("p(a). q(a). r(x). g.",
                         ["p(var__(9))", "q(var__(9))", "r(var__(42))", "g"],
                         heads=("answer(var__(9))", "answer(a)", "other(var__(42))"), max_body=4,
                         examples="positive(answer(a)).", constraints="""
        body(g).
        yes :- not no.
        no :- not yes.
        :- yes, body(g), not body(r(var__(42))).
        :- no, not body(q(var__(9))).
        :- head(other(var__(42))), body(q(var__(9))).
        :- 3 <= #sum { 2,p: body(p(var__(9))); 1,q: body(q(var__(9))) }.
        """)
        checkers = {head: _HeadSolver(task, head, []) for head in task.heads}
        search = _ProgramSearch(task, checkers, task.max_body + 2, lambda: None)
        # Inspect one active slot independently of positive/negative semantics.
        with search.control.backend() as backend:
            backend.add_rule([], [-search.active[0]])
            if len(search.active) > 1:
                backend.add_rule([], [search.active[1]])
        actual = set()
        while (proposal := search.next()) is not None:
            rules, _ = proposal
            self.assertEqual(set(rules), {0})
            head_index, body = rules[0]
            key = task.heads[head_index], body
            self.assertNotIn(key, actual)
            actual.add(key)
            with search.control.backend() as backend:
                backend.add_rule([], [search.gates[0][head_index],
                    *(atom if i in body else -atom for i, atom in enumerate(search.selected[0]))])
        expected = {(head, body) for head in task.heads
                    for size in range(task.max_body + 1)
                    for body in combinations(range(len(task.literals)), size)
                    if task.accepts(head, body)}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
