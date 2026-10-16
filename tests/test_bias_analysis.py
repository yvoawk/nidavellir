"""Exact bias guard proofs, checked against exhaustive independent legality."""
from itertools import combinations
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import clingo

from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.task import Task, TaskError


class BiasAnalysisTests(unittest.TestCase):
    def task(self, literals, *, heads=("answer(var__(1))",), restrictions="", max_body=3,
             body_choice=None, background="p(a). q(a). r(b). s(a).", examples="positive(answer(a))."):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        bias = "\n".join([f"candidate_head({h})." for h in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        bias += body_choice if body_choice is not None else "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n"
        bias += restrictions
        for name, source in (("bk.lp", background), ("bias.lp", bias), ("examples.lp", examples)):
            (path / name).write_text(source)
        return Task.load(path / "bk.lp", path / "examples.lp", path / "bias.lp")

    def check_exhaustive(self, task):
        analysis = analyze(task)
        all_indices = range(len(task.literals))
        for head in task.heads:
            actual = analysis.heads[head]
            legal = [frozenset(body) for count in range(min(task.max_body, len(task.literals)) + 1)
                     for body in combinations(all_indices, count) if task.accepts(head, body)]
            self.assertEqual(actual.legal_body_exists, bool(legal))
            for count in range(len(task.literals) + 1):
                for body in combinations(all_indices, count):
                    self.assertEqual(actual.accepts(body), task.accepts(head, body))
            for index in all_indices:
                containing = [body for body in legal if index in body]
                if containing:
                    expected = set.intersection(*(set(body) for body in containing)) - {index}
                    self.assertEqual(actual.guards[index], expected)
                    self.assertNotIn(index, actual.unusable_literals)
                else:
                    self.assertIn(index, actual.unusable_literals)
                    self.assertEqual(actual.guards[index], frozenset())
        self.assertTrue(analysis.stats["complete"])
        return analysis

    @staticmethod
    def ids(task, *literals):
        return [task.literals.index(clingo.parse_term(literal)) for literal in literals]

    def test_derived_meta_predicate_proves_conjunctive_guards(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))", "s(var__(1))"],
                         restrictions="supported :- body(q(var__(1))), body(r(var__(1))).\n"
                                      ":- body(p(var__(1))), not supported.\n")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        p, q, r = self.ids(task, "p(var__(1))", "q(var__(1))", "r(var__(1))")
        self.assertEqual(result.guards[p], {q, r})

    def test_disjunctive_requirement_does_not_invent_pairwise_guards(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         restrictions=":- body(p(var__(1))), not 1 { body(q(var__(1))); body(r(var__(1))) }.\n")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertEqual(result.guards[self.ids(task, "p(var__(1))")[0]], frozenset())

    def test_mutual_implications_and_transitive_closure(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))", "s(var__(1))"], max_body=4,
                         restrictions=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                      ":- body(q(var__(1))), not body(r(var__(1))).\n"
                                      ":- body(r(var__(1))), not body(p(var__(1))).\n")
        self.check_exhaustive(task)

    def test_head_conditioned_dependencies_are_kept_separate(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         heads=("answer(left,var__(1))", "answer(right,var__(1))"), examples="",
                         restrictions=":- head(answer(left,var__(1))), body(p(var__(1))), not body(q(var__(1))).\n")
        analysis = self.check_exhaustive(task)
        p, q = self.ids(task, "p(var__(1))", "q(var__(1))")
        self.assertEqual(analysis.heads[clingo.parse_term("answer(left,var__(1))")].guards[p], {q})
        self.assertEqual(analysis.heads[clingo.parse_term("answer(right,var__(1))")].guards[p], frozenset())

    def test_actual_length_restriction_and_unusable_literals(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))"], max_body=2,
                         body_choice="{ body(L) : allowed_literal(L) }.\n",
                         restrictions=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                      ":- body(p(var__(1))), not body(r(var__(1))).\n")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertEqual(result.unusable_literals, set(self.ids(task, "p(var__(1))")))

    def test_head_safety_and_private_variables(self):
        task = self.task(["p(var__(1))", "q(var__(2))", "r(var__(2))", "pair(var__(3),var__(3))"])
        result = self.check_exhaustive(task).heads[task.heads[0]]
        p, q, pair = self.ids(task, "p(var__(1))", "q(var__(2))", "pair(var__(3),var__(3))")
        self.assertEqual(result.guards[q], {p})
        self.assertEqual(result.private_variables, {3: pair})

    def test_no_safe_body_and_absent_body_atoms(self):
        task = self.task(["q(var__(2))"], body_choice="")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertFalse(result.legal_body_exists)
        self.assertEqual(result.unusable_literals, {0})
        task = self.task(["q(var__(2))"], heads=("success",), examples="", body_choice="")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertTrue(result.legal_body_exists)
        self.assertEqual(result.unusable_literals, {0})

    def test_only_empty_legal_body(self):
        task = self.task([], heads=("success",), max_body=0, examples="")
        result = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertTrue(result.legal_body_exists)
        self.assertEqual(result.guards, ())

    def test_guarded_masks_preserve_every_legal_body_and_head(self):
        task = self.task(["p(var__(1),var__(2))", "q(var__(2))", "r(var__(2))", "s(var__(1))"],
                         background="p(a,x). p(a,y). p(b,y). q(x). q(z). r(y). s(a). s(b).",
                         examples="positive(answer(a)). negative(answer(b)).",
                         restrictions=":- body(q(var__(2))), not body(p(var__(1),var__(2))).\n"
                                      ":- body(r(var__(2))), not body(p(var__(1),var__(2))).\n")
        analysis = self.check_exhaustive(task)
        indices, _ = build_cartesian_indices(task)
        for index in indices:
            proof = analysis.heads[index.head]
            guarded = []
            for i, mask in enumerate(index.literal_masks):
                for guard in proof.guards[i]:
                    mask &= index.literal_masks[guard]
                guarded.append(0 if i in proof.unusable_literals else mask)
            for size in range(task.max_body + 1):
                for body in combinations(range(len(task.literals)), size):
                    if task.accepts(index.head, body):
                        actual = index.all_mask
                        for literal in body:
                            actual &= guarded[literal]
                        self.assertEqual(actual, index.body_mask(body))

    def test_supplied_deadline_fails_without_partial_analysis(self):
        task = self.task(["p(var__(1))"])
        with self.assertRaises(ResourceLimit) as caught:
            analyze(task, seconds=0)
        self.assertFalse(caught.exception.stats["complete"])
        for seconds in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                analyze(task, seconds=seconds)

    def test_callback_deadline_is_honored(self):
        task = self.task(["p(var__(1))"])
        def stop():
            raise ResourceLimit("caller deadline")
        with self.assertRaisesRegex(ResourceLimit, "caller deadline"):
            analyze(task, check=stop)

    def test_partial_completion_matches_exhaustive_legal_bodies(self):
        task = self.task(["p(var__(1))", "q(var__(1))", "r(var__(1))", "s(var__(1))"],
                         restrictions=":- body(p(var__(1))), not 1 { body(q(var__(1))); body(r(var__(1))) }.\n")
        analysis = self.check_exhaustive(task)
        head = task.heads[0]
        proof = analysis.heads[head]
        bodies = [frozenset(b) for n in range(task.max_body + 1)
                  for b in combinations(range(len(task.literals)), n) if task.accepts(head, b)]
        subsets = [frozenset(b) for n in range(len(task.literals) + 1)
                   for b in combinations(range(len(task.literals)), n)]
        for included in subsets:
            for excluded in subsets:
                expected = any(included <= body and not excluded & body for body in bodies)
                self.assertEqual(proof.possible_completion(included, excluded), expected)
        self.assertTrue(proof.checker_stats["completion_queries"])
        self.assertTrue(analysis.possible_completion(head))

    def test_reusable_head_checkers_keep_head_specific_controls(self):
        task = self.task(["p(var__(1))", "q(var__(1))"], examples="",
                         heads=("answer(left,var__(1))", "answer(right,var__(1))"),
                         restrictions=":- head(answer(left,var__(1))), body(p(var__(1))), not body(q(var__(1))).\n")
        analysis = self.check_exhaustive(task)
        p, = self.ids(task, "p(var__(1))")
        left, right = (clingo.parse_term(f"answer({side},var__(1))") for side in ("left", "right"))
        for _ in range(3):
            self.assertFalse(analysis.accepts(left, (p,)))
            self.assertTrue(analysis.accepts(right, (p,)))

    def test_guards_can_make_raw_private_variables_shared(self):
        task = self.task(["p(var__(1),var__(2))", "q(var__(1))"],
                         restrictions=":- body(q(var__(1))), not body(p(var__(1),var__(2))).\n")
        proof = self.check_exhaustive(task).heads[task.heads[0]]
        self.assertNotIn(2, proof.private_variables)

    def test_checker_deadline_and_invalid_candidate_handling(self):
        task = self.task(["p(var__(1))", "q(var__(1))"])
        proof = analyze(task).heads[task.heads[0]]
        for body in ((0, 0), (-1,), (2,), ("0",)):
            self.assertFalse(proof.accepts(body))
        self.assertFalse(proof.possible_completion((3,), ()))
        self.assertFalse(proof.possible_completion((0,), (0,)))
        def stop():
            raise ResourceLimit("current compiler deadline")
        with self.assertRaisesRegex(ResourceLimit, "current compiler deadline"):
            proof.accepts((0,), check=stop)

    def test_unknown_query_is_not_a_proof_of_impossibility(self):
        task = self.task(["p(var__(1))", "q(var__(1))"])
        proof = analyze(task).heads[task.heads[0]]
        # Exercise an actual solver fallback, rather than a separately proved
        # complete-body result or a previously obtained satisfiable witness.
        proof._checker.compiled_legality = None
        proof._checker.witnesses.clear()
        original = proof._checker.control
        proof._checker.control = SimpleNamespace(solve=lambda **_: SimpleNamespace(unknown=True))
        try:
            with self.assertRaisesRegex(TaskError, "UNKNOWN"):
                proof.accepts((0,))
        finally:
            proof._checker.control = original


if __name__ == "__main__":
    unittest.main()
