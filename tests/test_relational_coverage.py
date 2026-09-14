"""Lazy bound-oracle joins and certificates checked directly with Clingo."""

from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import clingo

from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import Task, instantiate, variables


def unique_model(source):
    control = clingo.Control(["2"], logger=lambda *_: None)
    control.add("base", [], source)
    control.ground([("base", [])])
    models = []
    answer = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if answer.unknown or not answer.satisfiable or not answer.exhausted or len(models) != 1:
        raise AssertionError("Expected exactly one exhausted Clingo model")
    return models[0]


class RelationalCoverageTests(unittest.TestCase):
    def task(self, background, literals, positives, negatives=(), head="answer(var__(1))"):
        temporary = tempfile.TemporaryDirectory(prefix="nidavellir-relational-check-")
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "background.lp").write_text(background)
        (folder / "examples.lp").write_text("\n".join(
            [f"positive({atom})." for atom in positives] + [f"negative({atom})." for atom in negatives]))
        (folder / "bias.lp").write_text(
            f"candidate_head({head}).\n" + "\n".join(f"allowed_literal({literal})." for literal in literals)
            + f"\nmax_body({len(literals)}).\n1 {{head(H):candidate_head(H)}} 1.\n"
            + "0 {body(L):allowed_literal(L)} N :- max_body(N).\n")
        return Task.load(folder / "background.lp", folder / "examples.lp", folder / "bias.lp")

    def check_all(self, task):
        oracle = CoverageOracle(task)
        head, = task.heads
        positive_order = sorted(task.positives)
        for size in range(len(task.literals) + 1):
            for body in combinations(range(len(task.literals)), size):
                used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                if not variables(head) <= used:
                    continue
                rule = task.rule(head, body)
                model = unique_model(task.background + "\n" + rule)
                target_signature = head.name, len(head.arguments)
                predicted = {atom for atom in model if (atom.name, len(atom.arguments)) == target_signature}
                with self.subTest(rule=rule):
                    self.assertEqual(oracle.predictions(head, body), predicted)
                    self.assertEqual(oracle.coverage(head, body), sum(
                        1 << i for i, atom in enumerate(positive_order) if atom in predicted))
                    for label in task.positives | task.negatives:
                        self.assertEqual(oracle.exists(head, body, label), label in predicted)
                        core = oracle.core(head, body, label)
                        if label in predicted:
                            self.assertIsNone(core)
                        else:
                            self.assertTrue(set(core) <= set(body))
                            self.assertFalse(oracle.exists(head, core, label))
                            for literal in core:
                                self.assertTrue(oracle.exists(head, tuple(i for i in core if i != literal), label))
                    witness = oracle.negative(head, body)
                    self.assertEqual(witness is not None, bool(predicted & task.negatives))
                    if witness:
                        self.assertIn(witness["head"], task.negatives)
                        self.assertEqual(instantiate(head, witness["binding"]), witness["head"])
                        self.assertTrue(all(instantiate(task.literals[i], witness["binding"]) in task.atoms for i in body))
                        expected_truth = sum(1 << i for i, literal in enumerate(task.literals)
                                             if instantiate(literal, witness["binding"]) in task.atoms)
                        self.assertEqual(witness["truth"], expected_truth)
        return oracle

    def test_shared_variables_preserve_correlated_joins_and_minimal_cores(self):
        task = self.task("p(a,x). q(a,y). p(b,x). q(b,x).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))"],
                         ["answer(a)"], ["answer(b)"])
        oracle = self.check_all(task)
        self.assertGreater(oracle.stats["cache_hits"], 0)

    def test_nested_terms_repeated_variables_and_unused_slots_in_witness(self):
        self.check_all(self.task(
            "edge(a,pair(x,x)). edge(b,pair(y,z)). edge(c,pair(y,y)). mark(x). mark(y). tag(z).",
            ["edge(var__(1),pair(var__(2),var__(2)))", "mark(var__(2))", "tag(var__(3))"],
            ["answer(a)"], ["answer(b)", "answer(c)"]))

    def test_unlabelled_outputs_are_predictions_but_not_negatives(self):
        task = self.task("p(a). p(unlabelled). q(a).", ["p(var__(1))", "q(var__(1))"],
                         ["answer(a)"], ["answer(negative)"])
        oracle = self.check_all(task)
        p_index = next(i for i, literal in enumerate(task.literals) if literal.name == "p")
        self.assertIn(clingo.parse_term("answer(unlabelled)"), oracle.predictions(task.heads[0], (p_index,)))
        self.assertIsNone(oracle.negative(task.heads[0], (p_index,)))

    def test_ground_head_zero_arity_and_empty_body(self):
        self.check_all(self.task("p(a).", ["p(var__(1))", "missing(var__(2))"], ["success"], head="success"))
        self.check_all(self.task("p(a).", [], [], ["failure"], head="failure"))

    def test_missing_head_match_has_an_empty_impossibility_core(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a,a)"], ["answer(a,b)"],
                         head="answer(var__(1),var__(1))")
        oracle = CoverageOracle(task)
        self.assertFalse(oracle.exists(task.heads[0], (0,), clingo.parse_term("answer(a,b)")))
        self.assertEqual(oracle.core(task.heads[0], (0,), clingo.parse_term("answer(a,b)")), ())

    def test_negative_safety_cache_is_scoped_to_the_task_labels(self):
        first = self.task("p(a). q(a).", ["p(var__(1))", "q(var__(1))"], ["answer(a)"])
        safe = CoverageOracle(first)
        self.assertIsNone(safe.negative(first.heads[0], (0,)))
        self.assertIsNone(safe.negative(first.heads[0], (0, 1)))
        self.assertEqual(safe.stats["safe_subsumption_hits"], 1)
        second = self.task("p(a). q(a).", ["p(var__(1))", "q(var__(1))"], [], ["answer(a)"])
        unsafe = CoverageOracle(second)
        self.assertIsNotNone(unsafe.negative(second.heads[0], (0, 1)))

    def test_head_unsafe_queries_are_rejected_and_deadline_is_checked(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"])
        oracle = CoverageOracle(task)
        for method in (oracle.coverage, oracle.negative, oracle.predictions):
            with self.assertRaises(ValueError):
                method(task.heads[0], ())
        stopped = False

        def check():
            if stopped:
                raise TimeoutError("test deadline")

        oracle = CoverageOracle(task, check)
        stopped = True
        with self.assertRaises(TimeoutError):
            oracle.predictions(task.heads[0], (0,))

    def test_real_negative_witness_cache_reuses_a_proof_without_another_join(self):
        task = self.task("p(a). q(a).", ["p(var__(1))", "q(var__(1))"], [], ["answer(a)"])
        oracle = CoverageOracle(task)
        head = task.heads[0]
        first = oracle.negative(head, (0, 1))
        matched = oracle.stats["matched_atoms"]
        # Mutating a returned certificate must not corrupt the cached proof.
        first["binding"].clear()
        first["truth"] = 0
        repeated = oracle.negative(head, (0,))
        self.assertEqual(instantiate(head, repeated["binding"]), clingo.parse_term("answer(a)"))
        self.assertEqual(repeated["truth"], 3)
        self.assertEqual(oracle.stats["matched_atoms"], matched)
        self.assertEqual(oracle.stats["negative_witness_cache_hits"], 1)
        self.assertEqual(oracle.profile.export()["timings"]["oracle.negative"]["calls"], 2)
        self.assertIsNone(oracle.negative(clingo.parse_term("other(var__(1))"), (0,)))

    def test_negative_profiles_never_merge_different_bindings_into_a_fake_witness(self):
        task = self.task("p(a). q(a). p(b). q(c).",
                         ["p(var__(1))", "q(var__(1))"], ["answer(a)"], ["answer(b)", "answer(c)"])
        oracle = CoverageOracle(task)
        self.assertIsNotNone(oracle.negative(task.heads[0], (0,)))
        self.assertIsNotNone(oracle.negative(task.heads[0], (1,)))
        self.assertEqual(oracle.stats["negative_witnesses_cached"], 2)
        self.assertIsNone(oracle.negative(task.heads[0], (0, 1)))
        self.assertEqual(oracle.predictions(task.heads[0], (0, 1)), task.positives)

    def test_bounded_or_disabled_negative_cache_preserves_exact_answers(self):
        task = self.task("p(a). p(b). q(a). q(c). r(a). r(d).",
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         ["answer(a)"], ["answer(b)", "answer(c)", "answer(d)"])
        for limit in (0, 1, 2):
            oracle = CoverageOracle(task, max_negative_witnesses=limit)
            for size in range(1, 4):
                for body in combinations(range(3), size):
                    predicted = unique_model(task.background + task.rule(task.heads[0], body))
                    self.assertEqual(oracle.negative(task.heads[0], body) is not None,
                                     bool(predicted & task.negatives))
                    self.assertLessEqual(oracle.stats["negative_witnesses_cached"], limit)
        with self.assertRaises(ValueError):
            CoverageOracle(task, max_negative_witnesses=-1)

    def test_coverage_fixes_labelled_heads_without_enumerating_all_outputs(self):
        task = self.task("p(a,x). p(a,y). p(unlabelled,z). q(x). q(y). q(z).",
                         ["p(var__(1),var__(2))", "q(var__(2))"], ["answer(a)", "answer(missing)"])
        oracle = CoverageOracle(task)
        original = oracle.bindings
        fixed_queries = []

        def labelled_bindings(body, fixed=None):
            self.assertIn(1, fixed)
            fixed_queries.append(dict(fixed))
            yield from original(body, fixed)

        oracle.bindings = labelled_bindings
        expected = sum(1 << i for i, atom in enumerate(sorted(task.positives))
                       if atom == clingo.parse_term("answer(a)"))
        self.assertEqual(oracle.coverage(task.heads[0], (0, 1)), expected)
        self.assertEqual(len(fixed_queries), 2)
        oracle.bindings = original
        self.assertEqual(oracle.predictions(task.heads[0], (0, 1)),
                         frozenset(map(clingo.parse_term, ["answer(a)", "answer(unlabelled)"])))


if __name__ == "__main__":
    unittest.main()
