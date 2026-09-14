import tempfile
import unittest
from pathlib import Path

import clingo

from nidavellir.task import Task, TaskError, instantiate, variables


BASE_BIAS = """
candidate_head(target(var__(12))).
allowed_literal(feature(var__(12))).
max_body(2).
1 { head(H) : candidate_head(H) } 1.
0 { body(L) : allowed_literal(L) } 2.
"""


class TaskContractTests(unittest.TestCase):
    def task(self, bk="feature(a).", examples="positive(target(a)).", bias=BASE_BIAS):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("bk.lp", "examples.lp", "bias.lp")]
            for path, content in zip(paths, (bk, examples, bias)):
                path.write_text(content)
            return Task.load(*paths)

    def test_generic_variable_ids_actual_bias_and_rendering(self):
        task = self.task()
        self.assertTrue(task.accepts(task.heads[0], (0,)))
        self.assertFalse(task.accepts(task.heads[0], ()))
        self.assertEqual(task.rule(task.heads[0], (0,)), "target(V12) :- feature(V12).")
        constrained = self.task(bias=BASE_BIAS + ":- body(feature(var__(12))).")
        self.assertFalse(constrained.accepts(constrained.heads[0], (0,)))

    def test_signature_not_predicate_name_and_not_nested_constructor(self):
        task = self.task(bk="target(a,b). feature(a). marker(target(a)).")
        self.assertIn(clingo.parse_term("target(a,b)"), task.atoms)
        with self.assertRaisesRegex(TaskError, "target predicate"):
            self.task(bk="feature(a). helper(X) :- target(X).")

    def test_rejects_acyclic_cross_target_dependency(self):
        bias = BASE_BIAS + "candidate_head(feature(var__(12)))."
        with self.assertRaisesRegex(TaskError, "cross-target"):
            self.task(bias=bias)

    def test_rejects_bk_recursion_choice_and_external(self):
        for bk in ("feature(X) :- feature(X).",
                   "{ feature(a) }.", "feature(@f())."):
            with self.subTest(bk=bk), self.assertRaises(TaskError):
                self.task(bk=bk)

    def test_acyclic_background_default_negation_is_exact(self):
        task = self.task(bk="seed(a). seed(b). absent(b). feature(X) :- seed(X), not absent(X).")
        self.assertIn(clingo.parse_term("feature(a)"), task.atoms)
        self.assertNotIn(clingo.parse_term("feature(b)"), task.atoms)
        self.assertIn("not absent(X)", task.background)
        for bk in ("feature(a) :- not target(a).", "feature(X) :- not absent(X).",
                   "feature(a) :- not absent(a). absent(a) :- not feature(a).",
                   "feature(a) :- not not absent(a)."):
            with self.subTest(bk=bk), self.assertRaises(TaskError):
                self.task(bk=bk)

    def test_rejects_unsatisfiable_or_unsafe_bk(self):
        for bk in ("feature(a). :- feature(a).", "feature(X)."):
            with self.subTest(bk=bk), self.assertRaises(TaskError):
                self.task(bk=bk)

    def test_positive_rules_comparisons_and_constants(self):
        task = self.task(bk="#const n=3. seed(1..n). feature(X) :- seed(X), X != 2.",
                         examples="positive(target(1)). negative(target(2)).")
        self.assertIn(clingo.parse_term("feature(1)"), task.atoms)
        self.assertNotIn(clingo.parse_term("feature(2)"), task.atoms)

    def test_pooled_facts_rules_and_ground_examples(self):
        task = self.task(bk="seed(a;b). feature(X) :- seed(X). pair(a;b,1;2).",
                         examples="positive(target(a;b)). negative(target(c;d)).")
        self.assertEqual(task.positives, frozenset(map(clingo.parse_term, ("target(a)", "target(b)"))))
        self.assertEqual(len(task.negatives), 2)
        self.assertIn(clingo.parse_term("feature(b)"), task.atoms)
        self.assertIn(clingo.parse_term("pair(b,1)"), task.atoms)
        for bad_bk in ("feature(a);feature(b).", "feature(a;target(a)). target(a;b)."):
            with self.subTest(bk=bad_bk), self.assertRaises(TaskError):
                self.task(bk=bad_bk)

    def test_explicit_length_and_factual_metadata_required(self):
        for bias in (BASE_BIAS.replace("max_body(2).", ""),
                     BASE_BIAS.replace("max_body(2).", "n(2). max_body(X) :- n(X).")):
            with self.subTest(bias=bias), self.assertRaises(TaskError):
                self.task(bias=bias)

    def test_ground_hard_labels_only_and_conflicts(self):
        for examples in ("positive(target(a)). negative(target(a)).", "positive(other(a)).",
                         "positive(target(X)).", "negative(target(a)) :- feature(a)."):
            with self.subTest(examples=examples), self.assertRaises(TaskError):
                self.task(examples=examples)

    def test_include_snapshot_and_display_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "facts.lp").write_text("feature(a).")
            (folder / "bk.lp").write_text('#include "facts.lp". #show nonexistent/1.')
            (folder / "examples.lp").write_text("positive(target(a)).")
            (folder / "bias.lp").write_text(BASE_BIAS)
            task = Task.load(folder / "bk.lp", folder / "examples.lp", folder / "bias.lp")
        self.assertIn(clingo.parse_term("feature(a)"), task.atoms)
        self.assertNotIn("#include", task.background)

    def test_ground_head_empty_rule_and_empty_examples(self):
        task = self.task(bias="candidate_head(ok). max_body(0). head(ok).", examples="")
        self.assertTrue(task.accepts(task.heads[0], ()))
        self.assertEqual(task.rule(task.heads[0], ()), "ok.")

    def test_instantiation_nested_terms_strings_and_repeated_variable(self):
        term = clingo.parse_term('p(f(var__(12)),"x%y",var__(12))')
        self.assertEqual(variables(term), {12})
        self.assertEqual(str(instantiate(term, {12: clingo.Number(4)})), 'p(f(4),"x%y",4)')


if __name__ == "__main__":
    unittest.main()
