from types import SimpleNamespace
import unittest

import clingo

from nidavellir.verification import verify_rules
from nidavellir.task import _model, TaskError


class VerificationTests(unittest.TestCase):
    def test_background_display_suppression_keeps_all_true_atoms(self):
        source = "p(a;b). blocked(b). q(X) :- p(X), not blocked(X). irrelevant(1..10000)."
        expected = []
        control = clingo.Control(["2", "--warn=none"])
        control.add("base", [], source)
        control.ground([("base", [])])
        control.solve(on_model=lambda model: expected.append(frozenset(model.symbols(atoms=True))))
        actual, control = _model(source, "BK", [])
        self.assertEqual(actual, expected[0])
        self.assertEqual(len(actual), 10004)
        shown = []
        control.solve(on_model=lambda model: shown.extend(model.symbols(shown=True)))
        self.assertEqual(shown, [])
        for invalid in ("{ hidden }.", ":-."):
            with self.subTest(source=invalid), self.assertRaises(TaskError):
                _model(invalid, "BK", [])

    def test_projection_preserves_unlabelled_outputs_and_full_bk_negation(self):
        parse = clingo.parse_term
        background = "p(a;b;c). blocked(b). irrelevant(1..10000). t(extra,arity). q(X) :- p(X), not blocked(X)."
        task = SimpleNamespace(background=background, heads=(parse("t(var__(1))"),),
                               positives=frozenset({parse("t(a)")}),
                               negatives=frozenset({parse("t(b)")}))
        rules = ["t(X) :- q(X)."]
        actual = verify_rules(task, rules)
        ctl = clingo.Control(["2", "--warn=none"])
        ctl.add("base", [], background + "\n" + "\n".join(rules))
        ctl.ground([("base", [])])
        expected = []
        ctl.solve(on_model=lambda m: expected.extend(str(a) for a in m.symbols(atoms=True)
            if a.name == "t" and len(a.arguments) == 1))
        self.assertTrue(actual["matched"])
        self.assertEqual(actual["predictions"], sorted(expected))
        self.assertEqual(actual["predictions"], ["t(a)", "t(c)"])
        rejected = verify_rules(task, ["t(X) :- p(X)."])
        self.assertFalse(rejected["matched"])
        self.assertEqual(rejected["covered_negatives"], ["t(b)"])

    def test_projection_does_not_hide_multiple_or_absent_stable_models(self):
        for background in ("{ hidden }.", ":-."):
            task = SimpleNamespace(background=background, heads=(clingo.parse_term("t(a)"),),
                                   positives=frozenset(), negatives=frozenset())
            with self.subTest(background=background), self.assertRaises(RuntimeError):
                verify_rules(task, [])


if __name__ == "__main__":
    unittest.main()
