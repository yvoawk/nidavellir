"""Liveness projection avoids private products without dropping FD join keys."""
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir import reduction
from nidavellir.task import Task


def atoms(source):
    control = clingo.Control(["2"], logger=lambda *_: None)
    control.add("base", [], source)
    control.ground([("base", [])])
    models = []
    result = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if not result.satisfiable or not result.exhausted or len(models) != 1:
        raise AssertionError("Expected a unique complete interpretation")
    return models[0]


class ReductionLivenessTests(unittest.TestCase):
    def task(self, background, literals, *, heads=("answer(var__(1))",),
             examples="positive(answer(b)). negative(answer(a)).", constraints="", max_body=3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        bias = "\n".join([f"candidate_head({head})." for head in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n" + constraints
        for name, source in (("bk.lp", background), ("bias.lp", bias), ("examples.lp", examples)):
            (directory / name).write_text(source)
        return Task.load(directory / "bk.lp", directory / "examples.lp", directory / "bias.lp")

    def compare(self, task):
        optimized, stats = build_optimized_indices(task)
        reference, _ = build_cartesian_indices(task, max_witnesses=100000)
        positive_order = sorted(task.positives)
        for actual, raw in zip(optimized, reference):
            for length in range(min(task.max_body, len(task.literals)) + 1):
                for body in combinations(range(len(task.literals)), length):
                    if not task.accepts(actual.head, body):
                        continue
                    prediction = atoms(task.background + "\n" + task.rule(actual.head, body))
                    mask = actual.body_mask(body)
                    self.assertEqual(actual.coverage(mask), raw.coverage(raw.body_mask(body)))
                    self.assertEqual(actual.coverage(mask), sum(1 << i for i, p in enumerate(positive_order) if p in prediction))
                    self.assertEqual(bool(mask & actual.negative_mask), bool(prediction & task.negatives))
        return optimized, stats

    def test_independent_private_guards_do_not_materialize_four_million_pairs(self):
        task = self.task("p(1..2000). q(1..2000).", ["p(var__(1))", "q(var__(2))"],
                         heads=("success",), examples="positive(success).", max_body=2,
                         constraints=":- not body(p(var__(1))).\n:- not body(q(var__(2))).\n")
        observed = []
        original = reduction.join_relations
        def measured(left, right, check):
            observed.append((len(left[1]), len(right[1])))
            # Checking inputs rather than peak RSS makes this test stable and
            # fails before allocating a large Cartesian intermediate.
            self.assertLessEqual(len(left[1]) * len(right[1]), 1)
            return original(left, right, check)
        with patch.object(reduction, "join_relations", measured):
            indices, stats = build_optimized_indices(task)
        self.assertTrue(observed)
        self.assertEqual(stats["compressed_witnesses"], 1)
        self.assertEqual(indices[0].coverage(indices[0].body_mask((0, 1))), 1)
        self.assertEqual(stats["reductions"]["success"]["private_variables"], [1, 2])
        self.assertIn(clingo.parse_term("success"), atoms(task.background + "success :- p(X), q(Y)."))

    def test_eliminated_shared_fd_output_is_kept_until_conflicting_guard_join(self):
        task = self.task("r(a,x). s(a,y). r(b,z). s(b,z).",
                         ["r(var__(1),var__(2))", "s(var__(1),var__(2))"], max_body=2,
                         constraints=":- body(r(var__(1),var__(2))), not body(s(var__(1),var__(2))).\n"
                                     ":- body(s(var__(1),var__(2))), not body(r(var__(1),var__(2))).\n")
        indices, stats = self.compare(task)
        plan = stats["reductions"][str(task.heads[0])]
        self.assertIn(2, plan["eliminated_variables"])
        self.assertTrue(plan["functional_outputs"])
        mask = indices[0].body_mask((0, 1))
        self.assertEqual(indices[0].coverage(mask), 1)
        self.assertFalse(mask & indices[0].negative_mask)

    def test_fd_determinant_lives_until_a_later_consumer_joins(self):
        task = self.task("a_carrier(a,x). a_carrier(b,y). link(x,u). link(y,v). good(v).",
                         ["a_carrier(var__(1),var__(2))", "link(var__(2),var__(3))", "good(var__(3))"],
                         constraints=":- body(link(var__(2),var__(3))), not body(a_carrier(var__(1),var__(2))).\n"
                                     ":- body(good(var__(3))), not body(link(var__(2),var__(3))).\n")
        _, stats = self.compare(task)
        plan = stats["reductions"][str(task.heads[0])]
        self.assertIn(2, plan["eliminated_variables"])

    def test_nonfunctional_output_does_not_split_incompatible_consumers(self):
        task = self.task("r(a,x). r(a,y). r(b,z). low(x). low(z). high(y). high(z).",
                         ["r(var__(1),var__(2))", "low(var__(2))", "high(var__(2))"],
                         constraints=":- body(low(var__(2))), not body(r(var__(1),var__(2))).\n"
                                     ":- body(high(var__(2))), not body(r(var__(1),var__(2))).\n")
        _, stats = self.compare(task)
        self.assertNotIn(2, stats["reductions"][str(task.heads[0])]["eliminated_variables"])

    def test_functional_scan_checks_deadline_periodically(self):
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ResourceLimit("FD scan interrupted")
        with self.assertRaisesRegex(ResourceLimit, "FD scan"):
            reduction._functional((1, 2), [(i, i) for i in range(5000)], {2}, check)
        self.assertEqual(calls, 2)

    def test_large_single_join_bucket_checks_deadline_inside_bucket(self):
        # 4096 rows: two table-building checks and one left-loop check happen
        # first. The fourth check must occur while traversing the sole bucket.
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ResourceLimit("join bucket interrupted")
        with self.assertRaisesRegex(ResourceLimit, "join bucket"):
            reduction.join_relations(((), frozenset({()})), ((2,), frozenset((i,) for i in range(4096))), check)
        self.assertEqual(calls, 4)


if __name__ == "__main__":
    unittest.main()
