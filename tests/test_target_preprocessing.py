"""Target-specific unary filtering and suffix capacity versus independent ASP."""
from itertools import combinations
import unittest

import clingo

from nidavellir.oracle import build_cartesian_indices
from nidavellir.search import _TargetPlan, _Traversal, compile_catalogue, mine_variants
from tests import test_search_optimizations as fixtures


class TargetPreprocessingTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task
    expected = fixtures.SearchOptimizationTests.expected
    actual = staticmethod(fixtures.SearchOptimizationTests.actual)

    def independent_bodies(self, task, head):
        positives = sorted(task.positives)
        rows = []
        for size in range(min(len(task.literals), task.max_body) + 1):
            for body in combinations(range(len(task.literals)), size):
                if task.accepts(head, body):
                    prediction = fixtures.atoms(task.background + "\n" + task.rule(head, body))
                    covered = sum(1 << i for i, positive in enumerate(positives) if positive in prediction)
                    rows.append((sum(1 << i for i in body), covered))
        return rows

    def assert_plan_bounds(self, task):
        indices, _ = build_cartesian_indices(task)
        for order in ("source", "selectivity"):
            traversal = _Traversal(task, indices, lambda: None, lambda: 30, order=order)
            for index in indices:
                proof, closures, _, unavailable, permutation, *_ = traversal.contexts[index.head]
                bodies = self.independent_bodies(task, index.head)
                for cost in range(1, task.max_body + 2):
                    for wanted in range(1, 1 << len(task.positives)):
                        plan = _TargetPlan.build(index, proof, closures, unavailable, permutation,
                                                 cost, wanted, lambda: None)
                        matching = [body for body, coverage in bodies
                                    if body.bit_count() + 1 == cost and coverage & wanted == wanted]
                        if plan is None:
                            self.assertFalse(matching)
                            continue
                        for body in matching:
                            self.assertEqual(body & ~plan.eligible, 0)
                            # Include prefixes whose closures already select
                            # literals beyond the current enumeration position.
                            for selected in range(1 << len(task.literals)):
                                if selected & ~body or any(closures[i] & ~selected for i in range(len(closures))
                                                           if selected >> i & 1):
                                    continue
                                for start in range(len(plan.order) + 1):
                                    if body & plan.prefix[start] & ~selected:
                                        continue
                                    self.assertLessEqual(body.bit_count(), selected.bit_count()
                                                         + plan.remaining_capacity(start, selected))
        return indices

    def test_static_filter_propagates_an_ineligible_obligatory_guard(self):
        task = self.task("p(a). p(b). q(a). r(a). r(b).", [f"{p}(var__(1))" for p in "pqr"],
                         constraints=":- body(r(var__(1))), not body(q(var__(1))).\n")
        indices = self.assert_plan_bounds(task)
        traversal = _Traversal(task, indices, lambda: None, lambda: 30)
        rows = []
        traversal.walk(2, lambda *_: rows.append(True), targets={(3, 3)})
        self.assertFalse(rows)
        self.assertGreaterEqual(traversal.stats["target_static_rejected_literals"], 2)
        self.assertEqual(traversal.stats["pruned_target_capacity"], 1)
        self.assertEqual(traversal.stats["nodes"], 1)

    def test_guard_cycles_count_literals_and_already_occupied_future_groups(self):
        task = self.task("p(a). q(a). r(a).", [f"{p}(var__(1))" for p in "pqr"], max_body=3,
                         examples="positive(answer(a)). negative(answer(b)).",
                         constraints=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                     ":- body(q(var__(1))), not body(p(var__(1))).\n")
        indices = self.assert_plan_bounds(task)
        traversal = _Traversal(task, indices, lambda: None, lambda: 30, order="source")
        index = indices[0]
        proof, closures, _, unavailable, order, *_ = traversal.contexts[index.head]
        plan = _TargetPlan.build(index, proof, closures, unavailable, order, 4, 1, lambda: None)
        p, q = [task.literals.index(clingo.parse_term(f"{name}(var__(1))")) for name in ("p", "q")]
        self.assertEqual(plan.remaining_capacity(0, 0), 3)
        self.assertEqual(plan.remaining_capacity(1, (1 << p) | (1 << q)), 1)

    def test_overlapping_at_most_groups_and_arbitrary_disjunctions(self):
        task = self.task("p(a). q(a). r(a). s(a).", [f"{p}(var__(1))" for p in "pqrs"], max_body=4,
                         examples="positive(answer(a)). negative(answer(b)).",
                         constraints=":- 2 { body(p(var__(1))); body(q(var__(1))) }.\n"
                                     ":- 2 { body(q(var__(1))); body(r(var__(1))) }.\n"
                                     ":- body(s(var__(1))), not 1 { body(p(var__(1))); body(r(var__(1))) }.\n")
        self.assert_plan_bounds(task)

    def test_targeted_and_reconstructed_catalogues_preserve_all_heads_and_ties(self):
        task = self.task("p(a). p(b). q(a). r(a). r(b). s(b).", [f"{p}(var__(1))" for p in "pqrs"],
                         heads=("answer(left,var__(1))", "answer(right,var__(1))"),
                         examples="positive(answer(left,a)). positive(answer(left,b)). "
                                  "positive(answer(right,a)). negative(answer(left,c)). negative(answer(right,b)).",
                         constraints=":- head(answer(left,var__(1))), body(q(var__(1))), not body(s(var__(1))).\n")
        indices = self.assert_plan_bounds(task)
        expected = self.expected(task)
        for order in ("source", "selectivity"):
            for preprocessing in (False, True):
                result, stats = compile_catalogue(task, indices, order=order,
                                                  preprocessing=preprocessing, variants="targeted")
                self.assertTrue(stats["complete"])
                self.assertEqual(self.actual(result), expected)
        representatives, _ = compile_catalogue(task, indices, variants="representative")
        recovered, stats = mine_variants(task, indices, representatives)
        self.assertEqual(self.actual(recovered), expected)
        self.assertEqual([row["id"] for row in recovered], [row["id"] for row in representatives])
        self.assertTrue(stats["complete"])

    def test_static_singletons_do_not_claim_a_correlated_joint_witness(self):
        task = self.task("edge(a,x). good(y).", ["edge(var__(1),var__(2))", "good(var__(2))"], max_body=2,
                         examples="positive(answer(a)). negative(answer(b)).",
                         constraints=":- body(edge(var__(1),var__(2))), not body(good(var__(2))).\n")
        indices = self.assert_plan_bounds(task)
        traversal = _Traversal(task, indices, lambda: None, lambda: 30)
        rows = []
        traversal.walk(2, lambda *_: rows.append(True), targets={(3, 1)})
        self.assertFalse(rows)
        self.assertGreater(traversal.stats["pruned_no_positive"] + traversal.stats["pruned_incompatible"], 0)

    def test_empty_body_ground_head_keeps_cost_one_target(self):
        task = self.task("", [], heads=("answer(a)",), max_body=0,
                         examples="positive(answer(a)). negative(answer(b)).")
        indices = self.assert_plan_bounds(task)
        representatives, _ = compile_catalogue(task, indices, variants="representative")
        recovered, stats = mine_variants(task, indices, representatives)
        self.assertEqual(self.actual(recovered), {(1, 1): {"answer(a)."}})
        self.assertTrue(stats["complete"])


if __name__ == "__main__":
    unittest.main()
