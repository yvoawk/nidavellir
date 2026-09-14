"""Singleton projections checked against independent guarded Clingo rules."""
import unittest

import clingo

from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import asp_text
from tests import test_relational_coverage as fixtures
from tests import test_optimized_oracle as optimized_fixtures


class AtomicProjectionTests(unittest.TestCase):
    task = fixtures.RelationalCoverageTests.task

    def compare(self, task):
        oracle = CoverageOracle(task)
        for head in task.heads:
            for i, literal in enumerate(task.literals):
                labels = "\n".join(f"rc_label({label})." for label in task.positives)
                rule = f"rc_hit({asp_text(head)}) :- rc_label({asp_text(head)}), {asp_text(literal)}."
                model = fixtures.unique_model(task.background + "\n" + labels + "\n" + rule)
                expected = sum(1 << position for label, position in oracle.positives.items()
                               if clingo.Function("rc_hit", [label]) in model)
                self.assertEqual(oracle.atomic_coverage(head, i), expected, (head, literal))
                self.assertEqual(oracle.atomic_coverage(head, i), expected)
        return oracle

    def test_nested_repeated_hidden_variables_and_partial_heads(self):
        self.compare(self.task(
            "edge(a,pair(x,x)). edge(b,pair(y,z)). edge(c,pair(y,y)). tag(x). flag.",
            ["edge(var__(1),pair(var__(2),var__(2)))", "edge(var__(1),pair(var__(2),var__(3)))",
             "tag(var__(2))", "flag", "missing(var__(1))"],
            ["answer(a,a)", "answer(a,b)", "answer(b,b)", "answer(c,c)", "answer(unseen,unseen)"],
            head="answer(var__(1),var__(1))"))

    def test_ground_heads_constants_and_empty_relations(self):
        self.compare(self.task("p(a).", ["p(var__(1))", "p(b)", "absent"],
                               ["success"], head="success"))

    def test_cache_key_preserves_head_role_and_shares_hidden_renamings(self):
        task = self.task("r(a,b). r(a,c). r(b,b).",
                         ["r(var__(1),var__(2))", "r(var__(1),var__(3))", "r(var__(2),var__(1))"],
                         ["answer(a)", "answer(b)", "answer(c)"])
        oracle = self.compare(task)
        self.assertEqual(oracle.stats["atomic_projection_builds"], 2)
        self.assertGreater(oracle.stats["atomic_projection_cache_hits"], 0)

    def test_projection_avoids_per_label_existential_queries(self):
        task = self.task("p(a,x). p(b,y).", ["p(var__(1),var__(2))"],
                         ["answer(a)", "answer(b)", "answer(c)"])
        oracle = CoverageOracle(task)
        def forbidden(*_):
            raise AssertionError("A shared-head singleton must use its relation projection")
        oracle.has_binding = forbidden
        self.assertEqual(oracle.atomic_coverage(task.heads[0], 0), 3)

    def test_partial_indexed_answers_do_not_refute_unqueried_keys_of_another_head(self):
        task = optimized_fixtures.OptimizedOracleContractTests.task(
            self, "p(a). p(c). p(d). p(e).", ["p(var__(1))"],
            ["first(a)", "first(x)", "second(c)", "second(y)"],
            heads=("first(var__(1))", "second(var__(1))"), max_body=1)
        oracle = self.compare(task)
        self.assertTrue(any(kind == "answers" for kind, _ in oracle._atomic_projections.values()))

    def test_deadline_is_checked_before_reusing_a_projection(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"])
        oracle = self.compare(task)
        def expired():
            raise TimeoutError("expired")
        oracle.check = expired
        with self.assertRaises(TimeoutError):
            oracle.atomic_coverage(task.heads[0], 0)


if __name__ == "__main__":
    unittest.main()
