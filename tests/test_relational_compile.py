"""Exact relational catalogues versus independent exhaustive legal-rule search."""

from itertools import combinations
import unittest

from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import compile_catalogue, mine_variants
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.search import compile_catalogue as bitmask_compile
from tests import test_optimized_oracle as fixtures

model_of = fixtures.model_of


def normalized(catalogue):
    return {(row["coverage"], row["cost"], tuple(sorted(row["variants"]))) for row in catalogue}


def exhaustive_catalogue(task, bound=None):
    classes = {}
    positive_order = sorted(task.positives)
    maximum = min(task.max_body, bound - 1) if bound is not None else task.max_body
    for head in task.heads:
        for size in range(maximum + 1):
            for body in combinations(range(len(task.literals)), size):
                if not task.accepts(head, body):
                    continue
                rule = task.rule(head, body)
                predictions = model_of(task.background + "\n" + rule)
                coverage = sum(1 << i for i, atom in enumerate(positive_order) if atom in predictions)
                if not coverage or predictions & task.negatives:
                    continue
                cost = 1 + size
                if coverage not in classes or cost < classes[coverage][0]:
                    classes[coverage] = cost, {rule}
                elif cost == classes[coverage][0]:
                    classes[coverage][1].add(rule)
    return {(coverage, cost, tuple(sorted(rules))) for coverage, (cost, rules) in classes.items()
            if not any(other_cost < cost and other | coverage == other
                       for other, (other_cost, _) in classes.items())}


class RelationalCompilationTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def compare(self, task, bound=None):
        expected = exhaustive_catalogue(task, bound)
        indices, _ = build_optimized_indices(task, seconds=10)
        reference, _ = bitmask_compile(task, indices, seconds=10, bound=bound)
        self.assertEqual(normalized(reference), expected)
        for order in ("source", "selectivity"):
            for variant_mode in ("exhaustive", "targeted"):
                with self.subTest(order=order, variants=variant_mode):
                    result, stats = compile_catalogue(task, CoverageOracle(task), seconds=10, bound=bound,
                                                      order=order, variants=variant_mode)
                    self.assertTrue(stats["complete"])
                    self.assertEqual(normalized(result), expected)
        return reference

    def test_incompatible_positive_witnesses_do_not_create_safe_conjunction(self):
        self.compare(self.task("p(a,x). q(a,y). p(b,x). q(b,x).",
                               ["p(var__(1),var__(2))", "q(var__(1),var__(2))"],
                               ["answer(a)"], ["answer(b)"], max_body=2))

    def test_cyclic_guards_and_disjunctive_legality(self):
        self.compare(self.task("p(a). q(a). r(b). s(a). s(b).",
                               ["p(var__(1))", "q(var__(1))", "r(var__(1))", "s(var__(1))"],
                               ["answer(a)"], ["answer(b)"], max_body=3,
                               bias=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                    ":- body(q(var__(1))), not body(p(var__(1))).\n"
                                    ":- not body(p(var__(1))), not body(r(var__(1)))."))

    def test_equal_masks_do_not_remove_structurally_required_literals(self):
        self.compare(self.task("p(a). q(a).", ["p(var__(1))", "q(var__(1))"],
                               ["answer(a)"], ["answer(b)"], max_body=2,
                               bias=":- not body(p(var__(1))).\n:- not body(q(var__(1)))."))

    def test_overlapping_rules_equal_cost_dominance_and_syntactic_ties(self):
        self.compare(self.task("p(a). p(b). q(b). q(c). r(a). r(c). s(a). s(b).",
                               ["p(var__(1))", "q(var__(1))", "r(var__(1))", "s(var__(1))"],
                               ["answer(a)", "answer(b)", "answer(c)"], ["answer(n)"], max_body=3))
        self.compare(self.task("p(a). q(a). q(b).", ["p(var__(1))", "q(var__(1))"],
                               ["answer(a)", "answer(b)"], max_body=2))

    def test_multiple_heads_and_bound_keep_equal_cost_variants(self):
        task = self.task("p(a). p(b). q(a).", ["p(var__(1))", "q(var__(1))"],
                         ["first(a)", "second(b)"], ["first(b)", "second(a)"],
                         heads=("first(var__(1))", "second(var__(1))"), max_body=2)
        self.compare(task, bound=3)

    def test_ground_head_empty_body_and_no_positive_examples(self):
        self.compare(self.task("p(a).", ["p(var__(1))"], ["success"], heads=("success",), max_body=1))
        task = self.task("p(a).", ["p(var__(1))"], [], ["answer(a)"], max_body=1)
        result, stats = compile_catalogue(task, CoverageOracle(task))
        self.assertEqual(result, [])
        self.assertTrue(stats["complete"])

    def test_targeted_variants_after_selection_preserve_candidate_ids(self):
        task = self.task("p(a). q(a). r(b).", ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         ["answer(a)", "answer(b)"], max_body=2)
        full, _ = compile_catalogue(task, CoverageOracle(task))
        representatives, _ = compile_catalogue(task, CoverageOracle(task), variants="representative")
        chosen = [dict(row, id="chosen:" + row["id"]) for row in representatives]
        reconstructed, stats = mine_variants(task, CoverageOracle(task), chosen)
        self.assertTrue(stats["complete"])
        self.assertEqual(normalized(reconstructed), normalized(full))
        self.assertEqual([row["id"] for row in reconstructed], [row["id"] for row in chosen])

    def test_resource_limits_never_return_an_incomplete_catalogue(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"], max_body=1)
        oracle = CoverageOracle(task)
        original = oracle.check
        with self.assertRaises(ResourceLimit):
            compile_catalogue(task, oracle, seconds=0)
        self.assertIs(oracle.check, original)

    def test_positive_coverage_does_not_enumerate_every_rule_output(self):
        task = self.task("p(a). p(b). p(unlabelled). q(a).",
                         ["p(var__(1))", "q(var__(1))"], ["answer(a)"], ["answer(b)"], max_body=2)
        oracle = CoverageOracle(task)

        def all_outputs_forbidden(*args):
            raise AssertionError("Coverage must use focused positive-label queries")

        oracle.coverage = all_outputs_forbidden
        oracle.predictions = all_outputs_forbidden
        result, stats = compile_catalogue(task, oracle)
        self.assertTrue(stats["complete"])
        self.assertEqual(normalized(result), exhaustive_catalogue(task))


if __name__ == "__main__":
    unittest.main()
