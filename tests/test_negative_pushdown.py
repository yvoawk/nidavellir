"""Early negative-head filtering preserves full conjunctive witnesses."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import clingo

from nidavellir.labels import AtomRecord, CompactNegativeSet
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import instantiate
from tests import test_relational_coverage as fixtures


class NegativePushdownTests(unittest.TestCase):
    task = fixtures.RelationalCoverageTests.task
    check_all = fixtures.RelationalCoverageTests.check_all

    def test_nonnegative_heads_skip_hidden_witness_products_without_outer_symbols(self):
        task = self.task("p(a). p(unlabelled). q(1..20). r(1..20).",
                         ["p(var__(1))", "q(var__(2))", "r(var__(3))"],
                         ["answer(a)"], ["answer(absent)"])
        oracle = CoverageOracle(task, negative_join_backend="bindings")
        body = tuple(range(3))
        original_function = clingo.Function
        def forbid_target(name, *args, **kwargs):
            self.assertNotEqual(name, "answer")
            return original_function(name, *args, **kwargs)
        with patch.object(AtomRecord, "to_symbol", side_effect=AssertionError("outer target Symbol")), \
             patch("nidavellir.task.clingo.Function", side_effect=forbid_target):
            self.assertIsNone(oracle.negative(task.heads[0], body))
        self.assertEqual(oracle.stats["matched_atoms"], 2)
        self.assertEqual(oracle.stats["negative_head_prunes"], 2)
        # The public join and prediction APIs still include every hidden
        # witness and unlabelled target; only the negative query is filtered.
        self.assertEqual(len(list(oracle.bindings(body))), 2 * 20 * 20)
        self.assertEqual(oracle.predictions(task.heads[0], body), frozenset(map(
            clingo.parse_term, ["answer(a)", "answer(unlabelled)"])))

    def test_retained_negative_has_a_complete_real_witness_and_truth_profile(self):
        task = self.task("p(a). p(b). q(1..4). r(2). extra(z).",
                         ["p(var__(1))", "q(var__(2))", "r(var__(2))", "extra(var__(3))"],
                         ["answer(a)"], ["answer(b)"])
        head = task.heads[0]
        body = tuple(i for i, literal in enumerate(task.literals) if literal.name != "extra")
        oracle = CoverageOracle(task)
        witness = oracle.negative(head, body)
        self.assertIsNotNone(witness)
        self.assertEqual(witness["head"], clingo.parse_term("answer(b)"))
        self.assertEqual(set(witness["binding"]), {1, 2, 3})
        self.assertTrue(all(instantiate(task.literals[i], witness["binding"]) in task.atoms for i in body))
        truth = sum(1 << i for i, literal in enumerate(task.literals)
                    if instantiate(literal, witness["binding"]) in task.atoms)
        self.assertEqual(witness["truth"], truth)
        self.assertEqual(instantiate(head, witness["binding"]), witness["head"])

    def test_fixed_negative_head_does_not_bypass_incompatible_hidden_join(self):
        task = self.task("p(a). q(x). r(y).",
                         ["p(var__(1))", "q(var__(2))", "r(var__(2))"], [], ["answer(a)"])
        oracle = CoverageOracle(task, negative_join_backend="bindings")
        self.assertIsNone(oracle.negative(task.heads[0], tuple(range(3))))
        self.assertGreater(oracle.stats["negative_head_checks"], 0)
        self.assertEqual(oracle.stats["negative_head_prunes"], 0)

    def test_holes_and_nested_repeated_head_terms_keep_exact_label_membership(self):
        task = self.task("pair(a,a). pair(b,b). hidden(1..10).",
                         ["pair(var__(1),var__(2))", "hidden(var__(3))"],
                         ["answer(pair(a,a),a)"],
                         ["answer(pair(a,b),a)", "answer(pair(b,a),b)"],
                         head="answer(pair(var__(1),var__(2)),var__(1))")
        oracle = self.check_all(task)
        # The historical check counts head-membership pruning in BK traversal.
        # Label-table pruning is covered separately by test_negative_label_join.
        oracle = CoverageOracle(task, negative_join_backend="bindings")
        self.assertIsNone(oracle.negative(task.heads[0], tuple(range(2))))
        self.assertGreater(oracle.stats["negative_head_prunes"], 0)
        self.check_all(self.task("p(a).", ["p(var__(1))"], [], ["answer(a,b)"],
                                 head="answer(var__(1),var__(1))"))

    def test_partial_head_binding_is_not_filtered_before_other_slots_are_bound(self):
        task = self.task("p(a). q(b). q(c).", ["p(var__(1))", "q(var__(2))"],
                         [], ["answer(a,b)"], head="answer(var__(1),var__(2))")
        oracle = self.check_all(task)
        witness = oracle.negative(task.heads[0], tuple(range(2)))
        self.assertEqual(witness["head"], clingo.parse_term("answer(a,b)"))

    def test_ordinary_symbol_sets_and_ground_nullary_heads_remain_supported(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"], ["answer(a)"], ["answer(b)"])
        ordinary = replace(task, negatives=frozenset(task.negatives))
        self.check_all(ordinary)
        self.check_all(self.task("p(a).", ["p(var__(1))"], [], ["failure"], head="failure"))
        safe = self.task("q(1..10).", ["q(var__(1))"], ["success"], head="success")
        oracle = CoverageOracle(safe)
        self.assertIsNone(oracle.negative(safe.heads[0], (0,)))
        self.assertEqual(oracle.stats["matched_atoms"], 0)

    def test_interruption_on_final_rejected_head_does_not_publish_safety(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"], ["answer(b)"])
        expired = False
        def check():
            if expired:
                raise TimeoutError("pushdown deadline")
        oracle = CoverageOracle(task, check, negative_join_backend="bindings")
        original = CompactNegativeSet.__contains__
        def interrupted_membership(labels, record):
            nonlocal expired
            result = original(labels, record)
            expired = True
            return result
        with patch.object(CompactNegativeSet, "__contains__", interrupted_membership):
            with self.assertRaises(TimeoutError):
                oracle.negative(task.heads[0], (0,))
        self.assertFalse(oracle._safe)
        self.assertFalse(oracle._negative_witnesses)
        expired = False
        self.assertIsNone(oracle.negative(task.heads[0], (0,)))
        expired = True
        with self.assertRaises(TimeoutError):
            oracle.negative(task.heads[0], (0,))

    def test_expired_cached_witness_is_not_returned(self):
        task = self.task("p(a).", ["p(var__(1))"], [], ["answer(a)"])
        oracle = CoverageOracle(task)
        self.assertIsNotNone(oracle.negative(task.heads[0], (0,)))
        def expired():
            raise TimeoutError("cached witness deadline")
        oracle.check = expired
        with self.assertRaises(TimeoutError):
            oracle.negative(task.heads[0], (0,))


if __name__ == "__main__":
    unittest.main()
