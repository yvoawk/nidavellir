"""Negative sampling changes order only; exact small catalogues are preserved."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import clingo

from nidavellir.bias_analysis import analyze
from nidavellir.labels import AtomRecord
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.relational_order import order_literals
from tests import test_optimized_oracle as fixtures


term = clingo.parse_term


class RelationalOrderTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def prepared(self, task):
        oracle = CoverageOracle(task)
        analysis = analyze(task)
        head = task.heads[0]
        masks = tuple(oracle.atomic_coverage(head, i) for i in range(len(task.literals)))
        return oracle, head, analysis.heads[head], masks

    def test_equal_positive_retention_distinguished_by_negative_rejection(self):
        task = self.task("a_bad(p). a_bad(n1). a_bad(n2). z_good(p).",
                         ["a_bad(var__(1))", "z_good(var__(1))"],
                         ["answer(p)"], ["answer(n1)", "answer(n2)"])
        oracle, head, proof, masks = self.prepared(task)
        order, stats = order_literals(task, oracle, head, proof, masks, lambda: None)
        names = [task.literals[i].name for i in order]
        self.assertEqual(names, ["z_good", "a_bad"])
        self.assertEqual(stats["positive_retention"], [1, 1])
        self.assertEqual(stats["negative_scores"], [0, 1])
        self.assertEqual(stats["completed_literals"], 2)
        self.assertEqual(stats["sample_size"], 2)
        self.assertTrue(stats["population_known"])
        self.assertEqual(stats["compatible_negative_population"], 2)

    def test_guards_are_queried_jointly_and_cyclic_closures_share_answers(self):
        task = self.task("p(a,x). q(a,x). p(n,x). q(n,y).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))"],
                         ["answer(a)"], ["answer(n)"], max_body=2,
                         bias=":- body(p(var__(1),var__(2))), not body(q(var__(1),var__(2))).\n"
                              ":- body(q(var__(1),var__(2))), not body(p(var__(1),var__(2))).")
        oracle, head, proof, masks = self.prepared(task)
        self.assertTrue(oracle.exists(head, (0,), term("answer(n)")))
        self.assertTrue(oracle.exists(head, (1,), term("answer(n)")))
        _, stats = order_literals(task, oracle, head, proof, masks, lambda: None)
        self.assertEqual(stats["negative_scores"], [1, 1])
        self.assertEqual(stats["negative_queries"], 1)
        self.assertEqual(stats["completed_literals"], 2)

    def test_proved_groups_and_guard_dependencies_are_respected(self):
        task = SimpleNamespace(literals=(term("p"), term("q"), term("r")),
                               negatives=(), positives=())
        proof = SimpleNamespace(guards=(frozenset(), frozenset(), frozenset({0})),
                                prefix_filter=SimpleNamespace(bounds=((3, 0, 0, 1),)))
        oracle = SimpleNamespace(positives={}, check=lambda: None)
        order, stats = order_literals(task, oracle, term("answer"), proof, (0, 0, 0), lambda: None)
        self.assertEqual(order, (0, 1, 2))
        self.assertEqual(stats["head_groups"], 2)
        self.assertEqual(stats["negative_scores"], [None] * 3)

    def test_record_stream_converts_only_sampled_compatible_outer_symbols(self):
        class Records:
            def __len__(self):
                return 10_000_000
            def __iter__(self):
                raise AssertionError("Do not materialize negative outer symbols")
            def iter_records(self):
                for i in range(4):
                    yield AtomRecord("other", (clingo.Number(i),))
                for i in range(10_000_000):
                    yield AtomRecord("answer", (clingo.Number(i),))
        task = SimpleNamespace(literals=(term("p(var__(1))"),), negatives=Records())
        proof = SimpleNamespace(guards=(frozenset(),), prefix_filter=None)
        oracle = SimpleNamespace(positives={term("answer(p)"): 0}, check=lambda: None,
                                 exists=lambda head, body, label: False)
        converted = []
        original = AtomRecord.to_symbol
        def to_symbol(record):
            converted.append(record)
            return original(record)
        with patch.object(AtomRecord, "to_symbol", to_symbol):
            _, stats = order_literals(task, oracle, term("answer(var__(1))"), proof, (1,),
                                      lambda: None, sample_size=2, max_records=8)
        self.assertEqual(stats["scanned_records"], 6)
        self.assertEqual(stats["sample_size"], 2)
        self.assertEqual(len(converted), 2)
        self.assertEqual(stats["negative_queries"], 2)
        self.assertFalse(stats["population_known"])
        self.assertIsNone(stats["compatible_negative_population"])

    def test_sparse_compatibility_scan_is_bounded_without_symbol_iteration(self):
        class Records:
            def __len__(self): return 100_000
            def __iter__(self): raise AssertionError("Symbol iteration forbidden")
            def iter_records(self):
                for i in range(100_000):
                    yield AtomRecord("other", (clingo.Number(i),))
        task = SimpleNamespace(literals=(term("p(var__(1))"),), negatives=Records())
        proof = SimpleNamespace(guards=(frozenset(),), prefix_filter=None)
        oracle = SimpleNamespace(positives={term("answer(p)"): 0}, check=lambda: None)
        _, stats = order_literals(task, oracle, term("answer(var__(1))"), proof, (1,),
                                  lambda: None, max_records=7)
        self.assertEqual(stats["scanned_records"], 7)
        self.assertTrue(stats["scan_limit_reached"])
        self.assertEqual(stats["sample_size"], 0)
        self.assertEqual(stats["negative_scores"], [None])
        self.assertFalse(stats["population_known"])

    def test_local_timeout_keeps_only_completed_scores_and_restores_check(self):
        clock = [0.0]
        task = SimpleNamespace(literals=(term("p(var__(1))"), term("q(var__(1))")),
                               negatives=(term("answer(n1)"), term("answer(n2)")))
        proof = SimpleNamespace(guards=(frozenset(), frozenset()), prefix_filter=None)
        original = lambda: None
        oracle = SimpleNamespace(positives={term("answer(p)"): 0}, check=original)
        calls = []
        def exists(head, body, label):
            calls.append((body, label))
            if body == (1,):
                clock[0] = 5
                oracle.check()
            return False
        oracle.exists = exists
        with patch("nidavellir.relational_order.time.monotonic", lambda: clock[0]):
            order, stats = order_literals(task, oracle, term("answer(var__(1))"), proof, (1, 1),
                                          lambda: None, seconds=3)
        self.assertIs(oracle.check, original)
        self.assertTrue(stats["local_timeout"])
        self.assertEqual(stats["completed_literal_indices"], [0])
        self.assertEqual(stats["negative_scores"], [1, None])
        self.assertEqual(stats["scores"], [1, 0])
        self.assertEqual(stats["negative_queries"], 3)
        self.assertEqual(stats["completed_negative_queries"], 2)
        self.assertEqual(set(order), {0, 1})

    def test_zero_local_budget_is_positive_fallback_but_global_expiry_propagates(self):
        task = self.task("p(a). q(a). q(b).", ["p(var__(1))", "q(var__(1))"],
                         ["answer(a)", "answer(b)"], ["answer(n)"])
        oracle, head, proof, masks = self.prepared(task)
        original = oracle.check
        order, stats = order_literals(task, oracle, head, proof, masks, lambda: None, seconds=0)
        self.assertEqual(order, (0, 1))
        self.assertTrue(stats["local_timeout"])
        self.assertEqual(stats["completed_literals"], 0)
        self.assertEqual(stats["negative_scores"], [None, None])
        def global_expired():
            raise ResourceLimit("global deadline")
        with self.assertRaisesRegex(ResourceLimit, "global deadline"):
            order_literals(task, oracle, head, proof, masks, global_expired, seconds=0)
        self.assertIs(oracle.check, original)
        oracle.check = global_expired
        with self.assertRaisesRegex(ResourceLimit, "global deadline"):
            order_literals(task, oracle, head, proof, masks, lambda: None)
        self.assertIs(oracle.check, global_expired)

    def test_multiple_heads_constants_and_repeated_slots_keep_global_positions(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"],
                         ["answer(a,a)", "answer(b,c)", "other(a)"],
                         ["answer(n,n)", "answer(n,m)", "other(n)"],
                         heads=("answer(var__(1),var__(1))", "other(var__(1))"))
        oracle = CoverageOracle(task)
        analysis = analyze(task)
        head = next(h for h in task.heads if h.name == "answer")
        masks = (oracle.atomic_coverage(head, 0),)
        _, stats = order_literals(task, oracle, head, analysis.heads[head], masks, lambda: None)
        self.assertEqual(stats["positive_population"], 1)
        self.assertEqual(stats["positive_retention"], [1])
        self.assertEqual(stats["negative_population"], 3)
        self.assertEqual(stats["compatible_negative_population"], 1)
        self.assertEqual(stats["sample_size"], 1)

    def test_sampled_order_and_source_order_preserve_exhaustive_catalogues(self):
        from nidavellir.relational_compile import compile_catalogue
        from tests.test_relational_compile import exhaustive_catalogue, normalized
        task = self.task("p(a,x). p(b,x). p(n,x). q(a,y). q(b,x). r(a). r(n).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))", "r(var__(1))"],
                         ["answer(a)", "answer(b)"], ["answer(n)"], max_body=3)
        expected = exhaustive_catalogue(task)
        source, source_stats = compile_catalogue(task, CoverageOracle(task), order="source", seconds=10)
        self.assertTrue(source_stats["complete"])
        self.assertEqual(source_stats["negative_ordering"], {})
        self.assertEqual(normalized(source), expected)
        result, stats = compile_catalogue(task, CoverageOracle(task), order="selectivity", seconds=10)
        self.assertTrue(stats["complete"])
        self.assertTrue(stats["negative_ordering"])
        self.assertEqual(normalized(result), expected)

    def test_bad_budgets_or_masks_fail_explicitly(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"], ["answer(n)"])
        oracle, head, proof, masks = self.prepared(task)
        for kwargs in ({"seconds": -1}, {"seconds": float("inf")}, {"sample_size": True},
                       {"sample_size": -1}, {"max_records": -1}):
            with self.assertRaises(ValueError):
                order_literals(task, oracle, head, proof, masks, lambda: None, **kwargs)
        for bad_masks in ((), (-1,), (True,)):
            with self.assertRaises(ValueError):
                order_literals(task, oracle, head, proof, bad_masks, lambda: None)


if __name__ == "__main__":
    unittest.main()
