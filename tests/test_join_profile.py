"""Diagnostics are bounded, replayable and retain interrupted calls."""
import unittest
from unittest.mock import patch

from nidavellir.join_profile import JoinProfile
from nidavellir.positive_join import has_positive
from nidavellir.relational_coverage import CoverageOracle
from tests import test_positive_join as fixtures
from tests import test_search_optimizations as rule_fixtures


class JoinProfileTests(unittest.TestCase):
    def test_top_calls_exact_totals_and_memory_bound(self):
        task = fixtures.PositiveJoinTests.task(self, "p(a).", ["p(var__(1))"], ["answer(a)"])
        profile = JoinProfile(limit=3)
        for i in range(20):
            profile.record(task, task.heads[0], (0,), 1, i, "false",
                           dict(cache="miss", relations={}))
        report = profile.export()
        self.assertEqual(report["calls"], 20)
        self.assertEqual(report["seconds"], 190)
        self.assertEqual([r["seconds"] for r in report["slowest_calls"]], [19, 18, 17])
        self.assertEqual(report["retained_calls"], 3)

    def test_hits_misses_relation_sizes_and_interrupted_queries(self):
        task = fixtures.PositiveJoinTests.task(self, "p(a). p(b).", ["p(var__(1))"], ["answer(a)"])
        oracle = CoverageOracle(task)
        for _ in range(2):
            self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))
        with patch.object(oracle, "check", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                has_positive(oracle, task.heads[0], (0,), 1)
        report = oracle.profile.export()["diagnostics"]["positive_join"]
        self.assertEqual(report["calls"], 3)
        self.assertEqual(oracle.stats["positive_join_cache_hits"], 1)
        self.assertEqual(oracle.stats["positive_join_cache_misses"], 1)
        self.assertEqual(report["literals"]["0"]["max_rows"], 2)
        self.assertEqual({r["outcome"] for r in report["slowest_calls"]}, {"true", "interrupted"})
        self.assertTrue(all(int(r["possible_hex"], 16) == 1 for r in report["slowest_calls"]))

    def test_negative_profiles_keep_real_results_cache_hits_and_interrupted_bodies(self):
        task=rule_fixtures.SearchOptimizationTests.task(self,'p(a). p(c). q(a).',
            ['p(var__(1))','q(var__(1))'])
        oracle=CoverageOracle(task)
        p=next(i for i,l in enumerate(task.literals) if l.name=='p')
        q=next(i for i,l in enumerate(task.literals) if l.name=='q')
        self.assertIsNotNone(oracle.negative(task.heads[0],(p,)))
        self.assertIsNotNone(oracle.negative(task.heads[0],(p,)))
        self.assertIsNone(oracle.negative(task.heads[0],(q,)))
        with patch.object(oracle,'check',side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                oracle.negative(task.heads[0],(q,))
        report=oracle.profile.export()['diagnostics']['negative_join']
        self.assertEqual(report['calls'],4)
        self.assertEqual({r['outcome'] for r in report['slowest_calls']},{'safe','witness','interrupted'})
        self.assertTrue(any(r['cache']=='hit' for r in report['slowest_calls']))
        self.assertTrue(all(r['possible_hex'] is None for r in report['slowest_calls']))
