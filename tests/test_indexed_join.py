"""Integer joins and pair pruning independently compared with complete Clingo."""
from itertools import combinations
import unittest

import clingo

from nidavellir.indexed_join import JoinRelation, matched_relation, incompatible_pairs
from nidavellir.oracle import ResourceLimit
from nidavellir.positive_join import has_positive
from nidavellir.relational_coverage import CoverageOracle
from nidavellir import relational_compile
from nidavellir.task import variables
from tests import test_positive_join as fixtures
from tests.test_relational_coverage import unique_model


class IndexedJoinTests(unittest.TestCase):
    def task(self, *args, **kwargs):
        return fixtures.PositiveJoinTests.task(self, *args, **kwargs)

    def compare_all(self, task):
        oracles = [CoverageOracle(task, positive_join_backend=backend, positive_pair_pruning=pairs)
                   for backend, pairs in (("reference", False), ("indexed", False), ("indexed", True))]
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(head) <= used:
                        continue
                    model = unique_model(task.background + "\n" + task.rule(head, body))
                    for possible in range(1 << len(task.positives)):
                        expected = any(label in model and possible >> position & 1
                                       for label, position in oracles[0].positives.items())
                        for oracle in oracles:
                            self.assertEqual(has_positive(oracle, head, body, possible), expected,
                                             (head, body, possible, oracle.positive_join_backend))
        catalogues = [relational_compile.compile_catalogue(task, o)[0] for o in oracles]
        norm = lambda rows: {(r["cost"], r["coverage"], tuple(r["variants"])) for r in rows}
        self.assertTrue(all(norm(c) == norm(catalogues[0]) for c in catalogues))

    def test_pairwise_compatible_odd_cycle_still_requires_a_full_witness(self):
        task = self.task("edge(0,1). edge(1,0).", ["edge(var__(1),var__(2))",
            "edge(var__(2),var__(3))", "edge(var__(3),var__(1))"], ["answer(0)", "answer(1)"])
        self.compare_all(task)
        oracle = CoverageOracle(task)
        self.assertFalse(incompatible_pairs(oracle.space, (0, 1, 2)))
        self.assertFalse(has_positive(oracle, task.heads[0], (0, 1, 2), 3))

    def test_pair_proof_keeps_common_column_correlations_and_orientation(self):
        task = self.task("r(a,x). r(b,y). s(a,y). s(b,x).",
            ["r(var__(1),var__(2))", "s(var__(1),var__(2))", "s(var__(2),var__(1))"],
            ["answer(a)", "answer(b)"])
        self.compare_all(task)
        oracle = CoverageOracle(task)
        body = tuple(i for i, l in enumerate(task.literals) if str(l) != "s(var__(2),var__(1))")
        self.assertTrue(incompatible_pairs(oracle.space, body))

    def test_nested_constants_repetition_and_global_positive_positions(self):
        self.compare_all(self.task("r(tag,pair(a,x),x). r(tag,pair(b,y),y). r(tag,pair(a,x),y). q(x).",
            ["r(tag,pair(var__(1),var__(2)),var__(2))", "r(tag,pair(var__(1),var__(3)),var__(2))",
             "q(var__(2))"], ["alpha(a)", "omega(pair(a,x),a)", "omega(pair(b,y),b)"],
            ["alpha(var__(1))", "omega(pair(var__(1),var__(2)),var__(1))"]))

    def test_all_renamings_share_one_complete_table(self):
        task = self.task("p(a,x). p(b,y).", [f"p(var__(1),var__({i}))" for i in range(2, 80)],
                         ["answer(a)"])
        oracle = CoverageOracle(task)
        tables = [matched_relation(oracle.space, i)[1] for i in range(len(task.literals))]
        self.assertTrue(all(t is tables[0] for t in tables))
        self.assertEqual(len(oracle.space._integer_relations), 1)
        self.assertEqual(oracle.space.stats["integer_relation_rows"], 2)

    def test_interrupted_relation_index_and_projection_are_never_published(self):
        task = self.task("\n".join(f"p({i},{i % 7}) ." for i in range(600)),
                         ["p(var__(1),var__(2))"], ["answer(0)"])
        oracle = CoverageOracle(task)
        count = 0
        def deadline():
            nonlocal count
            count += 1
            if count >= 3:
                raise ResourceLimit("construction deadline")
        oracle.check = deadline
        with self.assertRaises(ResourceLimit):
            matched_relation(oracle.space, 0)
        self.assertFalse(oracle.space._integer_relations)
        self.assertFalse(oracle.space._integer_templates)
        oracle.check = lambda: None
        _, table = matched_relation(oracle.space, 0)
        for operation, cache in ((lambda: table.select(((0, 0),), {}), table.indexes),
                                 (lambda: table.project((1,)), table.projections)):
            count = 0
            oracle.check = deadline
            with self.assertRaises(ResourceLimit):
                operation()
            self.assertFalse(cache)
        oracle.check = lambda: None
        self.assertEqual(len(table.project((1,))), 7)

    def test_joint_filter_cache_is_bounded_by_rows_and_entries(self):
        table = JoinRelation([(0, x, y) for x in range(40) for y in range(300)], lambda: None)
        for x in range(40):
            self.assertEqual(len(table.select(((0, 0), (1, x)), {})), 300)
        self.assertLessEqual(table.filter_rows, 8192)
        self.assertEqual(table.filter_rows, sum(map(len, table.filters.values())))
        self.assertLess(len(table.filters), 40)

    def test_expired_final_relation_check_leaves_no_cached_answer(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"])
        oracle = CoverageOracle(task)
        _, table = matched_relation(oracle.space, 0)
        oracle.check = lambda: (_ for _ in ()).throw(ResourceLimit("deadline"))
        with self.assertRaises(ResourceLimit):
            has_positive(oracle, task.heads[0], (0,), 1)
        self.assertFalse(getattr(oracle, "_positive_join_results", {}))
