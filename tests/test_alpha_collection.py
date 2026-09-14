"""Early alpha collection against exhaustive legal Clingo catalogues."""
from itertools import combinations, permutations
import unittest
from unittest.mock import patch

import clingo

from nidavellir.alpha_collection import VariantBucket
from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.relational_coverage import CoverageOracle
from nidavellir import relational_compile, search
from nidavellir.solvers import solve
from tests import test_search_optimizations as fixtures


class AlphaCollectionTests(unittest.TestCase):
    def task(self, *args, **kwargs):
        return fixtures.SearchOptimizationTests.task(self, *args, **kwargs)

    def compare(self, task):
        indices, _ = build_cartesian_indices(task)
        outputs = []
        for engine, oracle in ((search, indices), (relational_compile, CoverageOracle(task))):
            for mode in ("exhaustive", "targeted"):
                raw, _ = engine.compile_catalogue(task, oracle, variants=mode, alpha_collect=False)
                compact, stats = engine.compile_catalogue(task, oracle, variants=mode, alpha_collect=True)
                normalized = lambda rows: {(r["cost"], r["coverage"]):
                    {min(group) for group in r["alpha_variants"]} for r in rows}
                self.assertEqual(normalized(raw), normalized(compact))
                self.assertEqual({(r["cost"], r["coverage"]) for r in compact},
                                 set(fixtures.SearchOptimizationTests.expected(self, task)))
                for row in compact:
                    for rule in row["variants"]:
                        predictions = fixtures.atoms(task.background + "\n" + rule)
                        self.assertFalse(predictions & task.negatives)
                        self.assertEqual(sum(1 << i for i, p in enumerate(sorted(task.positives))
                                             if p in predictions), row["coverage"])
                opt = solve(compact, len(task.positives), enumerate_optima=True, expand_variants=True)
                ref = solve(raw, len(task.positives), enumerate_optima=True, expand_variants=True)
                self.assertEqual(opt["optimum_cost"], ref["optimum_cost"])
                self.assertEqual(opt["variant_expansion"]["programs"], ref["variant_expansion"]["programs"])
                self.assertTrue(stats["alpha_collection"]["enabled"])
                recovered, recovery = engine.mine_variants(task, oracle, compact, alpha_collect=True)
                self.assertEqual(normalized(recovered), normalized(raw))
                self.assertEqual([r["id"] for r in recovered], [r["id"] for r in compact])
                outputs.append(compact)
        return outputs[0]

    def test_exchangeable_hidden_slots_and_multiple_rules(self):
        task = self.task("p(a,x). q(b,y).", ["p(var__(1),var__(2))", "p(var__(1),var__(3))",
            "q(var__(1),var__(2))", "q(var__(1),var__(3))"], max_body=2)
        rows = self.compare(task)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(r["variants"]) == 1 for r in rows))
        self.assertTrue(all(r["alpha_collection"]["duplicates_avoided"] > 0 for r in rows))

    def test_oriented_slots_with_asymmetric_constraints_never_invent_illegal_spelling(self):
        task = self.task("edge(a,x). edge(b,x). edge(c,y). good(x).",
            ["edge(var__(1),var__(2))", "edge(var__(1),var__(3))", "good(var__(2))", "good(var__(3))"],
            constraints=":- body(good(var__(2))).", max_body=2)
        rows = self.compare(task)
        self.assertTrue(all("good(V3)" in r["rule"] for r in rows))

    def test_guards_cycles_and_structurally_distinct_joins(self):
        task = self.task("edge(a,x). edge(b,x). edge(c,y). good(x). link(x,x).",
            ["edge(var__(1),var__(2))", "edge(var__(1),var__(3))", "good(var__(2))", "good(var__(3))",
             "link(var__(2),var__(3))", "link(var__(3),var__(2))"], max_body=3,
            constraints=":- body(good(var__(2))), not body(edge(var__(1),var__(2))).\n"
                        ":- body(edge(var__(1),var__(2))), not body(good(var__(2))).")
        self.compare(task)

    def test_constants_nested_and_repeated_variables_are_not_lost(self):
        task = self.task("edge(pair(a,x),x,ok). edge(pair(b,x),x,ok). edge(pair(c,x),y,ok).",
            ["edge(pair(var__(1),var__(2)),var__(2),ok)",
             "edge(pair(var__(1),var__(3)),var__(3),ok)",
             "edge(pair(var__(1),var__(2)),var__(3),ok)"], max_body=1)
        self.compare(task)

    def test_interrupted_key_or_legality_does_not_publish_half_a_class(self):
        task = self.task("p(a,x). p(b,x).", ["p(var__(1),var__(2))"])
        bucket = VariantBucket(task, lambda: None, task.accepts)
        row = dict(head=str(task.heads[0]), body=[0], rule=task.rule(task.heads[0], (0,)))
        with patch.object(search, "_alpha_key", side_effect=ResourceLimit("class deadline")):
            with self.assertRaises(ResourceLimit):
                bucket.append(row)
        self.assertEqual(bucket.export()["syntax_encounters"], 0)
        self.assertFalse(bucket.positions)
        bucket.append(row)
        bucket.check = lambda: (_ for _ in ()).throw(ResourceLimit("cached deadline"))
        with self.assertRaises(ResourceLimit):
            bucket.append(row)
        self.assertEqual(bucket.export()["syntax_encounters"], 1)

    def test_repeated_encounters_retain_only_one_row_and_one_key(self):
        task = self.task("p(a,x). p(b,x).", ["p(var__(1),var__(2))", "p(var__(1),var__(3))"])
        bucket = VariantBucket(task, lambda: None, task.accepts)
        for i in range(200):
            body = (i % 2,)
            bucket.append(dict(head=str(task.heads[0]), body=list(body), rule=task.rule(task.heads[0], body)))
        self.assertEqual((len(bucket), len(bucket.positions)), (1, 1))
        self.assertEqual(bucket.export()["duplicates_avoided"], 199)

    def test_canonicalization_deadline_precedes_factorial_materialization(self):
        task = self.task("p(a).", [f"p(var__({i}))" for i in range(1, 12)], max_body=11)
        with self.assertRaises(ResourceLimit):
            search._alpha_key(task, task.heads[0], tuple(range(11)),
                              lambda: (_ for _ in ()).throw(ResourceLimit("deadline")))

    def test_key_equivalence_matches_explicit_variable_bijections(self):
        task = self.task("p(a,a).", ["p(var__(1),var__(2))", "p(var__(2),var__(1))",
            "p(var__(1),var__(3))", "p(var__(3),var__(1))", "p(var__(2),var__(3))",
            "p(var__(3),var__(2))", "p(var__(2),var__(2))", "p(var__(3),var__(3))"], max_body=2)
        def rename(term, mapping):
            if term.type != clingo.SymbolType.Function:
                return term
            if term.name == 'var__':
                return clingo.Function('var__', [clingo.Number(mapping[term.arguments[0].number])])
            return clingo.Function(term.name, [rename(a, mapping) for a in term.arguments], term.positive)
        bodies = [b for n in (1, 2) for b in combinations(range(len(task.literals)), n)]
        for first in bodies:
            key = search._alpha_key(task, task.heads[0], first)
            orbit = set()
            for order in permutations((1, 2, 3)):
                mapping = dict(zip((1, 2, 3), order))
                if rename(task.heads[0], mapping) == task.heads[0]:
                    orbit.add(frozenset(rename(task.literals[i], mapping) for i in first))
            for second in bodies:
                expected = frozenset(task.literals[i] for i in second) in orbit
                self.assertEqual(key == search._alpha_key(task, task.heads[0], second), expected,
                                 (first, second))
