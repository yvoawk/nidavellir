"""Bound-aware pruning preserves optimal programs evaluated independently."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import compile_catalogue, mine_variants
from nidavellir.relational_coverage import CoverageOracle
from tests import test_cli_extended as cli_fixtures
from tests import test_optimized_oracle as fixtures
from tests.test_relational_compile import exhaustive_catalogue, normalized


def optimal_programs(signatures, full, bound):
    """Independent powerset reference; no production set-cover implementation."""
    rows = sorted(set(signatures))
    best, winners = bound + 1, set()
    for count in range(1, min(len(rows), bound) + 1):
        for selected in combinations(rows, count):
            cost = sum(cost for _, cost in selected)
            if cost > min(bound, best):
                continue
            coverage = 0
            for mask, _ in selected:
                coverage |= mask
            if coverage != full:
                continue
            if cost < best:
                best, winners = cost, set()
            winners.add(frozenset(selected))
    return (best if winners else None), winners


class ProgramBoundPruningTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def compare(self, task, bound):
        expected = exhaustive_catalogue(task, bound)
        full = (1 << len(task.positives)) - 1
        optimum = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        self.assertIsNotNone(optimum[0])
        last = None
        for order in ('source', 'selectivity'):
            ordinary, _ = compile_catalogue(task, CoverageOracle(task), bound=bound, order=order)
            self.assertEqual(normalized(ordinary), expected)
            for variants in ('exhaustive', 'targeted', 'representative'):
                with self.subTest(order=order, variants=variants):
                    rows, stats = compile_catalogue(task, CoverageOracle(task), bound=bound,
                        program_bound=True, order=order, variants=variants)
                    self.assertTrue(stats['complete'])
                    self.assertTrue(stats['program_bound'])
                    self.assertEqual(optimal_programs(((c['coverage'], c['cost']) for c in rows), full, bound), optimum)
                    last = rows, stats
        return expected, last

    def test_irrelevant_expensive_partial_signature_is_removed(self):
        task = self.task('p(a). p(b). p(d). q(c). q(d). '
            'x(b). x(c). x(n1). x(n2). y(b). y(c). y(n1). y(n3). '
            'z(b). z(c). z(n2). z(n3).',
            [f'{p}(var__(1))' for p in ('p', 'q', 'x', 'y', 'z')],
            ['answer(a)', 'answer(b)', 'answer(c)', 'answer(d)'],
            ['answer(n1)', 'answer(n2)', 'answer(n3)'])
        expected, (rows, stats) = self.compare(task, 4)
        self.assertTrue(any(cost == 4 for _, cost, _ in expected))
        self.assertFalse(any(c['cost'] == 4 for c in rows))
        self.assertEqual(stats['minimum_useful_rule_cost'], 2)
        self.assertGreater(stats['pruned_program_bound'], 0)

    def test_equality_retains_two_rules_of_cost_two_and_three(self):
        task = self.task('p(a). q(b). q(n1). r(b). r(n2).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['answer(a)', 'answer(b)'], ['answer(n1)', 'answer(n2)'])
        _, (rows, _) = self.compare(task, 5)
        self.assertEqual(sorted(c['cost'] for c in rows), [2, 3])

    def test_full_coverage_rule_is_not_charged_an_unnecessary_second_rule(self):
        task = self.task('p(a). p(b). p(n1). q(a). q(b). q(n2).',
            ['p(var__(1))', 'q(var__(1))'],
            ['answer(a)', 'answer(b)'], ['answer(n1)', 'answer(n2)'])
        _, (rows, _) = self.compare(task, 3)
        self.assertEqual([(c['cost'], c['coverage']) for c in rows], [(3, 3)])

    def test_legal_ground_empty_rule_requires_lower_bound_one(self):
        task = self.task('p(a). p(n1). q(a). q(n2).',
            ['p(var__(1))', 'q(var__(1))'], ['answer(a)', 'flag'],
            ['answer(n1)', 'answer(n2)'], heads=('answer(var__(1))', 'flag'))
        _, (rows, stats) = self.compare(task, 4)
        self.assertEqual(stats['minimum_useful_rule_cost'], 1)
        self.assertEqual(sorted(c['cost'] for c in rows), [1, 3])

    def test_multiple_heads_use_global_coverage_and_reconstruct_selected_variants(self):
        task = self.task('p(a). q(a).', ['p(var__(1))', 'q(var__(1))'],
            ['first(a)', 'second(a)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'second(var__(1))'))
        _, (rows, stats) = self.compare(task, 4)
        self.assertEqual(stats['minimum_useful_rule_cost'], 2)
        rebuilt, reconstruction = mine_variants(task, CoverageOracle(task), rows)
        self.assertTrue(reconstruction['complete'])
        self.assertEqual([c['id'] for c in rebuilt], [c['id'] for c in rows])
        self.assertTrue(all(len(c['variants']) == 2 for c in rebuilt))

    def test_guards_are_counted_in_actual_rule_cost(self):
        task = self.task('p(a). q(a). r(b).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['answer(a)', 'answer(b)'], ['answer(n)'],
            bias=':- body(p(var__(1))), not body(q(var__(1))).\n'
                 ':- body(q(var__(1))), not body(p(var__(1))).')
        _, (rows, _) = self.compare(task, 5)
        self.assertEqual(sorted(c['cost'] for c in rows), [2, 3])

    def test_invalid_bound_and_deadline_do_not_produce_a_complete_catalogue(self):
        task = self.task('p(a).', ['p(var__(1))'], ['answer(a)'], ['answer(n)'])
        for invalid in (None, 0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                compile_catalogue(task, CoverageOracle(task), bound=invalid, program_bound=True)
        with self.assertRaises(ResourceLimit):
            compile_catalogue(task, CoverageOracle(task), bound=2, program_bound=True, seconds=0)

    def test_cli_activates_filter_only_with_verified_positive_cost_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, extra, enabled in [('bound', (), True), ('unbounded', ('--bound-seconds', '0'), False)]:
                process, output, status = cli_fixtures.ExtendedCLITests.invoke(
                    self, folder, name, '--compiler-backend', 'relational', *extra)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                self.assertEqual(status['optimal_cost'], 4)
                stats = json.loads((output / 'catalogue.json').read_text())['stats']
                self.assertEqual(stats['program_bound'], enabled)


if __name__ == '__main__':
    unittest.main()
