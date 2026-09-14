"""Correlated support projections are necessary filters, never full witnesses."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.indexed_join import matched_relation
from nidavellir.oracle import ResourceLimit
from nidavellir.positive_join import has_positive
from nidavellir.positive_support import filter_supported
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import variables
from tests import test_cli_extended as cli_fixtures
from tests import test_search_optimizations as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class PositiveLookaheadTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    def compare_all(self, task):
        oracles = [CoverageOracle(task, positive_lookahead=enabled, positive_join_backend=backend,
                                  positive_pair_pruning=pairs, positive_residual_cache=residual)
                   for enabled, backend, pairs, residual in (
                       (False, 'indexed', False, False), (True, 'indexed', False, False),
                       (True, 'indexed', True, True), (False, 'reference', False, False),
                       (True, 'reference', False, False))]
        with patch('nidavellir.positive_support.MIN_ROWS', 1):
            for head in task.heads:
                for size in range(len(task.literals) + 1):
                    for body in combinations(range(len(task.literals)), size):
                        used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                        if not variables(head) <= used:
                            continue
                        model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                        masks = list(range(1 << len(task.positives)))
                        masks += [mask | (1 << (len(task.positives) + 2)) for mask in masks]
                        for possible in masks + masks[::-1]:
                            expected = any(label in model and possible & (1 << position)
                                           for label, position in oracles[0].positives.items())
                            for oracle in oracles:
                                with self.subTest(head=str(head), body=body, possible=possible,
                                                  enabled=oracle.positive_lookahead,
                                                  backend=oracle.positive_join_backend):
                                    self.assertEqual(has_positive(oracle, head, body, possible), expected)

    def support_task(self, success=False, copies=12):
        text = ' '.join(f'candidate({i},{i}).' for i in range(copies))
        text += ' ' + ' '.join(f'support({i},{(i + 1) % copies},{z}).'
                               for i in range(copies) for z in range(4))
        if success:
            text += f' support({copies - 1},{copies - 1},ok).'
        return self.task(text, ['candidate(var__(1),var__(2))',
                               'support(var__(1),var__(2),var__(3))'],
                         heads=('answer(ok)',), examples='positive(answer(ok)).')

    def correlated_task(self):
        return self.task('candidate(x,u). candidate(x,v). candidate(y,u). candidate(y,v). '
                         'support(a,x,u,z). support(a,y,v,z). support(b,x,v,z). support(b,y,u,z).',
                         ['candidate(var__(2),var__(3))',
                          'support(var__(1),var__(2),var__(3),var__(4))'],
                         heads=('answer(ok)',), examples='positive(answer(ok)).')

    @staticmethod
    def index(task, name):
        return next(i for i, literal in enumerate(task.literals) if literal.name == name)

    def test_all_global_masks_overlapping_heads_and_hidden_correlations(self):
        self.compare_all(self.task('p(a,x). p(a,y). p(b,y). q(a,y). q(b,x). mark(x).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))', 'mark(var__(2))'],
            heads=('answer(var__(1))', 'answer(a)', 'omega(var__(1))'),
            examples='positive(answer(a)). positive(answer(b)). positive(omega(a)). positive(omega(b)).'))

    def test_nested_repeated_terms_constants_and_head_permutations(self):
        self.compare_all(self.task(
            'p(tag,pair(a,x),x). p(tag,pair(b,y),y). p(tag,pair(a,x),y). '
            'p(other,pair(a,a),a). q(x,x). q(x,y). q(y,y). mark(x).',
            ['p(tag,pair(var__(1),var__(2)),var__(2))',
             'p(tag,pair(var__(1),var__(3)),var__(2))', 'q(var__(2),var__(2))', 'mark(var__(2))'],
            heads=('answer(pair(var__(1),var__(2)),var__(1),ok)',
                   'answer(pair(var__(2),var__(1)),var__(1),ok)',
                   'answer(pair(var__(1),var__(1)),var__(1),ok)'),
            examples='positive(answer(pair(a,x),a,ok)). positive(answer(pair(x,a),a,ok)). '
                     'positive(answer(pair(b,y),b,ok)). positive(answer(pair(a,x),b,ok)).', max_body=4))

    def test_supports_keep_multiple_columns_and_already_fixed_hidden_values(self):
        task = self.correlated_task()
        oracle = CoverageOracle(task)
        chosen, other = self.index(task, 'candidate'), self.index(task, 'support')
        slots, table = matched_relation(oracle.space, chosen)
        original_rows, original_members = table.rows, table.members
        for fixed_value, expected in (
                ('a', {('x', 'u'), ('y', 'v')}),
                ('b', {('x', 'v'), ('y', 'u')}), ('missing', set())):
            binding = {1: oracle.space.intern(clingo.parse_term(fixed_value))}
            with patch('nidavellir.positive_support.MIN_ROWS', 1):
                result = filter_supported(oracle, chosen, table.rows, (other,), binding, {})
            actual = {tuple(str(oracle.space.symbols[row[slots.index(slot)]]) for slot in (2, 3))
                      for row in result}
            self.assertEqual(actual, expected)
            self.assertIs(table.rows, original_rows)
            self.assertIs(table.members, original_members)
        self.assertEqual(set(table.rows), set(original_rows))

    def test_pairwise_supported_odd_cycle_still_needs_a_complete_witness(self):
        text = ' '.join(f'edge({x},{y}). edge({y},{x}).' for x in range(3) for y in range(3, 6))
        task = self.task(text, ['edge(var__(1),var__(2))', 'edge(var__(2),var__(3))',
                                'edge(var__(3),var__(1))'], heads=('answer(ok)',),
                         examples='positive(answer(ok)).')
        oracle = CoverageOracle(task, positive_pair_pruning=False, positive_residual_cache=False)
        body = tuple(range(3))
        _, table = matched_relation(oracle.space, 0)
        self.assertEqual(set(filter_supported(oracle, 0, table.rows, (1, 2), {}, {})), set(table.rows))
        self.assertNotIn(clingo.parse_term('answer(ok)'), fixtures.atoms(task.background + '\n' + task.rule(task.heads[0], body)))
        self.assertFalse(has_positive(oracle, task.heads[0], body, 1))

    def test_false_and_late_true_queries_reject_rows_before_recursive_search(self):
        from nidavellir import positive_join
        original = positive_join.measured_rows
        for success in (False, True):
            task = self.support_task(success)
            chosen = self.index(task, 'candidate')
            body = tuple(range(len(task.literals)))
            expected = clingo.parse_term('answer(ok)') in fixtures.atoms(task.background + '\n' + task.rule(task.heads[0], body))
            self.assertEqual(expected, success)
            def late_last(oracle, index, binding, metrics):
                rows = original(oracle, index, binding, metrics)
                if index == chosen:
                    # Deliberately place the only successful extension last;
                    # correctness cannot depend on relation row ordering.
                    slots, _ = matched_relation(oracle.space, index)
                    return tuple(sorted(rows, key=lambda row:
                        oracle.space.symbols[row[slots.index(1)]] == clingo.Number(11)))
                return rows
            results = []
            for enabled in (False, True):
                oracle = CoverageOracle(task, positive_lookahead=enabled, positive_pair_pruning=False,
                                        positive_residual_cache=False)
                with patch.object(positive_join, 'measured_rows', side_effect=late_last):
                    self.assertEqual(has_positive(oracle, task.heads[0], body, 1), expected)
                results.append(oracle.stats)
            with self.subTest(success=success):
                self.assertGreater(results[0]['positive_join_nodes'], 12)
                self.assertLess(results[1]['positive_join_nodes'] * 3, results[0]['positive_join_nodes'])
                self.assertGreaterEqual(results[1].get('positive_lookahead_rows_pruned', 0), 11)

    def test_projection_budget_skips_new_builds_but_reuses_complete_projections(self):
        task = self.correlated_task()
        oracle = CoverageOracle(task)
        chosen, other = self.index(task, 'candidate'), self.index(task, 'support')
        _, table = matched_relation(oracle.space, chosen)
        slots, support = matched_relation(oracle.space, other)
        binding = {1: oracle.space.intern(clingo.parse_term('a'))}
        columns = tuple(slots.index(slot) for slot in (1, 2, 3))
        with patch('nidavellir.positive_support.MIN_ROWS', 1), \
             patch('nidavellir.positive_support.MAX_PROJECT_ROWS', 1):
            self.assertEqual(set(filter_supported(oracle, chosen, table.rows, (other,), binding, {})), set(table.rows))
            self.assertNotIn(columns, support.projections)
            support.project(columns)
            result = filter_supported(oracle, chosen, table.rows, (other,), binding, {})
            self.assertEqual(len(result), 2)
            self.assertGreater(oracle.stats.get('positive_lookahead_projection_hits', 0), 0)

    def test_projection_cache_remains_bounded_across_many_bound_column_sets(self):
        text = ' '.join(f'candidate({i},{i}). wide(' + ','.join([str(i)] * 9) + ').' for i in range(10))
        task = self.task(text, ['candidate(var__(1),var__(2))',
                               'wide(' + ','.join(f'var__({i})' for i in range(1, 10)) + ')'],
                         heads=('answer(ok)',), examples='positive(answer(ok)).')
        oracle = CoverageOracle(task)
        chosen, other = self.index(task, 'candidate'), self.index(task, 'wide')
        _, table = matched_relation(oracle.space, chosen)
        _, support = matched_relation(oracle.space, other)
        zero = oracle.space.intern(clingo.Number(0))
        for count in range(8):
            for fixed in combinations(range(3, 10), count):
                result = filter_supported(oracle, chosen, table.rows, (other,), dict.fromkeys(fixed, zero), {})
                self.assertEqual(set(result), {(zero, zero)} if fixed else set(table.rows))
                self.assertLessEqual(len(support.projections), 64)
                self.assertLessEqual(support.projection_rows, 262144)
                self.assertEqual(support.projection_rows, sum(map(len, support.projections.values())))
        self.assertGreater(len(support.projections), 0)

    def test_interruption_never_publishes_partial_projection_or_query_answer(self):
        for success in (False, True):
            task = self.support_task(success=success, copies=100)
            head, body = task.heads[0], tuple(range(len(task.literals)))
            calls = 0
            def count():
                nonlocal calls
                calls += 1
            baseline = CoverageOracle(task, positive_pair_pruning=False, positive_residual_cache=False)
            baseline.check = count
            self.assertEqual(has_positive(baseline, head, body, 1), success)
            total = calls
            for stop in range(1, total + 1):
                oracle = CoverageOracle(task, positive_pair_pruning=False, positive_residual_cache=False)
                calls = 0
                def interrupt():
                    nonlocal calls
                    calls += 1
                    if calls >= stop:
                        raise ResourceLimit('lookahead interruption')
                oracle.check = interrupt
                with self.subTest(success=success, stop=stop), self.assertRaisesRegex(ResourceLimit, 'lookahead interruption'):
                    has_positive(oracle, head, body, 1)
                self.assertFalse(getattr(oracle, '_positive_join_results', {}))
                self.assertFalse(oracle._safe)
                for relation in getattr(oracle.space, '_integer_relations', {}).values():
                    for columns, projection in relation.projections.items():
                        self.assertEqual(projection, frozenset(tuple(row[i] for i in columns) for row in relation.rows))
                oracle.check = lambda: None
                self.assertEqual(has_positive(oracle, head, body, 1), success)

    def test_complete_catalogues_variants_and_all_optima_match_clingo(self):
        task = self.task('p(a,x). p(b,x). p(n1,x). q(a,x). q(b,x). q(n2,x). '
                         'r(b). r(c). s(a). s(c).',
                         ['p(var__(1),var__(2))', 'q(var__(1),var__(2))', 'r(var__(1))', 's(var__(1))'],
                         examples='positive(answer(a)). positive(answer(b)). positive(answer(c)). '
                                  'negative(answer(n1)). negative(answer(n2)).', max_body=3)
        expected = exhaustive_catalogue(task)
        full, bound = (1 << len(task.positives)) - 1, 6
        optima = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        self.assertIsNotNone(optima[0])
        with patch('nidavellir.positive_support.MIN_ROWS', 1):
            for enabled in (False, True):
                for order in ('source', 'selectivity'):
                    for variants in ('exhaustive', 'targeted'):
                        oracle = CoverageOracle(task, positive_lookahead=enabled)
                        rows, stats = compile_catalogue(task, oracle, order=order, variants=variants, alpha_collect=False)
                        with self.subTest(enabled=enabled, order=order, variants=variants):
                            self.assertTrue(stats['complete'])
                            self.assertEqual(normalized(rows), expected)
                            self.assertEqual(optimal_programs(((row['coverage'], row['cost']) for row in rows), full, bound), optima)

    def test_disabled_reference_small_and_unrelated_branches_skip_filter(self):
        task = self.support_task()
        chosen, other = self.index(task, 'candidate'), self.index(task, 'support')
        for options in ({'positive_lookahead': False}, {'positive_join_backend': 'reference'}):
            oracle = CoverageOracle(task, **options)
            _, table = matched_relation(oracle.space, chosen)
            self.assertIs(filter_supported(oracle, chosen, table.rows, (other,), {}, {}), table.rows)
            self.assertFalse(has_positive(oracle, task.heads[0], tuple(range(2)), 1))
            self.assertEqual(oracle.stats.get('positive_lookahead_queries', 0), 0)
        oracle = CoverageOracle(task)
        _, table = matched_relation(oracle.space, chosen)
        small = table.rows[:7]
        self.assertIs(filter_supported(oracle, chosen, small, (other,), {}, {}), small)
        self.assertIs(filter_supported(oracle, chosen, table.rows, (), {}, {}), table.rows)

    def test_cli_default_and_ablation_reach_both_sequential_targets(self):
        task = self.task('p(a). p(b). p(n1). q(a). q(b). q(n2).',
                         ['p(var__(1))', 'q(var__(1))'], heads=('first(var__(1))', 'second(var__(1))'),
                         examples='positive(first(a)). positive(first(b)). positive(second(b)). '
                                  'negative(first(n1)). negative(first(n2)). negative(second(n1)). negative(second(n2)).')
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / 'inputs'
            inputs.mkdir()
            (inputs / 'background.lp').write_text(task.background)
            (inputs / 'bias.lp').write_text(task.bias_source)
            task.write_examples(inputs / 'examples.lp')
            costs = []
            for disabled in (False, True):
                options = ['--compiler-backend', 'relational', '--target-mode', 'separate']
                if disabled:
                    options.append('--no-positive-lookahead')
                process, output, status = cli_fixtures.ExtendedCLITests.invoke(
                    self, folder, 'disabled' if disabled else 'default', *options, inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                self.assertEqual(status['status'], 'complete')
                self.assertEqual(status['arguments']['no_positive_lookahead'], disabled)
                targets = sorted((output / 'targets').glob('*/catalogue.json'))
                self.assertEqual(len(targets), 2)
                for catalogue in targets:
                    compiled = json.loads(catalogue.read_text())
                    manifest = json.loads((catalogue.parent / 'manifest.json').read_text())
                    self.assertEqual(manifest['arguments']['no_positive_lookahead'], disabled)
                    self.assertEqual(compiled['stats']['positive_lookahead'], not disabled)
                    self.assertTrue(compiled['stats']['complete'])
                costs.append(status['optimal_cost'])
            self.assertEqual(costs[0], costs[1])

    def test_lookahead_option_requires_boolean(self):
        task = self.support_task()
        for value in (0, 1, None, 'yes'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CoverageOracle(task, positive_lookahead=value)


if __name__ == '__main__':
    unittest.main()
