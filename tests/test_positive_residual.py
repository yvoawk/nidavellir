"""Residual failures preserve exact positive joins, labels and optimal variants."""
from collections import OrderedDict
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.positive_join import has_positive
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import variables
from tests import test_search_optimizations as fixtures
from tests import test_cli_extended as cli_fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class PositiveResidualTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    def compare_all(self, task):
        oracles = [CoverageOracle(task, positive_residual_cache=residual,
                                  positive_join_backend=backend, positive_pair_pruning=pairs)
                   for residual, backend, pairs in ((False, 'indexed', False), (True, 'indexed', False),
                       (True, 'indexed', True), (False, 'reference', False), (True, 'reference', False))]
        saw_true = saw_false = False
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(head) <= used:
                        continue
                    model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                    masks = list(range(1 << len(task.positives)))
                    # Global positions include other heads and out-of-range bits.
                    masks += [mask | (1 << (len(task.positives) + 2)) for mask in masks]
                    for possible in masks + masks[::-1]:
                        expected = any(label in model and possible & (1 << position)
                                       for label, position in oracles[0].positives.items())
                        saw_true |= expected
                        saw_false |= not expected
                        for oracle in oracles:
                            with self.subTest(head=str(head), body=body, possible=possible,
                                              residual=oracle.positive_residual_cache,
                                              backend=oracle.positive_join_backend):
                                self.assertEqual(has_positive(oracle, head, body, possible), expected)
        self.assertTrue(saw_true)
        self.assertTrue(saw_false)

    def odd_cycle_task(self, copies=12, width=3):
        background = ' '.join(f'p(a,{i}).' for i in range(copies))
        background += ' ' + ' '.join(f'edge({x},{y}). edge({y},{x}).'
                                     for x in range(width) for y in range(width, 2 * width))
        return self.task(background,
            ['p(var__(1),var__(2))', 'edge(var__(3),var__(4))',
             'edge(var__(4),var__(5))', 'edge(var__(5),var__(3))'],
            examples='positive(answer(a)).', max_body=4)

    def test_all_masks_and_overlapping_heads_match_complete_clingo(self):
        self.compare_all(self.task(
            'p(a,x). p(a,y). p(b,y). q(a,y). q(b,x). marker(x).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))', 'marker(var__(2))'],
            heads=('answer(var__(1))', 'answer(a)', 'omega(var__(1))'),
            examples='positive(answer(a)). positive(answer(b)). positive(omega(a)). positive(omega(b)).'))

    def test_nested_repeated_variables_and_constants_match_clingo(self):
        self.compare_all(self.task(
            'edge(tag,pair(a,x),x). edge(tag,pair(b,y),y). '
            'edge(tag,pair(a,x),y). edge(other,pair(a,a),a). mark(x).',
            ['edge(tag,pair(var__(1),var__(2)),var__(2))',
             'edge(tag,pair(var__(1),var__(3)),var__(2))', 'mark(var__(2))'],
            heads=('answer(pair(var__(1),var__(2)),var__(1),ok)',
                   'answer(pair(var__(2),var__(1)),var__(1),ok)',
                   'answer(pair(var__(1),var__(1)),var__(1),ok)'),
            examples='positive(answer(pair(a,x),a,ok)). positive(answer(pair(x,a),a,ok)). '
                     'positive(answer(pair(b,y),b,ok)). positive(answer(pair(a,x),b,ok)).'))

    def test_live_hidden_variables_are_not_discarded_with_dead_variables(self):
        self.compare_all(self.task(
            'p(a,x,1). p(a,x,2). p(a,y,1). p(a,y,2). p(b,z,1). '
            'q(x,u). q(y,v). q(z,u). r(v).',
            ['p(var__(1),var__(2),var__(3))', 'q(var__(2),var__(4))', 'r(var__(4))'],
            examples='positive(answer(a)). positive(answer(b)). positive(answer(missing)).'))

    def test_dead_variable_repetition_reuses_full_failed_suffixes(self):
        task = self.odd_cycle_task()
        body, head = tuple(range(len(task.literals))), task.heads[0]
        self.assertNotIn(clingo.parse_term('answer(a)'), fixtures.atoms(task.background + '\n' + task.rule(head, body)))
        for backend in ('indexed', 'reference'):
            ordinary = CoverageOracle(task, positive_residual_cache=False, positive_join_backend=backend,
                                      positive_pair_pruning=False)
            cached = CoverageOracle(task, positive_residual_cache=True, positive_join_backend=backend,
                                    positive_pair_pruning=False)
            self.assertFalse(has_positive(ordinary, head, body, 1))
            self.assertFalse(has_positive(cached, head, body, 1))
            with self.subTest(backend=backend):
                self.assertLess(cached.stats['positive_join_matched_atoms'] * 3,
                                ordinary.stats['positive_join_matched_atoms'])
                self.assertLess(cached.stats['positive_join_nodes'] * 3,
                                ordinary.stats['positive_join_nodes'])
                self.assertGreater(cached.stats.get('positive_join_residual_hits', 0), 0)
                self.assertEqual(ordinary.stats.get('positive_join_residual_stores', 0), 0)

    def test_cache_capacity_only_changes_repeated_work(self):
        task = self.odd_cycle_task()
        head, body = task.heads[0], tuple(range(len(task.literals)))
        for limit in (0, 1, 3, 7):
            oracle = CoverageOracle(task, positive_residual_cache=True, positive_pair_pruning=False)
            with self.subTest(limit=limit), patch('nidavellir.positive_join.RESIDUAL_CACHE_LIMIT', limit):
                self.assertFalse(has_positive(oracle, head, body, 1))
                self.assertLessEqual(oracle.stats.get('positive_join_residual_peak_entries', 0), limit)
                if limit:
                    self.assertGreater(oracle.stats.get('positive_join_residual_stores', 0), limit)
                    self.assertGreater(oracle.stats.get('positive_join_residual_evictions', 0), 0)
                else:
                    self.assertEqual(oracle.stats.get('positive_join_residual_stores', 0), 0)
                    self.assertEqual(oracle.stats.get('positive_join_residual_hits', 0), 0)

    def test_success_is_not_stored_as_a_residual_failure(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='positive(answer(a)).')
        oracle = CoverageOracle(task, positive_residual_cache=True)
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))
        self.assertEqual(oracle.stats.get('positive_join_residual_stores', 0), 0)
        self.assertEqual(oracle.stats.get('positive_join_residual_hits', 0), 0)
        # The existing complete top-level result cache still supports success.
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))
        self.assertEqual(oracle.stats.get('positive_join_cache_hits', 0), 1)

    def test_query_local_cache_releases_references_on_return_and_interruption(self):
        task = self.odd_cycle_task(copies=3, width=2)
        head, body = task.heads[0], tuple(range(len(task.literals)))
        singleton = next(i for i, literal in enumerate(task.literals) if literal.name == 'p')
        for interrupted in (False, True):
            oracle = CoverageOracle(task, positive_residual_cache=True, positive_pair_pruning=False)
            # Initialize persistent top-level caches before tracking query-local allocations.
            self.assertTrue(has_positive(oracle, head, (singleton,), 1))
            allocated = []
            def tracked(*args, **kwargs):
                value = OrderedDict(*args, **kwargs)
                allocated.append(value)
                return value
            if interrupted:
                def stop_after_a_complete_failure():
                    if oracle.stats.get('positive_join_residual_stores', 0):
                        raise ResourceLimit('stop after retained failure')
                oracle.check = stop_after_a_complete_failure
            with self.subTest(interrupted=interrupted), \
                 patch('nidavellir.positive_join.OrderedDict', side_effect=tracked):
                if interrupted:
                    with self.assertRaisesRegex(ResourceLimit, 'stop after retained failure'):
                        has_positive(oracle, head, body, 1)
                    self.assertNotIn((head, body, 1), oracle._positive_join_results)
                else:
                    self.assertFalse(has_positive(oracle, head, body, 1))
                self.assertGreater(oracle.stats.get('positive_join_residual_stores', 0), 0)
                self.assertTrue(allocated)
                self.assertTrue(all(not cache for cache in allocated))

    def test_failures_do_not_escape_their_query_or_hide_a_shorter_success(self):
        task = self.odd_cycle_task(copies=4, width=2)
        oracle = CoverageOracle(task, positive_residual_cache=True, positive_pair_pruning=False)
        head, body = task.heads[0], tuple(range(len(task.literals)))
        self.assertFalse(has_positive(oracle, head, body, 1))
        for excluded in range(len(task.literals)):
            shorter = tuple(i for i in body if i != excluded)
            used = set().union(*(variables(task.literals[i]) for i in shorter))
            if not variables(head) <= used:
                continue
            model = fixtures.atoms(task.background + '\n' + task.rule(head, shorter))
            self.assertIn(clingo.parse_term('answer(a)'), model)
            self.assertTrue(has_positive(oracle, head, shorter, 1))
        self.assertFalse(has_positive(oracle, head, body, 1))

    def test_complete_catalogues_tied_variants_and_all_optima_match_clingo(self):
        task = self.task('p(a,x). p(b,x). p(n1,x). q(a,x). q(b,x). q(n2,x). '
                         'r(b). r(c). s(a). s(c).',
                         ['p(var__(1),var__(2))', 'q(var__(1),var__(2))',
                          'r(var__(1))', 's(var__(1))'],
                         examples='positive(answer(a)). positive(answer(b)). positive(answer(c)). '
                                  'negative(answer(n1)). negative(answer(n2)).', max_body=3)
        expected = exhaustive_catalogue(task)
        full, bound = (1 << len(task.positives)) - 1, 6
        optima = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        self.assertIsNotNone(optima[0])
        for enabled in (False, True):
            for order in ('source', 'selectivity'):
                for variants in ('exhaustive', 'targeted'):
                    oracle = CoverageOracle(task, positive_residual_cache=enabled)
                    rows, stats = compile_catalogue(task, oracle, order=order, variants=variants,
                                                    alpha_collect=False)
                    with self.subTest(enabled=enabled, order=order, variants=variants):
                        self.assertTrue(stats['complete'])
                        self.assertGreater(oracle.stats.get('positive_join_queries', 0), 0)
                        self.assertEqual(normalized(rows), expected)
                        self.assertEqual(optimal_programs(((r['coverage'], r['cost']) for r in rows),
                                                         full, bound), optima)

    def test_interruption_never_publishes_a_partial_positive_answer(self):
        tasks = [self.odd_cycle_task(copies=3, width=2),
                 self.task('p(a,x). p(a,y). q(y).',
                           ['p(var__(1),var__(2))', 'q(var__(2))'],
                           examples='positive(answer(a)).')]
        for task in tasks:
            head, body = task.heads[0], tuple(range(len(task.literals)))
            model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
            expected = clingo.parse_term('answer(a)') in model
            calls = 0
            def count():
                nonlocal calls
                calls += 1
            baseline = CoverageOracle(task, positive_residual_cache=True, positive_pair_pruning=False)
            baseline.check = count
            self.assertEqual(has_positive(baseline, head, body, 1), expected)
            total = calls
            for stop in sorted({1, 2, 3, total // 3, total // 2, total - 2, total - 1, total}):
                oracle = CoverageOracle(task, positive_residual_cache=True, positive_pair_pruning=False)
                calls = 0
                def interrupt():
                    nonlocal calls
                    calls += 1
                    if calls >= stop:
                        raise ResourceLimit('positive residual deadline')
                oracle.check = interrupt
                with self.subTest(expected=expected, stop=stop):
                    with self.assertRaisesRegex(ResourceLimit, 'positive residual deadline'):
                        has_positive(oracle, head, body, 1)
                    self.assertFalse(getattr(oracle, '_positive_join_results', {}))
                    self.assertFalse(oracle._safe)
                    self.assertFalse(oracle._negative_witnesses)
                    diagnostics = oracle.profile.export()['diagnostics']['positive_join']
                    self.assertEqual(diagnostics['slowest_calls'][0]['outcome'], 'interrupted')
                    oracle.check = lambda: None
                    self.assertEqual(has_positive(oracle, head, body, 1), expected)

    def test_residual_option_requires_an_explicit_boolean(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='positive(answer(a)).')
        for value in (0, 1, None, 'yes'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CoverageOracle(task, positive_residual_cache=value)

    def test_cli_default_and_ablation_reach_every_sequential_target(self):
        task = self.task('p(a). p(b). p(n1). q(a). q(b). q(n2).',
                         ['p(var__(1))', 'q(var__(1))'],
                         heads=('first(var__(1))', 'second(var__(1))'),
                         examples='positive(first(a)). positive(first(b)). positive(second(b)). '
                                  'negative(first(n1)). negative(first(n2)). '
                                  'negative(second(n1)). negative(second(n2)).')
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / 'inputs'
            inputs.mkdir()
            (inputs / 'background.lp').write_text(task.background)
            (inputs / 'bias.lp').write_text(task.bias_source)
            task.write_examples(inputs / 'examples.lp')
            costs = []
            for disabled in (False, True):
                options = ['--compiler-backend', 'relational', '--target-mode', 'separate',
                           '--proof-strategy', 'catalogue']
                if disabled:
                    options.append('--no-positive-residual-cache')
                process, output, status = cli_fixtures.ExtendedCLITests.invoke(
                    self, folder, 'disabled' if disabled else 'default', *options, inputs=inputs)
                with self.subTest(disabled=disabled):
                    self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                    self.assertEqual(status['status'], 'complete')
                    self.assertEqual(status['arguments']['no_positive_residual_cache'], disabled)
                    targets = sorted((output / 'targets').glob('*/catalogue.json'))
                    self.assertEqual(len(targets), 2)
                    for catalogue in targets:
                        compiled = json.loads(catalogue.read_text())
                        manifest = json.loads((catalogue.parent / 'manifest.json').read_text())
                        self.assertEqual(manifest['arguments']['no_positive_residual_cache'], disabled)
                        self.assertEqual(compiled['stats']['positive_residual_cache'], not disabled)
                        self.assertTrue(compiled['stats']['complete'])
                    costs.append(status['optimal_cost'])
            self.assertEqual(costs[0], costs[1])


if __name__ == '__main__':
    unittest.main()
