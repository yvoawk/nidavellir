"""Only verified bound literals may disappear from a residual failure key."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir import positive_join
from nidavellir.indexed_join import matched_relation
from nidavellir.oracle import ResourceLimit
from nidavellir.positive_join import has_positive
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import variables
from tests import test_search_optimizations as fixtures
from tests import test_cli_extended as cli_fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class PositiveNormalizationTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    def compare_all(self, task):
        oracles = [CoverageOracle(task, positive_residual_normalization=enabled,
                    positive_residual_cache=cached, positive_join_backend=backend,
                    positive_pair_pruning=False, positive_lookahead=False)
                   for enabled, cached, backend in ((False, True, 'indexed'),
                       (True, True, 'indexed'), (True, False, 'indexed'),
                       (False, True, 'reference'), (True, True, 'reference'))]
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(head) <= used:
                        continue
                    model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                    masks = list(range(1 << len(task.positives)))
                    masks += [mask | (1 << (len(task.positives) + 1)) for mask in masks]
                    for possible in masks + masks[::-1]:
                        expected = any(label in model and possible >> position & 1
                                       for label, position in oracles[0].positives.items())
                        for oracle in oracles:
                            with self.subTest(head=str(head), body=body, possible=possible,
                                    normalized=oracle.positive_residual_normalization,
                                    cached=oracle.positive_residual_cache,
                                    backend=oracle.positive_join_backend):
                                self.assertEqual(has_positive(oracle, head, body, possible), expected)

    def repeated_guard(self, copies=12):
        background = ' '.join(f'p(a,{i}).' for i in range(copies))
        background += ' ' + ' '.join(f'guard({i}).' for i in range(copies + 30))
        background += ' ' + ' '.join(f'edge({x},{y}). edge({y},{x}).'
                                     for x in range(3) for y in range(3, 6))
        return self.task(background, ['p(var__(1),var__(2))', 'guard(var__(2))',
                         'edge(var__(3),var__(4))', 'edge(var__(4),var__(5))',
                         'edge(var__(5),var__(3))'], max_body=5,
                         examples='positive(answer(a)).')

    def failure_then_success(self):
        background = 'p(a,bad). p(a,good). guard(good). '
        background += ' '.join(f'guard(extra{i}). q(a,{i}).' for i in range(20))
        return self.task(background, ['p(var__(1),var__(2))', 'guard(var__(2))',
                         'q(var__(1),var__(3))'], examples='positive(answer(a)).')

    @staticmethod
    def ordered_rows(original):
        def ordered(oracle, index, binding, metrics):
            rows = original(oracle, index, binding, metrics)
            if oracle.task.literals[index].name == 'p':
                if oracle.positive_join_backend == 'indexed':
                    return tuple(sorted(rows, key=lambda row: tuple(str(oracle.space.symbols[v]) for v in row)))
                return tuple(sorted(rows, key=str))
            return rows
        return ordered

    def test_global_masks_overlapping_heads_and_live_hidden_values_match_clingo(self):
        self.compare_all(self.task('p(a,x,1). p(a,y,2). p(b,x,2). guard(1). guard(2). '
            'q(x,u). q(y,v). r(v).', ['p(var__(1),var__(2),var__(3))',
             'guard(var__(3))', 'q(var__(2),var__(4))', 'r(var__(4))'], max_body=4,
            heads=('answer(var__(1))', 'answer(a)', 'omega(var__(1))'),
            examples='positive(answer(a)). positive(answer(b)). positive(omega(a)). positive(omega(b)).'))

    def test_nested_repeated_variables_constants_and_ground_heads_match_clingo(self):
        self.compare_all(self.task('p(tag,pair(a,x),x). p(tag,pair(b,y),y). '
            'p(tag,pair(a,x),y). guard(x). q(x,x). q(y,y). marker.',
            ['p(tag,pair(var__(1),var__(2)),var__(2))', 'guard(var__(2))',
             'q(var__(2),var__(2))', 'marker'], max_body=4,
            heads=('answer(pair(var__(1),var__(2)),var__(1),ok)',
                   'answer(pair(var__(1),var__(1)),var__(1),ok)', 'answer(pair(a,x),a,ok)'),
            examples='positive(answer(pair(a,x),a,ok)). positive(answer(pair(b,y),b,ok)). '
                     'positive(answer(pair(a,a),a,ok)).'))

    def test_failed_bound_literal_cannot_poison_a_successful_normalized_suffix(self):
        task = self.failure_then_success()
        head, body = task.heads[0], tuple(range(len(task.literals)))
        self.assertIn(clingo.parse_term('answer(a)'), fixtures.atoms(task.background + '\n' + task.rule(head, body)))
        for backend in ('indexed', 'reference'):
            oracle = CoverageOracle(task, positive_join_backend=backend,
                       positive_pair_pruning=False, positive_lookahead=False)
            with patch.object(positive_join, 'measured_rows',
                              side_effect=self.ordered_rows(positive_join.measured_rows)):
                self.assertTrue(has_positive(oracle, head, body, 1))
            self.assertGreater(oracle.stats.get('positive_join_normalization_failures', 0), 0)
            self.assertGreater(oracle.stats.get('positive_join_normalized_literals', 0), 0)
            self.assertTrue(oracle._positive_join_results[head, body, 1])

    def test_true_bound_guards_allow_reuse_before_repeating_the_next_join(self):
        task = self.repeated_guard()
        head, body = task.heads[0], tuple(range(len(task.literals)))
        self.assertNotIn(clingo.parse_term('answer(a)'), fixtures.atoms(task.background + '\n' + task.rule(head, body)))
        for backend in ('indexed', 'reference'):
            results = []
            for enabled in (False, True):
                oracle = CoverageOracle(task, positive_join_backend=backend,
                    positive_residual_normalization=enabled, positive_pair_pruning=False, positive_lookahead=False)
                self.assertFalse(has_positive(oracle, head, body, 1))
                results.append(oracle.stats)
            self.assertLess(results[1]['positive_join_nodes'] * 2, results[0]['positive_join_nodes'])
            self.assertGreater(results[1].get('positive_join_normalized_literals', 0), 0)
            self.assertGreater(results[1].get('positive_join_residual_hits', 0), 0)
            self.assertEqual(results[0].get('positive_join_normalization_checks', 0), 0)

    def test_bound_hidden_values_needed_by_the_remaining_body_are_kept(self):
        task = self.task('p(a,bad,1). p(a,good,2). guard(1). guard(2). '
            + ' '.join(f'guard(extra{i}). q(bad,x{i}). q(good,y{i}). r(y{i}).' for i in range(10)),
            ['p(var__(1),var__(2),var__(3))', 'guard(var__(3))',
             'q(var__(2),var__(4))', 'r(var__(4))'], max_body=4, examples='positive(answer(a)).')
        oracle = CoverageOracle(task, positive_pair_pruning=False, positive_lookahead=False)
        with patch.object(positive_join, 'measured_rows',
                          side_effect=self.ordered_rows(positive_join.measured_rows)):
            self.assertTrue(has_positive(oracle, task.heads[0], tuple(range(4)), 1))
        self.assertGreater(oracle.stats.get('positive_join_normalized_states', 0), 0)

    def test_cache_limits_and_disabling_cache_preserve_exact_answers(self):
        task = self.repeated_guard(copies=5)
        for limit in (0, 1, 3, 8192):
            oracle = CoverageOracle(task, positive_pair_pruning=False, positive_lookahead=False)
            with patch.object(positive_join, 'RESIDUAL_CACHE_LIMIT', limit):
                self.assertFalse(has_positive(oracle, task.heads[0], tuple(range(5)), 1))
            self.assertLessEqual(oracle.stats.get('positive_join_residual_peak_entries', 0), limit)
        oracle = CoverageOracle(task, positive_residual_cache=False,
                                positive_pair_pruning=False, positive_lookahead=False)
        self.assertFalse(has_positive(oracle, task.heads[0], tuple(range(5)), 1))
        self.assertGreater(oracle.stats.get('positive_join_normalized_literals', 0), 0)
        self.assertEqual(oracle.stats.get('positive_join_residual_stores', 0), 0)

    def test_interruption_at_every_checkpoint_never_publishes_a_partial_answer(self):
        for task, expected in ((self.repeated_guard(copies=2), False),
                               (self.failure_then_success(), True)):
            head, body = task.heads[0], tuple(range(len(task.literals)))
            for backend in ('indexed', 'reference'):
                calls = 0
                def count():
                    nonlocal calls
                    calls += 1
                oracle = CoverageOracle(task, positive_join_backend=backend,
                                         positive_pair_pruning=False, positive_lookahead=False)
                oracle.check = count
                self.assertEqual(has_positive(oracle, head, body, 1), expected)
                total = calls
                for stop in range(1, total + 1):
                    fresh = CoverageOracle(task, positive_join_backend=backend,
                                           positive_pair_pruning=False, positive_lookahead=False)
                    calls = 0
                    def interrupt():
                        nonlocal calls
                        calls += 1
                        if calls >= stop:
                            raise ResourceLimit('normalization checkpoint')
                    fresh.check = interrupt
                    with self.subTest(backend=backend, expected=expected, stop=stop), \
                         self.assertRaisesRegex(ResourceLimit, 'normalization checkpoint'):
                        has_positive(fresh, head, body, 1)
                    self.assertFalse(getattr(fresh, '_positive_join_results', {}))
                    self.assertFalse(fresh._safe)
                    fresh.check = lambda: None
                    self.assertEqual(has_positive(fresh, head, body, 1), expected)

    def test_complete_catalogues_variants_and_optima_are_identical(self):
        task = self.task('p(a,x). p(b,x). p(n1,x). q(a,x). q(b,x). q(n2,x). '
                         'r(b). r(c). s(a). s(c).',
                         ['p(var__(1),var__(2))', 'q(var__(1),var__(2))', 'r(var__(1))', 's(var__(1))'],
                         examples='positive(answer(a)). positive(answer(b)). positive(answer(c)). '
                                  'negative(answer(n1)). negative(answer(n2)).')
        expected = exhaustive_catalogue(task)
        full, bound = (1 << len(task.positives)) - 1, 6
        optima = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        for enabled in (False, True):
            for order in ('source', 'selectivity'):
                for variants in ('exhaustive', 'targeted'):
                    oracle = CoverageOracle(task, positive_residual_normalization=enabled)
                    rows, stats = compile_catalogue(task, oracle, order=order, variants=variants, alpha_collect=False)
                    self.assertTrue(stats['complete'])
                    self.assertEqual(normalized(rows), expected)
                    self.assertEqual(optimal_programs(((row['coverage'], row['cost']) for row in rows), full, bound), optima)

    def test_cli_ablation_reaches_each_sequential_target(self):
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
                    options.append('--no-positive-residual-normalization')
                process, output, status = cli_fixtures.ExtendedCLITests.invoke(
                    self, folder, 'disabled' if disabled else 'default', *options, inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status['status'], 'complete')
                paths = sorted((output / 'targets').glob('*/catalogue.json'))
                self.assertEqual(len(paths), 2)
                for path in paths:
                    self.assertEqual(json.loads(path.read_text())['stats']['positive_residual_normalization'], not disabled)
                    manifest = json.loads(path.with_name('manifest.json').read_text())
                    self.assertEqual(manifest['arguments']['no_positive_residual_normalization'], disabled)
                costs.append(status['optimal_cost'])
            self.assertEqual(costs[0], costs[1])

    def test_normalization_flag_requires_boolean(self):
        task = self.failure_then_success()
        for value in (0, 1, None, 'yes'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CoverageOracle(task, positive_residual_normalization=value)


if __name__ == '__main__':
    unittest.main()
