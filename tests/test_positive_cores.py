"""Global failure certificates versus independent labelled Clingo joins."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir import positive_cores
from nidavellir.oracle import ResourceLimit
from nidavellir.positive_join import has_positive, pattern_has_positive
from nidavellir.relational_compile import compile_catalogue, mine_variants
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import asp_text
from tests import test_search_optimizations as fixtures
from tests import test_cli_extended as cli_fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


def expected(task, head, body, possible):
    labels = '\n'.join(f'core_label({label}).' for i, label in enumerate(sorted(task.positives))
                       if possible >> i & 1)
    parts = [f'core_label({asp_text(head)})', *(asp_text(task.literals[i]) for i in body)]
    model = fixtures.atoms(task.background + '\n' + labels + '\ncore_answer :- ' + ', '.join(parts) + '.')
    return clingo.Function('core_answer') in model


class PositiveCoreTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    def test_partial_bodies_masks_nested_heads_and_hidden_correlations(self):
        task = self.task('p(a,x). p(b,y). q(x,x). q(y,z). r(tag,pair(a,x)). marker.',
            ['p(var__(1),var__(2))', 'q(var__(2),var__(2))',
             'r(tag,pair(var__(1),var__(3)))', 'marker'], max_body=4,
            heads=('answer(pair(var__(1),var__(2)),var__(1))', 'answer(pair(a,x),a)', 'other(var__(1))'),
            examples='positive(answer(pair(a,x),a)). positive(answer(pair(b,y),b)). positive(other(b)).')
        oracles = [CoverageOracle(task, positive_join_backend=backend, positive_residual_cache=cache,
            positive_lookahead=cache, positive_residual_normalization=cache)
            for backend in ('indexed', 'reference') for cache in (False, True)]
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    for mask in range(1 << len(task.positives)):
                        result = expected(task, head, body, mask)
                        for oracle in oracles:
                            for possible in (mask, mask | (1 << 20), mask):
                                self.assertEqual(pattern_has_positive(oracle, head, body, possible), result)

    def test_restricted_failure_is_not_a_global_certificate(self):
        task = self.task('p(b).', ['p(var__(1))'],
                         examples='positive(answer(a)). positive(answer(b)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        head = task.heads[0]
        mask = 1 << oracle.positives[clingo.parse_term('answer(a)')]
        self.assertFalse(has_positive(oracle, head, (0,), mask))
        oracle.positive_cores.learn(head, (0,), 1)
        self.assertFalse(oracle.positive_cores.cores)
        self.assertTrue(has_positive(oracle, head, (0,), 3))

    def test_global_core_can_be_head_unsafe_and_is_reused_only_for_its_head(self):
        task = self.task('p(a,x). p(b,y). q(y). link(a). link(b).',
            ['p(var__(1),var__(2))', 'q(var__(2))', 'link(var__(3))'],
            heads=('first(var__(1),var__(3))', 'second(var__(1),var__(3))'),
            examples='positive(first(a,a)). positive(second(b,b)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        first = clingo.parse_term('first(var__(1),var__(3))')
        second = clingo.parse_term('second(var__(1),var__(3))')
        core = tuple(i for i, literal in enumerate(task.literals) if literal.name in ('p', 'q'))
        oracle.positive_cores.learn(first, (0, 1, 2), 1)
        self.assertEqual(list(oracle.positive_cores.cores[first].values()), [core])
        self.assertIsNotNone(oracle.positive_cores.match(first, 7))
        self.assertIsNone(oracle.positive_cores.match(second, 7))
        self.assertTrue(pattern_has_positive(oracle, second, (0, 1, 2), 3))
        with self.assertRaises(ValueError):
            has_positive(oracle, first, core, 3)
        self.assertFalse(CoverageOracle(task, positive_core_pruning=True).positive_cores.cores)

    def test_retained_proofs_are_global_and_all_supersets_are_impossible(self):
        task = self.task('p(a,x). p(b,y). q(y). r(a). s(b). marker.',
            ['p(var__(1),var__(2))', 'q(var__(2))', 'r(var__(1))', 's(var__(1))', 'marker'],
            max_body=5, examples='positive(answer(a)). positive(answer(b)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        head = task.heads[0]
        for size in range(6):
            for body in combinations(range(5), size):
                oracle.positive_cores.learn(head, body, 1)
                for core in oracle.positive_cores.cores.get(head, {}).values():
                    self.assertFalse(expected(task, head, core, 3))
                if oracle.positive_cores.match(head, sum(1 << i for i in body)) is not None:
                    self.assertFalse(expected(task, head, body, 3))

    def test_antichain_capacity_and_eviction_do_not_invent_proofs(self):
        task = self.task('p(z). q(z). r(z).', [f'{p}(var__(1))' for p in 'pqr'],
                         examples='positive(answer(a)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        bank, head = oracle.positive_cores, task.heads[0]
        # All these conjunctions are independently impossible on answer(a).
        with patch.object(positive_cores, 'MAX_CORES_PER_HEAD', 2):
            for body in ((0, 1), (0,), (1,), (2,)):
                self.assertFalse(expected(task, head, body, 1))
                bank._store(head, body)
            self.assertEqual(set(bank.cores[head].values()), {(1,), (2,)})
        self.assertIsNone(bank.match(head, 1))
        self.assertGreater(oracle.stats['positive_core_subsumed'], 0)
        self.assertEqual(oracle.stats['positive_core_evictions'], 1)
        with patch.object(positive_cores, 'MAX_CORES_PER_HEAD', 0):
            fresh = CoverageOracle(task, positive_core_pruning=True)
            fresh.positive_cores.learn(head, (0,), 1)
            self.assertIsNone(fresh.positive_cores.match(head, 1))

    def test_local_timeouts_are_unknown_and_restore_checks(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='positive(answer(a)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        original = oracle.check
        with patch.object(positive_cores, 'MAX_PROBE_SECONDS', 0):
            oracle.positive_cores.learn(task.heads[0], (0,), 1)
        self.assertIs(oracle.check, original)
        self.assertFalse(oracle.positive_cores.cores)
        self.assertGreater(oracle.stats['positive_core_probe_timeouts'], 0)
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))

    def test_learning_timeout_may_keep_only_an_already_completed_proof(self):
        task = self.task('p(z). q(z).', [f'{p}(var__(1))' for p in 'pq'],
                         examples='positive(answer(a)).')
        oracle = CoverageOracle(task, positive_core_pruning=True)
        bank, head = oracle.positive_cores, task.heads[0]
        probe = bank._probe
        def one_completed_proof(h, body):
            if body == (1,):
                return probe(h, body)
            raise positive_cores._LearningExpired()
        with patch.object(bank, '_probe', one_completed_proof), patch.object(positive_cores, 'MAX_SMALL_CORE_SIZE', 0):
            bank.learn(head, (0, 1), 1)
        self.assertEqual(list(bank.cores[head].values()), [(1,)])
        self.assertFalse(expected(task, head, (1,), 1))

    def test_caller_deadlines_propagate_at_every_checkpoint_without_partial_proofs(self):
        task = self.task('p(a,x). q(y). link(a).',
            ['p(var__(1),var__(2))', 'q(var__(2))', 'link(var__(3))'],
            heads=('answer(var__(1),var__(3))',), examples='positive(answer(a,a)).')
        calls = 0
        def count():
            nonlocal calls
            calls += 1
        oracle = CoverageOracle(task, positive_core_pruning=True)
        oracle.check = count
        oracle.positive_cores.learn(task.heads[0], (0, 1, 2), 1)
        for stop in range(1, calls + 1):
            fresh = CoverageOracle(task, positive_core_pruning=True)
            seen = 0
            def interrupt():
                nonlocal seen
                seen += 1
                if seen >= stop:
                    raise ResourceLimit('caller deadline')
            fresh.check = interrupt
            with self.assertRaises(ResourceLimit):
                fresh.positive_cores.learn(task.heads[0], (0, 1, 2), 1)
            self.assertIs(fresh.check, interrupt)
            self.assertFalse(fresh.positive_cores.cores)
            fresh.check = lambda: None
            for (head, body, mask), answer in getattr(fresh, '_positive_join_results', {}).items():
                self.assertEqual(answer, expected(task, head, body, mask))

    def test_catalogues_variants_and_multi_rule_optima_match_exhaustive_reference(self):
        task = self.task('p(a,x). p(b,x). p(n1,x). p(unlabelled,y). '
            'q(a,x). q(b,x). q(n2,x). r(b). r(c). s(a). s(c). t(y). u(z).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))', 'r(var__(1))',
             's(var__(1))', 't(var__(2))', 'u(var__(3))'], max_body=3,
            heads=('first(var__(1))', 'second(var__(1))'),
            examples='positive(first(a)). positive(first(b)). positive(first(c)). positive(second(b)). '
                     'negative(first(n1)). negative(first(n2)). negative(second(n1)). negative(second(n2)).')
        reference = exhaustive_catalogue(task)
        optimum = optimal_programs(((m, c) for m, c, _ in reference), 15, 10)
        hits = 0
        with patch.object(positive_cores, 'MIN_QUERY_NODES', 0):
            for enabled in (False, True):
                for order in ('source', 'selectivity'):
                    for variants in ('exhaustive', 'targeted', 'representative'):
                        oracle = CoverageOracle(task, positive_core_pruning=enabled)
                        rows, stats = compile_catalogue(task, oracle, order=order, variants=variants, alpha_collect=False)
                        self.assertTrue(stats['complete'])
                        self.assertEqual(stats['positive_core_pruning'], enabled)
                        hits += stats['pruned_positive_core']
                        if variants == 'representative':
                            rows, reconstruction = mine_variants(task, oracle, rows, order=order, alpha_collect=False)
                            self.assertTrue(reconstruction['complete'])
                        self.assertEqual(normalized(rows), reference)
                        self.assertEqual(optimal_programs(((r['coverage'], r['cost']) for r in rows), 15, 10), optimum)
        self.assertGreater(hits, 0)

    def test_cli_option_reaches_sequential_children_and_variants(self):
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
            for enabled in (False, True):
                process, output, status = cli_fixtures.ExtendedCLITests.invoke(self, folder, str(enabled),
                    '--compiler-backend', 'relational', '--target-mode', 'separate',
                    '--positive-core-pruning' if enabled else '--no-positive-core-pruning',
                    '--enumerate-optima', '--expand-variants', inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertTrue(json.loads((output / 'variants.json').read_text())['all_optimal_programs_complete'])
                children = list((output / 'targets').glob('*/catalogue.json'))
                self.assertEqual(len(children), 2)
                for path in children:
                    self.assertEqual(json.loads(path.read_text())['stats']['positive_core_pruning'], enabled)
                    self.assertEqual(json.loads(path.with_name('manifest.json').read_text())
                                     ['arguments']['positive_core_pruning'], enabled)
                    self.assertEqual(json.loads(path.with_name('variants.json').read_text())
                                     ['reconstruction']['positive_core_pruning'], enabled)

    def test_benchmark_forwards_ablation_and_exact_target(self):
        from nidavellir.benchmarks.run_benchmarks import command_for, parser
        common = ['--case', 'synthetic', '--inputs-root', 'inputs', '--output', 'output']
        args = parser().parse_args(common)
        self.assertNotIn('--positive-core-pruning', command_for(args, Path('inputs'), Path('output')))
        for flag in ('--positive-core-pruning', '--no-positive-core-pruning'):
            args = parser().parse_args(common + [flag, '--target-signature', 'answer/1'])
            command = command_for(args, Path('inputs'), Path('output'))
            self.assertIn(flag, command)
            self.assertEqual(command[command.index('--target-signature') + 1], 'answer/1')

    def test_frontier_stop_hook_compares_dfs_position_not_pruned_node_count(self):
        from nidavellir.benchmarks.positive_core_prefix import instrument
        from nidavellir import relational_compile
        task = self.task('p(x). q(y). r(a). r(n1). s(a). s(n2).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(3))', 's(var__(3))'],
            max_body=3, heads=('answer(var__(3))',),
            examples='positive(answer(a)). negative(answer(n1)). negative(answer(n2)).')
        reference = None
        for enabled in (False, True):
            oracle = CoverageOracle(task, positive_core_pruning=enabled)
            if enabled:
                oracle.positive_cores.learn(task.heads[0], (0, 1), 1)
            method, _ = instrument(reference, 12)
            with patch.object(relational_compile._Search, 'walk', method), self.assertRaises(ResourceLimit) as caught:
                compile_catalogue(task, oracle, order='source')
            stats = caught.exception.stats
            self.assertTrue(stats['validation_frontier_reached'])
            if reference is None:
                reference = stats
            else:
                self.assertGreaterEqual(tuple(stats['frontier']), tuple(reference['frontier']))
                self.assertEqual(stats['partial_classes'], reference['partial_classes'])

    def test_invalid_flag_and_empty_positive_domain(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='')
        for flag in (None, 0, 'on'):
            with self.assertRaises(ValueError):
                CoverageOracle(task, positive_core_pruning=flag)
        oracle = CoverageOracle(task, positive_core_pruning=True)
        oracle.positive_cores.learn(task.heads[0], (), 1)
        self.assertEqual(oracle.positive_cores.match(task.heads[0], 1), 0)

    def test_default_is_opt_in_for_api_and_cli(self):
        from nidavellir.__main__ import parser
        task = self.task('p(a).', ['p(var__(1))'], examples='positive(answer(a)).')
        self.assertIsNone(CoverageOracle(task).positive_cores)
        self.assertIs(parser().get_default('positive_core_pruning'), False)


if __name__ == '__main__':
    unittest.main()
