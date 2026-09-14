"""Bitmask target bounds versus exhaustive rules, including worker failures."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import clingo

from nidavellir.bitmask_bound import discover_target_bounds
from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.oracle import ResourceLimit
from nidavellir.search import compile_catalogue, mine_variants
from nidavellir.target_bounds import costs_from_certificates, target_view
from nidavellir.verification import verify_rules
from tests import test_optimized_oracle as fixtures
from tests import test_cli_extended as cli_fixtures
from tests import test_per_target_bounds as target_fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class BitmaskTargetBoundTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task
    invoke = cli_fixtures.ExtendedCLITests.invoke
    sample = target_fixtures.PerTargetBoundTests.sample

    def compare(self, task, bounds):
        exhaustive = exhaustive_catalogue(task)
        full = (1 << len(task.positives)) - 1
        optimum = optimal_programs(((m, c) for m, c, _ in exhaustive), full, 30)
        spelling = {(m, c): v for m, c, v in exhaustive}
        indices, _ = build_optimized_indices(task, seconds=10)
        before = [(i.head, i.literal_masks, i.positive_heads, i.negative_mask) for i in indices]
        for order in ("source", "selectivity"):
            for mode in ("exhaustive", "targeted", "representative"):
                for global_bound in (None, 30):
                    rows, stats = compile_catalogue(task, indices, target_bounds=bounds,
                        bound=global_bound, order=order, variants=mode, alpha_collect=False)
                    self.assertTrue(stats['complete'])
                    self.assertTrue(all(s['search_complete'] for s in stats['target_search'].values()))
                    self.assertEqual(optimal_programs(((r['coverage'], r['cost']) for r in rows), full, 30), optimum)
                    if mode == 'representative':
                        rows, rebuilt = mine_variants(task, indices, rows, alpha_collect=False)
                        self.assertTrue(rebuilt['complete'])
                    for mask, cost, variants in normalized(rows):
                        self.assertEqual(variants, spelling[(mask, cost)])
        self.assertEqual(before, [(i.head, i.literal_masks, i.positive_heads, i.negative_mask) for i in indices])
        return rows, stats

    def test_different_costs_keep_joint_optima_global_positions_and_variants(self):
        rows, stats = self.compare(self.sample(), {('first', 1): 2, ('second', 1): 3})
        self.assertEqual({r['coverage'] for r in rows}, {1, 2})
        self.assertEqual(stats['target_effective_body_limits'], {'first/1': 1, 'second/1': 2})

    def test_missing_bound_leaves_other_target_grammar(self):
        self.compare(self.sample(), {('first', 1): 2})

    def test_multi_rule_cover_and_equal_cost_alternatives_survive(self):
        task = self.task('p(a). q(b). r(a). s(c).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))', 's(var__(1))'],
            ['first(a)', 'first(b)', 'second(c)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'second(var__(1))'), max_body=3)
        self.compare(task, {('first', 1): 4, ('second', 1): 2})
        indices, _ = build_optimized_indices(task)
        cover, report = discover_target_bounds(task, indices, 5, lambda rules: verify_rules(task, rules))
        self.assertEqual(len(cover), 3)
        self.assertEqual(costs_from_certificates(task, report['target_bounds']), {('first', 1): 4, ('second', 1): 2})

    def test_overlapping_templates_share_empty_rule_minimum_cost(self):
        task = self.task('p(b). q(c).', ['p(var__(1))', 'q(var__(1))'],
            ['first(a)', 'first(b)', 'second(c)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'first(a)', 'second(var__(1))'))
        rows, stats = self.compare(task, {('first', 1): 3, ('second', 1): 2})
        self.assertEqual(stats['target_minimum_useful_rule_cost'], {'first/1': 1, 'second/1': 2})
        indices, _ = build_optimized_indices(task)
        cover, report = discover_target_bounds(task, indices, 5, lambda rules: verify_rules(task, rules))
        self.assertEqual(report['cost'], 5)
        self.assertEqual(costs_from_certificates(task, report['target_bounds']), {('first', 1): 3, ('second', 1): 2})

    def test_cyclic_guards_preserve_complete_legal_bodies(self):
        task = self.task('p(a). q(a). r(b).', ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['first(a)', 'second(b)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'second(var__(1))'), max_body=3,
            bias=':- body(p(var__(1))), not body(q(var__(1))).\n:- body(q(var__(1))), not body(p(var__(1))).')
        self.compare(task, {('first', 1): 3, ('second', 1): 2})

    def test_nested_repeated_hidden_variables_and_stratified_negative_BK(self):
        task = self.task('base(f(a),x,x). base(f(n),x,y). edge(f(b),x). blocked(f(n)).\n'
                         'p(X,Y,Z) :- base(X,Y,Z), not blocked(X).',
            ['p(var__(1),var__(2),var__(2))', 'p(var__(1),var__(2),var__(3))', 'edge(var__(1),var__(2))'],
            ['first(f(a))', 'second(f(b))'], ['first(f(n))', 'second(f(n))'],
            heads=('first(var__(1))', 'second(var__(1))'), max_body=2)
        self.compare(task, {('first', 1): 2, ('second', 1): 2})

    def test_invalid_maps_and_zero_budget_never_return_partial_catalogue(self):
        task = self.sample()
        indices, _ = build_optimized_indices(task)
        for value in ([], {('unknown', 1): 2}, {('first', 1): 0}, {('first', 1): True}, {('first', 1): -1}):
            with self.assertRaises(ValueError):
                compile_catalogue(task, indices, target_bounds=value)
        with self.assertRaises(ResourceLimit):
            compile_catalogue(task, indices, target_bounds={('first', 1): 2}, seconds=0)

    def test_partial_certificate_survives_interruption_without_global_cover(self):
        task = self.sample()
        indices, _ = build_optimized_indices(task)
        snapshots = []
        def stop(state):
            snapshots.append(copy.deepcopy(state))
            raise ResourceLimit('after first independently verified target')
        cover, report = discover_target_bounds(task, indices, 5, lambda rules: verify_rules(task, rules),
            checkpoint=stop, fingerprint='current')
        self.assertIsNone(cover)
        self.assertEqual(report['status'], 'timeout')
        self.assertEqual(costs_from_certificates(task, report['target_bounds']), {('first', 1): 2})
        self.assertIsNone(snapshots[0]['incumbent'])
        self.assertEqual(snapshots[0]['fingerprint'], 'current')

    def test_no_bound_does_not_mean_unsatisfiable(self):
        task = self.sample()
        indices, _ = build_optimized_indices(task)
        cover, report = discover_target_bounds(task, indices, 0, lambda rules: verify_rules(task, rules))
        self.assertIsNone(cover)
        self.assertEqual(report['status'], 'disabled')
        self.assertEqual(report['target_bounds'], {})
        self.compare(task, {})

    def test_cli_auto_and_explicit_bitmask_preserve_complete_program_enumeration(self):
        task = self.sample()
        programs = []
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / 'inputs'
            inputs.mkdir()
            (inputs / 'background.lp').write_text(task.background)
            (inputs / 'bias.lp').write_text(task.bias_source)
            task.write_examples(inputs / 'examples.lp')
            for mode, extra in [('global', ['--no-per-target-bounds']), ('auto', ['--per-target-bounds']),
                                ('explicit', ['--per-target-bounds', '--compiler-backend', 'bitmask'])]:
                process, output, status = self.invoke(folder, mode, *extra,
                    '--target-mode', 'grouped', '--enumerate-optima', '--expand-variants', inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status['compiler_backend'], 'bitmask')
                self.assertEqual(status['optimal_cost'], 5)
                variants = json.loads((output / 'variants.json').read_text())
                self.assertTrue(variants['all_optimal_programs_complete'])
                programs.append({tuple(p['rules']) for p in variants['programs']})
                if mode != 'global':
                    bound = json.loads((output / 'bound.json').read_text())
                    self.assertEqual(costs_from_certificates(task, bound['target_bounds']),
                                     {('first', 1): 2, ('second', 1): 3})
                    self.assertTrue((output / 'bound_checkpoint.json').exists())
            self.assertTrue(all(p == programs[0] for p in programs))

    def test_worker_streams_actual_negative_labels_for_target_certification(self):
        from nidavellir.__main__ import bound_in_subprocess
        task = self.task('p(a). p(n). q(b).', ['p(var__(1))', 'q(var__(1))'],
            ['first(a)', 'second(b)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'second(var__(1))'))
        indices, _ = build_optimized_indices(task)
        lying_indices = [replace(index, negative_mask=0) for index in indices]
        with self.assertRaisesRegex(RuntimeError, 'Target incumbent failed original-BK verification'):
            bound_in_subprocess(task, lying_indices, 5, per_target_bounds=True)

    def test_hard_bitmask_worker_timeout_preserves_only_matching_certificates(self):
        from nidavellir.__main__ import bound_in_subprocess
        task = self.sample()
        indices, _ = build_optimized_indices(task)
        snapshots = []
        def stop(state):
            snapshots.append(copy.deepcopy(state))
            raise ResourceLimit('after first target')
        discover_target_bounds(task, indices, 5, lambda rules: verify_rules(task, rules),
                               checkpoint=stop, fingerprint='current')
        for fingerprint in ('current', 'stale'):
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / 'checkpoint.json'
                state = dict(snapshots[0], fingerprint=fingerprint, _verification={})
                class Worker:
                    pid = 123456789
                    calls = 0
                    def communicate(self, timeout=None):
                        self.calls += 1
                        if self.calls == 1:
                            checkpoint.write_text(json.dumps(state))
                            raise subprocess.TimeoutExpired('fake-worker', timeout)
                        return b'', b''
                with patch('nidavellir.__main__.subprocess.Popen', return_value=Worker()), \
                     patch('nidavellir.__main__.os.killpg'):
                    cover, report = bound_in_subprocess(task, indices, 5, per_target_bounds=True,
                        fingerprint='current', checkpoint_path=checkpoint)
                self.assertIsNone(cover)
                self.assertEqual(costs_from_certificates(task, report.get('target_bounds', {})),
                                 {('first', 1): 2} if fingerprint == 'current' else {})

    def test_search_shares_one_traversal_and_original_indices(self):
        from nidavellir.search import _Traversal
        task = self.sample()
        indices, _ = build_optimized_indices(task)
        with patch('nidavellir.bitmask_bound._Traversal', wraps=_Traversal) as constructor:
            cover, report = discover_target_bounds(task, indices, 5, lambda rules: verify_rules(task, rules))
        self.assertEqual(constructor.call_count, 1)
        self.assertIs(constructor.call_args.args[0], task)
        self.assertIs(constructor.call_args.args[1], indices)
        self.assertEqual(report['cost'], 5)
        self.assertTrue(verify_rules(task, [c['rule'] for c in cover])['matched'])

    def test_non_stratified_bias_and_unsupervised_outputs(self):
        task = self.task('p(a). p(unlabelled). q(a). r(b).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['first(a)', 'second(b)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'second(var__(1))'),
            bias='left :- not right. right :- not left.\n:- left, body(p(var__(1))), body(q(var__(1))).')
        self.compare(task, {('first', 1): 2, ('second', 1): 2})

    def test_zero_cost_empty_target_does_not_shift_positive_positions(self):
        task = self.task('p(a).', ['p(var__(1))'], ['first(a)'], ['second(n)'],
                         heads=('first(var__(1))', 'second(var__(1))'))
        self.compare(task, {('first', 1): 2, ('second', 1): 0})
