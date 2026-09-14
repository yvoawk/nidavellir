"""Public sequential mode preserves grammar, optima and honest partial states."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nidavellir.__main__ import parser
from nidavellir import target_execution as execution
from nidavellir.task import _bias, _model
from tests import test_optimized_oracle as fixtures
from tests import test_cli_extended as cli
from tests import test_per_target_bounds as bounds


class TargetExecutionTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task
    invoke = cli.ExtendedCLITests.invoke
    sample = bounds.PerTargetBoundTests.sample

    def inputs(self, folder, task):
        path = Path(folder) / 'inputs'
        path.mkdir()
        (path / 'background.lp').write_text(task.background)
        (path / 'bias.lp').write_text(task.bias_source)
        task.write_examples(path / 'examples.lp')
        return path

    def args(self, inputs, output, *extra):
        return parser().parse_args(['--background', str(inputs / 'background.lp'),
            '--bias', str(inputs / 'bias.lp'), '--examples', str(inputs / 'examples.lp'),
            '--output', str(output), '--bound-seconds', '5', *extra])

    def test_default_fresh_sequential_calls_match_grouped_optimum_and_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            programs = []
            for mode, extra in [('default', []), ('grouped', ['--target-mode', 'grouped'])]:
                process, output, status = self.invoke(folder, mode, *extra, '--enumerate-optima',
                                                      '--expand-variants', inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status['optimal_cost'], 5)
                self.assertTrue(status['arguments']['per_target_bounds'])
                self.assertEqual(json.loads(process.stdout), status)
                expansion = execution.read_json(output / 'variants.json')
                self.assertTrue(expansion['all_optimal_programs_complete'])
                programs.append({tuple(sorted(p['rules'])) for p in expansion['programs']})
                if mode == 'default':
                    plan = execution.read_json(output / 'targets.json')['targets']
                    self.assertEqual([p['signature'] for p in plan], ['first/1', 'second/1'])
                    self.assertEqual(len({p['pid'] for p in plan}), 2)
                    self.assertLessEqual(plan[0]['finished_seconds'], plan[1]['started_seconds'])
                    self.assertEqual([p['result']['optimal_cost'] for p in plan], [2, 3])
                    self.assertTrue(all(p['result']['compiler_backend'] == 'bitmask' for p in plan))
                    self.assertEqual(execution.read_json(output / 'bound.json')['cost'], 5)
                    self.assertTrue(execution.read_json(output / 'verification.json')['matched'])
            self.assertEqual(*programs)

    def test_full_bias_metadata_and_same_signature_templates_are_preserved(self):
        task = self.task('p(b). q(c).', ['p(var__(1))', 'q(var__(1))'],
            ['first(a)', 'first(b)', 'second(c)'], ['first(n)', 'second(n)'],
            heads=('first(var__(1))', 'first(a)', 'second(var__(1))'),
            bias=':- #count { H : candidate_head(H) } != 3.')
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, task)
            process, output, status = self.invoke(folder, 'run', '--enumerate-optima', '--expand-variants', inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(status['optimal_cost'], 5)
            plan = execution.read_json(output / 'targets.json')['targets']
            self.assertEqual(len(plan), 2)
            self.assertEqual(len(plan[0]['head_templates']), 2)
            for entry in plan:
                self.assertEqual((output / entry['output'] / 'inputs/bias.lp').read_text(), task.bias_source)

    def test_predicate_arity_partition_and_empty_positive_target(self):
        task = self.task('p(a).', ['p(var__(1))'], ['answer(a)'], ['answer(b)', 'answer(b,c)'],
                         heads=('answer(var__(1))', 'answer(var__(1),var__(2))'))
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, task)
            process, output, status = self.invoke(folder, 'run', '--enumerate-optima', '--expand-variants', inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(status['optimal_cost'], 2)
            plan = execution.read_json(output / 'targets.json')['targets']
            self.assertEqual([p['signature'] for p in plan], ['answer/1', 'answer/2'])
            self.assertEqual([p['result']['optimal_cost'] for p in plan], [2, 0])
            self.assertEqual(execution.read_json(output / 'variants.json')['program_count'], 1)

    def test_preparation_flattens_includes_and_pools_without_grounding_BK(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            (inputs / 'facts.lp').write_text((inputs / 'background.lp').read_text())
            (inputs / 'background.lp').write_text('#include "facts.lp".')
            (inputs / 'examples.lp').write_text('positive(first(a);second(b)). negative(first(n1);second(n1);second(n2)).')
            output = Path(folder) / 'run'
            output.mkdir()
            args = self.args(inputs, output)
            def labels_only(source, kind, diagnostics):
                self.assertNotEqual(kind, 'BK', 'No parent BK grounding')
                return _model(source, kind, diagnostics)
            with patch('nidavellir.task._model', side_effect=labels_only):
                plan = execution.prepare(args, output, _bias(args.bias))
            self.assertNotIn('#include', (output / 'inputs/background.lp').read_text())
            self.assertEqual([p['positive_count'] for p in plan], [1, 1])
            self.assertEqual([p['negative_count'] for p in plan], [1, 2])
            self.assertNotIn('second(', (output / plan[0]['examples']).read_text())

    def test_single_signature_preserves_flat_artifacts_and_unknown_signature_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.task('p(a).', ['p(var__(1))'], ['first(a)'], heads=('first(var__(1))',)))
            process, output, status = self.invoke(folder, 'single', inputs=inputs)
            self.assertEqual(process.returncode, 0)
            self.assertTrue((output / 'catalogue.json').exists())
            self.assertFalse((output / 'targets').exists())
            process, output, status = self.invoke(folder, 'invalid', '--target-signature', 'other/1', inputs=inputs)
            self.assertEqual(status['status'], 'invalid_input')
            self.assertEqual(process.returncode, 2)

    def test_child_options_preserve_ablation_cache_and_resource_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            output = Path(folder) / 'run'
            args = self.args(inputs, output, '--no-per-target-bounds', '--compiler-backend', 'relational',
                '--no-prefix-completion', '--enumerate-optima', '--expand-variants', '--max-programs', '2',
                '--oracle-cache', folder, '--compile-seconds', '17')
            command = execution.child_command(args, output, {'examples': 'labels.lp', 'output': 'target', 'signature': 'first/1'})
            forwarded = parser().parse_args(command[3:])
            self.assertFalse(forwarded.per_target_bounds)
            self.assertEqual(forwarded.target_mode, 'grouped')
            self.assertEqual(forwarded.target_signature, 'first/1')
            self.assertEqual(forwarded.compile_seconds, 17)
            self.assertEqual(forwarded.max_programs, 2)
            self.assertTrue(forwarded.no_prefix_completion)
            self.assertEqual(forwarded.oracle_cache, Path(folder))

    def test_unsatisfiable_target_does_not_hide_completed_target(self):
        task = self.task('p(a).', ['p(var__(1))'], ['first(a)', 'second(b)'],
                         heads=('first(var__(1))', 'second(var__(1))'))
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, task)
            process, output, status = self.invoke(folder, 'run', '--enumerate-optima', '--expand-variants', inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(status['status'], 'unsatisfiable')
            self.assertIsNone(status['optimal_cost'])
            self.assertFalse((output / 'learned.lp').exists())
            self.assertTrue(execution.read_json(output / 'variants.json')['all_optimal_programs_complete'])
            plan = execution.read_json(output / 'targets.json')['targets']
            self.assertTrue((output / plan[0]['output'] / 'learned.lp').exists())

    def test_failed_first_child_does_not_prevent_later_target_or_invent_global_optimum(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            output = Path(folder) / 'run'
            args = self.args(inputs, output)
            original = execution.launch
            calls = []
            def first_fails(command, *rest):
                calls.append(command)
                return -9 if len(calls) == 1 else original(command, *rest)
            with (patch.object(execution, 'launch', side_effect=first_fails),
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
                code = execution.dispatch_separate(args)
            self.assertEqual(code, 2)
            self.assertEqual(len(calls), 2)
            status = execution.read_json(output / 'status.json')
            self.assertEqual(status['status'], 'error')
            self.assertIsNone(status['optimal_cost'])
            self.assertFalse((output / 'learned.lp').exists())
            plan = execution.read_json(output / 'targets.json')['targets']
            self.assertEqual(plan[1]['result']['optimal_cost'], 3)
            self.assertTrue((output / plan[1]['output'] / 'learned.lp').exists())

    def test_cartesian_semantic_limits_and_deadlines_keep_namespaced_ids(self):
        parts = [('x/1', {'complete': True, 'semantic_hypotheses': [['c0'], ['c1']]}),
                 ('y/1', {'complete': True, 'semantic_hypotheses': [['c0'], ['c1']]})]
        result = execution.product_report(parts, variants=False, cost=4, limit=4, seconds=10)
        self.assertTrue(result['complete'])
        self.assertEqual(result['hypothesis_count'], 4)
        self.assertEqual(result['semantic_hypotheses'][0], ['x/1:c0', 'y/1:c0'])
        result = execution.product_report(parts, variants=False, cost=4, limit=3, seconds=10)
        self.assertFalse(result['complete'])
        self.assertEqual(result['hypothesis_count'], 3)
        result = execution.product_report(parts, variants=False, cost=4, limit=None, seconds=0)
        self.assertEqual(result['status'], 'timeout')
        parts[0][1]['complete'] = False
        result = execution.product_report(parts, variants=False, cost=4, limit=None, seconds=10)
        self.assertFalse(result['complete'])

    def test_global_variant_cap_preserves_proven_cost_without_claiming_completeness(self):
        task = self.task('p(a). q(a).', ['p(var__(1))', 'q(var__(1))'], ['first(a)', 'second(a)'],
                         heads=('first(var__(1))', 'second(var__(1))'))
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, task)
            process, output, status = self.invoke(folder, 'run', '--enumerate-optima', '--expand-variants',
                                                  '--max-programs', '3', inputs=inputs)
            self.assertEqual(process.returncode, 3, process.stdout + process.stderr)
            self.assertEqual(status['status'], 'enumeration_incomplete')
            self.assertEqual(status['optimal_cost'], 4)
            expansion = execution.read_json(output / 'variants.json')
            self.assertEqual(expansion['program_count'], 3)
            self.assertFalse(expansion['all_optimal_programs_complete'])

    def test_resume_requires_one_checkpoint_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, self.sample())
            process, output, status = self.invoke(folder, 'run', '--resume-bound', 'checkpoint.json', inputs=inputs)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(status['status'], 'invalid_input')
            self.assertIn('one target checkpoint', status['reason'])

    def test_shared_disk_cache_reuses_profiles_but_recomputes_target_label_masks(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'],
            ['first(a)', 'second(b)'], ['first(b)', 'second(a)'],
            heads=('first(var__(1))', 'second(var__(1))'))
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.inputs(folder, task)
            cache = Path(folder) / 'cache'
            for attempt in ('cold', 'warm'):
                process, output, status = self.invoke(folder, attempt, '--oracle-cache', cache, inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(status['optimal_cost'], 4)
                plan = execution.read_json(output / 'targets.json')['targets']
                records = [execution.read_json(output / p['output'] / 'oracle.json')['stats']['cache'] for p in plan]
                if attempt == 'cold':
                    self.assertFalse(records[0]['hit'])
                    self.assertTrue(records[1]['hit'])
                    self.assertEqual(records[1]['scope'], 'unlabelled_profiles')
                else:
                    self.assertTrue(all(c['hit'] and c['scope'] == 'exact' for c in records))
                self.assertIn('first(V1) :- p(V1).', (output / 'learned.lp').read_text())
                self.assertIn('second(V1) :- q(V1).', (output / 'learned.lp').read_text())
