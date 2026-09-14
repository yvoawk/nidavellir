"""Signed bias cores checked against independent exhaustive ASP legality."""
from itertools import combinations, product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nidavellir.bias_analysis import analyze, _HeadSolver
from nidavellir import bias_cores
from nidavellir.oracle import ResourceLimit
from nidavellir.task import TaskError
from tests import test_bias_analysis as fixtures
from tests import test_bias_certificates_review as certificates


class BiasCoreTests(unittest.TestCase):
    task = fixtures.BiasAnalysisTests.task
    ids = staticmethod(fixtures.BiasAnalysisTests.ids)

    def sample(self):
        return self.task([f'{p}(var__(1))' for p in 'pqrs'], max_body=4,
            restrictions=':- body(p(var__(1))), not body(q(var__(1))).')

    def test_every_signed_prefix_matches_exhaustive_legality(self):
        for name, restrictions in certificates.BiasCertificateReviewTests.CASES.items():
            task = self.task([f'{p}(var__(1))' for p in 'pqrs'], restrictions=restrictions, max_body=4)
            head = task.heads[0]
            legal = [frozenset(body) for size in range(5) for body in combinations(range(4), size)
                     if task.accepts(head, body)]
            for checked in (False, True):
                with self.subTest(bias=name, checked=checked):
                    solver = _HeadSolver(task, head, [], core_pruning=True)
                    for states in product(range(3), repeat=4):
                        included = {i for i,s in enumerate(states) if s == 1}
                        excluded = {i for i,s in enumerate(states) if s == 2}
                        expected = any(included <= body and not excluded & body for body in legal)
                        self.assertEqual(solver.find(included, excluded,
                            check=(lambda: None) if checked else None) is not None, expected)
                        for core in solver.cores.cores:
                            for body in legal:
                                assumptions = {atom if i in body else -atom
                                    for i,atom in enumerate(solver.body_atoms) if atom is not None}
                                self.assertFalse(core <= assumptions)

    def test_absence_is_required_and_adding_literal_repairs_illegality(self):
        task = self.sample()
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        p,q,r = self.ids(task, 'p(var__(1))','q(var__(1))','r(var__(1))')
        self.assertIsNone(solver.find((p,), (q,)))
        before = solver.stats['solves']
        self.assertIsNone(solver.find((p,r), (q,)))
        self.assertEqual(solver.stats['solves'], before)
        self.assertIsNotNone(solver.find((p,q,r), ()))
        self.assertGreater(solver.stats['bias_core_hits'], 0)

    def test_guard_gate_proofs_are_not_exported_as_unconditional_cores(self):
        task = self.sample()
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        p,q = self.ids(task, 'p(var__(1))','q(var__(1))')
        for _ in range(4):
            self.assertIsNone(solver.find((p,), omit_some=(q,), check=lambda: None))
            self.assertFalse(solver.cores.cores)
            self.assertIsNotNone(solver.find((p,q), check=lambda: None))
        self.assertIsNone(solver.find((p,), (q,), check=lambda: None))
        self.assertTrue(solver.cores.cores)
        self.assertIsNotNone(solver.find((p,q)))

    def test_scopes_are_separate_for_heads_and_tasks(self):
        task = self.task(['p(var__(1))','q(var__(1))'], heads=('left(var__(1))','right(var__(1))'),
            examples='', restrictions=':- head(left(var__(1))), body(p(var__(1))), not body(q(var__(1))).')
        proof = analyze(task, core_pruning=True)
        p,q = self.ids(task, 'p(var__(1))','q(var__(1))')
        left, right = (proof.heads[head]._checker for head in task.heads)
        self.assertIsNone(left.find((p,), (q,)))
        self.assertIsNotNone(right.find((p,), (q,)))
        self.assertIsNot(left.cores, right.cores)
        fresh = self.sample()
        self.assertFalse(_HeadSolver(fresh, fresh.heads[0], [], core_pruning=True).cores.cores)

    def test_pure_negative_and_empty_cores(self):
        task = self.task(['p(var__(1))','q(var__(1))'], heads=('answer',), examples='',
            restrictions=':- not body(p(var__(1))), not body(q(var__(1))).')
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        self.assertIsNone(solver.find((), (0,1)))
        self.assertTrue(any(core and all(lit < 0 for lit in core) for core in solver.cores.cores))
        self.assertIsNotNone(solver.find((0,), (1,)))
        impossible = self.task(['q(var__(2))'])  # No body can bind the head variable.
        other = _HeadSolver(impossible, impossible.heads[0], [], core_pruning=True)
        self.assertIsNone(other.find())
        self.assertIn(frozenset(), other.cores.cores)
        before = other.stats['solves']
        self.assertIsNone(other.find((0,), ()))
        self.assertEqual(other.stats['solves'], before)

    def test_aliases_for_forced_atoms_and_absent_inputs(self):
        task = self.task(['p(var__(1))','q(var__(1))','r(var__(1))'],
            body_choice='body(p(var__(1))). body(q(var__(1))).')
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        for states in product(range(3), repeat=3):
            included = {i for i,s in enumerate(states) if s == 1}
            excluded = {i for i,s in enumerate(states) if s == 2}
            expected = any(included <= set(body) and not excluded & set(body) and task.accepts(task.heads[0], body)
                           for n in range(4) for body in combinations(range(3), n))
            self.assertEqual(solver.find(included,excluded) is not None, expected)

    def test_interrupted_unsat_is_not_published_or_cached(self):
        task = self.sample()
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        p,q = self.ids(task, 'p(var__(1))','q(var__(1))')
        def stop_after_launch():
            if solver.stats['solves']:
                raise ResourceLimit('caller deadline')
        with self.assertRaises(ResourceLimit):
            solver.completion((p,), (q,), check=stop_after_launch)
        self.assertFalse(solver.cores.cores)
        self.assertFalse(solver.cache)
        self.assertFalse(solver.completion((p,), (q,)))
        self.assertTrue(solver.cores.cores)

    def test_unknown_and_incomplete_unsat_do_not_publish_proofs(self):
        for unknown, exhausted in ((True, False), (False, False)):
            task = self.sample()
            solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
            def solve(**kwargs):
                kwargs['on_core']([])
                return SimpleNamespace(unknown=unknown, unsatisfiable=True, exhausted=exhausted, satisfiable=False)
            solver.control = SimpleNamespace(solve=solve)
            with self.assertRaises(TaskError):
                solver.completion((0,), (1,))
            self.assertFalse(solver.cores.cores)
            self.assertFalse(solver.cache)

    def test_capacity_antichain_and_unmapped_assumptions(self):
        bank = bias_cores.BiasCores({})
        with patch.object(bias_cores, 'MAX_CORES', 2):
            bank.store((1,-2,3), {1,-2,3})
            bank.store((1,-2), {1,-2})
            self.assertEqual(len(bank.cores), 1)
            self.assertTrue(bank.match((1,3), {1,3,-2}))
            self.assertFalse(bank.match((1,2,3), {1,2,3}))
            bank.store((4,), {4})
            bank.store((5,), {5})
            self.assertEqual(len(bank.cores), 2)
            self.assertFalse(bank.match((1,), {1,-2}))
            bank.store((999,), {1})
            self.assertNotIn(frozenset((999,)), bank.cores)
        with patch.object(bias_cores, 'MAX_CORES', 0):
            fresh = bias_cores.BiasCores({})
            fresh.store((), set())
            self.assertFalse(fresh.match((), set()))

    def test_grouped_lookup_matches_naive_signed_subset_for_every_assignment(self):
        bank = bias_cores.BiasCores({})
        for core in ((1,-2),(-1,3),(2,-4,5),(1,2,6),(-1,-3,-5),(2,3,4),(2,3,5),(1,4,5)):
            bank.store(core,set(core))
        for states in product(range(3),repeat=6):
            present = {i+1 for i,state in enumerate(states) if state == 1}
            assumptions = present | {-i-1 for i,state in enumerate(states) if state == 2}
            expected = any(core <= assumptions for core in bank.cores)
            self.assertEqual(bank.match(present,assumptions), expected)

    def test_enabled_analysis_has_same_guards_and_validates_flag(self):
        task = self.sample()
        off, on = analyze(task), analyze(task, core_pruning=True)
        for head in task.heads:
            self.assertEqual(off.heads[head], on.heads[head])
        for flag in (0, 'yes'):
            with self.assertRaises(ValueError):
                analyze(task, core_pruning=flag)
        task._bias_core_pruning = True
        self.assertIsNotNone(analyze(task).heads[task.heads[0]]._checker.cores)
        self.assertIsNone(analyze(task, core_pruning=False).heads[task.heads[0]]._checker.cores)

    def test_full_size_core_does_not_evict_a_reusable_one(self):
        task = self.task(['p(var__(1))','q(var__(1))','r(var__(1))'], max_body=2,
            restrictions=':- body(p(var__(1))), body(q(var__(1))).')
        solver = _HeadSolver(task, task.heads[0], [], core_pruning=True)
        p,q = self.ids(task,'p(var__(1))','q(var__(1))')
        self.assertIsNone(solver.find((p,q)))
        self.assertFalse(solver.cores.cores)
        self.assertEqual(solver.stats['bias_core_full_body_skips'], 1)
        self.assertIsNotNone(solver.find((p,), (q,)))

    def test_catalogues_variants_and_multi_rule_optima_match_reference(self):
        from nidavellir.relational_coverage import CoverageOracle
        from nidavellir.relational_compile import compile_catalogue, mine_variants
        from tests.test_relational_compile import exhaustive_catalogue, normalized
        from tests.test_program_bound_pruning import optimal_programs
        hits = 0
        for restriction in certificates.BiasCertificateReviewTests.CASES.values():
            task = self.task([f'{p}(var__(1))' for p in 'pqrs'], restrictions=restriction,
                background='p(a). p(c). q(a). q(b). r(b). r(c). s(a). s(b).', max_body=4,
                examples='positive(answer(a)). positive(answer(b)). negative(answer(c)).')
            reference = exhaustive_catalogue(task)
            expected = optimal_programs(((m,c) for m,c,_ in reference), 3, 20)
            for enabled in (False, True):
                for ground in (False, True):
                    for mode in ('exhaustive', 'representative'):
                        task._bias_analysis = analyze(task, core_pruning=enabled)
                        oracle = CoverageOracle(task)
                        rows, stats = compile_catalogue(task, oracle, variants=mode, alpha_collect=False,
                                                        ground_prefix_filter=ground)
                        self.assertTrue(stats['complete'])
                        if mode == 'representative':
                            rows, reconstruction = mine_variants(task, oracle, rows, alpha_collect=False,
                                                                ground_prefix_filter=ground)
                            self.assertTrue(reconstruction['complete'])
                        self.assertEqual(normalized(rows), reference)
                        self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),3,20), expected)
                        hits += sum(p.checker_stats.get('bias_core_hits',0) for p in task._bias_analysis.heads.values())
        self.assertGreater(hits, 0)

    def test_cli_propagates_to_sequential_targets_and_variants(self):
        from tests import test_cli_extended as cli
        task = self.task(['p(var__(1))','q(var__(1))'],
            heads=('first(var__(1))','second(var__(1))'),
            background='p(a). p(b). p(n1). q(a). q(b). q(n2).',
            examples='positive(first(a)). positive(first(b)). positive(second(b)). '
                     'negative(first(n1)). negative(first(n2)). negative(second(n1)). negative(second(n2)).')
        outputs = []
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / 'inputs'
            inputs.mkdir()
            (inputs / 'background.lp').write_text(task.background)
            (inputs / 'bias.lp').write_text(task.bias_source)
            task.write_examples(inputs / 'examples.lp')
            for enabled in (False, True):
                process, output, status = cli.ExtendedCLITests.invoke(self,folder,str(enabled),
                    '--compiler-backend','relational','--target-mode','separate',
                    '--bias-core-pruning' if enabled else '--no-bias-core-pruning',
                    '--enumerate-optima','--expand-variants', inputs=inputs)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                variants = json.loads((output / 'variants.json').read_text())
                self.assertTrue(variants['all_optimal_programs_complete'])
                outputs.append(variants['programs'])
                for path in (output / 'targets').glob('*/catalogue.json'):
                    manifest = json.loads(path.with_name('manifest.json').read_text())
                    self.assertEqual(manifest['arguments']['bias_core_pruning'], enabled)
                    for name in ('catalogue.json','variant_catalogue.json'):
                        stats = json.loads(path.with_name(name).read_text())['stats']
                        for proof in stats['profile']['diagnostics']['bias_queries'].values():
                            self.assertEqual(proof['core_pruning'], enabled)
            self.assertEqual(outputs[0], outputs[1])

    def test_benchmark_ablation_reaches_command(self):
        from nidavellir.benchmarks.run_benchmarks import parser, command_for
        common = ['--case','synthetic','--inputs-root','inputs','--output','output']
        for flag in ('--bias-core-pruning','--no-bias-core-pruning'):
            args = parser().parse_args(common + [flag])
            self.assertIn(flag, command_for(args,Path('inputs'),Path('output')))

    def test_independent_core_check_detects_missing_negative_condition(self):
        from nidavellir.benchmarks.verify_bias_cores import verify
        task = self.sample()
        proof = analyze(task, core_pruning=True)
        head = task.heads[0]
        p,q = self.ids(task,'p(var__(1))','q(var__(1))')
        self.assertIsNone(proof.heads[head]._checker.find((p,), (q,)))
        report = proof.export()
        self.assertTrue(verify(task.bias_source,list(map(str,task.literals)),task.max_body,report)['matched'])
        report[str(head)]['cores']['cores'] = [[proof.heads[head]._checker.body_atoms[p]]]
        self.assertFalse(verify(task.bias_source,list(map(str,task.literals)),task.max_body,report)['matched'])

    def test_bitmask_catalogues_and_reconstruction_keep_every_variant(self):
        from nidavellir.oracle import build_cartesian_indices
        from nidavellir.search import compile_catalogue, mine_variants
        from tests import test_search_optimizations as search_fixtures
        for restriction in certificates.BiasCertificateReviewTests.CASES.values():
            task = self.task([f'{p}(var__(1))' for p in 'pqrs'], restrictions=restriction,
                background='p(a). p(c). q(a). q(b). r(b). r(c). s(a). s(b).', max_body=4,
                examples='positive(answer(a)). positive(answer(b)). negative(answer(c)).')
            expected = search_fixtures.SearchOptimizationTests.expected(self, task)
            for enabled in (False, True):
                task._bias_analysis = analyze(task, core_pruning=enabled)
                indices, _ = build_cartesian_indices(task)
                rows, stats = compile_catalogue(task, indices, variants='representative', alpha_collect=False)
                self.assertTrue(stats['complete'])
                rows, stats = mine_variants(task, indices, rows, alpha_collect=False)
                self.assertTrue(stats['complete'])
                self.assertEqual(search_fixtures.SearchOptimizationTests.actual(rows), expected)


if __name__ == '__main__':
    unittest.main()
