"""Short-pool seeding versus independent complete Clingo interpretations."""
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import clingo

from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.short_bound import seed_short_components
from tests import test_relational_bound as fixtures


class ShortBoundTests(unittest.TestCase):
    task = fixtures.RelationalBoundTests.task

    def run_phase(self, task, pool=None, *, on_component=lambda _: None, seconds=10, check=lambda: None):
        analysis = analyze(task, seconds=10)
        oracle = CoverageOracle(task)
        pool = {} if pool is None else pool
        original = oracle.check
        result = seed_short_components(task, oracle, analysis, pool,
                                       seconds=seconds, check=check, on_component=on_component)
        self.assertIs(oracle.check, original)
        return pool, result, oracle

    def exhaustive_pool(self, task):
        pool = {}
        positives = sorted(task.positives)
        for head in task.heads:
            for size in range(min(2, task.max_body) + 1):
                for body in combinations(range(len(task.literals)), size):
                    if not task.accepts(head, body):
                        continue
                    rule = task.rule(head, body)
                    actual = fixtures.model(task.background + '\n' + rule)
                    coverage = sum(1 << i for i, atom in enumerate(positives) if atom in actual)
                    if not coverage or actual & task.negatives:
                        continue
                    candidate = {'head': str(head), 'body': list(body), 'cost': size + 1,
                                 'coverage': coverage, 'rule': rule}
                    previous = pool.get(coverage)
                    if previous is None or (candidate['cost'], rule) < (previous['cost'], previous['rule']):
                        pool[coverage] = candidate
        return pool

    def test_completed_short_phase_preserves_exhaustive_reachable_coverage(self):
        # The unreachable positive forces complete short enumeration rather
        # than an early return when a feasible union has already been found.
        task = self.task('p(a,x). p(c,z). q(x). r(b). r(c). mark(b).',
            ['p(var__(1),var__(2))', 'q(var__(2))', 'r(var__(1))', 'mark(var__(1))'],
            examples='positive(answer(a)). positive(answer(b)). positive(answer(missing)). negative(answer(c)).')
        pool, report, _ = self.run_phase(task)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['completed_body_levels'], [0, 1, 2])
        expected = self.exhaustive_pool(task)
        def union(rows):
            mask = 0
            for candidate in rows.values():
                mask |= candidate['coverage']
            return mask
        self.assertEqual(union(pool), union(expected))
        for coverage, candidate in pool.items():
            self.assertIn(coverage, expected)
            actual = fixtures.model(task.background + '\n' + candidate['rule'])
            self.assertFalse(task.negatives & actual)
            self.assertEqual(coverage, sum(1 << i for i, atom in enumerate(sorted(task.positives)) if atom in actual))
        self.assertNotIn('optimal_cost', report)
        self.assertIn('not an exhaustive catalogue', report['scope'])
        self.assertGreater(report['rejected_negative'], 0)

    def test_negative_singleton_does_not_discard_its_safe_extension(self):
        task = self.task('p(a,x). p(c,y). keep(x).',
            ['p(var__(1),var__(2))', 'keep(var__(2))'],
            examples='positive(answer(a)). negative(answer(c)).')
        pool, report, _ = self.run_phase(task)
        self.assertEqual(report['status'], 'cover_found')
        self.assertEqual(len(pool), 1)
        self.assertEqual(next(iter(pool.values()))['cost'], 3)
        self.assertGreater(report['rejected_negative'], 0)

    def test_atomic_masks_are_not_used_as_correlated_coverage(self):
        task = self.task('p(a,x). p(c,x). q(a,y). q(c,x).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))'],
            examples='positive(answer(a)). negative(answer(c)).')
        pool, report, _ = self.run_phase(task)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(pool, {})
        self.assertGreater(report['pruned_no_joint_positive'], 0)

    def test_ground_empty_body_is_tried_before_nonempty_bodies(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='positive(success).', heads=('success',))
        pool, report, _ = self.run_phase(task)
        self.assertEqual(report['status'], 'cover_found')
        candidate, = pool.values()
        self.assertEqual(candidate['body'], [])
        self.assertEqual(candidate['cost'], 1)
        self.assertEqual(report['atomic_queries'], 0)
        self.assertEqual(candidate['rule'], 'success.')

    def test_multiple_heads_and_overlapping_components_use_global_positions(self):
        task = self.task('p(a). p(b). q(b). q(d). end(z).',
            ['p(var__(1))', 'q(var__(1))', 'end(var__(1))'],
            examples=('positive(alpha(a)). positive(alpha(b)). positive(alpha(d)). '
                      'positive(omega(z)). negative(alpha(z)). '
                      'negative(omega(a)). negative(omega(b)). negative(omega(d)).'),
            heads=('alpha(var__(1))', 'omega(var__(1))'))
        seen = []
        pool, report, _ = self.run_phase(task, on_component=lambda row: seen.append(row))
        self.assertEqual(report['status'], 'cover_found')
        self.assertEqual(report['covered_positives'], 4)
        self.assertEqual(len(pool), 3)
        self.assertEqual(len(seen), 3)
        rows = list(pool.values())
        self.assertTrue(any(a['coverage'] & b['coverage'] for i, a in enumerate(rows) for b in rows[i + 1:]))
        actual = fixtures.model(task.background + '\n' + '\n'.join(row['rule'] for row in rows))
        self.assertTrue(task.positives <= actual)
        self.assertFalse(task.negatives & actual)

    def test_guard_closure_uses_a_fixed_point_and_respects_effective_length_cap(self):
        task = self.task('p(a). q(x). r(y).',
            ['p(var__(1))', 'q(var__(2))', 'r(var__(3))'],
            constraints=(':- body(p(var__(1))), not body(q(var__(2))).\n'
                         ':- body(q(var__(2))), not body(r(var__(3))).\n'),
            examples='positive(answer(a)). negative(answer(c)).')
        analysis = analyze(task, seconds=10)
        head = task.heads[0]
        idx = {literal.name: i for i, literal in enumerate(task.literals)}
        # Direct guards intentionally omit the transitive p -> r edge; the
        # helper must still compute it rather than assume a closed input.
        guards = [frozenset() for _ in task.literals]
        guards[idx['p']] = frozenset({idx['q']})
        guards[idx['q']] = frozenset({idx['r']})
        proof = replace(analysis.heads[head], guards=tuple(guards))
        analysis = replace(analysis, heads={head: proof})
        oracle = CoverageOracle(task)
        with patch.object(oracle, 'atomic_coverage', side_effect=AssertionError('Illegal body reached BK')):
            pool = {}
            report = seed_short_components(task, oracle, analysis, pool,
                                           seconds=10, check=lambda: None, on_component=lambda _: None)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(pool, {})
        self.assertGreater(report['guard_closures_above_limit'], 0)

    def test_arbitrary_asp_legality_is_checked_before_semantics(self):
        task = self.task('p(a). q(a). r(a).', ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            examples='positive(answer(a)). negative(answer(c)).',
            constraints=':- #count { L : body(L) } < 2.\n')
        analysis = analyze(task, seconds=10)
        oracle = CoverageOracle(task)
        original = oracle.atomic_coverage
        checked = []
        def observed(head, literal):
            checked.append(literal)
            return original(head, literal)
        with patch.object(oracle, 'atomic_coverage', side_effect=observed):
            pool = {}
            report = seed_short_components(task, oracle, analysis, pool,
                                           seconds=10, check=lambda: None, on_component=lambda _: None)
        self.assertEqual(report['status'], 'cover_found')
        self.assertEqual(next(iter(pool.values()))['cost'], 3)
        self.assertGreater(report['pruned_illegal'], 0)
        self.assertEqual(len(checked), 2)

    def test_local_timeout_preserves_partial_pool_and_restores_oracle_check(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'])
        analysis = analyze(task, seconds=10)
        oracle = CoverageOracle(task)
        original_check = oracle.check
        clock = [0.0]
        pool = {}
        def stop_after_component(_):
            clock[0] = 11.0
        with patch('nidavellir.short_bound.time.monotonic', side_effect=lambda: clock[0]):
            report = seed_short_components(task, oracle, analysis, pool, seconds=10,
                                           check=lambda: None, on_component=stop_after_component)
        self.assertEqual(report['status'], 'budget_exhausted')
        self.assertEqual(len(pool), 1)
        self.assertEqual(report['covered_positives'], 1)
        self.assertIs(oracle.check, original_check)
        resumed = seed_short_components(task, oracle, analysis, pool, seconds=10,
                                        check=lambda: None, on_component=lambda _: None)
        self.assertEqual(resumed['status'], 'cover_found')
        self.assertEqual(len(pool), 2)

    def test_global_and_callback_resource_limits_are_propagated(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'])
        analysis = analyze(task, seconds=10)
        def expired():
            raise ResourceLimit('global deadline')
        for source in ('caller', 'oracle', 'callback'):
            oracle = CoverageOracle(task)
            if source == 'oracle':
                oracle.check = expired
            original = oracle.check
            pool = {}
            with self.subTest(source=source), self.assertRaisesRegex(ResourceLimit, 'global deadline'):
                seed_short_components(task, oracle, analysis, pool, seconds=10,
                                      check=expired if source == 'caller' else lambda: None,
                                      on_component=(lambda _: expired()) if source == 'callback' else lambda _: None)
            self.assertIs(oracle.check, original)
            self.assertEqual(len(pool), 1 if source == 'callback' else 0)

    def test_empty_positive_domain_and_existing_cover_stop_immediately(self):
        task = self.task('p(a).', ['p(var__(1))'], examples='negative(answer(c)).')
        pool, report, _ = self.run_phase(task)
        self.assertEqual(report['status'], 'cover_found')
        self.assertEqual(pool, {})
        task = self.task('p(a). p(b).', ['p(var__(1))'])
        p = next(i for i, literal in enumerate(task.literals) if literal.name == 'p')
        candidate = {'head': str(task.heads[0]), 'body': [p], 'cost': 2,
                     'coverage': 3, 'rule': task.rule(task.heads[0], (p,))}
        pool, report, _ = self.run_phase(task, {3: candidate},
                                         on_component=lambda _: self.fail('Existing cover was rediscovered'))
        self.assertEqual(report['status'], 'cover_found')
        self.assertEqual(report['generated_bodies'], 0)

    def test_zero_local_budget_and_invalid_parameters(self):
        task = self.task('p(a).', ['p(var__(1))'])
        pool, report, _ = self.run_phase(task, seconds=0)
        self.assertEqual(report['status'], 'budget_exhausted')
        self.assertEqual(pool, {})
        for seconds in (-1, float('nan'), float('inf'), True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.run_phase(task, seconds=seconds)

    def test_already_covered_singletons_and_extensions_skip_semantic_queries(self):
        task = self.task('p(a). q(a). r(a).',
                         ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
                         examples='positive(answer(a)). positive(answer(missing)). negative(answer(c)).')
        analysis = analyze(task, seconds=10)
        oracle = CoverageOracle(task)
        original_negative = oracle.negative
        original_coverage = oracle.coverage
        negative_bodies, coverage_bodies = [], []
        def negative(head, body):
            negative_bodies.append(body)
            return original_negative(head, body)
        def coverage(head, body):
            coverage_bodies.append(body)
            return original_coverage(head, body)
        with patch.object(oracle, 'negative', side_effect=negative), patch.object(oracle, 'coverage', side_effect=coverage):
            pool = {}
            report = seed_short_components(task, oracle, analysis, pool,
                                           seconds=10, check=lambda: None, on_component=lambda _: None)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['completed_body_levels'], [0, 1, 2])
        self.assertEqual(len(pool), 1)
        self.assertEqual(len(negative_bodies), 1)
        self.assertEqual(len(coverage_bodies), 1)
        self.assertEqual(report['joint_positive_queries'], 1)
        self.assertEqual(report['pruned_already_covered'], 5)
        self.assertIn('minimum rule-cost certificate', report['scope'])

    def test_existing_pool_is_not_improved_when_coverage_is_already_present(self):
        task = self.task('p(a). q(a).', ['p(var__(1))', 'q(var__(1))'],
                         examples='positive(answer(a)). positive(answer(missing)). negative(answer(c)).')
        body = tuple(range(len(task.literals)))
        candidate = {'head': str(task.heads[0]), 'body': list(body), 'cost': 3,
                     'coverage': 1, 'rule': task.rule(task.heads[0], body)}
        pool, report, _ = self.run_phase(task, {1: candidate},
                                         on_component=lambda _: self.fail('Already covered pool must not be improved'))
        self.assertIs(pool[1], candidate)
        self.assertEqual(report['joint_positive_queries'], 0)
        self.assertEqual(report['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
