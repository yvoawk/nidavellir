"""Independent regressions for bound replacement and short-phase deadlines."""
import unittest
from unittest.mock import patch

from nidavellir import relational_bound as bound
from nidavellir.bias_analysis import analyze
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.short_bound import seed_short_components
from tests import test_relational_bound as fixtures


class BoundTimeoutReviewTests(unittest.TestCase):
    task = fixtures.RelationalBoundTests.task
    verifier = fixtures.RelationalBoundTests.verifier

    def resumed_task(self):
        task = self.task('p(a). q(a). r(a).',
                         ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
                         examples='positive(answer(a)). negative(answer(missing)).')
        head = task.heads[0]
        indices = {literal.name: i for i, literal in enumerate(task.literals)}
        def candidate(body):
            return {'head': str(head), 'body': list(body), 'cost': len(body) + 1,
                    'coverage': 1, 'rule': task.rule(head, tuple(body))}
        incumbent = candidate([indices['p']])
        component = candidate([indices['q'], indices['r']])
        state = {'version': 1, 'fingerprint': bound._fingerprint(task),
                 'components': [bound._dump_candidate(component)],
                 'incumbent': [bound._dump_candidate(incumbent)],
                 'positive_cores': [], 'negative_witnesses': [], 'rounds': 0}
        return task, incumbent, state

    def assert_incumbent_consistent(self, cover, report, incumbent):
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(cover, [incumbent])
        self.assertEqual(report['cost'], incumbent['cost'])
        self.assertEqual(sum(row['cost'] for row in cover), report['cost'])
        self.assertEqual(report['rules'], [row['rule'] for row in cover])
        self.assertEqual(report['state']['incumbent'], [bound._dump_candidate(incumbent)])
        self.assertTrue(report['verification']['matched'])

    def test_deadline_after_greedy_does_not_replace_the_verified_resumed_program(self):
        task, incumbent, state = self.resumed_task()
        clock, verified = [0.0], []
        actual_verifier = self.verifier(task)
        def verify(rules):
            verified.append(list(rules))
            return actual_verifier(rules)
        original_greedy = bound.greedy_cover
        def greedy_then_deadline(rows, count):
            proposed = original_greedy(rows, count)
            self.assertEqual(sum(row['cost'] for row in proposed), 3)
            clock[0] = 11.0
            return proposed
        with patch('nidavellir.relational_bound.time.monotonic', side_effect=lambda: clock[0]), \
             patch.object(bound, 'greedy_cover', side_effect=greedy_then_deadline):
            cover, report = bound.discover_relational_bound(task, 10, verify, state=state,
                                                             short_phase_seconds=0)
        self.assertEqual(verified, [[incumbent['rule']]])
        self.assert_incumbent_consistent(cover, report, incumbent)

    def test_a_worse_fresh_pool_cover_never_replaces_the_lower_resumed_incumbent(self):
        task, incumbent, state = self.resumed_task()
        cover, report = bound.discover_relational_bound(task, 10, self.verifier(task),
                                                         state=state, short_phase_seconds=0)
        self.assert_incumbent_consistent(cover, report, incumbent)
        self.assertTrue(self.verifier(task)([row['rule'] for row in cover])['matched'])

    def test_callback_on_last_short_component_still_obeys_the_local_deadline(self):
        task = self.task('p(a).', ['p(var__(1))'],
                         examples='positive(answer(a)). negative(answer(missing)).')
        oracle, analysis = CoverageOracle(task), analyze(task)
        original_check = oracle.check
        clock, pool = [0.0], {}
        def completed_callback(_):
            clock[0] = 11.0
        with patch('nidavellir.short_bound.time.monotonic', side_effect=lambda: clock[0]):
            report = seed_short_components(task, oracle, analysis, pool, seconds=10,
                                           check=lambda: None, on_component=completed_callback)
        self.assertEqual(report['status'], 'budget_exhausted')
        self.assertEqual(report['covered_positives'], 1)
        self.assertEqual(len(pool), 1)
        self.assertIs(oracle.check, original_check)
        self.assertTrue(self.verifier(task)([row['rule'] for row in pool.values()])['matched'])


if __name__ == '__main__':
    unittest.main()
