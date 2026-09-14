"""Optional short seeding and CEGIS share verified components and deadlines."""
import unittest
from unittest.mock import patch

from nidavellir.relational_bound import ComponentSearch, discover_relational_bound
from tests import test_relational_bound as fixtures


class ShortBoundIntegrationTests(unittest.TestCase):
    task = fixtures.RelationalBoundTests.task
    verifier = fixtures.RelationalBoundTests.verifier

    def small_components(self):
        return self.task('p(a). observation(b,x). observation(n,y). keep(x).',
            ['p(var__(1))', 'observation(var__(1),var__(2))', 'keep(var__(2))'],
            examples='positive(answer(a)). positive(answer(b)). negative(answer(n)).')

    def test_default_short_phase_verifies_small_program_without_cegis_proposals(self):
        task = self.small_components()
        with patch.object(ComponentSearch, 'next', side_effect=AssertionError('CEGIS should not be needed')):
            cover, report = discover_relational_bound(task, 10, self.verifier(task))
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['short_phase']['status'], 'cover_found')
        self.assertEqual(report['examined'], 0)
        self.assertEqual(report['cost'], 5)
        self.assertEqual(sorted(component['cost'] for component in cover), [2, 3])
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])

    def test_zero_short_budget_uses_the_existing_cegis_fallback(self):
        task = self.small_components()
        original_next = ComponentSearch.next
        calls = []
        def observed(search, check):
            calls.append(search.head)
            return original_next(search, check)
        with patch('nidavellir.relational_bound.seed_short_components',
                   side_effect=AssertionError('Disabled phase must not run')), \
             patch.object(ComponentSearch, 'next', observed):
            cover, report = discover_relational_bound(task, 10, self.verifier(task), short_phase_seconds=0)
        self.assertTrue(calls)
        self.assertNotIn('short_phase', report)
        self.assertEqual(report['status'], 'verified')
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])

    def test_expired_local_phase_passes_its_pool_and_real_negative_witness_to_cegis(self):
        # p is refuted by the real negative, then q is accepted. Expire only
        # the short phase at that checkpoint; CEGIS can finish using r.
        task = self.task('p(a). p(n). q(b). r(a).',
                         ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
                         examples='positive(answer(a)). positive(answer(b)). negative(answer(n)).')
        clock = [0.0]
        first_component = []
        def checkpoint(state):
            if state['components'] and not first_component:
                first_component.append(dict(state['components'][0]))
                clock[0] = 2.0
        original_init = ComponentSearch.__init__
        supplied_witnesses = []
        def observed(search, *args, **kwargs):
            supplied_witnesses.extend(kwargs.get('negative_witnesses', ()))
            original_init(search, *args, **kwargs)
        with patch('nidavellir.relational_bound.time.monotonic', side_effect=lambda: clock[0]), \
             patch.object(ComponentSearch, '__init__', observed):
            cover, report = discover_relational_bound(task, 10, self.verifier(task),
                                                      short_phase_seconds=1, checkpoint=checkpoint)
        self.assertEqual(report['short_phase']['status'], 'budget_exhausted')
        self.assertEqual(report['short_phase']['pool_components'], 1)
        self.assertGreater(report['short_phase']['rejected_negative'], 0)
        self.assertTrue(supplied_witnesses)
        self.assertTrue(report['state']['negative_witnesses'])
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['cost'], 4)
        self.assertIn(first_component[0]['rule'], [component['rule'] for component in cover])
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])
        self.assertGreater(report['examined'], 0)

    def test_exhausting_short_levels_does_not_restrict_the_cegis_grammar(self):
        task = self.task('p(a,x). p(n,y). q(x,z). q(y,w). r(z).',
                         ['p(var__(1),var__(2))', 'q(var__(2),var__(3))', 'r(var__(3))'],
                         examples='positive(answer(a)). negative(answer(n)).')
        cover, report = discover_relational_bound(task, 10, self.verifier(task))
        self.assertEqual(report['short_phase']['status'], 'complete')
        self.assertEqual(report['short_phase']['completed_body_levels'], [0, 1, 2])
        self.assertEqual(report['status'], 'verified')
        self.assertGreater(report['examined'], 0)
        self.assertEqual(report['cost'], 4)
        self.assertEqual(len(cover), 1)
        self.assertEqual(len(cover[0]['body']), 3)
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])


if __name__ == '__main__':
    unittest.main()
