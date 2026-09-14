"""Progressive CEGIS caps preserve exact cuts, pilot scope and deadlines."""
import unittest
from unittest.mock import patch

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import TaskError
from tests import test_optimized_oracle as fixtures
from nidavellir.verification import verify_rules

from nidavellir import relational_bound as bound
term = clingo.parse_term


class ProgressiveBoundTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def triple(self):
        return self.task('p(a). q(a). r(a). p(b). q(b). p(c). r(c). q(d). r(d).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['answer(a)'], ['answer(b)', 'answer(c)', 'answer(d)'], max_body=3)

    def learn(self, task, **kwargs):
        return bound.discover_relational_bound(task, 5, lambda rules: verify_rules(task, rules),
             short_phase_seconds=0, fingerprint='unit-progressive', **kwargs)

    def test_component_cap_counts_all_actual_literals_and_none_retains_api(self):
        task = self.triple()
        head, pilot = task.heads[0], next(iter(task.positives))
        self.assertIsNone(bound.ComponentSearch(task, head, pilot, max_body=0).next(lambda: None))
        for cap in (1, 2, 3):
            body = bound.ComponentSearch(task, head, pilot, max_body=cap).next(lambda: None)
            self.assertTrue(task.accepts(head, body))
            self.assertLessEqual(len(body), cap)
        body = bound.ComponentSearch(task, head, pilot).next(lambda: None)
        self.assertTrue(task.accepts(head, body))
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                bound.ComponentSearch(task, head, pilot, max_body=invalid)

    def test_three_literals_required_and_negative_cuts_replayed_across_levels(self):
        task = self.triple()
        built = []
        original = bound.ComponentSearch.__init__
        def observe(search, *args, **kwargs):
            built.append((kwargs['max_body'], len(kwargs['negative_witnesses'])))
            original(search, *args, **kwargs)
        with patch.object(bound.ComponentSearch, '__init__', observe):
            cover, report = self.learn(task)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['cost'], 4)
        self.assertEqual(len(cover), 1)
        self.assertEqual(set(cover[0]['body']), {0, 1, 2})
        self.assertEqual([k for k, _ in built], [0, 1, 2, 3])
        self.assertGreater(built[2][1], 0)
        self.assertGreaterEqual(built[3][1], built[2][1])
        self.assertEqual([v['max_body_literals'] for v in report['completed_body_levels']], [0, 1, 2])
        self.assertEqual(report['unary_literals_checked'], 3)
        self.assertEqual(report['capped_searches'], 4)
        self.assertTrue(verify_rules(task, [c['rule'] for c in cover])['matched'])

    def test_real_positive_join_cores_are_scoped_and_replayed_at_all_caps(self):
        task = self.task('p(a,x). q(a,y). r(a,x). s(a,x). '
                         'p(b,x). r(b,x). p(c,x). s(c,x). r(d,x). s(d,x). q(e,y).',
                         ['p(var__(1),var__(2))', 'q(var__(1),var__(2))',
                          'r(var__(1),var__(2))', 's(var__(1),var__(2))'],
                         ['answer(a)'], ['answer(b)', 'answer(c)', 'answer(d)', 'answer(e)'], max_body=3)
        head, pilot = task.heads[0], next(iter(task.positives))
        ids = {literal.name: i for i, literal in enumerate(task.literals)}
        cores = [sorted((ids['q'], ids[name])) for name in ('p', 'r', 's')]
        oracle = CoverageOracle(task)
        for core in cores:
            self.assertFalse(oracle.exists(head, core, pilot))
            self.assertTrue(all(oracle.exists(head, (i,), pilot) for i in core))
        state = {'version': 1, 'fingerprint': 'unit-progressive', 'components': [],
                 'positive_cores': [{'head': str(head), 'pilot': str(pilot), 'body': core} for core in cores],
                 'negative_witnesses': [], 'rounds': 0, 'incumbent': None}
        built = []
        original = bound.ComponentSearch.__init__
        def observe(search, *args, **kwargs):
            built.append((kwargs['max_body'], [(str(label), list(core)) for label, core in kwargs['pilot_cores']]))
            original(search, *args, **kwargs)
        with patch.object(bound.ComponentSearch, '__init__', observe):
            cover, report = self.learn(task, state=state)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['cost'], 4)
        self.assertEqual([k for k, _ in built], [0, 1, 2, 3])
        for _, replayed in built:
            self.assertTrue(all((str(pilot), core) in replayed for core in cores))
        self.assertEqual(set(cover[0]['body']), {ids['p'], ids['r'], ids['s']})

    def test_positive_claims_remain_an_or_under_a_cap(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'],
                         ['answer(a)', 'answer(b)'], ['answer(n)'], max_body=2)
        head = task.heads[0]
        a, b = term('answer(a)'), term('answer(b)')
        p, q = (next(i for i, literal in enumerate(task.literals) if literal.name == name) for name in ('p', 'q'))
        search = bound.ComponentSearch(task, head, (a, b), max_body=1,
                                       pilot_cores=((a, (q,)), (b, (p,))))
        body = search.next(lambda: None)
        self.assertIsNotNone(body)
        self.assertEqual(len(body), 1)
        self.assertEqual(len(search.compatible_pilots(body)), 1)

    def test_all_heads_are_considered_at_shorter_level_before_widening(self):
        task = self.task('p(a). q(a). r(a).',
                         ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
                         ['first(a)', 'second(a)'], ['first(n)', 'second(n)'], max_body=3,
                         heads=('first(var__(1))', 'second(var__(1))'),
                         bias=':- head(first(var__(1))), not body(p(var__(1))).\n'
                              ':- head(first(var__(1))), not body(q(var__(1))).\n'
                              ':- head(first(var__(1))), not body(r(var__(1))).')
        observed = []
        def checkpoint(state):
            for component in state['components']:
                if component['rule'] not in observed:
                    observed.append(component['rule'])
        cover, report = self.learn(task, checkpoint=checkpoint)
        self.assertEqual(report['status'], 'verified')
        self.assertTrue(observed[0].startswith('second('))
        self.assertEqual(report['cost'], 6)
        first_levels = [v['max_body_literals'] for v in report['completed_body_levels'] if v['component'] == 0]
        self.assertEqual(first_levels, [0])

    def test_timeout_does_not_finish_current_level_and_resume_restarts_zero(self):
        task = self.triple()
        original_next = bound.ComponentSearch.next
        def expired(search, check):
            if search.max_body == 2:
                raise ResourceLimit('synthetic deadline during cap 2')
            return original_next(search, check)
        with patch.object(bound.ComponentSearch, 'next', expired):
            cover, partial = self.learn(task)
        self.assertIsNone(cover)
        self.assertEqual(partial['status'], 'timeout')
        self.assertNotIn('reason', partial)
        self.assertEqual([v['max_body_literals'] for v in partial['completed_body_levels']], [0, 1])
        self.assertEqual(partial['current_body_level'], 2)
        replayed_caps = []
        original_init = bound.ComponentSearch.__init__
        def observe(search, *args, **kwargs):
            replayed_caps.append(kwargs['max_body'])
            original_init(search, *args, **kwargs)
        with patch.object(bound.ComponentSearch, '__init__', observe):
            cover, resumed = self.learn(task, state=partial['state'])
        self.assertEqual(replayed_caps[0], 0)
        self.assertEqual(resumed['status'], 'verified')
        self.assertEqual(resumed['cost'], 4)
        self.assertEqual(resumed['unary_literals_checked'], 0)

    def test_unknown_or_solver_errors_never_become_an_unsat_level(self):
        task = self.triple()
        progress = []
        with patch.object(bound.ComponentSearch, 'next', side_effect=TaskError('UNKNOWN')):
            with self.assertRaisesRegex(TaskError, 'UNKNOWN'):
                self.learn(task, progress=progress.append)
        self.assertEqual(progress[-1]['completed_body_levels'], [])


if __name__ == '__main__':
    unittest.main()
