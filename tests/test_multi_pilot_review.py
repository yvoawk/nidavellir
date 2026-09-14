"""Independent semantic review of the disjunctive multi-pilot bound."""
from itertools import combinations
import json
import unittest
from unittest.mock import patch

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from tests import test_relational_bound as fixtures


from nidavellir import relational_bound as staged


class MultiPilotReviewTests(unittest.TestCase):
    task = fixtures.RelationalBoundTests.task
    verifier = fixtures.RelationalBoundTests.verifier

    def legal_bodies(self, task):
        head = task.heads[0]
        for size in range(task.max_body + 1):
            for body in combinations(range(len(task.literals)), size):
                if task.accepts(head, body):
                    yield body

    def test_opposing_pilot_cores_preserve_disjunction_not_conjunction(self):
        task = self.task('p(a). q(b). r(a). r(b).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'])
        head = task.heads[0]
        a, b = map(clingo.parse_term, ('answer(a)', 'answer(b)'))
        idx = {literal.name: i for i, literal in enumerate(task.literals)}
        search = staged.ComponentSearch(task, head, (a, b),
                                        pilot_cores=[(a, (idx['q'],)), (b, (idx['p'],))])
        self.assertEqual(search.compatible_pilots((idx['p'],)), (a,))
        self.assertEqual(search.compatible_pilots((idx['q'],)), (b,))
        self.assertEqual(search.compatible_pilots((idx['r'],)), (a, b))
        for body in self.legal_bodies(task):
            expected = idx['p'] not in body or idx['q'] not in body
            excluded = set(range(len(task.literals))) - set(body)
            self.assertEqual(search.solver.find(body, excluded) is not None, expected, body)

    def test_full_refinement_equals_exact_safe_coverage_of_any_pilot(self):
        task = self.task('p(a). p(c). q(b). q(c). r(a). r(b).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'])
        head = task.heads[0]
        pilots = tuple(sorted(task.positives))
        oracle = CoverageOracle(task)
        search = staged.ComponentSearch(task, head, pilots)
        legal = list(self.legal_bodies(task))
        predictions = {body: fixtures.model(task.background + '\n' + task.rule(head, body)) for body in legal}
        for body in legal:
            for pilot in pilots:
                if pilot not in predictions[body]:
                    core = oracle.core(head, body, pilot)
                    self.assertIsNotNone(core)
                    search.positive_core(core, pilot)
            if predictions[body] & task.negatives:
                witness = oracle.negative(head, body)
                self.assertIsNotNone(witness)
                search.negative_witness(witness['truth'])
        for body in legal:
            expected = bool(predictions[body] & task.positives) and not bool(predictions[body] & task.negatives)
            excluded = set(range(len(task.literals))) - set(body)
            self.assertEqual(search.solver.find(body, excluded) is not None, expected, body)

    def test_multiple_heads_and_disjoint_components_are_verified_globally(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'],
            examples='positive(alpha(a)). positive(omega(b)). negative(alpha(b)). negative(omega(a)).',
            heads=('alpha(var__(1))', 'omega(var__(1))'))
        cover, report = staged.discover_relational_bound(task, 10, self.verifier(task))
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['cost'], 4)
        self.assertEqual(len(cover), 2)
        self.assertEqual(report['initial_positives'], 24)
        for component in cover:
            head = clingo.parse_term(component['head'])
            self.assertTrue(task.accepts(head, component['body']))
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])
        self.assertNotIn('optimum_cost', report)

    def test_timeout_and_resume_do_not_relabel_partial_pool_as_complete(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'])
        stopped = False
        def checkpoint(state):
            nonlocal stopped
            if state['components'] and not stopped:
                stopped = True
                raise ResourceLimit('independent checkpoint interruption')
        cover, report = staged.discover_relational_bound(task, 10, self.verifier(task), checkpoint=checkpoint)
        self.assertEqual(report['status'], 'timeout')
        self.assertIsNone(cover)
        self.assertIsNone(report['cost'])
        saved = json.loads(json.dumps(report['state']))
        self.assertEqual(len(saved['components']), 1)
        cover, resumed = staged.discover_relational_bound(task, 10, self.verifier(task), state=saved,
                                                        initial_positives=1)
        self.assertEqual(resumed['status'], 'verified')
        self.assertTrue(self.verifier(task)([component['rule'] for component in cover])['matched'])
        with patch.object(staged.ComponentSearch, 'next', side_effect=ResourceLimit('interrupted solve')):
            cover, failed = staged.discover_relational_bound(task, 10, self.verifier(task), short_phase_seconds=0)
        self.assertEqual(failed['status'], 'timeout')
        self.assertIsNone(cover)
        self.assertIsNone(failed['cost'])


if __name__ == '__main__':
    unittest.main()
