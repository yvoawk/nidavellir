"""Recover an interrupted global incumbent from independently checked targets."""
import copy
import unittest

import clingo

from nidavellir.__main__ import recover_target_cover
from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_bound import discover_relational_bound
from nidavellir.target_bounds import collect_verified_target_bounds
from nidavellir.verification import verify_rules
from tests import test_optimized_oracle as fixtures


class RecoveredTargetCoverTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def interrupted(self):
        task = self.task('p(a). q(b).', ['p(var__(1))', 'q(var__(1))'],
                         ['first(a)', 'second(b)'], ['first(b)', 'second(a)', 'unused(a)'],
                         heads=('first(var__(1))', 'second(var__(1))', 'unused(var__(1))'),
                         max_body=1)
        stopped = False
        def stop_after_target_certificates(state):
            nonlocal stopped
            if not stopped and {'first/1', 'second/1'} <= state.get('target_bounds', {}).keys():
                self.assertIsNone(state['incumbent'])
                stopped = True
                raise ResourceLimit('interrupted before global incumbent publication')
        cover, report = discover_relational_bound(
            task, 5, lambda rules: verify_rules(task, rules), per_target_bounds=True,
            short_phase_seconds=0, checkpoint=stop_after_target_certificates)
        self.assertIsNone(cover)
        self.assertEqual(report['status'], 'timeout')
        self.assertIsNone(report['state']['incumbent'])
        return task, report

    def test_interrupted_worker_certificates_form_independently_verified_incumbent(self):
        task, bound = self.interrupted()
        original = copy.deepcopy(bound)
        cover, recovered = recover_target_cover(task, None, bound)
        self.assertEqual(recovered['status'], 'verified')
        self.assertEqual(recovered['discovery_status'], 'timeout')
        self.assertEqual(recovered['incumbent_source'], 'verified_target_certificates')
        self.assertEqual(recovered['cost'], 4)
        self.assertEqual(recovered['rules'], [row['rule'] for row in cover])
        self.assertTrue(all(type(row['coverage']) is int for row in cover))
        self.assertTrue(recovered['verification']['matched'])
        self.assertEqual(recovered['verification'], verify_rules(task, recovered['rules']))
        self.assertNotIn('optimality_proven', recovered)
        self.assertEqual(bound, original)

    def test_missing_positive_target_certificate_does_not_promote_partial_cover(self):
        task, bound = self.interrupted()
        del bound['target_bounds']['second/1']
        cover, recovered = recover_target_cover(task, None, bound)
        self.assertIsNone(cover)
        self.assertIs(recovered, bound)
        self.assertEqual(recovered['status'], 'timeout')

    def test_unlabelled_metadata_head_needs_no_target_certificate(self):
        task, bound = self.interrupted()
        del bound['target_bounds']['unused/1']
        cover, recovered = recover_target_cover(task, None, bound)
        self.assertEqual(recovered['status'], 'verified')
        self.assertEqual(recovered['cost'], 4)
        self.assertTrue(verify_rules(task, [row['rule'] for row in cover])['matched'])
        self.assertFalse(any(clingo.parse_term(row['head']).name == 'unused' for row in cover))
        self.assertEqual(len(task.heads), 3)

    def test_zero_positive_task_can_recover_verified_empty_program(self):
        task = self.task('p(a).', ['p(var__(1))'], [], ['first(a)', 'second(a)'],
                         heads=('first(var__(1))', 'second(var__(1))'))
        certificates = collect_verified_target_bounds(task, {}, {}, analyze(task), lambda: None)
        bound = {'status': 'timeout', 'target_bounds': certificates}
        cover, recovered = recover_target_cover(task, None, bound)
        self.assertEqual(cover, [])
        self.assertEqual(recovered['cost'], 0)
        self.assertEqual(recovered['rules'], [])
        self.assertTrue(recovered['verification']['matched'])
        self.assertEqual(recovered['verification']['covered_negatives'], [])

    def test_malformed_components_and_unverified_certificates_are_rejected(self):
        task, bound = self.interrupted()
        for corruption in ('cost', 'syntax', 'coverage', 'verification'):
            with self.subTest(corruption=corruption):
                malformed = copy.deepcopy(bound)
                certificate = malformed['target_bounds']['first/1']
                row, = certificate['cover']
                if corruption == 'cost':
                    row['cost'] += 1
                elif corruption == 'syntax':
                    row['rule'] = 'first(a).'
                    certificate['rules'] = [row['rule']]
                elif corruption == 'coverage':
                    row['coverage'] = '0x3'
                else:
                    certificate['verification']['matched'] = False
                with self.assertRaises(ValueError):
                    recover_target_cover(task, None, malformed)

    def test_structurally_consistent_but_false_coverage_fails_original_BK_check(self):
        task, bound = self.interrupted()
        certificate = bound['target_bounds']['first/1']
        row, = certificate['cover']
        row['body'] = [task.literals.index(clingo.parse_term('q(var__(1))'))]
        row['rule'] = task.rule(clingo.parse_term(row['head']), row['body'])
        certificate['rules'] = [row['rule']]
        # Syntax, cost and claimed masks are coherent, but this first-target
        # rule derives its negative b and misses its positive a.
        with self.assertRaisesRegex(AssertionError, 'failed independent verification'):
            recover_target_cover(task, None, bound)

    def test_existing_global_cover_and_absent_certificates_are_left_unchanged(self):
        task, bound = self.interrupted()
        cover, recovered = recover_target_cover(task, None, bound)
        result, metadata = recover_target_cover(task, cover, recovered)
        self.assertIs(result, cover)
        self.assertIs(metadata, recovered)
        absent = {'status': 'timeout'}
        result, metadata = recover_target_cover(task, None, absent)
        self.assertIsNone(result)
        self.assertIs(metadata, absent)


if __name__ == '__main__':
    unittest.main()
