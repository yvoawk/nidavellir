"""Empty target incumbents use the already validated fixed interpretation."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import clingo
from nidavellir.bias_analysis import analyze
from nidavellir.labels import CompactNegativeSet
from nidavellir.target_bounds import collect_verified_target_bounds, costs_from_certificates, target_view
from nidavellir.verification import verify_rules
from tests import test_optimized_oracle as fixtures


class EmptyTargetBoundTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def test_negative_only_empty_targets_skip_regrounding_without_expanding_labels(self):
        task = self.task('p(a). p(n).', ['p(var__(1))'], ['first(a)'],
                         ['second(n)', 'third(n)'],
                         heads=('first(var__(1))', 'second(var__(1))', 'third(var__(1))'))
        analysis = analyze(task)
        with patch('nidavellir.target_bounds.verify_rules', side_effect=AssertionError('BK re-grounded')), \
             patch.object(CompactNegativeSet, '__iter__', side_effect=AssertionError('expanded negatives')):
            result = collect_verified_target_bounds(task, {}, {}, analysis, lambda: None,
                verify_target=lambda *_: self.fail('custom BK verifier invoked for empty rules'))
        self.assertEqual(costs_from_certificates(task, result), {('second', 1): 0, ('third', 1): 0})
        for key in (('second', 1), ('third', 1)):
            self.assertTrue(verify_rules(target_view(task, key), [])['matched'])
        self.assertEqual(len(task.heads), 3)
        self.assertTrue(all(proof.legal_body_exists for proof in analysis.heads.values()))

    def test_materialized_target_atoms_in_invalid_manual_task_are_rejected(self):
        task = self.task('p(a).', ['p(var__(1))'], [], ['answer(a)'])
        task = replace(task, atoms=task.atoms | {clingo.parse_term('answer(a)')})
        with self.assertRaisesRegex(ValueError, 'absent from the fixed BK'):
            collect_verified_target_bounds(task, {}, {}, analyze(task), lambda: None)

    def test_positive_target_still_requires_independent_original_BK_verification(self):
        task = self.task('p(a). p(n).', ['p(var__(1))'], ['answer(a)'], ['answer(n)'])
        head, = task.heads
        row = {'head': str(head), 'body': [0], 'rule': task.rule(head, [0]), 'cost': 2, 'coverage': 1}
        with self.assertRaisesRegex(ValueError, 'failed original-BK verification'):
            collect_verified_target_bounds(task, {1: row}, {}, analyze(task), lambda: None)


if __name__ == '__main__':
    unittest.main()
