"""Verified online incumbents preserve exhaustive optimal-program contracts."""
import unittest
from unittest.mock import patch

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.target_bounds import costs_from_certificates
from nidavellir.verification import verify_rules
from tests import test_optimized_oracle as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue


class DynamicCompilationBoundTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def improving_task(self):
        return self.task(
            "p(a). q(b). r(a). r(b). r(n1). s(a). s(b). s(n2). "
            "u(a). u(b). u(n3). v(a). v(b). v(n4).",
            [f"{name}(var__(1))" for name in ("p", "q", "r", "s", "u", "v")],
            ["answer(a)", "answer(b)"],
            ["answer(n1)", "answer(n2)", "answer(n3)", "answer(n4)"], max_body=4)

    def test_later_cheaper_rule_tightens_bound_and_keeps_all_optimal_variants(self):
        task = self.improving_task()
        reference = exhaustive_catalogue(task)
        expected = optimal_programs(((mask, cost) for mask, cost, _ in reference), 3, 20)
        expected_variants = {rule for mask, cost, rules in reference if mask == 3 and cost == 3
                             for rule in rules}
        for variants in ("exhaustive", "targeted", "representative"):
            for order in ("source", "selectivity"):
                with self.subTest(variants=variants, order=order):
                    callbacks = []
                    rows, stats = compile_catalogue(task, CoverageOracle(task),
                        dynamic_program_bound=True, dynamic_bound_interval=1,
                        variants=variants, order=order, on_incumbent=callbacks.append)
                    self.assertEqual(optimal_programs(((row['coverage'], row['cost']) for row in rows), 3, 20), expected)
                    self.assertEqual([item['cost'] for item in callbacks], [4, 3])
                    self.assertEqual(stats['program_bound_value'], 3)
                    self.assertEqual(stats['target_search']['answer/1']['effective_body_limit'], 2)
                    self.assertEqual(stats['dynamic_bound_verifications'], 2)
                    self.assertTrue(verify_rules(task, stats['dynamic_incumbent']['rules'])['matched'])
                    self.assertTrue(stats['complete'])
                    if variants != 'representative':
                        actual = {rule for row in rows if row['coverage'] == 3 and row['cost'] == 3
                                  for rule in row['variants']}
                        self.assertEqual(actual, expected_variants)

    def test_default_does_not_change_unrestricted_catalogue_contract(self):
        task = self.improving_task()
        rows, stats = compile_catalogue(task, CoverageOracle(task))
        actual = {(row['coverage'], row['cost'], tuple(sorted(row['variants']))) for row in rows}
        self.assertEqual(actual, exhaustive_catalogue(task))
        self.assertNotIn('dynamic_incumbent', stats)

    def test_no_initial_bound_discovers_joint_program_and_certifies_each_target(self):
        task = self.task('p(a). q(b). q(n1). r(b). r(n2).',
            ['p(var__(1))', 'q(var__(1))', 'r(var__(1))'],
            ['first(a)', 'second(b)'], ['first(n1)', 'second(n1)', 'second(n2)'],
            heads=('first(var__(1))', 'second(var__(1))', 'unused(var__(1))'))
        reference = exhaustive_catalogue(task)
        rows, stats = compile_catalogue(task, CoverageOracle(task), dynamic_program_bound=True,
            target_bounds={('first', 1): 2, ('second', 1): 4})
        self.assertEqual(optimal_programs(((row['coverage'], row['cost']) for row in rows), 3, 20),
                         optimal_programs(((mask, cost) for mask, cost, _ in reference), 3, 20))
        self.assertEqual(costs_from_certificates(task, stats['dynamic_target_bounds']),
                         {('first', 1): 2, ('second', 1): 3})
        self.assertEqual(stats['dynamic_incumbent']['cost'], 5)
        self.assertTrue(verify_rules(task, stats['dynamic_incumbent']['rules'])['matched'])

    def test_existing_scalar_and_target_bounds_never_increase(self):
        task = self.improving_task()
        _, stats = compile_catalogue(task, CoverageOracle(task), bound=3, program_bound=True,
            target_bounds={('answer', 1): 3}, dynamic_program_bound=True, dynamic_bound_interval=1)
        self.assertEqual(stats['program_bound_value'], 3)
        self.assertEqual(stats['program_bounds'], {'answer/1': 3})
        self.assertIsNone(stats['dynamic_incumbent'])
        self.assertEqual(stats['dynamic_bound_verifications'], 1)

    def test_ground_head_empty_rules_and_equal_cost_complete_rule_all_survive(self):
        task = self.task('p(a). p(b).', ['p(var__(1))'], ['answer(a)', 'answer(b)'],
            ['answer(n)'], heads=('answer(a)', 'answer(b)', 'answer(var__(1))'))
        reference = exhaustive_catalogue(task)
        rows, stats = compile_catalogue(task, CoverageOracle(task), dynamic_program_bound=True,
            variants='targeted')
        self.assertEqual(optimal_programs(((row['coverage'], row['cost']) for row in rows), 3, 20),
                         optimal_programs(((mask, cost) for mask, cost, _ in reference), 3, 20))
        self.assertEqual(stats['dynamic_incumbent']['cost'], 2)
        self.assertEqual(stats['target_minimum_useful_rule_cost']['answer/1'], 1)

    def test_partial_target_progress_does_not_claim_a_complete_incumbent(self):
        task = self.task('p(a).', ['p(var__(1))'], ['first(a)', 'second(b)'],
                         heads=('first(var__(1))', 'second(var__(1))'))
        _, stats = compile_catalogue(task, CoverageOracle(task), dynamic_program_bound=True)
        self.assertEqual(costs_from_certificates(task, stats['dynamic_target_bounds']), {('first', 1): 2})
        self.assertIsNone(stats['dynamic_incumbent'])
        self.assertFalse(stats['program_bound'])

    def test_unlimited_budget_uses_no_numeric_time_sentinel(self):
        task = self.task('p(a).', ['p(var__(1))'], ['answer(a)'], ['answer(n)'])
        _, stats = compile_catalogue(task, CoverageOracle(task), dynamic_program_bound=True, seconds=None)
        self.assertTrue(stats['complete'])
        self.assertEqual(stats['dynamic_incumbent']['cost'], 2)

    def test_timeout_after_independent_verification_keeps_feasible_incumbent(self):
        task = self.task('p(a).', ['p(var__(1))'], ['answer(a)'], ['answer(n)'])
        oracle = CoverageOracle(task)
        interrupted = [False]
        def check():
            if interrupted[0]:
                raise ResourceLimit('External resource limit immediately after verification')
        oracle.check = check
        def verify(*args, **kwargs):
            result = verify_rules(*args, **kwargs)
            interrupted[0] = True
            return result
        with patch('nidavellir.verification.verify_rules', verify), self.assertRaises(ResourceLimit) as caught:
            compile_catalogue(task, oracle, dynamic_program_bound=True)
        stats = caught.exception.stats
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['dynamic_incumbent']['cost'], 2)
        self.assertTrue(stats['dynamic_incumbent']['verification']['matched'])
        self.assertEqual(costs_from_certificates(task, stats['dynamic_target_bounds']), {('answer', 1): 2})
        self.assertIs(oracle.check, check)

    def test_callback_can_transfer_verified_incumbent_to_another_strategy(self):
        class IncumbentReady(Exception):
            pass
        task = self.improving_task()
        oracle = CoverageOracle(task)
        original_check = oracle.check
        accepted = []
        def callback(certificate):
            accepted.append(certificate)
            raise IncumbentReady()
        with self.assertRaises(IncumbentReady):
            compile_catalogue(task, oracle, dynamic_program_bound=True, on_incumbent=callback)
        self.assertEqual(accepted[0]['cost'], 4)
        self.assertTrue(verify_rules(task, accepted[0]['rules'])['matched'])
        self.assertEqual(oracle.check, original_check)

    def test_invalid_options_are_rejected(self):
        task = self.improving_task()
        for options in ({'dynamic_program_bound': 1}, {'dynamic_bound_interval': 0},
                        {'dynamic_bound_interval': True}, {'on_incumbent': lambda _: None},
                        {'dynamic_program_bound': True, 'on_incumbent': 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compile_catalogue(task, CoverageOracle(task), **options)


if __name__ == '__main__':
    unittest.main()
