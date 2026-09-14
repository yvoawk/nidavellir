"""Necessary ground constraints never replace final ASP legality checks."""
import unittest
from unittest.mock import patch

from nidavellir.bias_analysis import HeadAnalysis
from nidavellir.relational_compile import compile_catalogue, mine_variants
from nidavellir.relational_coverage import CoverageOracle
from tests import test_optimized_oracle as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class RelationalPrefixFilterTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def sample(self, bias=''):
        return self.task('p(a). p(n). q(a). q(n). r(a). r(b). s(b). s(n).',
            ['p(var__(1))','q(var__(1))','r(var__(1))','s(var__(1))'],
            ['answer(a)','answer(b)'], ['answer(n)'], max_body=3, bias=bias)

    def test_full_catalogues_variants_and_all_optima_match_clingo(self):
        biases = ('',
            ':- body(p(var__(1))), not body(q(var__(1))).\n'
            ':- body(q(var__(1))), not body(p(var__(1))).',
            ':- 2 { body(p(var__(1))); body(q(var__(1))); body(r(var__(1))) }.',
            'aux :- body(p(var__(1))).\naux :- body(q(var__(1))).\n:- not aux.',
            'left :- not right.\nright :- not left.\n'
            ':- left, body(p(var__(1))).\n:- right, body(q(var__(1))).')
        for bias in biases:
            task=self.sample(bias)
            expected=exhaustive_catalogue(task)
            optimum=optimal_programs(((m,c) for m,c,_ in expected),3,20)
            for order in ('source','selectivity'):
                for mode in ('exhaustive','targeted'):
                    for enabled in (False,True):
                        rows,stats=compile_catalogue(task,CoverageOracle(task),order=order,
                            variants=mode,ground_prefix_filter=enabled)
                        self.assertTrue(stats['complete'])
                        self.assertEqual(normalized(rows),expected,(bias,order,mode,enabled))
                        self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),3,20),optimum)
                        rebuilt,_=mine_variants(task,CoverageOracle(task),rows,
                            ground_prefix_filter=enabled,order=order)
                        self.assertEqual(normalized(rebuilt),expected)

    def test_complete_constraint_filter_avoids_completion_solves_but_keeps_legality(self):
        task=self.sample(':- 2 { body(p(var__(1))); body(q(var__(1))) }.')
        calls=[]
        original=HeadAnalysis.accepts
        def accept(proof,body,**kwargs):
            calls.append(tuple(body))
            return original(proof,body,**kwargs)
        with patch.object(HeadAnalysis,'possible_completion',side_effect=AssertionError('redundant completion solve')):
            with patch.object(HeadAnalysis,'accepts',accept):
                rows,stats=compile_catalogue(task,CoverageOracle(task),ground_prefix_filter=True)
        self.assertEqual(normalized(rows),exhaustive_catalogue(task))
        self.assertGreater(stats['ground_prefix_checks'],0)
        self.assertEqual(stats['ground_prefix_fallback_heads'],0)
        self.assertTrue(calls)

    def test_unsupported_bias_still_calls_the_exact_completion_solver(self):
        task=self.sample('left :- not right.\nright :- not left.\n'
                         ':- left, body(p(var__(1))).\n:- right, body(q(var__(1))).')
        calls=[]
        original=HeadAnalysis.possible_completion
        def completion(proof,*args,**kwargs):
            calls.append(args)
            return original(proof,*args,**kwargs)
        with patch.object(HeadAnalysis,'possible_completion',completion):
            rows,stats=compile_catalogue(task,CoverageOracle(task),ground_prefix_filter=True)
        self.assertGreater(stats['ground_prefix_fallback_heads'],0)
        self.assertTrue(calls)
        self.assertEqual(normalized(rows),exhaustive_catalogue(task))

    def test_invalid_switch_is_rejected(self):
        for value in (1,None,'yes'):
            with self.assertRaises(ValueError):
                compile_catalogue(self.sample(),CoverageOracle(self.sample()),ground_prefix_filter=value)
