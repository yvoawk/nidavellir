"""Exact ground lowering, including recursive bias, against independent Clingo."""
from itertools import combinations, product
from dataclasses import replace
import unittest
from unittest.mock import patch

from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit
from tests import test_bias_analysis as fixtures


class GroundLegalityTests(unittest.TestCase):
    task = fixtures.BiasAnalysisTests.task

    def compare(self, restrictions, *, supported=True, body_choice=None, head_support=None):
        task = self.task(['p', 'q', 'r', 's'], heads=('answer',), examples='',
                         max_body=4, restrictions=restrictions, body_choice=body_choice)
        if head_support is not None:
            task = replace(task, bias_source=task.bias_source.replace(
                '1 <= { head(H): candidate_head(H) } <= 1.', head_support))
        proof = analyze(task).heads[task.heads[0]]
        self.assertEqual(proof._checker.compiled_legality is not None, supported)
        legal = [frozenset(body) for size in range(5) for body in combinations(range(4), size)
                 if task.accepts(task.heads[0], body)]
        for states in product(range(3), repeat=4):
            included = frozenset(i for i, state in enumerate(states) if state == 1)
            excluded = frozenset(i for i, state in enumerate(states) if state == 2)
            expected = any(included <= body and not excluded & body for body in legal)
            self.assertEqual(proof.possible_completion(included, excluded), expected, states)
            if len(included | excluded) == 4:
                self.assertEqual(proof.accepts(included), expected, states)
        return proof

    def test_recursive_connectivity_uses_the_least_fixed_point(self):
        proof = self.compare('''
            edge(0,1) :- body(p). edge(1,2) :- body(q).
            edge(2,1) :- body(r). edge(2,3) :- body(s).
            used(1) :- body(p). used(2) :- body(q).
            used(1) :- body(r). used(3) :- body(s).
            reachable(0). reachable(Y) :- reachable(X), edge(X,Y).
            :- used(X), not reachable(X).
        ''')
        self.assertGreater(proof.checker_stats['compiled_legality_hits'], 0)

    def test_stratified_negation_above_and_below_a_positive_cycle(self):
        self.compare('''
            blocked :- body(r). eligible :- body(p), not blocked.
            left :- eligible. right :- left, body(q). left :- right.
            absent :- not right. :- body(s), absent.
        ''')

    def test_recursive_cardinality_definitions(self):
        self.compare('''
            left :- 2 { body(p); body(q); right }.
            right :- left, body(r).
            :- body(s), not right.
        ''')

    def test_negative_cycles_even_when_disconnected_keep_solver_fallback(self):
        for source in ('left :- not right. right :- not left.',
                       'left :- not left, body(p).'):
            self.compare(source, supported=False)

    def test_auxiliary_choices_and_nonunit_weights_keep_solver_fallback(self):
        self.compare('{ chosen }. :- body(p), not chosen.', supported=False)
        self.compare(':- 3 #sum { 2,p:body(p); 1,q:body(q); 1,r:body(r) }.', supported=False)

    def test_conditional_input_support_and_input_definitions_keep_fallback(self):
        self.compare('', body_choice='{body(p)}. {body(q)} :- body(p).', supported=False)
        self.compare('', body_choice='{body(L)} :- allowed_literal(L). body(p) :- body(q).',
                     supported=False)

    def test_fixed_head_assumptions_are_never_used_as_support(self):
        for support in ('{head(answer)} :- body(q).',
                        'ready :- body(q). {head(answer)} :- ready.',
                        'blocked :- not body(q). {head(answer)} :- not blocked.',
                        'head(answer) :- body(q).',
                        'ready :- head(answer). ready :- body(q). {head(answer)} :- ready.'):
            with self.subTest(support=support):
                self.compare('', head_support=support, supported=False)

    def test_forced_and_absent_inputs_match_original_solver(self):
        self.compare('', body_choice='body(p). {body(q)}.')

    def test_complete_assignments_need_no_more_solver_calls(self):
        proof = self.compare(':- body(p), not body(q).')
        proof._checker.cache.clear()
        before = proof.checker_stats['solves']
        with patch.object(proof._checker, 'find', side_effect=AssertionError('unexpected solve')):
            for size in range(5):
                for body in combinations(range(4), size):
                    proof.accepts(body)
        self.assertEqual(proof.checker_stats['solves'], before)

    def test_no_time_limit_still_honors_external_cancellation(self):
        task = self.task(['p'], heads=('answer',), examples='')
        self.assertTrue(analyze(task, seconds=None).stats['complete'])
        def stop():
            raise ResourceLimit('cancelled')
        with self.assertRaisesRegex(ResourceLimit, 'cancelled'):
            analyze(task, seconds=None, check=stop)


if __name__ == '__main__':
    unittest.main()
