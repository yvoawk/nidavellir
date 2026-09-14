"""Ground prefix filters and deletion certificates against actual stable models."""
from itertools import combinations, product
import unittest

import clingo

from nidavellir.bias_analysis import analyze
from tests import test_bias_analysis


class PrefixFilterTests(unittest.TestCase):
    task = test_bias_analysis.BiasAnalysisTests.task

    def check_filter(self, task):
        analysis = analyze(task)
        count = len(task.literals)
        for head, proof in analysis.heads.items():
            legal = [sum(1 << i for i in body)
                     for size in range(min(count, task.max_body) + 1)
                     for body in combinations(range(count), size) if task.accepts(head, body)]
            prefix = proof.prefix_filter
            for states in product(range(3), repeat=count):
                included = sum(1 << i for i, state in enumerate(states) if state == 1)
                excluded = sum(1 << i for i, state in enumerate(states) if state == 2)
                if any(body & included == included and not body & excluded for body in legal):
                    self.assertTrue(prefix.allows(included, excluded), (head, included, excluded))
                for i in range(count):
                    if included >> i & 1 and prefix.can_delete(i, included, excluded):
                        for body in legal:
                            if body & included == included and not body & excluded:
                                self.assertIn(body & ~(1 << i), legal, (head, included, excluded, body, i))
            for body in legal:
                for i in range(count):
                    if prefix.deletion_monotone >> i & 1 and body >> i & 1:
                        self.assertIn(body & ~(1 << i), legal, (head, body, i))
                    if body >> i & 1:
                        self.assertFalse(prefix.incompatible[i] & body)
        return analysis

    def test_cardinalities_and_guard_closures(self):
        task = self.task([f'{p}(var__(1))' for p in 'pqrs'], max_body=4,
                         restrictions=':- 2 { body(p(var__(1))); body(q(var__(1))); body(r(var__(1))) }.\n'
                                      ':- not 1 { body(p(var__(1))); body(s(var__(1))) }.\n'
                                      ':- body(q(var__(1))), not body(s(var__(1))).\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        self.assertTrue(proof.prefix_filter.all_constraints_compiled)
        self.assertTrue(any(proof.prefix_filter.incompatible))

    def test_head_conditioned_cardinalities(self):
        task = self.task([f'{p}(var__(1))' for p in 'pqr'], max_body=3, examples='',
                         heads=('answer(left,var__(1))', 'answer(right,var__(1))'),
                         restrictions=':- head(answer(left,var__(1))), 2 { body(p(var__(1))); body(q(var__(1))) }.\n'
                                      ':- head(answer(right,var__(1))), not body(r(var__(1))).\n')
        analysis = self.check_filter(task)
        self.assertTrue(all(p.prefix_filter.all_constraints_compiled for p in analysis.heads.values()))
        left, right = [analysis.heads[clingo.parse_term(f'answer({s},var__(1))')] for s in ('left', 'right')]
        self.assertNotEqual(left.prefix_filter.incompatible, right.prefix_filter.incompatible)

    def test_deletion_certificate_protects_safety_guards_and_minimums(self):
        task = self.task(['p(var__(1))', 'q(var__(2))', 'r(var__(2))'], max_body=3,
                         restrictions=':- body(r(var__(2))), not body(q(var__(2))).\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        p, q, r = [task.literals.index(clingo.parse_term(f'{s}(var__({v}))')) for s, v in [('p', 1), ('q', 2), ('r', 2)]]
        self.assertFalse(proof.prefix_filter.deletion_monotone & ((1 << p) | (1 << q)))
        self.assertTrue(proof.prefix_filter.deletion_monotone >> r & 1)

    def test_choice_support_and_normal_body_derivations_are_not_assumed_free(self):
        for body_choice in ('{ body(p(var__(1))) }. { body(q(var__(2))) } :- body(p(var__(1))).\n',
                            '{ body(L) : allowed_literal(L) }. body(p(var__(1))) :- body(q(var__(2))).\n'):
            task = self.task(['p(var__(1))', 'q(var__(2))'], body_choice=body_choice)
            proof = self.check_filter(task).heads[task.heads[0]]
            self.assertEqual(proof.prefix_filter.deletion_monotone, 0)

    def test_contextual_deletion_preserves_future_guards_and_existing_safety(self):
        task = self.task(['p(var__(1))', 'q(var__(1))', 'r(var__(2))'], max_body=3,
                         restrictions=':- body(r(var__(2))), not body(q(var__(1))).\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        p, q, r = [task.literals.index(clingo.parse_term(f'{s}(var__({v}))')) for s, v in [('p', 1), ('q', 1), ('r', 2)]]
        selected = (1 << p) | (1 << q)
        self.assertFalse(proof.prefix_filter.deletion_monotone >> p & 1)
        self.assertTrue(proof.prefix_filter.can_delete(p, selected, 0))
        self.assertFalse(proof.prefix_filter.can_delete(q, selected, 0))
        self.assertTrue(proof.prefix_filter.can_delete(q, selected, 1 << r))

    def test_fact_body_literal_is_protected(self):
        task = self.task(['p(var__(1))', 'q(var__(2))'], body_choice='body(p(var__(1))). { body(q(var__(2))) }.\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        p = task.literals.index(clingo.parse_term('p(var__(1))'))
        self.assertFalse(proof.prefix_filter.deletion_monotone >> p & 1)

    def test_auxiliary_nonmonotonic_cycles_disable_deletion_certificate(self):
        task = self.task(['p(var__(1))', 'q(var__(2))'],
                         restrictions='left :- not right. right :- not left.\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        self.assertEqual(proof.prefix_filter.deletion_monotone, 0)

    def test_unhandled_auxiliary_choice_keeps_sat_fallback(self):
        task = self.task(['p(var__(1))', 'q(var__(2))'],
                         restrictions='{ chosen }. :- body(q(var__(2))), not chosen.\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        self.assertFalse(proof.prefix_filter.all_constraints_compiled)
        self.assertGreater(proof.prefix_filter.skipped_constraints, 0)
        self.assertEqual(proof.prefix_filter.deletion_monotone, 0)

    def test_weighted_aggregate_with_nonunit_weights_keeps_fallback(self):
        task = self.task([f'{p}(var__(1))' for p in 'pqr'], max_body=3,
                         restrictions=':- 3 #sum { 2,p : body(p(var__(1))); 1,q : body(q(var__(1))); 1,r : body(r(var__(1))) }.\n')
        proof = self.check_filter(task).heads[task.heads[0]]
        self.assertGreater(proof.prefix_filter.skipped_constraints, 0)


if __name__ == '__main__':
    unittest.main()
