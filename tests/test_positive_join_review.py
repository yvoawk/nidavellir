"""Independent correlation review of the positive-label semijoin."""
from itertools import combinations
import unittest

import clingo

from nidavellir.positive_join import has_positive
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import variables
from tests import test_positive_join as fixtures


def model(program):
    control = clingo.Control(['0', '--warn=none'])
    control.add('base', [], program)
    control.ground([('base', [])])
    models = []
    result = control.solve(on_model=lambda value: models.append(set(value.symbols(atoms=True))))
    if not result.satisfiable or not result.exhausted or len(models) != 1:
        raise AssertionError('The independent positive reference must have one complete model')
    return models[0]


class PositiveJoinReviewTests(unittest.TestCase):
    def make_task(self, background, literals, labels, heads):
        return fixtures.PositiveJoinTests.task(self, background, literals, labels, heads)

    def compare_all(self, task):
        oracle = CoverageOracle(task)
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(head) <= used:
                        continue
                    predictions = model(task.background + '\n' + task.rule(head, body))
                    # Repeated calls in both mask orders exercise shared indexes
                    # without allowing an answer for one mask to leak to another.
                    masks = list(range(1 << len(oracle.positives)))
                    for possible in masks + masks[::-1]:
                        expected = any(atom in predictions and possible & (1 << position)
                                       for atom, position in oracle.positives.items())
                        with self.subTest(head=str(head), body=body, mask=possible):
                            self.assertEqual(has_positive(oracle, head, body, possible), expected)
        return oracle

    def test_positive_column_domains_do_not_admit_missing_crossed_pairs(self):
        task = self.make_task('edge(a,x). edge(b,y). anchor(a).',
            ['edge(var__(1),var__(2))', 'anchor(var__(1))'],
            ['answer(a,y)', 'answer(b,x)'], ['answer(var__(1),var__(2))'])
        oracle = self.compare_all(task)
        edge = next(i for i, literal in enumerate(task.literals) if literal.name == 'edge')
        self.assertFalse(has_positive(oracle, task.heads[0], (edge,), 3))

    def test_same_predicate_heads_preserve_reversal_repetition_and_nested_terms(self):
        task = self.make_task('edge(a,x). edge(b,y). edge(a,a). mark(a). mark(x).',
            ['edge(var__(1),var__(2))', 'mark(var__(1))', 'mark(var__(2))'],
            ['answer(pair(a,x),a)', 'answer(pair(x,a),a)', 'answer(pair(a,a),a)'],
            ['answer(pair(var__(1),var__(2)),var__(1))',
             'answer(pair(var__(2),var__(1)),var__(1))',
             'answer(pair(var__(1),var__(1)),var__(1))'])
        self.compare_all(task)

    def test_partial_head_binding_filters_the_joint_positive_relation(self):
        task = self.make_task('anchor(a). wide(x). wide(y). relation(a,x,z). relation(a,y,w). good(z).',
            ['anchor(var__(1))', 'wide(var__(2))',
             'relation(var__(1),var__(2),var__(3))', 'good(var__(3))'],
            ['alpha(a,y)', 'alpha(b,x)', 'omega(a,x)'],
            ['alpha(var__(1),var__(2))', 'omega(var__(1),var__(2))'])
        self.compare_all(task)


if __name__ == '__main__':
    unittest.main()
