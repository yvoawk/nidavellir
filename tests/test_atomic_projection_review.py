"""Independent cache review of complete versus indexed atomic projections."""
import unittest

import clingo

from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import asp_text
from tests import test_optimized_oracle as fixtures


class AtomicProjectionReviewTests(unittest.TestCase):
    def task(self, background, literals, positives, heads):
        return fixtures.OptimizedOracleContractTests.task(
            self, background, literals, positives, heads=heads, max_body=len(literals))

    def expected(self, task, head, literal, oracle):
        control = clingo.Control(['0', '--warn=none'])
        labels = '\n'.join(f'review_label({label}).' for label in task.positives)
        rule = (f'review_hit({asp_text(head)}) :- review_label({asp_text(head)}), '
                f'{asp_text(literal)}.')
        control.add('base', [], task.background + '\n' + labels + '\n' + rule)
        control.ground([('base', [])])
        models = []
        result = control.solve(on_model=lambda m: models.append(set(m.symbols(atoms=True))))
        self.assertTrue(result.satisfiable and result.exhausted)
        self.assertEqual(len(models), 1)
        return sum(1 << pos for label, pos in oracle.positives.items()
                   if clingo.Function('review_hit', [label]) in models[0])

    def test_indexed_keys_survive_alpha_renaming_reversal_and_nested_heads(self):
        task = self.task('r(a,pair(x,z)). r(b,pair(y,w)). ' +
                         ' '.join(f'r(extra{i},pair(v{i},w{i})).' for i in range(8)),
            ['r(var__(1),pair(var__(2),var__(3)))',
             'r(var__(4),pair(var__(5),var__(6)))',
             'r(var__(2),pair(var__(1),var__(3)))'],
            ['first(pair(a,x),a)', 'first(pair(no,x),no)',
             'second(pair(b,y),b)', 'reverse(pair(y,b),b)'],
            ['first(pair(var__(1),var__(2)),var__(1))',
             'second(pair(var__(4),var__(5)),var__(4))',
             'reverse(pair(var__(5),var__(4)),var__(4))'])
        for heads in (task.heads, tuple(reversed(task.heads))):
            oracle = CoverageOracle(task)
            for head in heads:
                for i, literal in enumerate(task.literals):
                    expected = self.expected(task, head, literal, oracle)
                    self.assertEqual(oracle.atomic_coverage(head, i), expected)
                    self.assertEqual(oracle.atomic_coverage(head, i), expected)
            self.assertTrue(any(kind == 'answers' for kind, _ in oracle._atomic_projections.values()))

    def test_one_indexed_key_restores_every_matching_positive_position(self):
        task = self.task('p(a). p(x). p(c). p(d). p(e).',
            ['p(var__(1))', 'p(var__(2))', 'p(var__(3))'],
            ['answer(a,x)', 'answer(a,y)', 'answer(b,x)', 'answer(b,y)'],
            ['answer(var__(1),var__(2))'])
        oracle = CoverageOracle(task)
        for i, literal in enumerate(task.literals):
            self.assertEqual(oracle.atomic_coverage(task.heads[0], i),
                             self.expected(task, task.heads[0], literal, oracle))

    def test_interrupted_indexed_answer_does_not_cache_a_false_for_unknown_key(self):
        task = self.task('p(a). p(b). p(c). p(d).', ['p(var__(1))'],
                         ['answer(a)', 'answer(b)', 'answer(x)'], ['answer(var__(1))'])
        oracle = CoverageOracle(task)
        real = oracle.has_binding
        calls = 0
        def stop_second(body, fixed=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError('independent interruption')
            return real(body, fixed)
        oracle.has_binding = stop_second
        with self.assertRaises(TimeoutError):
            oracle.atomic_coverage(task.heads[0], 0)
        kind, answers = next(iter(oracle._atomic_projections.values()))
        self.assertEqual(kind, 'answers')
        self.assertEqual(len(answers), 1)
        oracle.has_binding = real
        self.assertEqual(oracle.atomic_coverage(task.heads[0], 0),
                         self.expected(task, task.heads[0], task.literals[0], oracle))


if __name__ == '__main__':
    unittest.main()
