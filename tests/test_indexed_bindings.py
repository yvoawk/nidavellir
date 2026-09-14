"""All integer bindings, negative witnesses and predictions match the reference."""
from itertools import combinations
import unittest
from unittest.mock import patch

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import variables
from tests import test_search_optimizations as fixtures


class IndexedBindingTests(unittest.TestCase):
    def task(self, *args, **kwargs):
        return fixtures.SearchOptimizationTests.task(self, *args, **kwargs)

    def compare(self, task):
        reference = CoverageOracle(task, binding_backend='reference', max_negative_witnesses=0)
        indexed = CoverageOracle(task, binding_backend='indexed', max_negative_witnesses=0)
        norm = lambda rows: {tuple(sorted(binding.items())) for binding in rows}
        for size in range(len(task.literals) + 1):
            for body in combinations(range(len(task.literals)), size):
                for fixed in ({}, {1: clingo.parse_term('a')}, {2: clingo.parse_term('missing')},
                              {99: clingo.parse_term('pair(var__(4),k)')}):
                    self.assertEqual(norm(reference.bindings(body, fixed)), norm(indexed.bindings(body, fixed)),
                                     (body, fixed))
                used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                for head in task.heads:
                    if not variables(head) <= used:
                        continue
                    expected = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                    predictions = {a for a in expected if a.name == head.name and len(a.arguments) == len(head.arguments)}
                    self.assertEqual(indexed.predictions(head, body), predictions)
                    self.assertEqual(indexed.coverage(head, body), reference.coverage(head, body))
                    witness = indexed.negative(head, body)
                    self.assertEqual(witness is None, not bool(predictions & task.negatives))
                    if witness is not None:
                        self.assertIn(witness['head'], predictions & task.negatives)
                        self.assertTrue(all(witness['truth'] >> i & 1 for i in body))

    def test_hidden_correlations_cycles_and_unlabelled_outputs(self):
        self.compare(self.task('p(a,x). p(b,y). p(unlabelled,z). q(x). q(z). edge(x,y). edge(y,x).',
            ['p(var__(1),var__(2))', 'q(var__(2))', 'edge(var__(2),var__(3))',
             'edge(var__(3),var__(2))'], max_body=4,
            examples='positive(answer(a)). negative(answer(b)).'))

    def test_nested_repeated_terms_constants_and_fixed_values(self):
        self.compare(self.task('p(tag,pair(a,x),x). p(tag,pair(b,y),x). p(tag,pair(c,y),y). q(x).',
            ['p(tag,pair(var__(1),var__(2)),var__(2))',
             'p(tag,pair(var__(1),var__(3)),var__(2))', 'q(var__(2))'],
            heads=('answer(pair(var__(1),var__(2)),var__(1))',),
            examples='positive(answer(pair(a,x),a)). negative(answer(pair(c,y),c)).'))

    def test_ground_head_and_empty_body(self):
        self.compare(self.task('p(a).', ['p(var__(1))'], heads=('answer(ok)',),
                               examples='positive(answer(ok)). negative(answer(other)).'))

    def test_interruption_after_index_query_never_caches_an_exists_answer(self):
        task = self.task('p(a). p(b).', ['p(var__(1))'])
        oracle = CoverageOracle(task)
        head, label = task.heads[0], min(task.positives)
        self.assertTrue(oracle.has_binding((0,)))
        def expired_bindings(*args, **kwargs):
            oracle.check = lambda: (_ for _ in ()).throw(ResourceLimit('after last native operation'))
            yield {1: clingo.parse_term('a')}
        with patch.object(oracle, 'bindings', expired_bindings):
            with self.assertRaises(ResourceLimit):
                oracle.exists(head, (0,), label)
        self.assertFalse(oracle._exists_cache)

    def test_fixed_nonnegative_head_is_rejected_before_building_unused_tables(self):
        task = self.task('p(a,x).', ['p(var__(1),var__(2))'],
                        examples='positive(answer(a)). negative(answer(b)).')
        oracle = CoverageOracle(task)
        rows = list(oracle._bindings((0,), {1: clingo.parse_term('a')}, negative_head=task.heads[0]))
        self.assertEqual(rows, [])
        self.assertFalse(getattr(oracle.space, '_integer_relations', {}))
        self.assertEqual(oracle.stats['negative_head_prunes'], 1)
