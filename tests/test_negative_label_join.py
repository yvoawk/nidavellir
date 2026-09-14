"""Negative-label joins preserve complete witnesses and unconstrained outputs."""
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import clingo

from nidavellir.labels import CompactNegativeSet
from nidavellir.negative_join import negative_relation
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import instantiate, variables
from tests import test_search_optimizations as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class NegativeLabelJoinTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    @staticmethod
    def normalized_bindings(bindings):
        return {tuple(sorted(binding.items())) for binding in bindings}

    def assert_witness(self, oracle, head, body, witness, expected):
        self.assertIn(witness['head'], expected)
        self.assertEqual(instantiate(head, witness['binding']), witness['head'])
        self.assertTrue(oracle.space.all_vars <= witness['binding'].keys())
        self.assertTrue(all(instantiate(oracle.task.literals[i], witness['binding'])
                            in oracle.task.atoms for i in body))
        truth = sum(1 << i for i, literal in enumerate(oracle.task.literals)
                    if instantiate(literal, witness['binding']) in oracle.task.atoms)
        self.assertEqual(witness['truth'], truth)

    def compare_all(self, task):
        oracles = [CoverageOracle(task, negative_join_backend=negative_backend,
                                  binding_backend=binding_backend, max_negative_witnesses=0)
                   for negative_backend, binding_backend in
                   [('labels', 'indexed'), ('bindings', 'indexed'), ('labels', 'reference')]]
        negatives = frozenset(task.negatives)
        for size in range(len(task.literals) + 1):
            for body in combinations(range(len(task.literals)), size):
                expected_bindings = self.normalized_bindings(oracles[-1].bindings(body))
                for oracle in oracles:
                    self.assertEqual(self.normalized_bindings(oracle.bindings(body)), expected_bindings)
                used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                for head in task.heads:
                    if not variables(head) <= used:
                        continue
                    model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                    predictions = {atom for atom in model if atom.name == head.name
                                   and len(atom.arguments) == len(head.arguments)}
                    expected = predictions & negatives
                    for oracle in oracles:
                        with self.subTest(head=str(head), body=body,
                                          backend=oracle.negative_join_backend,
                                          bindings=oracle.binding_backend):
                            self.assertEqual(oracle.predictions(head, body), predictions)
                            witness = oracle.negative(head, body)
                            self.assertEqual(witness is None, not bool(expected))
                            if witness is not None:
                                self.assert_witness(oracle, head, body, witness, expected)

    def test_exhaustive_head_and_hidden_correlations(self):
        self.compare_all(self.task(
            'p(a,x). p(b,y). p(unlabelled,z). q(a,y). q(b,x). q(unlabelled,z). '
            'edge(x,y). edge(y,x). good(x). good(z).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))',
             'edge(var__(2),var__(3))', 'good(var__(3))'],
            heads=('answer(var__(1),var__(2))',), max_body=4,
            examples='positive(answer(unlabelled,z)). negative(answer(a,x)). '
                     'negative(answer(b,y)). negative(answer(a,y)).'))

    def test_exhaustive_nested_repeated_terms_constants_and_multiple_heads(self):
        self.compare_all(self.task(
            'p(tag,pair(a,x),x). p(tag,pair(b,y),y). p(tag,pair(a,y),x). '
            'p(other,pair(b,x),x). good(x). good(y).',
            ['p(tag,pair(var__(1),var__(2)),var__(2))',
             'p(tag,pair(var__(1),var__(3)),var__(2))', 'good(var__(2))'],
            heads=('answer(pair(var__(1),var__(2)),var__(1))', 'other(var__(1))'),
            examples='positive(answer(pair(a,x),a)). positive(other(a)). '
                     'negative(answer(pair(b,y),b)). negative(answer(pair(a,x),b)). '
                     'negative(other(b)).'))

    def test_ground_and_nullary_heads_empty_body_and_ordinary_label_sets(self):
        task = self.task('p(a). p(b).', ['p(var__(1))'],
                         heads=('failure', 'answer(ok)', 'answer(var__(1),var__(1))'),
                         examples='positive(answer(ok)). negative(failure). '
                                  'negative(answer(a,b)). negative(answer(b,b)).')
        self.compare_all(task)
        self.compare_all(replace(task, negatives=frozenset(task.negatives)))

    def test_complete_label_witness_does_not_skip_hidden_join_or_free_outputs(self):
        task = self.task('p(a). p(b). p(unlabelled). q(x). q(y). r(y). unused(z).',
                         ['p(var__(1))', 'q(var__(2))', 'r(var__(2))', 'unused(var__(3))'],
                         examples='positive(answer(a)). negative(answer(b)).')
        head = task.heads[0]
        body = tuple(i for i, literal in enumerate(task.literals) if literal.name != 'unused')
        oracle = CoverageOracle(task)
        witness = oracle.negative(head, body)
        self.assert_witness(oracle, head, body, witness, {clingo.parse_term('answer(b)')})
        self.assertEqual(witness['binding'][2], clingo.parse_term('y'))
        self.assertEqual(oracle.predictions(head, body), frozenset(map(clingo.parse_term,
                         ['answer(a)', 'answer(b)', 'answer(unlabelled)'])))
        self.assertEqual(len(list(oracle.bindings(body))), 3)
        # A supplied label fixes only the head. Its two incompatible hidden
        # substitutions must not be combined into an invented full witness.
        bad = self.task('p(a). q(x). r(y).',
                        ['p(var__(1))', 'q(var__(2))', 'r(var__(2))'],
                        examples='negative(answer(a)).')
        self.assertIsNone(CoverageOracle(bad).negative(bad.heads[0], tuple(range(3))))

    def test_fixed_partial_heads_keep_correlations_and_extra_binding_values(self):
        task = self.task('p(a,x). p(b,y). p(a,y). q(x,z). q(y,w).',
                         ['p(var__(1),var__(2))', 'q(var__(2),var__(3))'],
                         heads=('answer(var__(1),var__(2))',),
                         examples='positive(answer(a,x)). negative(answer(a,y)). '
                                  'negative(answer(b,y)).')
        head, body = task.heads[0], tuple(range(2))
        reference = CoverageOracle(task, negative_join_backend='bindings', binding_backend='reference')
        indexed = CoverageOracle(task)
        for fixed in ({}, {1: clingo.parse_term('a')}, {2: clingo.parse_term('y')},
                      {1: clingo.parse_term('absent')},
                      {99: clingo.parse_term('pair(var__(4),k)')}):
            expected = self.normalized_bindings(reference._bindings(body, fixed, negative_head=head))
            self.assertEqual(self.normalized_bindings(indexed._bindings(body, fixed, negative_head=head)),
                             expected)

    def test_catalogues_and_all_bounded_optima_match_exhaustive_clingo(self):
        task = self.task('p(a). p(b). p(n1). q(a). q(b). q(n2). '
                         'r(b). r(c). s(a). s(c).',
                         [f'{name}(var__(1))' for name in ('p', 'q', 'r', 's')],
                         examples='positive(answer(a)). positive(answer(b)). positive(answer(c)). '
                                  'negative(answer(n1)). negative(answer(n2)).', max_body=3)
        expected = exhaustive_catalogue(task)
        full, bound = (1 << len(task.positives)) - 1, 6
        optimum = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        self.assertIsNotNone(optimum[0])
        for backend, limit in [('labels', 65536), ('labels', 0), ('bindings', 65536)]:
            for order in ('source', 'selectivity'):
                for variants in ('exhaustive', 'targeted'):
                    oracle = CoverageOracle(task, negative_join_backend=backend, negative_label_limit=limit)
                    rows, stats = compile_catalogue(task, oracle, order=order,
                                                     variants=variants, alpha_collect=False)
                    self.assertTrue(stats['complete'])
                    self.assertEqual(normalized(rows), expected)
                    self.assertEqual(optimal_programs(((r['coverage'], r['cost']) for r in rows),
                                                     full, bound), optimum)

    def test_compact_label_construction_never_uses_symbol_iteration(self):
        task = self.task('p(a). p(b).', ['p(var__(1))'])
        self.assertIsInstance(task.negatives, CompactNegativeSet)
        oracle = CoverageOracle(task)
        with patch.object(CompactNegativeSet, '__iter__', side_effect=AssertionError('outer Symbol iteration')):
            slots, table = negative_relation(oracle, task.heads[0])
            self.assertEqual(len(slots), 1)
            self.assertEqual(len(table.rows), 1)
            self.assertIsNone(oracle.negative(task.heads[0], (0,)))

    def test_alpha_renamed_heads_share_one_complete_table(self):
        heads = tuple(f'answer(pair(var__({i}),var__({i + 1})),var__({i}))'
                      for i in range(1, 80, 2))
        task = self.task('p(a).', ['p(var__(1))'], heads=heads,
                         examples='positive(answer(pair(a,x),a)). '
                                  'negative(answer(pair(b,y),b)). '
                                  'negative(answer(pair(c,z),c)). '
                                  'negative(answer(pair(a,x),wrong)).')
        oracle = CoverageOracle(task)
        tables = []
        for head in task.heads:
            slots, table = negative_relation(oracle, head)
            self.assertEqual(set(slots), variables(head))
            tables.append(table)
        self.assertTrue(all(table is tables[0] for table in tables))
        self.assertEqual(len(oracle._negative_join_relations), 1)
        self.assertEqual(oracle._negative_join_rows, 2)

    def test_global_row_budget_and_disabled_join_use_exact_fallback(self):
        task = self.task('p(a). p(b). p(c).', ['p(var__(1))'],
                         heads=('first(var__(1))', 'second(var__(1))'),
                         examples='positive(first(a)). negative(first(b)). negative(first(c)). '
                                  'negative(second(b)). negative(second(c)).')
        # Ordinary sets still require materialization; compact labels have a
        # separate bounded bucket join when the expanded row budget is full.
        task = replace(task, negatives=frozenset(task.negatives))
        oracle = CoverageOracle(task, negative_label_limit=4)
        self.assertIsNotNone(negative_relation(oracle, task.heads[0]))
        self.assertIsNone(negative_relation(oracle, task.heads[1]))
        self.assertEqual(oracle._negative_join_rows, 2)
        self.assertLessEqual(oracle._negative_join_rows, 4)
        self.assertIsNotNone(oracle.negative(task.heads[1], (0,)))
        for options in ({'negative_label_limit': 0}, {'negative_label_limit': 1},
                        {'negative_join_backend': 'bindings'}):
            oracle = CoverageOracle(task, **options)
            for head in task.heads:
                self.assertIsNone(negative_relation(oracle, head))
                self.assertIsNotNone(oracle.negative(head, (0,)))
            self.assertFalse(getattr(oracle, '_negative_join_relations', {}))
            self.assertEqual(getattr(oracle, '_negative_join_rows', 0), 0)

    def test_table_count_budget_falls_back_without_forgetting_real_negatives(self):
        heads = tuple(f'answer_{i}(var__(1))' for i in range(70))
        task = self.task('p(b).', ['p(var__(1))'], heads=heads,
                         examples=' '.join(f'negative(answer_{i}(b)).' for i in range(70)))
        oracle = CoverageOracle(task)
        results = [negative_relation(oracle, head) for head in task.heads]
        self.assertEqual(sum(result is not None for result in results), 64)
        self.assertEqual(len(oracle._negative_join_relations), 64)
        self.assertEqual(oracle._negative_join_rows, 64)
        for head, result in zip(task.heads, results):
            if result is None:
                self.assertIsNotNone(oracle.negative(head, (0,)))
        self.assertEqual(len(oracle._negative_join_relations), 64)

    def test_interrupted_construction_publishes_neither_table_nor_safety(self):
        task = self.task('p(0..10).', ['p(var__(1))'],
                         examples='positive(answer(10)). ' +
                         ' '.join(f'negative(answer({i})).' for i in range(10)))
        original = CompactNegativeSet.iter_records
        for query in (lambda oracle: negative_relation(oracle, task.heads[0]),
                      lambda oracle: oracle.negative(task.heads[0], (0,))):
            expired, visited = False, 0
            oracle = CoverageOracle(task)
            def check():
                if expired:
                    raise ResourceLimit('during negative-label construction')
            def interrupted(labels):
                nonlocal expired, visited
                for record in original(labels):
                    visited += 1
                    yield record
                    expired = True
            oracle.check = check
            with patch.object(CompactNegativeSet, 'iter_records', interrupted):
                with self.assertRaises(ResourceLimit):
                    query(oracle)
            self.assertGreater(visited, 0)
            self.assertFalse(getattr(oracle, '_negative_join_relations', {}))
            self.assertEqual(getattr(oracle, '_negative_join_rows', 0), 0)
            self.assertFalse(oracle._safe)
            self.assertFalse(oracle._negative_witnesses)
            expired = False
            self.assertIsNotNone(oracle.negative(task.heads[0], (0,)))

    def test_invalid_backend_or_budget_is_rejected(self):
        task = self.task('p(a).', ['p(var__(1))'])
        for value in (-1, 1.5, True, None):
            with self.assertRaises(ValueError):
                CoverageOracle(task, negative_label_limit=value)
        for value in ('unknown', '', None):
            with self.assertRaises(ValueError):
                CoverageOracle(task, negative_join_backend=value)


if __name__ == '__main__':
    unittest.main()
