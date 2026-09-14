"""Compact label joins agree with exhaustive rules without expanding labels."""
from contextlib import ExitStack
from dataclasses import replace
from itertools import combinations, product
import unittest
from unittest.mock import patch

import clingo

from nidavellir.labels import CompactNegativeSet, _Table, _TableBuilder
from nidavellir.negative_join import negative_relation
from nidavellir.oracle import ResourceLimit, match
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import instantiate, variables
from tests import test_search_optimizations as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


def forbid_expansion():
    stack = ExitStack()
    for owner, name in ((CompactNegativeSet, '__iter__'),
                        (CompactNegativeSet, 'iter_records'), (_Table, 'arguments')):
        stack.enter_context(patch.object(owner, name, side_effect=AssertionError('expanded labels')))
    return stack


def force_pivot(atoms, pivot):
    """Exercise valid storage layouts independently of the layout heuristic."""
    tables = {}
    for atom in atoms:
        signature = atom.name, len(atom.arguments)
        if signature not in tables:
            tables[signature] = _TableBuilder(*signature)
            tables[signature].pivot = pivot if atom.arguments else None
        tables[signature].add(atom.arguments)
    return CompactNegativeSet([table.freeze() for table in tables.values()])


class CompactNegativeJoinTests(unittest.TestCase):
    task = fixtures.SearchOptimizationTests.task

    @staticmethod
    def binding_set(bindings):
        return {tuple(sorted(binding.items())) for binding in bindings}

    def assert_witness(self, oracle, head, body, witness, expected):
        self.assertIn(witness['head'], expected)
        self.assertEqual(instantiate(head, witness['binding']), witness['head'])
        self.assertTrue(oracle.space.all_vars <= witness['binding'].keys())
        self.assertTrue(all(instantiate(oracle.task.literals[i], witness['binding'])
                            in oracle.task.atoms for i in body))
        self.assertEqual(witness['truth'], sum(1 << i for i, literal in enumerate(oracle.task.literals)
                         if instantiate(literal, witness['binding']) in oracle.task.atoms))

    def compare_all_bodies(self, task):
        self.assertGreater(len(task.negatives), 1)
        negatives = frozenset(task.negatives)
        optimized = CoverageOracle(task, negative_label_limit=1, max_negative_witnesses=0)
        reference = CoverageOracle(task, negative_join_backend='bindings', binding_backend='reference',
                                   max_negative_witnesses=0)
        saw_safe = saw_negative = False
        for size in range(len(task.literals) + 1):
            for body in combinations(range(len(task.literals)), size):
                used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                for head in task.heads:
                    if not variables(head) <= used:
                        continue
                    expected_model = fixtures.atoms(task.background + '\n' + task.rule(head, body))
                    predictions = {atom for atom in expected_model if atom.name == head.name
                                   and len(atom.arguments) == len(head.arguments)}
                    expected = predictions & negatives
                    with self.subTest(head=str(head), body=body):
                        self.assertEqual(optimized.predictions(head, body), predictions)
                        self.assertEqual(self.binding_set(optimized._bindings(body, negative_head=head)),
                                         self.binding_set(reference._bindings(body, negative_head=head)))
                        witness = optimized.negative(head, body)
                        self.assertEqual(witness is None, not bool(expected))
                        if witness is None:
                            saw_safe = True
                        else:
                            saw_negative = True
                            self.assert_witness(optimized, head, body, witness, expected)
        self.assertTrue(saw_safe)
        self.assertTrue(saw_negative)

    def test_exhaustive_head_and_hidden_correlations_match_clingo(self):
        self.compare_all_bodies(self.task(
            'p(a,x). p(b,y). p(unlabelled,z). q(a,y). q(b,x). q(unlabelled,z). '
            'edge(x,y). edge(y,x). good(x). good(z).',
            ['p(var__(1),var__(2))', 'q(var__(1),var__(2))',
             'edge(var__(2),var__(3))', 'good(var__(3))'],
            heads=('answer(var__(1),var__(2))',), max_body=4,
            examples='positive(answer(unlabelled,z)). negative(answer(a,x)). '
                     'negative(answer(b,y)). negative(answer(a,y)).'))

    def test_repeated_and_nested_heads_retain_exact_fallback(self):
        task = self.task(
            'p(tag,pair(a,x),x). p(tag,pair(b,y),y). p(tag,pair(a,y),x). '
            'p(other,pair(b,x),x). good(x).',
            ['p(tag,pair(var__(1),var__(2)),var__(2))',
             'p(tag,pair(var__(1),var__(3)),var__(2))', 'good(var__(2))'],
            heads=('answer(pair(var__(1),var__(2)),var__(1))', 'other(var__(1),var__(1))'),
            examples='positive(answer(pair(a,x),a)). positive(other(a,a)). '
                     'negative(answer(pair(b,y),b)). negative(answer(pair(a,x),b)). '
                     'negative(other(b,b)). negative(other(a,b)).')
        for head in task.heads:
            self.assertIsNone(negative_relation(CoverageOracle(task, negative_label_limit=1), head))
        self.compare_all_bodies(task)

    def test_all_pivots_and_partial_head_columns_preserve_correlated_holes(self):
        texts = ['answer(tag,a,x)', 'answer(tag,a,y)', 'answer(tag,b,y)',
                 'answer(other,b,x)', 'answer(other,c,z)']
        atoms = tuple(map(clingo.parse_term, texts))
        task = self.task('p(a).', ['p(var__(1))'],
                         heads=('answer(var__(3),var__(1),var__(2))',
                                'answer(tag,var__(1),var__(2))',
                                'answer(absent,var__(1),var__(2))'),
                         examples=' '.join('negative(' + text + ').' for text in texts))
        for pivot in range(3):
            compact_task = replace(task, negatives=force_pivot(atoms, pivot))
            self.assertEqual(compact_task.negatives.stats()['pivots']['answer/3'], pivot)
            for head in task.heads:
                oracle = CoverageOracle(compact_task, negative_label_limit=1)
                # Symbol IDs in the oracle must not be confused with per-column label IDs.
                for symbol in reversed(tuple(map(clingo.parse_term, ['x', 'tag', 'a', 'y', 'b', 'z']))):
                    oracle.space.intern(symbol)
                with forbid_expansion():
                    result = negative_relation(oracle, head)
                    self.assertIsNotNone(result)
                    slots, relation = result
                    expected = {tuple(oracle.space.intern(binding[slot]) for slot in slots)
                                for atom in atoms if (binding := match(head, atom)) is not None}
                    self.assertEqual(len(relation.rows), len(expected))
                    self.assertEqual(set(relation.rows), expected)
                    domains = [set(row[column] for row in expected) |
                               {oracle.space.intern(clingo.parse_term('missing'))}
                               for column in range(len(slots))]
                    for count in range(len(slots) + 1):
                        for columns in combinations(range(len(slots)), count):
                            for values in product(*(domains[column] for column in columns)):
                                fixed = tuple(zip(columns, values))
                                actual = relation.select(fixed, oracle.space.stats)
                                wanted = {row for row in expected if all(row[c] == v for c, v in fixed)}
                                with self.subTest(pivot=pivot, head=str(head), fixed=fixed):
                                    self.assertEqual(len(actual), len(wanted))
                                    self.assertEqual(set(actual), wanted)
                                    self.assertEqual(set(actual), wanted)  # Reusable selection, not a generator.

    def test_sparse_buckets_work_with_every_storage_pivot(self):
        atoms = tuple(clingo.Function('answer', [clingo.Number(i)] * 3) for i in range(4205))
        task = self.task('p(0).', ['p(var__(1))'],
                         heads=('answer(var__(1),var__(2),var__(3))',), examples='')
        for pivot in range(3):
            labels = force_pivot(atoms, pivot)
            self.assertLess(labels.stats()['bitmap_buckets'], labels.stats()['buckets'])
            oracle = CoverageOracle(replace(task, negatives=labels), negative_label_limit=1)
            with forbid_expansion():
                _, relation = negative_relation(oracle, task.heads[0])
                self.assertEqual(len(relation.rows), len(atoms))
                for i in (0, 4095, 4096, 4204):
                    value = oracle.space.intern(clingo.Number(i))
                    wrong = oracle.space.intern(clingo.Number((i + 1) % len(atoms)))
                    for column in range(3):
                        with self.subTest(pivot=pivot, value=i, column=column):
                            rows = relation.select(((column, value),), oracle.space.stats)
                            self.assertEqual(len(rows), 1)
                            self.assertEqual(set(rows), {(value,) * 3})
                    self.assertFalse(relation.select(((0, value), (1, wrong)), oracle.space.stats))

    def test_ground_nullary_and_missing_signatures(self):
        task = self.task('p(a).', ['p(var__(1))'],
                         heads=('failure', 'missing', 'answer(ok)', 'answer(absent)'),
                         examples='negative(failure). negative(answer(ok)). negative(answer(other)).')
        oracle = CoverageOracle(task, negative_label_limit=1)
        with forbid_expansion():
            for head in task.heads:
                expected = {()} if str(head) in {'failure', 'answer(ok)'} else set()
                result = negative_relation(oracle, head)
                if head.arguments:
                    slots, relation = result
                    self.assertEqual(slots, ())
                    self.assertEqual(set(relation.rows), expected)
                    self.assertEqual(len(relation.select((), oracle.space.stats)), len(expected))
                else:
                    self.assertIsNone(result)
                self.assertEqual(oracle.negative(head, ()) is None, not bool(expected))

    def test_fixed_heads_keep_hidden_join_and_unrelated_binding_values(self):
        task = self.task('p(a,x). p(a,y). p(b,y). q(x,z). q(y,w). r(w).',
                         ['p(var__(1),var__(2))', 'q(var__(2),var__(3))', 'r(var__(3))'],
                         heads=('answer(var__(1),var__(2))',),
                         examples='positive(answer(a,x)). negative(answer(a,y)). negative(answer(b,y)).')
        head, body = task.heads[0], tuple(range(3))
        optimized = CoverageOracle(task, negative_label_limit=1)
        reference = CoverageOracle(task, negative_join_backend='bindings', binding_backend='reference')
        for fixed in ({}, {1: clingo.parse_term('a')}, {2: clingo.parse_term('y')},
                      {3: clingo.parse_term('z')}, {1: clingo.parse_term('missing')},
                      {99: clingo.parse_term('pair(var__(4),k)')}):
            expected = self.binding_set(reference._bindings(body, fixed, negative_head=head))
            with forbid_expansion(), self.subTest(fixed=fixed):
                self.assertEqual(self.binding_set(optimized._bindings(body, fixed, negative_head=head)), expected)
        witness = optimized.negative(head, body)
        self.assert_witness(optimized, head, body, witness, frozenset(task.negatives))
        self.assertEqual(witness['binding'][3], clingo.parse_term('w'))

    def test_catalogue_variants_and_all_bounded_optima_match_exhaustive_clingo(self):
        task = self.task('p(a). p(b). p(n1). q(a). q(b). q(n2). '
                         'r(b). r(c). s(a). s(c).',
                         [f'{name}(var__(1))' for name in ('p', 'q', 'r', 's')],
                         examples='positive(answer(a)). positive(answer(b)). positive(answer(c)). '
                                  'negative(answer(n1)). negative(answer(n2)).', max_body=3)
        expected = exhaustive_catalogue(task)
        full, bound = (1 << len(task.positives)) - 1, 6
        optimum = optimal_programs(((mask, cost) for mask, cost, _ in expected), full, bound)
        self.assertIsNotNone(optimum[0])
        for order in ('source', 'selectivity'):
            for variants in ('exhaustive', 'targeted'):
                oracle = CoverageOracle(task, negative_label_limit=1)
                rows, stats = compile_catalogue(task, oracle, order=order, variants=variants,
                                                 alpha_collect=False)
                self.assertTrue(stats['complete'])
                self.assertGreater(oracle.stats.get('negative_compact_tables', 0), 0)
                self.assertEqual(normalized(rows), expected)
                self.assertEqual(optimal_programs(((r['coverage'], r['cost']) for r in rows), full, bound),
                                 optimum)

    def test_disabled_or_ordinary_set_fallbacks_remain_exact(self):
        task = self.task('p(b).', ['p(var__(1))'],
                         examples='negative(answer(a)). negative(answer(b)).')
        for task_value, options in ((task, {'negative_label_limit': 0}),
                                    (task, {'negative_label_limit': 1, 'negative_join_backend': 'bindings'}),
                                    (replace(task, negatives=frozenset(task.negatives)),
                                     {'negative_label_limit': 1})):
            oracle = CoverageOracle(task_value, **options)
            self.assertIsNone(negative_relation(oracle, task.heads[0]))
            self.assert_witness(oracle, task.heads[0], (0,), oracle.negative(task.heads[0], (0,)),
                                {clingo.parse_term('answer(b)')})

    def test_compact_bucket_budget_is_global_and_fallback_preserves_negatives(self):
        task = self.task('p(a,x). p(b,y).', ['p(var__(1),var__(2))'],
                         heads=('first(var__(1),var__(2))', 'second(var__(1),var__(2))'),
                         examples='negative(first(a,x)). negative(first(b,y)). '
                                  'negative(second(a,x)). negative(second(b,y)).')
        oracle = CoverageOracle(task, negative_label_limit=1)
        first, second = task.heads
        with patch('nidavellir.compact_negative_join.COMPACT_BUCKET_LIMIT', 3):
            self.assertIsNotNone(negative_relation(oracle, first))
            self.assertEqual(oracle.stats['negative_compact_buckets'], 2)
            self.assertIsNone(negative_relation(oracle, second))
            self.assertEqual(oracle.stats['negative_compact_buckets'], 2)
            self.assertIsNotNone(oracle.negative(second, (0,)))

    def test_lazy_selection_checks_deadline_after_last_row(self):
        task = self.task('p(a,x).', ['p(var__(1),var__(2))'],
                         heads=('answer(var__(1),var__(2))',),
                         examples='negative(answer(a,x)). negative(answer(b,y)).')
        oracle = CoverageOracle(task, negative_label_limit=1)
        _, relation = negative_relation(oracle, task.heads[0])
        selected = relation.select(((0, oracle.space.intern(clingo.parse_term('a'))),), oracle.space.stats)
        self.assertEqual(len(selected), 1)
        rows = iter(selected)
        next(rows)
        def expired():
            raise ResourceLimit('lazy rows deadline')
        oracle.check = expired
        with self.assertRaisesRegex(ResourceLimit, 'lazy rows deadline'):
            next(rows)
        with self.assertRaisesRegex(ResourceLimit, 'lazy rows deadline'):
            relation.select((), oracle.space.stats)

    def test_every_interruption_checkpoint_publishes_neither_safety_nor_witness(self):
        for background, found in (('p(a). q(x). r(y).', False), ('p(a). q(x). r(x).', True)):
            task = self.task(background, ['p(var__(1))', 'q(var__(2))', 'r(var__(2))'],
                             examples='negative(answer(a)). negative(answer(b)). negative(answer(c)).')
            head, body = task.heads[0], tuple(range(3))
            calls = 0
            def count():
                nonlocal calls
                calls += 1
            baseline = CoverageOracle(task, negative_label_limit=1)
            baseline.check = count
            self.assertEqual(baseline.negative(head, body) is not None, found)
            total = calls
            self.assertGreater(total, 3)
            for stop in range(1, total + 1):
                oracle = CoverageOracle(task, negative_label_limit=1)
                calls = 0
                def interrupt():
                    nonlocal calls
                    calls += 1
                    if calls >= stop:
                        raise ResourceLimit('compact query checkpoint')
                oracle.check = interrupt
                with self.subTest(found=found, stop=stop), forbid_expansion():
                    with self.assertRaisesRegex(ResourceLimit, 'compact query checkpoint'):
                        oracle.negative(head, body)
                    self.assertFalse(oracle._safe)
                    self.assertFalse(oracle._negative_witnesses)
                    oracle.check = lambda: None
                    self.assertEqual(oracle.negative(head, body) is not None, found)


if __name__ == '__main__':
    unittest.main()
