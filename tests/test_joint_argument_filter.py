"""All known argument equalities constrain join cardinalities and row caches."""
from types import SimpleNamespace
import unittest

import clingo

from nidavellir.oracle import ResourceLimit, match
from nidavellir.witness_space import WitnessSpace


term = clingo.parse_term


class JointArgumentFilterTests(unittest.TestCase):
    def space(self, atoms, literals):
        task = SimpleNamespace(atoms=frozenset(map(term, atoms)),
                               literals=tuple(map(term, literals)))
        return WitnessSpace(task, lambda: None)

    def test_two_equalities_reduce_dense_relation_before_join(self):
        space = self.space([f'r({x},{y})' for x in range(40) for y in range(40)],
                           ['r(var__(1),var__(2))'])
        literal = space.task.literals[0]
        fixed = {1: term('3'), 2: term('7')}
        self.assertEqual(list(space._rows(literal, fixed)), [term('r(3,7)')])
        scanned = space.stats['filtered_rows_scanned']
        self.assertEqual(scanned, 40)
        self.assertEqual(list(space._rows(literal, fixed)), [term('r(3,7)')])
        self.assertEqual(space.stats['filtered_rows_scanned'], scanned)

    def test_constants_nested_fixed_terms_and_renamed_slots(self):
        space = self.space(['r(tag,a,pair(x,y))', 'r(tag,b,pair(x,y))',
                            'r(other,a,pair(x,y))', 'r(tag,a,pair(x,z))'],
                           ['r(tag,var__(1),pair(var__(2),var__(3)))',
                            'r(tag,var__(7),pair(var__(8),var__(9)))'])
        first, second = space.task.literals
        fixed = {1: term('a'), 2: term('x'), 3: term('y')}
        self.assertEqual(set(space._rows(first, fixed)), {term('r(tag,a,pair(x,y))')})
        renamed = {7: term('a'), 8: term('x'), 9: term('y')}
        self.assertEqual(set(space._rows(second, renamed)), {term('r(tag,a,pair(x,y))')})
        self.assertEqual(space.stats['filtered_rows_cache_hits'], 1)
        self.assertEqual(set(space._rows(first, {1: term('b'), 2: term('x'), 3: term('y')})),
                         {term('r(tag,b,pair(x,y))')})

    def test_unbound_repeated_variables_still_checked_by_match(self):
        space = self.space(['r(tag,a,x,x)', 'r(tag,a,x,y)', 'r(other,a,x,x)'],
                           ['r(tag,var__(1),var__(2),var__(2))'])
        literal = space.task.literals[0]
        rows = space._rows(literal, {1: term('a')})
        self.assertEqual(len(rows), 2)
        self.assertEqual([atom for atom in rows if match(literal, atom, {1: term('a')}) is not None],
                         [term('r(tag,a,x,x)')])

    def test_interrupted_filter_does_not_cache_partial_rows(self):
        space = self.space([f'r(tag,a,{n})' for n in range(300)],
                           ['r(var__(1),var__(2),var__(3))'])
        literal = space.task.literals[0]
        space._rows(literal, {1: term('tag')})
        space._rows(literal, {2: term('a')})
        def stop():
            raise ResourceLimit('test deadline')
        space.check = stop
        with self.assertRaises(ResourceLimit):
            space._rows(literal, {1: term('tag'), 2: term('a')})
        self.assertFalse(space._filtered_rows)
        space.check = lambda: None
        self.assertEqual(len(space._rows(literal, {1: term('tag'), 2: term('a')})), 300)

    def test_cached_results_have_bounded_total_row_references(self):
        space = self.space([f'r(tag,{x},{y})' for x in range(80) for y in range(600)],
                           ['r(tag,var__(1),var__(2))'])
        literal = space.task.literals[0]
        for x in range(80):
            self.assertEqual(len(space._rows(literal, {1: clingo.Number(x)})), 600)
        self.assertLessEqual(space._filtered_row_count, 32768)
        self.assertEqual(space._filtered_row_count, sum(map(len, space._filtered_rows.values())))
        self.assertLess(len(space._filtered_rows), 80)


if __name__ == '__main__':
    unittest.main()
