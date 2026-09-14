"""Exact expensive-rule pruning keeps every bounded optimal program."""
import unittest
from unittest.mock import patch

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import _Search, compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from tests import test_optimized_oracle as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class ExactProgramBoundTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task

    def correlated(self):
        return self.task('base(a,x). base(b,x). base(c,x). base(n,x). '
            'gate(a,x). gate(b,y). gate(c,x). gate(n,z). p(a). p(b). q(b). q(c).',
            ['base(var__(1),var__(2))', 'gate(var__(1),var__(2))', 'p(var__(1))', 'q(var__(1))'],
            ['answer(a)', 'answer(b)', 'answer(c)'], ['answer(n)'])

    def test_correlated_partial_rule_is_rejected_despite_full_atomic_majorant(self):
        task = self.correlated()
        full, bound = 7, 4
        expected = exhaustive_catalogue(task, bound)
        optimum = optimal_programs(((mask,cost) for mask,cost,_ in expected), full, bound)
        for order in ('source','selectivity'):
            for mode in ('exhaustive','targeted'):
                raw, _ = compile_catalogue(task, CoverageOracle(task), bound=bound,
                    program_bound=True, exact_program_bound=False, order=order, variants=mode)
                rows, stats = compile_catalogue(task, CoverageOracle(task), bound=bound,
                    program_bound=True, exact_program_bound=True, order=order, variants=mode)
                self.assertEqual(normalized(raw), expected)
                eligible = [r for r in raw if r['coverage'] == full or r['cost']+2 <= bound]
                self.assertEqual(normalized(rows), normalized(eligible))
                self.assertLess(len(rows),len(raw))
                self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),full,bound),optimum)
                self.assertGreater(stats['pruned_exact_program_bound'],0)

    def test_full_rule_at_inclusive_bound_remains_available(self):
        task = self.task('p(a,x). p(b,x). p(n,x). q(a,x). q(b,x). q(n,y).',
            ['p(var__(1),var__(2))','q(var__(1),var__(2))'], ['answer(a)','answer(b)'], ['answer(n)'])
        raw,_ = compile_catalogue(task,CoverageOracle(task),bound=3,program_bound=True,exact_program_bound=False)
        rows,stats = compile_catalogue(task,CoverageOracle(task),bound=3,program_bound=True)
        self.assertEqual(normalized(rows),normalized(raw))
        self.assertEqual([(r['cost'],r['coverage']) for r in rows],[(3,3)])
        self.assertGreater(stats['full_positive_queries'],0)
        self.assertEqual(stats['pruned_exact_program_bound'],0)

    def test_inactive_without_a_program_bound_and_deadline_never_means_missing_label(self):
        task = self.correlated()
        with patch.object(_Search,'full_positive',side_effect=AssertionError('unverified bound')):
            _,stats = compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=False)
        self.assertFalse(stats['exact_program_bound'])
        with patch.object(_Search,'full_positive',side_effect=ResourceLimit('query interrupted')):
            with self.assertRaises(ResourceLimit) as stopped:
                compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True,bound_positive_probes=0)
        self.assertFalse(stopped.exception.stats.get('complete',False))

    def test_ground_empty_rule_lower_bound_keeps_the_equality_case(self):
        task = self.task('p(a). p(n1). q(a). q(n2).', ['p(var__(1))','q(var__(1))'],
            ['answer(a)','flag'], ['answer(n1)','answer(n2)'], heads=('answer(var__(1))','flag'))
        rows,stats = compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True)
        self.assertEqual(stats['minimum_useful_rule_cost'],1)
        self.assertEqual(sorted(r['cost'] for r in rows),[1,3])
        self.assertEqual(stats['pruned_exact_program_bound'],0)

    def test_unsafe_rules_do_not_trigger_all_positive_queries(self):
        task = self.task('p(a). p(b). p(n). q(a). q(b). q(unlabelled).',
                         ['p(var__(1))','q(var__(1))'], ['answer(a)','answer(b)'], ['answer(n)'])
        queried = []
        original = _Search.full_positive
        def observe(search, head, body):
            queried.append(tuple(task.literals[i].name for i in body))
            return original(search, head, body)
        with patch.object(_Search,'full_positive',observe):
            rows,stats=compile_catalogue(task,CoverageOracle(task),bound=2,program_bound=True)
        self.assertEqual(queried,[('q',)])
        self.assertEqual([(r['cost'],r['coverage']) for r in rows],[(2,3)])

    def test_unsafe_prefix_remains_open_for_its_safe_extension(self):
        task = self.task('p(a). p(n1). q(a). q(n2).', ['p(var__(1))','q(var__(1))'],
                         ['answer(a)'], ['answer(n1)','answer(n2)'])
        queried=[]
        original=_Search.full_positive
        def observe(search, head, body):
            queried.append(len(body))
            return original(search,head,body)
        with patch.object(_Search,'full_positive',observe):
            rows,_=compile_catalogue(task,CoverageOracle(task),bound=3,program_bound=True)
        self.assertEqual(queried,[2])
        self.assertEqual([r['cost'] for r in rows],[3])

    def test_probe_success_never_proves_full_coverage(self):
        task=self.correlated()
        reference,_=compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True,bound_positive_probes=0)
        for count in (1,2,24):
            rows,stats=compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True,bound_positive_probes=count)
            self.assertEqual(normalized(rows),normalized(reference))
            self.assertEqual(len(stats['bound_positive_pilot_positions']),min(count,3))
            if count==2:
                # The misleading rule covers both endpoints a/c, but misses b.
                self.assertEqual(stats['bound_positive_pilot_positions'],[0,2])
                self.assertGreater(stats['full_positive_queries'],0)

    def test_real_missing_pilot_avoids_the_negative_query_and_timeout_is_not_a_miss(self):
        task=self.correlated()
        body=tuple(i for i,literal in enumerate(task.literals) if literal.name in ('base','gate'))
        oracle=CoverageOracle(task)
        seen=[]
        original=oracle.negative
        def observe(head, selected):
            seen.append(tuple(selected))
            return original(head,selected)
        with patch.object(oracle,'negative',observe):
            _,stats=compile_catalogue(task,oracle,bound=4,program_bound=True,bound_positive_probes=24)
        self.assertNotIn(body,seen)
        self.assertGreater(stats['pruned_bound_positive_probe'],0)
        with patch.object(_Search,'probe_positive',side_effect=ResourceLimit('pilot interrupted')):
            with self.assertRaises(ResourceLimit) as stopped:
                compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True)
        self.assertFalse(stopped.exception.stats.get('complete',False))

    def test_invalid_probe_limits_are_errors(self):
        task=self.correlated()
        for value in (-1,1.5,True):
            with self.assertRaises(ValueError):
                compile_catalogue(task,CoverageOracle(task),bound=4,program_bound=True,bound_positive_probes=value)
