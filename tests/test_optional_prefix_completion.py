"""Optional prefix feasibility must preserve exact final legality and complete outputs."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.bias_analysis import HeadAnalysis
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_compile import compile_catalogue, mine_variants
from nidavellir.relational_coverage import CoverageOracle
from tests import test_optimized_oracle as fixtures
from tests import test_cli_extended as cli_fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class OptionalPrefixCompletionTests(unittest.TestCase):
    task = fixtures.OptimizedOracleContractTests.task
    invoke = cli_fixtures.ExtendedCLITests.invoke

    def sample(self, bias=''):
        return self.task('p(a). p(n). q(a). q(n). r(a). r(b). s(b). s(n).',
            ['p(var__(1))','q(var__(1))','r(var__(1))','s(var__(1))'],
            ['answer(a)','answer(b)'], ['answer(n)'], max_body=3, bias=bias)

    def test_complete_catalogues_variants_and_optima_match_independent_clingo(self):
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
                for variant_mode in ('exhaustive','targeted'):
                    for ground in (False,True):
                        for enabled in (False,True):
                            with self.subTest(bias=bias,order=order,mode=variant_mode,
                                              ground=ground,completion=enabled):
                                rows,stats=compile_catalogue(task,CoverageOracle(task),order=order,
                                    variants=variant_mode,ground_prefix_filter=ground,prefix_completion=enabled)
                                self.assertTrue(stats['complete'])
                                self.assertEqual(normalized(rows),expected)
                                self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),3,20),optimum)
                                rebuilt,reconstruction=mine_variants(task,CoverageOracle(task),rows,
                                    ground_prefix_filter=ground,prefix_completion=enabled,order=order)
                                self.assertTrue(reconstruction['complete'])
                                self.assertEqual(normalized(rebuilt),expected)
                                self.assertEqual([r['id'] for r in rebuilt],[r['id'] for r in rows])

    def test_uncompiled_bias_skips_feasibility_but_checks_actual_legality(self):
        task=self.task('p(a). p(n1). q(a). q(n2). r(a). r(n3).',
            ['p(var__(1))','q(var__(1))','r(var__(1))'],
            ['answer(a)'],['answer(n1)','answer(n2)','answer(n3)'],
            bias='left :- not right.\nright :- not left.\n'
                 ':- left, body(p(var__(1))).\n:- right, body(q(var__(1))).')
        calls=[]
        original=HeadAnalysis.accepts
        def accept(proof,body,**kwargs):
            result=original(proof,body,**kwargs)
            calls.append((tuple(body),result))
            return result
        with patch.object(HeadAnalysis,'possible_completion',
                          side_effect=AssertionError('prefix query must be skipped')):
            with patch.object(HeadAnalysis,'accepts',accept):
                # Even without either early structural filter, exact final
                # legality must reject illegal nonempty bodies.
                rows,stats=compile_catalogue(task,CoverageOracle(task),prefix_completion=False,
                                             ground_prefix_filter=False)
        self.assertGreater(stats['skipped_prefix_completion'],0)
        self.assertTrue(any(body and not ok for body,ok in calls))
        self.assertTrue(any(ok for _,ok in calls))
        self.assertEqual(normalized(rows),exhaustive_catalogue(task))
        for row in rows:
            self.assertTrue(task.accepts(clingo.parse_term(row['head']),row['body']))

    def test_program_cost_decisions_preserve_feasible_and_infeasible_limits(self):
        tasks=[
            self.task('p(a). q(b). q(n1). r(b). r(n2).',
                ['p(var__(1))','q(var__(1))','r(var__(1))'],
                ['answer(a)','answer(b)'],['answer(n1)','answer(n2)']),
            self.task('p(a). p(n1). q(a). q(n2).',
                ['p(var__(1))','q(var__(1))'],['answer(a)','flag'],
                ['answer(n1)','answer(n2)'],heads=('answer(var__(1))','flag')),
            self.task('p(a). q(b).',['p(var__(1))','q(var__(1))'],
                ['first(a)','second(b)'],heads=('first(var__(1))','second(var__(1))'))]
        for task in tasks:
            raw=exhaustive_catalogue(task)
            full=(1<<len(task.positives))-1
            for limit in (1,2,3,4,5):
                expected=optimal_programs(((m,c) for m,c,_ in raw),full,limit)
                reference=None
                for enabled in (True,False):
                    rows,stats=compile_catalogue(task,CoverageOracle(task),bound=limit,
                        program_bound=True,prefix_completion=enabled,ground_prefix_filter=False)
                    self.assertTrue(stats['complete'])
                    self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),full,limit),expected)
                    if reference is None:reference=normalized(rows)
                    else:self.assertEqual(normalized(rows),reference)

    def test_nested_terms_hidden_correlations_and_oriented_variables_survive(self):
        task=self.task('p(a,f(x),x). p(b,f(y),x). q(x,a). q(x,b). '
            'r(a,a). r(b,b). blocked(b). usable(X) :- r(X,X), not blocked(X).',
            ['p(var__(1),f(var__(2)),var__(2))','q(var__(2),var__(1))',
             'q(var__(1),var__(2))','r(var__(1),var__(1))','usable(var__(1))'],
            ['answer(a)'],['answer(b)'],max_body=3)
        expected=exhaustive_catalogue(task)
        for enabled in (True,False):
            rows,stats=compile_catalogue(task,CoverageOracle(task),prefix_completion=enabled,
                                         ground_prefix_filter=False)
            self.assertTrue(stats['complete'])
            self.assertEqual(normalized(rows),expected)

    def test_predicate_and_constant_renaming_preserves_results(self):
        background='edge(a,x). edge(b,y). mark(x). mark(y). blocked(b). free(X) :- edge(X,Y), not blocked(X).'
        literals=['edge(var__(1),var__(2))','mark(var__(2))','free(var__(1))']
        positives=['answer(a)'];negatives=['answer(b)']
        bias='aux :- body(free(var__(1))).\naux :- body(mark(var__(2))).\n:- not aux.'
        rename={'edge':'rel_a','mark':'rel_b','blocked':'rel_c','free':'rel_d',
                'answer':'label_z','aux':'meta_z','a':'u','b':'v','x':'w','y':'z'}
        def rewrite(text,mapping):
            return re.sub(r'\b[a-z][a-z_0-9]*\b',lambda m:mapping.get(m[0],m[0]),text)
        first=self.task(background,literals,positives,negatives,bias=bias,max_body=3)
        second=self.task(rewrite(background,rename),[rewrite(l,rename) for l in literals],
            [rewrite(p,rename) for p in positives],[rewrite(n,rename) for n in negatives],
            heads=(rewrite('answer(var__(1))',rename),),bias=rewrite(bias,rename),max_body=3)
        reverse={v:k for k,v in rename.items()}
        for enabled in (True,False):
            one,_=compile_catalogue(first,CoverageOracle(first),prefix_completion=enabled)
            two,_=compile_catalogue(second,CoverageOracle(second),prefix_completion=enabled)
            restored={(r['coverage'],r['cost'],tuple(sorted(rewrite(v,reverse) for v in r['variants']))) for r in two}
            self.assertEqual(normalized(one),restored)
            self.assertEqual(normalized(one),exhaustive_catalogue(first))

    def test_deadline_during_final_legality_is_not_a_complete_catalogue(self):
        task=self.sample()
        oracle=CoverageOracle(task)
        original_check=oracle.check
        with patch.object(HeadAnalysis,'accepts',side_effect=ResourceLimit('stop final legality')):
            with self.assertRaises(ResourceLimit) as caught:
                compile_catalogue(task,oracle,prefix_completion=False,ground_prefix_filter=False)
        self.assertFalse(caught.exception.stats.get('complete',False))
        self.assertIs(oracle.check,original_check)
        # A later complete call must still work; no partial answer is reused.
        rows,stats=compile_catalogue(task,oracle,prefix_completion=False,ground_prefix_filter=False)
        self.assertTrue(stats['complete'])
        self.assertEqual(normalized(rows),exhaustive_catalogue(task))

    def test_invalid_switch_is_rejected_by_compilation_and_reconstruction(self):
        task=self.sample()
        for value in (1,None,'yes'):
            for operation in ('compile','reconstruct'):
                with self.assertRaises(ValueError):
                    if operation=='compile':
                        compile_catalogue(task,CoverageOracle(task),prefix_completion=value)
                    else:
                        mine_variants(task,CoverageOracle(task),[],prefix_completion=value)

    def test_cli_switch_reaches_compilation_and_targeted_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            results=[]
            for enabled in (True,False):
                args=['--compiler-backend','relational','--bound-strategy','off',
                      '--enumerate-optima','--expand-variants','--no-ground-prefix-filter']
                if not enabled:args.append('--no-prefix-completion')
                process,output,status=self.invoke(folder,str(enabled),*args)
                self.assertEqual(process.returncode,0,process.stderr+process.stdout)
                self.assertEqual(status['optimal_cost'],4)
                catalogue=json.loads((output/'catalogue.json').read_text())
                rebuilt=json.loads((output/'variant_catalogue.json').read_text())
                for artifact in (catalogue,rebuilt):
                    self.assertEqual(artifact['stats']['prefix_completion'],enabled)
                    if not enabled:self.assertGreater(artifact['stats']['skipped_prefix_completion'],0)
                results.append((normalized(catalogue['candidates']),normalized(rebuilt['candidates'])))
            self.assertEqual(results[0],results[1])


if __name__=='__main__':
    unittest.main()
