"""Per-target bounds preserve the joint optimum and shared data contracts."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import clingo
from nidavellir.bias_analysis import analyze
from nidavellir.labels import CompactNegativeSet
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_bound import discover_relational_bound
from nidavellir.relational_compile import compile_catalogue
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.target_bounds import (collect_verified_target_bounds, costs_from_certificates,
                                              positive_scopes, target_view)
from nidavellir.task import signature
from nidavellir.verification import verify_rules
from tests import test_cli_extended as cli_fixtures
from tests import test_optimized_oracle as fixtures
from tests.test_program_bound_pruning import optimal_programs
from tests.test_relational_compile import exhaustive_catalogue, normalized


class PerTargetBoundTests(unittest.TestCase):
    task=fixtures.OptimizedOracleContractTests.task
    invoke=cli_fixtures.ExtendedCLITests.invoke

    def sample(self):
        return self.task('p(a). q(b). q(n1). r(b). r(n2).',
            ['p(var__(1))','q(var__(1))','r(var__(1))'],
            ['first(a)','second(b)'],['first(n1)','second(n1)','second(n2)'],
            heads=('first(var__(1))','second(var__(1))'))

    def compare(self,task,bounds):
        expected=exhaustive_catalogue(task)
        full=(1<<len(task.positives))-1
        optimum=optimal_programs(((m,c) for m,c,_ in expected),full,20)
        variants={(m,c):v for m,c,v in expected}
        reference=None
        for order in ('source','selectivity'):
            for prefix in (True,False):
                for global_limit in (None,20):
                    rows,stats=compile_catalogue(task,CoverageOracle(task),target_bounds=bounds,
                        prefix_completion=prefix,order=order,bound=global_limit,
                        program_bound=global_limit is not None)
                    self.assertTrue(stats['complete'])
                    self.assertEqual(optimal_programs(((r['coverage'],r['cost']) for r in rows),full,20),optimum)
                    for mask,cost,spelling in normalized(rows):
                        self.assertEqual(spelling,variants[(mask,cost)])
                    if reference is None:reference=normalized(rows)
                    else:self.assertEqual(normalized(rows),reference)
                    self.assertTrue(all(t['search_complete'] for t in stats['target_search'].values()))
        return rows,stats

    def test_different_target_bounds_keep_the_joint_optimum(self):
        rows,stats=self.compare(self.sample(),{('first',1):2,('second',1):3})
        self.assertEqual(sorted(r['cost'] for r in rows),[2,3])
        self.assertEqual(stats['target_search']['first/1']['effective_body_limit'],1)
        self.assertEqual(stats['target_search']['second/1']['effective_body_limit'],2)

    def test_universal_coverage_and_probes_use_only_the_target_global_positions(self):
        task=self.task('p(a). q(b).',['p(var__(1))','q(var__(1))'],
            ['first(a)','second(b)'],heads=('first(var__(1))','second(var__(1))'))
        rows,stats=self.compare(task,{('first',1):2,('second',1):2})
        self.assertEqual({r['coverage'] for r in rows},{1,2})
        self.assertEqual(stats['target_bound_positive_pilot_positions'],{'first/1':[0],'second/1':[1]})

    def test_overlapping_templates_stay_in_one_signature_and_empty_rules_count_one(self):
        task=self.task('p(a). p(b). q(c).',['p(var__(1))','q(var__(1))'],
            ['first(a)','first(b)','second(c)'],
            heads=('first(var__(1))','first(a)','first(b)','second(var__(1))'))
        rows,stats=self.compare(task,{('first',1):2,('second',1):2})
        self.assertEqual(stats['target_minimum_useful_rule_cost'],{'first/1':1,'second/1':2})
        self.assertTrue(any(r['cost']==1 for r in rows))

    def test_missing_bound_does_not_restrict_another_target(self):
        rows,stats=self.compare(self.sample(),{('first',1):2})
        self.assertEqual(sorted(r['cost'] for r in rows),[2,3])
        self.assertIsNone(stats['target_search']['second/1']['verified_target_bound'])
        self.assertEqual(stats['target_search']['second/1']['effective_body_limit'],3)

    def test_views_share_the_BK_and_compact_tables_without_outer_symbol_iteration(self):
        task=self.sample()
        with patch.object(CompactNegativeSet,'__iter__',side_effect=AssertionError('do not expand compact labels')):
            view=target_view(task,('second',1))
        self.assertIs(view.atoms,task.atoms)
        self.assertIs(view.background,task.background)
        self.assertIs(view.literals,task.literals)
        self.assertIs(view.negatives._tables[('second',1)],task.negatives._tables[('second',1)])
        self.assertEqual(view.positives,frozenset([clingo.parse_term('second(b)')]))
        self.assertEqual(len(view.negatives),2)
        self.assertEqual({signature(h) for h in view.heads},{('second',1)})
        self.assertIsNone(view.examples_path)

    def test_partial_pool_certifies_one_target_without_a_global_solution(self):
        task=self.task('p(a).',['p(var__(1))'],['first(a)','second(b)'],
            heads=('first(var__(1))','second(var__(1))'))
        snapshots=[]
        cover,report=discover_relational_bound(task,5,lambda rules:verify_rules(task,rules),
            per_target_bounds=True,checkpoint=lambda s:snapshots.append(copy.deepcopy(s)))
        self.assertIsNone(cover)
        self.assertEqual(costs_from_certificates(task,report['target_bounds']),{('first',1):2})
        self.assertTrue(any('first/1' in s.get('target_bounds',{}) for s in snapshots))
        certificate=report['target_bounds']['first/1']
        self.assertTrue(verify_rules(target_view(task,('first',1)),certificate['rules'])['matched'])
        rows,stats=compile_catalogue(task,CoverageOracle(task),
            target_bounds=costs_from_certificates(task,report['target_bounds']))
        self.assertTrue(stats['complete'])
        self.assertIsNone(optimal_programs(((r['coverage'],r['cost']) for r in rows),3,20)[0])

    def test_bad_components_or_interrupted_verification_cannot_publish_a_bound(self):
        task=self.sample()
        head=task.heads[0]
        body=(task.literals.index(clingo.parse_term('p(var__(1))')),)
        row={'head':str(head),'body':list(body),'rule':task.rule(head,body),'cost':2,'coverage':1}
        analysis=analyze(task)
        for change in ({'cost':1},{'coverage':3},{'rule':'first(a).'}):
            with self.assertRaises(ValueError):
                collect_verified_target_bounds(task,{1:dict(row,**change)},{},analysis,lambda:None)
        previous={}
        wrong_body=(task.literals.index(clingo.parse_term('q(var__(1))')),)
        lying=dict(row,body=list(wrong_body),rule=task.rule(head,wrong_body))
        with self.assertRaises(ValueError):
            collect_verified_target_bounds(task,{1:lying},previous,analysis,lambda:None)
        self.assertEqual(previous,{})
        interrupted=[False]
        def check():
            if interrupted[0]:raise ResourceLimit('after verification')
        def verify(*args):
            result=verify_rules(*args)
            interrupted[0]=True
            return result
        with patch('nidavellir.target_bounds.verify_rules',verify):
            with self.assertRaises(ResourceLimit):
                collect_verified_target_bounds(task,{1:row},previous,analysis,check)
        self.assertEqual(previous,{})

    def test_resume_rechecks_target_incumbents_instead_of_trusting_saved_costs(self):
        task=self.sample()
        _,first=discover_relational_bound(task,5,lambda rules:verify_rules(task,rules),
                                         per_target_bounds=True,fingerprint='same')
        state=copy.deepcopy(first['state'])
        self.assertTrue(state['target_bounds'])
        for certificate in state['target_bounds'].values():certificate['cost']=0
        cover,second=discover_relational_bound(task,5,lambda rules:verify_rules(task,rules),
            state=state,per_target_bounds=True,fingerprint='same')
        self.assertIsNotNone(cover)
        self.assertEqual(costs_from_certificates(task,second['target_bounds']),
                         {('first',1):2,('second',1):3})

    def test_unknown_signature_bad_cost_and_corrupt_certificate_are_rejected(self):
        task=self.sample()
        for value in ([],{('missing',1):2},{('first',1):True},{('first',1):0},{('first',1):-1}):
            with self.assertRaises(ValueError):
                compile_catalogue(task,CoverageOracle(task),target_bounds=value)
        _,report=discover_relational_bound(task,5,lambda rules:verify_rules(task,rules),per_target_bounds=True)
        malformed=copy.deepcopy(report['target_bounds'])
        malformed['first/1']['positive_mask']='0x3'
        with self.assertRaises(ValueError):costs_from_certificates(task,malformed)

    def test_cli_shared_call_keeps_bounds_and_completely_enumerates_small_joint_task(self):
        task=self.sample()
        with tempfile.TemporaryDirectory() as folder:
            inputs=Path(folder)/'inputs'
            inputs.mkdir()
            (inputs/'background.lp').write_text(task.background)
            (inputs/'bias.lp').write_text(task.bias_source)
            task.write_examples(inputs/'examples.lp')
            process,output,status=self.invoke(folder,'run','--per-target-bounds',
                '--target-mode','grouped','--compiler-backend','relational','--enumerate-optima','--expand-variants',inputs=inputs)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            self.assertEqual(status['optimal_cost'],5)
            self.assertEqual(status['compiler_backend'],'relational')
            bound=json.loads((output/'bound.json').read_text())
            self.assertEqual(costs_from_certificates(task,bound['target_bounds']),
                             {('first',1):2,('second',1):3})
            compilation=json.loads((output/'catalogue.json').read_text())['stats']
            self.assertEqual(compilation['program_bounds'],{'first/1':2,'second/1':3})
            self.assertTrue(json.loads((output/'variants.json').read_text())['all_optimal_programs_complete'])

    def test_hard_worker_timeout_keeps_only_current_fingerprint_target_certificates(self):
        from nidavellir.__main__ import bound_in_subprocess
        task=self.task('p(a).',['p(var__(1))'],['first(a)','second(b)'],
            heads=('first(var__(1))','second(var__(1))'))
        _,report=discover_relational_bound(task,5,lambda rules:verify_rules(task,rules),
                                         per_target_bounds=True,fingerprint='current')
        for fingerprint in ('current','stale'):
            with tempfile.TemporaryDirectory() as folder:
                checkpoint=Path(folder)/'checkpoint.json'
                state=dict(report['state'],fingerprint=fingerprint,_verification={})
                class Worker:
                    pid=123456789
                    calls=0
                    def communicate(self,timeout=None):
                        self.calls+=1
                        if self.calls==1:
                            checkpoint.write_text(json.dumps(state))
                            raise subprocess.TimeoutExpired('fake-worker',timeout)
                        return b'',b''
                with patch('nidavellir.__main__.subprocess.Popen',return_value=Worker()), \
                     patch('nidavellir.__main__.os.killpg'):
                    cover,result=bound_in_subprocess(task,[],5,strategy='relational',
                        fingerprint='current',checkpoint_path=checkpoint,per_target_bounds=True)
                self.assertIsNone(cover)
                self.assertEqual(result['status'],'timeout')
                expected={('first',1):2} if fingerprint=='current' else {}
                self.assertEqual(costs_from_certificates(task,result.get('target_bounds',{})),expected)


if __name__=='__main__':
    unittest.main()
