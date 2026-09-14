"""Staged OR-pilot CEGIS semantics versus independent complete Clingo rules."""
from itertools import combinations
import json
import unittest
from unittest.mock import patch

import clingo

from nidavellir import relational_bound as bound
from nidavellir.oracle import ResourceLimit
from nidavellir.relational_coverage import CoverageOracle
from tests import test_relational_bound as fixtures

model = fixtures.model


class MultiPilotBoundTests(unittest.TestCase):
    task = fixtures.RelationalBoundTests.task
    verifier = fixtures.RelationalBoundTests.verifier

    def singleton_cores(self, task, head, pilots):
        oracle = CoverageOracle(task)
        return [(pilot, (i,)) for pilot in pilots for i in range(len(task.literals))
                if not oracle.exists(head, (i,), pilot)]

    def test_claims_are_or_not_and_and_cores_are_isolated(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"], max_body=1)
        head = task.heads[0]
        pilots = tuple(sorted(task.positives))
        search = bound.ComponentSearch(task, head, pilots,
                                       pilot_cores=self.singleton_cores(task, head, pilots))
        for i in range(len(task.literals)):
            self.assertIsNotNone(search.solver.find((i,), (1-i,)))
            expected = tuple(p for p in pilots if p in model(task.background + task.rule(head, (i,))))
            self.assertEqual(search.compatible_pilots((i,)), expected)
        with self.assertRaisesRegex(ValueError, "identify its pilot"):
            search.positive_core((0,))

    def test_empty_core_disables_only_its_own_pilot(self):
        task = self.task("q(b).", ["q(var__(1))"], max_body=1)
        pilots = tuple(sorted(task.positives))
        search = bound.ComponentSearch(task, task.heads[0], pilots, pilot_cores=[(pilots[0], ())])
        self.assertIsNotNone(search.next(lambda: None))
        self.assertEqual(search.compatible_pilots((0,)), (pilots[1],))
        search.positive_core((), pilots[1])
        self.assertIsNone(search.next(lambda: None))

    def test_all_refinements_preserve_every_safe_rule_covering_any_pilot(self):
        task = self.task("p(a,x). p(c,x). q(a,y). q(b,x). r(x). s(b).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))",
                          "r(var__(2))", "s(var__(1))"], max_body=3)
        head, pilots = task.heads[0], tuple(sorted(task.positives))
        oracle = CoverageOracle(task)
        search = bound.ComponentSearch(task, head, pilots)
        legal = []
        for size in range(task.max_body + 1):
            for body in combinations(range(len(task.literals)), size):
                if not task.accepts(head, body):
                    continue
                predictions = model(task.background + task.rule(head, body))
                legal.append((body, predictions))
                for pilot in pilots:
                    if pilot not in predictions:
                        core = oracle.core(head, body, pilot)
                        self.assertIsNotNone(core)
                        search.positive_core(core, pilot)
                witness = oracle.negative(head, body)
                if witness:
                    search.negative_witness(witness["truth"])
        for body, predictions in legal:
            excluded = set(range(len(task.literals))) - set(body)
            expected = bool(predictions & task.positives) and not bool(predictions & task.negatives)
            self.assertEqual(search.solver.find(body, excluded) is not None, expected, task.rule(head, body))

    def test_impossible_single_pilot_does_not_hide_an_easy_safe_component(self):
        task = self.task("p(a). p(c). q(b).", ["p(var__(1))", "q(var__(1))"], max_body=1)
        single, one = bound.discover_relational_bound(task, 5, self.verifier(task), initial_positives=1, short_phase_seconds=0)
        multiple, many = bound.discover_relational_bound(task, 5, self.verifier(task), initial_positives=24, short_phase_seconds=0)
        self.assertIsNone(single)
        self.assertIsNone(multiple)  # a is impossible to cover safely in either grammar.
        self.assertEqual(one["safe_components"], 0)
        self.assertEqual(many["safe_components"], 1)
        self.assertEqual(many["status"], "not_found")
        rule = many["state"]["components"][0]["rule"]
        self.assertIn(clingo.parse_term("answer(b)"), model(task.background + rule))
        self.assertNotIn("optimality_proven", many)

    def test_multihead_groups_cover_original_labels_and_keep_full_rule_grammar(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"],
                         examples="positive(first(a)). positive(second(b)). negative(first(b)). negative(second(a)).",
                         heads=("first(var__(1))", "second(var__(1))"), max_body=1)
        observed = []
        original = bound.ComponentSearch
        def recording(task, head, pilots, **kwargs):
            observed.append((head, tuple(pilots)))
            return original(task, head, pilots, **kwargs)
        with patch.object(bound, "ComponentSearch", recording):
            cover, report = bound.discover_relational_bound(task, 5, self.verifier(task))
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["cost"], 4)
        self.assertEqual(len(cover), 2)
        self.assertTrue(all(all(p.name == h.name for p in pilots) for h, pilots in observed))
        self.assertEqual(report["initial_positives"], 24)
        self.assertEqual(len(task.literals), 2)
        expected = set()
        for head in task.heads:
            for i in range(len(task.literals)):
                rule = task.rule(head, (i,))
                predictions = model(task.background + rule)
                if predictions & task.positives and not predictions & task.negatives:
                    expected.add(rule)
        self.assertEqual(set(report["rules"]), expected)
        self.assertTrue(self.verifier(task)(report["rules"])["matched"])

    def test_partial_checkpoint_can_resume_with_single_pilot_ablation(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"], max_body=1)
        stopped = False
        def checkpoint(state):
            nonlocal stopped
            if state["components"] and not stopped:
                stopped = True
                raise ResourceLimit("stop after safe component")
        _, interrupted = bound.discover_relational_bound(task, 5, self.verifier(task), checkpoint=checkpoint)
        self.assertEqual(interrupted["status"], "timeout")
        saved = json.loads(json.dumps(interrupted["state"]))
        cover, resumed = bound.discover_relational_bound(task, 5, self.verifier(task), state=saved, initial_positives=1)
        self.assertEqual(resumed["status"], "verified")
        self.assertEqual(resumed["cost"], 4)
        self.assertTrue(self.verifier(task)([row["rule"] for row in cover])["matched"])
        for entry in saved["positive_cores"]:
            self.assertIn(entry, resumed["state"]["positive_cores"])

    def test_timeout_during_multi_pilot_bootstrap_does_not_mark_it_complete(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"], max_body=1)
        class StoppedOracle(CoverageOracle):
            count = 0
            def exists(self, *args):
                self.count += 1
                if self.count == 2:
                    raise ResourceLimit("bootstrap deadline")
                return super().exists(*args)
        with patch.object(bound, "CoverageOracle", StoppedOracle):
            cover, report = bound.discover_relational_bound(task, 5, self.verifier(task))
        self.assertIsNone(cover)
        self.assertEqual(report["status"], "timeout")
        self.assertFalse(report["state"].get("unary_preprocessed"))
        cover, resumed = bound.discover_relational_bound(task, 5, self.verifier(task), state=report["state"])
        self.assertEqual(resumed["status"], "verified")
        self.assertEqual(resumed["cost"], 4)

    def test_initial_sample_size_is_validated_even_if_disabled(self):
        task = self.task("p(a).", ["p(var__(1))"])
        for value in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "sample size"):
                bound.discover_relational_bound(task, 0, self.verifier(task), initial_positives=value)
        search = bound.ComponentSearch(task, task.heads[0], clingo.parse_term("answer(a)"))
        with self.assertRaises(ResourceLimit):
            search.next(lambda: (_ for _ in ()).throw(ResourceLimit("cancelled")))


if __name__ == "__main__":
    unittest.main()
