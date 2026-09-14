"""CEGIS cuts and saved upper bounds versus independent Clingo execution."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import clingo

from nidavellir.oracle import ResourceLimit
from nidavellir.relational_bound import ComponentSearch, discover_relational_bound
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import Task


def model(program):
    control = clingo.Control(["0"], logger=lambda *_: None)
    control.add("base", [], program)
    control.ground([("base", [])])
    models = []
    result = control.solve(on_model=lambda answer: models.append(set(answer.symbols(atoms=True))))
    if not result.satisfiable or not result.exhausted or len(models) != 1:
        raise AssertionError("Expected one independent interpretation")
    return models[0]


class RelationalBoundTests(unittest.TestCase):
    def task(self, background, literals, *, examples="positive(answer(a)). positive(answer(b)). negative(answer(c)).",
             constraints="", max_body=3, heads=("answer(var__(1))",)):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        bias = "\n".join([f"candidate_head({head})." for head in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n" + constraints
        for name, source in (("bk.lp", background), ("bias.lp", bias), ("examples.lp", examples)):
            (path / name).write_text(source)
        return Task.load(path / "bk.lp", path / "examples.lp", path / "bias.lp")

    def verifier(self, task):
        def verify(rules):
            predicted = model(task.background + "\n" + "\n".join(rules))
            return {"matched": task.positives <= predicted and not task.negatives & predicted}
        return verify

    def test_overlapping_components_and_pool_cost_are_verified(self):
        task = self.task("p(a). p(b). q(b). q(d). p(c). r(a). r(b).",
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         examples="positive(answer(a)). positive(answer(b)). positive(answer(d)). negative(answer(c)).",
                         constraints=":- body(r(var__(1))), not body(p(var__(1))).\n")
        snapshots = []
        cover, report = discover_relational_bound(task, 30, self.verifier(task), checkpoint=lambda s: snapshots.append(json.loads(json.dumps(s))))
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["cost"], 5)
        self.assertEqual(len(cover), 2)
        self.assertTrue(report["verification"]["matched"])
        self.assertTrue(report["state"]["incumbent"])
        self.assertTrue(snapshots)
        self.assertNotIn("optimum_cost", report)
        self.assertEqual(report["pool_selection"]["scope"], "discovered_components_only")

    def test_unsatisfiable_pilot_reports_no_bound_not_global_optimum(self):
        task = self.task("p(a). p(c).", ["p(var__(1))"])
        cover, report = discover_relational_bound(task, 30, self.verifier(task))
        self.assertIsNone(cover)
        self.assertEqual(report["status"], "not_found")
        self.assertIsNone(report["cost"])

    def test_each_positive_core_cut_preserves_all_pilot_covering_rules(self):
        task = self.task("p(a,x). p(c,x). q(y). r(x).",
                         ["p(var__(1),var__(2))", "q(var__(2))", "r(var__(2))"],
                         examples="positive(answer(a)). negative(answer(c)).")
        head = task.heads[0]
        pilot = clingo.parse_term("answer(a)")
        oracle = CoverageOracle(task)
        body = tuple(i for i, literal in enumerate(task.literals) if literal.name in {"p", "q"})
        core = oracle.core(head, body, pilot)
        self.assertIsNotNone(core)
        search = ComponentSearch(task, head, pilot, positive_cores=[core])
        for size in range(task.max_body + 1):
            for selected in combinations(range(len(task.literals)), size):
                if not task.accepts(head, selected):
                    continue
                predictions = model(task.background + task.rule(head, selected))
                if pilot in predictions:
                    omitted = set(range(len(task.literals))) - set(selected)
                    self.assertIsNotNone(search.solver.find(selected, omitted))

    def test_real_negative_profile_cut_preserves_every_safe_rule(self):
        task = self.task("p(a,x). p(c,x). q(a,x). r(y).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))", "r(var__(3))"],
                         examples="positive(answer(a)). negative(answer(c)).")
        head = task.heads[0]
        body = (next(i for i, literal in enumerate(task.literals) if literal.name == "p"),)
        negative = CoverageOracle(task).negative(head, body)
        self.assertIsNotNone(negative)
        search = ComponentSearch(task, head, clingo.parse_term("answer(a)"))
        search.negative_witness(negative["truth"])
        for size in range(task.max_body + 1):
            for selected in combinations(range(len(task.literals)), size):
                if not task.accepts(head, selected):
                    continue
                predictions = model(task.background + task.rule(head, selected))
                if not predictions & task.negatives:
                    omitted = set(range(len(task.literals))) - set(selected)
                    self.assertIsNotNone(search.solver.find(selected, omitted))

    def test_partial_component_checkpoint_resumes_and_rejects_changed_input(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"])
        stopped = False
        def checkpoint(state):
            nonlocal stopped
            if state["components"] and not stopped:
                stopped = True
                raise ResourceLimit("simulated interruption after first safe component")
        cover, partial = discover_relational_bound(task, 30, self.verifier(task), checkpoint=checkpoint)
        self.assertIsNone(cover)
        self.assertEqual(partial["status"], "timeout")
        self.assertEqual(len(partial["state"]["components"]), 1)
        cover, complete = discover_relational_bound(task, 30, self.verifier(task), state=partial["state"])
        self.assertEqual(complete["status"], "verified")
        self.assertEqual(complete["cost"], 4)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            discover_relational_bound(task, 30, self.verifier(task), state=partial["state"], fingerprint="changed")

    def test_failed_independent_verification_never_certifies_bound(self):
        task = self.task("p(a). p(b).", ["p(var__(1))"])
        with self.assertRaisesRegex(AssertionError, "independent"):
            discover_relational_bound(task, 30, lambda _: {"matched": False})

    def test_disabled_and_zero_positive_tasks(self):
        task = self.task("p(a).", ["p(var__(1))"], examples="negative(answer(c)).")
        cover, report = discover_relational_bound(task, 0, self.verifier(task))
        self.assertIsNone(cover)
        self.assertEqual(report["status"], "disabled")
        cover, report = discover_relational_bound(task, 30, self.verifier(task))
        self.assertEqual(cover, [])
        self.assertEqual(report["cost"], 0)

    def test_unary_bootstrap_stores_only_exact_pilot_impossibility_cores(self):
        task = self.task("p(a). q(c). r(a).",
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         examples="positive(answer(a)). negative(answer(c)).")
        # Exercise CEGIS bootstrap itself, without the optional short-rule pass.
        cover, report = discover_relational_bound(task, 30, self.verifier(task), short_phase_seconds=0)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["unary_literals_checked"], 3)
        self.assertEqual(report["unary_cores_added"], 1)
        head = task.heads[0]
        pilot = clingo.parse_term("answer(a)")
        unary = [entry for entry in report["state"]["positive_cores"] if len(entry["body"]) == 1]
        self.assertEqual(len(unary), 1)
        for size in range(1, task.max_body + 1):
            for body in combinations(range(len(task.literals)), size):
                if task.accepts(head, body) and pilot in model(task.background + task.rule(head, body)):
                    self.assertTrue(all(not set(entry["body"]) <= set(body) for entry in unary))
        self.assertIn("bound.unary_bootstrap", report["profile"]["timings"])
        resumed, second = discover_relational_bound(task, 30, self.verifier(task), state=report["state"])
        self.assertEqual(second["status"], "verified")
        self.assertEqual(second["unary_literals_checked"], 0)
        self.assertEqual(second["state"]["positive_cores"], report["state"]["positive_cores"])


if __name__ == "__main__":
    unittest.main()
