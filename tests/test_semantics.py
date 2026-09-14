"""Independent Clingo and exhaustive checks of generic witness compilation."""

from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import clingo

from nidavellir.oracle import build_indices
from nidavellir.search import compile_catalogue, discover_bound
from nidavellir.solvers import solve
from nidavellir.task import Task, variables


def interpretation(program):
    control = clingo.Control(["0"], logger=lambda *_: None)
    control.add("base", [], program)
    control.ground([("base", [])])
    models = []
    answer = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if not answer.satisfiable or len(models) != 1:
        raise AssertionError(f"Expected one model: {answer}, {len(models)}")
    return models[0]


def exact_cover_cost(candidates, positive_count):
    """Independent dynamic program over subsets of example indices."""
    full = (1 << positive_count) - 1
    costs = {0: 0}
    for candidate in candidates:
        for covered, cost in list(costs.items()):
            union = covered | candidate["coverage"]
            costs[union] = min(costs.get(union, float("inf")), cost + candidate["cost"])
    return costs.get(full)


class SemanticTests(unittest.TestCase):
    def task(self, background, heads, literals, positives, negatives=(), max_body=3, constraints=""):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "background.lp").write_text(background)
        (root / "examples.lp").write_text("\n".join(
            [f"positive({atom})." for atom in positives] +
            [f"negative({atom})." for atom in negatives]))
        bias = "\n".join([f"candidate_head({atom})." for atom in heads] +
                         [f"allowed_literal({atom})." for atom in literals])
        bias += f"\nmax_body({max_body}).\n"
        bias += "1 { head(H) : candidate_head(H) } 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n"
        (root / "bias.lp").write_text(bias + constraints)
        return Task.load(root / "background.lp", root / "examples.lp", root / "bias.lp")

    def check_all_bodies(self, task):
        indices, stats = build_indices(task, seconds=30, max_witnesses=100000)
        labelled = task.positives | task.negatives
        positives = sorted(task.positives)
        raw = []
        for index in indices:
            for size in range(task.max_body + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(index.head) <= used:
                        continue  # An unsafe ASP rule has no executable interpretation.
                    rule = task.rule(index.head, body)
                    predictions = interpretation(task.background + "\n" + rule) & labelled
                    witness_mask = index.body_mask(body)
                    oracle_predictions = {
                        clingo.parse_term(entry["head"])
                        for i, entry in enumerate(index.classes) if witness_mask & (1 << i)
                    }
                    with self.subTest(rule=rule):
                        self.assertEqual(oracle_predictions, predictions)
                    if task.accepts(index.head, body) and not predictions & task.negatives:
                        coverage = sum(1 << i for i, atom in enumerate(positives) if atom in predictions)
                        if coverage:
                            raw.append({"id": f"r{len(raw)}", "cost": 1 + len(body),
                                        "coverage": coverage, "rule": rule})
        compiled, compile_stats = compile_catalogue(task, indices, seconds=30)
        self.assertTrue(compile_stats["complete"])
        result = solve(compiled, len(positives))
        self.assertEqual(result["optimum_cost"], exact_cover_cost(raw, len(positives)))
        self.assertEqual(result["status"], "unsatisfiable" if exact_cover_cost(raw, len(positives)) is None else "complete")
        # Every retained variant is a legal, equally costly, semantically equivalent rule.
        for candidate in compiled:
            for variant in candidate["variants"]:
                predictions = interpretation(task.background + "\n" + variant)
                self.assertFalse(predictions & task.negatives)
                coverage = sum(1 << i for i, atom in enumerate(positives) if atom in predictions)
                self.assertEqual(coverage, candidate["coverage"])
        return indices, compiled, result, stats, compile_stats

    def test_negative_prefix_must_be_extended_to_become_safe(self):
        task = self.task("p(a). p(b). q(a).", ["target(var__(1))"],
                         ["p(var__(1))", "q(var__(1))"], ["target(a)"], ["target(b)"],
                         max_body=2, constraints=":- not body(p(var__(1))).\n")
        _, catalogue, result, _, _ = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 3)
        self.assertEqual(len(catalogue), 1)
        self.assertEqual(len(catalogue[0]["body"]), 2)

    def test_structurally_illegal_safe_prefix_and_equal_cost_variants(self):
        task = self.task("a(x). b(x). c(x).", ["target(var__(1))"],
                         ["a(var__(1))", "b(var__(1))", "c(var__(1))"],
                         ["target(x)"], ["target(y)"], max_body=3,
                         constraints=":- #count { L : body(L) } < 2.\n")
        _, catalogue, result, _, stats = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 3)
        self.assertEqual(len(catalogue), 1)
        self.assertEqual(len(catalogue[0]["variants"]), 3)
        self.assertGreater(stats["pruned_safe_extensions"], 0)

    def test_unused_hidden_slots_with_empty_domains_preserve_witnesses(self):
        task = self.task("p(a). p(b).", ["target(var__(1))"],
                         ["p(var__(1))", "absent(var__(2))", "also_absent(pair(var__(2),var__(3)))"],
                         ["target(a)"], ["target(z)"], max_body=3)
        indices, _, result, stats, _ = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 2)
        self.assertEqual(stats["raw_witnesses"], 2)
        representatives = [item["representative"] for item in indices[0].classes]
        self.assertTrue(all("2" in row and "3" in row for row in representatives))

    def test_nested_terms_arbitrary_arity_and_two_head_templates(self):
        task = self.task("edge(a,pair(b,x)). edge(c,pair(d,y)). kind(x,left). kind(y,right). flag.",
                         ["answer(pack(var__(1)),left,var__(2),9)",
                          "answer(pack(var__(1)),right,var__(2),9)"],
                         ["edge(var__(1),pair(var__(2),var__(3)))", "kind(var__(3),left)",
                          "kind(var__(3),right)", "flag"],
                         ["answer(pack(a),left,b,9)", "answer(pack(c),right,d,9)"],
                         ["answer(pack(c),left,d,9)", "answer(pack(a),right,b,9)"], max_body=3)
        _, _, result, _, _ = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 6)
        self.assertEqual(len(result["selected"]), 2)

    def test_head_unification_repeated_variable_and_constant_mismatch(self):
        task = self.task("edge(a,a). edge(a,b).", ["target(pair(var__(1),var__(1)))"],
                         ["edge(var__(1),var__(2))"], ["target(pair(a,a))"],
                         ["target(pair(a,b))", "target(other(a,a))"], max_body=1)
        _, _, result, stats, _ = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 2)
        self.assertEqual(stats["labelled_head_instances"], 1)
        self.assertGreater(stats["raw_witnesses"], stats["compressed_witnesses"])

    def test_unlabelled_predictions_are_unconstrained(self):
        task = self.task("p(a). p(b). p(c).", ["target(var__(1))"],
                         ["p(var__(1))"], ["target(a)"], max_body=1)
        _, catalogue, result, stats, _ = self.check_all_bodies(task)
        self.assertEqual(stats["labelled_head_instances"], 1)
        self.assertEqual(result["optimum_cost"], 2)
        model = interpretation(task.background + "\n" + catalogue[0]["rule"])
        self.assertIn(clingo.parse_term("target(c)"), model)

    def test_no_positives_learns_the_empty_hypothesis(self):
        task = self.task("p(a).", ["target(var__(1))"], ["p(var__(1))"], [], ["target(a)"], max_body=1)
        _, catalogue, result, _, _ = self.check_all_bodies(task)
        self.assertEqual(catalogue, [])
        self.assertEqual(result["optimum_cost"], 0)
        self.assertEqual(result["selected"], [])

    def test_ground_zero_arity_head_and_empty_body(self):
        task = self.task("unrelated(a).", ["success"], [], ["success"], max_body=0)
        _, catalogue, result, _, _ = self.check_all_bodies(task)
        self.assertEqual(result["optimum_cost"], 1)
        self.assertEqual(catalogue[0]["rule"], "success.")

    def test_multi_rule_verified_bound_preserves_the_exact_optimum(self):
        task = self.task("p(a). q(b). p(c). r(a). r(b).", ["target(var__(1))"],
                         ["p(var__(1))", "q(var__(1))", "r(var__(1))"],
                         ["target(a)", "target(b)"], ["target(c)"], max_body=3,
                         constraints=":- body(r(var__(1))), not body(p(var__(1))).\n")
        indices, catalogue, expected, _, _ = self.check_all_bodies(task)
        def verify(rules):
            predictions = interpretation(task.background + "\n" + "\n".join(rules))
            return {"matched": task.positives <= predictions and not task.negatives & predictions}
        cover, report = discover_bound(task, indices, 30, verify)
        self.assertEqual(report["status"], "verified")
        self.assertGreater(len(cover), 1)
        bounded, stats = compile_catalogue(task, indices, bound=report["cost"], seconds=30)
        self.assertEqual(solve(bounded, len(task.positives))["optimum_cost"], expected["optimum_cost"])
        self.assertTrue(stats["complete"])


if __name__ == "__main__":
    unittest.main()
