"""Optimized traversal versus exhaustive rules and independent ASP coverage."""
from itertools import combinations
from pathlib import Path
import random
import tempfile
import unittest

import clingo

from nidavellir.oracle import build_cartesian_indices
from nidavellir.search import compile_catalogue, discover_bound, greedy_cover, mine_variants, pool_cover
from nidavellir.task import Task


def atoms(program):
    control = clingo.Control(["0"], logger=lambda *_: None)
    control.add("base", [], program)
    control.ground([("base", [])])
    models = []
    result = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if not result.satisfiable or not result.exhausted or len(models) != 1:
        raise AssertionError("Independent evaluation needs exactly one model")
    return models[0]


class SearchOptimizationTests(unittest.TestCase):
    def task(self, background, literals, *, constraints="", max_body=3, heads=("answer(var__(1))",),
             examples="positive(answer(a)). positive(answer(b)). negative(answer(c))."):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        bias = "\n".join([f"candidate_head({head})." for head in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n" + constraints
        for name, text in (("bk.lp", background), ("bias.lp", bias), ("examples.lp", examples)):
            (path / name).write_text(text)
        return Task.load(path / "bk.lp", path / "examples.lp", path / "bias.lp")

    def expected(self, task):
        signatures = {}
        positives = sorted(task.positives)
        for head in task.heads:
            for size in range(min(task.max_body, len(task.literals)) + 1):
                for body in combinations(range(len(task.literals)), size):
                    if not task.accepts(head, body):
                        continue
                    rule = task.rule(head, body)
                    prediction = atoms(task.background + "\n" + rule)
                    if prediction & task.negatives:
                        continue
                    coverage = sum(1 << i for i, p in enumerate(positives) if p in prediction)
                    if not coverage:
                        continue
                    cost = size + 1
                    previous = signatures.get(coverage)
                    if previous is None or cost < previous[0]:
                        signatures[coverage] = cost, {rule}
                    elif cost == previous[0]:
                        previous[1].add(rule)
        return {(cost, coverage): variants for coverage, (cost, variants) in signatures.items()
                if not any(other_cost < cost and other | coverage == other
                           for other, (other_cost, _) in signatures.items())}

    @staticmethod
    def actual(catalogue):
        return {(c["cost"], c["coverage"]): set(c["variants"]) for c in catalogue}

    def compare(self, task):
        expected = self.expected(task)
        indices, _ = build_cartesian_indices(task, seconds=30)
        all_catalogues = []
        for order in ("source", "selectivity"):
            for preprocessing in (False, True):
                for variants in ("exhaustive", "targeted"):
                    catalogue, stats = compile_catalogue(task, indices, order=order,
                                                        preprocessing=preprocessing, variants=variants,
                                                        alpha_collect=False)
                    self.assertEqual(self.actual(catalogue), expected)
                    self.assertTrue(stats["complete"])
                    all_catalogues.append(catalogue)
        ids = [{(c["cost"], c["coverage"]): c["id"] for c in catalogue} for catalogue in all_catalogues]
        self.assertTrue(all(mapping == ids[0] for mapping in ids))
        return indices, all_catalogues[0]

    def test_guard_cycles_disjunctions_and_at_most_groups(self):
        task = self.task("p(a). p(c). q(a). q(b). r(b). s(a). s(b).", [f"{p}(var__(1))" for p in "pqrs"],
                         constraints=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                     ":- body(q(var__(1))), not body(p(var__(1))).\n"
                                     ":- body(r(var__(1))), not 1 { body(q(var__(1))); body(s(var__(1))) }.\n"
                                     ":- 2 { body(q(var__(1))); body(s(var__(1))) }.\n")
        self.compare(task)

    def test_semantically_equal_masks_keep_structural_roles_and_all_ties(self):
        task = self.task("p(a). p(b). q(a). q(b). r(a). r(b).", [f"{p}(var__(1))" for p in "pqr"],
                         constraints=":- #count { L : body(L) } != 2.\n")
        _, catalogue = self.compare(task)
        self.assertEqual(len(catalogue[0]["variants"]), 3)

    def test_equal_cost_subset_coverage_is_not_dominated(self):
        task = self.task("p(a). p(b). q(a).", ["p(var__(1))", "q(var__(1))"], max_body=1)
        _, catalogue = self.compare(task)
        self.assertEqual({c["coverage"] for c in catalogue}, {1, 3})

    def test_alpha_variants_are_grouped_without_losing_legal_spelling(self):
        task = self.task("edge(a,x). edge(b,x). edge(c,y). good(x).",
                         ["edge(var__(1),var__(2))", "edge(var__(1),var__(3))",
                          "good(var__(2))", "good(var__(3))"], max_body=2)
        _, catalogue = self.compare(task)
        self.assertEqual(len(catalogue), 1)
        self.assertEqual(len(catalogue[0]["variants"]), 2)
        self.assertEqual(catalogue[0]["alpha_variant_count"], 1)

    def test_random_finite_tasks_match_every_optimum_relevant_variant(self):
        for seed in range(4):
            rng = random.Random(seed)
            background = " ".join(f"{p}({x})." for p in "pqrs" for x in "abc" if rng.randrange(2))
            task = self.task(background, [f"{p}(var__(1))" for p in "pqrs"],
                             constraints=":- body(p(var__(1))), not 1 { body(q(var__(1))); body(r(var__(1))) }.\n")
            with self.subTest(seed=seed):
                self.compare(task)

    def test_targeted_variant_reconstruction_preserves_candidate_identifiers(self):
        task = self.task("p(a). p(b). q(a). q(b).", ["p(var__(1))", "q(var__(1))"])
        indices, expected = self.compare(task)
        representatives, _ = compile_catalogue(task, indices, variants="representative")
        reconstructed, stats = mine_variants(task, indices, representatives)
        self.assertEqual(self.actual(reconstructed), self.actual(expected))
        self.assertEqual([c["id"] for c in reconstructed], [c["id"] for c in representatives])
        self.assertTrue(stats["complete"])

    def test_pool_selection_improves_a_suboptimal_greedy_cover(self):
        candidates = [dict(rule="a.", cost=3, coverage=0b001111),
                      dict(rule="b.", cost=3, coverage=0b010011),
                      dict(rule="c.", cost=3, coverage=0b101100),
                      dict(rule="d.", cost=2, coverage=0b010000),
                      dict(rule="e.", cost=2, coverage=0b100000)]
        self.assertEqual(sum(c["cost"] for c in greedy_cover(candidates, 6)), 7)
        exact, report = pool_cover(candidates, 6)
        self.assertEqual(sum(c["cost"] for c in exact), 6)
        self.assertEqual(report["optimum_cost"], 6)
        self.assertEqual(report["scope"], "discovered_components_only")

    def test_multi_component_bound_cannot_delete_structural_noops(self):
        task = self.task("p(a). q(a). r(b). s(b).", [f"{p}(var__(1))" for p in "pqrs"], max_body=2,
                         constraints=":- body(p(var__(1))), not body(q(var__(1))).\n"
                                     ":- body(q(var__(1))), not body(p(var__(1))).\n"
                                     ":- body(r(var__(1))), not body(s(var__(1))).\n"
                                     ":- body(s(var__(1))), not body(r(var__(1))).\n")
        indices, _ = self.compare(task)
        def verify(rules):
            prediction = atoms(task.background + "\n" + "\n".join(rules))
            return {"matched": task.positives <= prediction and not task.negatives & prediction}
        cover, report = discover_bound(task, indices, 30, verify)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(len(cover), 2)
        self.assertEqual(report["cost"], 6)
        self.assertTrue(all(c["cost"] == 3 for c in cover))

    def test_bitmask_cegis_full_cover_uses_real_negative_witnesses(self):
        task = self.task("edge(a,x). edge(b,y). edge(c,z). good(x). good(y).",
                         ["edge(var__(1),var__(2))", "good(var__(2))"], max_body=2)
        indices, _ = build_cartesian_indices(task)
        def verify(rules):
            prediction = atoms(task.background + "\n" + "\n".join(rules))
            return {"matched": task.positives <= prediction and not task.negatives & prediction}
        cover, report = discover_bound(task, indices, 30, verify)
        self.assertEqual(report["cost"], 3)
        self.assertEqual(len(cover), 1)
        self.assertGreater(report["single_rule_negative_witnesses"], 0)
        self.assertEqual(report["search"]["nodes"], 0)
        self.assertEqual(report["examined"], report["single_rule_cegis_models"])

    def test_full_cover_heuristic_falls_back_for_distinct_head_templates(self):
        task = self.task("p(a). q(b).", ["p(var__(1))", "q(var__(1))"], max_body=1,
                         heads=("answer(left,var__(1))", "answer(right,var__(1))"),
                         examples="positive(answer(left,a)). positive(answer(right,b)). "
                                  "negative(answer(left,b)). negative(answer(right,a)).")
        indices, _ = build_cartesian_indices(task)
        def verify(rules):
            prediction = atoms(task.background + "\n" + "\n".join(rules))
            return {"matched": task.positives <= prediction and not task.negatives & prediction}
        cover, report = discover_bound(task, indices, 30, verify)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(len(cover), 2)
        self.assertEqual(report["cost"], 4)
        self.assertEqual(report["single_rule_cegis_models"], 0)


if __name__ == "__main__":
    unittest.main()
