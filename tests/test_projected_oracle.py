"""Independent checks of projected truth classes against raw witnesses and ASP.

Class ordering and multiplicities are deliberately not part of the comparison:
the compiler needs the exact set of (labelled head, literal truth vector).
"""

from itertools import combinations
from pathlib import Path
import random
import tempfile
import unittest

import clingo

from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.projected_oracle import build_projected_indices
from nidavellir.task import Task, instantiate, variables


def model_of(program):
    control = clingo.Control(["0"], logger=lambda *_: None)
    control.add("base", [], program)
    control.ground([("base", [])])
    models = []
    result = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if result.unknown or not result.satisfiable or len(models) != 1:
        raise AssertionError(f"Expected one background-plus-rule model: {result}, {len(models)}")
    return models[0]


class ProjectedOracleTests(unittest.TestCase):
    def task(self, background, heads, literals, positives, negatives=(), max_body=3):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "background.lp").write_text(background)
        (root / "examples.lp").write_text("\n".join(
            [f"positive({atom})." for atom in positives]
            + [f"negative({atom})." for atom in negatives]))
        bias = "\n".join([f"candidate_head({head})." for head in heads]
                         + [f"allowed_literal({literal})." for literal in literals])
        bias += f"\nmax_body({max_body}).\n"
        bias += "1 { head(H) : candidate_head(H) } 1.\n"
        bias += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n"
        (root / "bias.lp").write_text(bias)
        return Task.load(root / "background.lp", root / "examples.lp", root / "bias.lp")

    @staticmethod
    def classes(index):
        return {(entry["head"], int(entry["truth"], 16)) for entry in index.classes}

    def check_representatives(self, task, indices):
        for index in indices:
            expected_all = (1 << len(index.classes)) - 1
            self.assertEqual(index.all_mask, expected_all)
            self.assertEqual(index.positive_mask | index.negative_mask, expected_all)
            self.assertEqual(index.positive_mask & index.negative_mask, 0)
            self.assertEqual(len(index.classes), len(self.classes(index)))
            for position, entry in enumerate(index.classes):
                binding = {int(v): clingo.parse_term(value)
                           for v, value in entry["representative"].items()}
                with self.subTest(head=entry["head"], truth=entry["truth"]):
                    self.assertEqual(str(instantiate(index.head, binding)), entry["head"])
                    expected = sum(1 << i for i, literal in enumerate(task.literals)
                                   if instantiate(literal, binding) in task.atoms)
                    self.assertEqual(int(entry["truth"], 16), expected)
                    for literal, mask in enumerate(index.literal_masks):
                        self.assertEqual(bool(mask & (1 << position)),
                                         bool(expected & (1 << literal)))

    def compare(self, task):
        projected, stats = build_projected_indices(task, seconds=30, max_classes=100000)
        reference, reference_stats = build_cartesian_indices(
            task, seconds=30, max_witnesses=100000)
        self.assertEqual([index.head for index in projected], [index.head for index in reference])
        self.assertEqual(stats["compressed_witnesses"], reference_stats["compressed_witnesses"])
        self.assertEqual(stats["conceptual_cartesian_witnesses"], reference_stats["raw_witnesses"])
        self.assertEqual(stats["materialized_witnesses"], 0)
        self.check_representatives(task, projected)
        labels = task.positives | task.negatives
        sorted_positives = sorted(task.positives)
        for actual, expected in zip(projected, reference):
            self.assertEqual(self.classes(actual), self.classes(expected))
            for size in range(min(task.max_body, len(task.literals)) + 1):
                for body in combinations(range(len(task.literals)), size):
                    # The Cartesian oracle is defined for every template subset,
                    # including structurally rejected or head-unsafe subsets.
                    actual_mask = actual.body_mask(body)
                    expected_mask = expected.body_mask(body)
                    self.assertEqual(actual.coverage(actual_mask), expected.coverage(expected_mask))
                    self.assertEqual(bool(actual_mask & actual.negative_mask),
                                     bool(expected_mask & expected.negative_mask))
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(actual.head) <= used:
                        continue
                    rule = task.rule(actual.head, body)
                    predictions = model_of(task.background + "\n" + rule) & labels
                    represented = {clingo.parse_term(entry["head"])
                                   for position, entry in enumerate(actual.classes)
                                   if actual_mask & (1 << position)}
                    with self.subTest(rule=rule):
                        self.assertEqual(represented, predictions)
                        expected_coverage = sum(1 << i for i, atom in enumerate(sorted_positives)
                                                if atom in predictions)
                        self.assertEqual(actual.coverage(actual_mask), expected_coverage)
        return projected, stats

    def test_shared_hidden_variable_preserves_join_correlation(self):
        task = self.task(
            "p(a,x). q(a,y). p(b,x). q(b,x).",
            ["answer(var__(1))"], ["p(var__(1),var__(2))", "q(var__(1),var__(2))"],
            ["answer(a)"], ["answer(b)"], max_body=2)
        indices, _ = self.compare(task)
        index, = indices
        # Both literals have a witness for a, but they cannot share a witness.
        self.assertTrue(index.coverage(index.body_mask((0,))))
        self.assertTrue(index.coverage(index.body_mask((1,))))
        self.assertEqual(index.coverage(index.body_mask((0, 1))), 0)
        self.assertTrue(index.body_mask((0, 1)) & index.negative_mask)

    def test_nested_terms_arbitrary_arity_and_multiple_head_templates(self):
        task = self.task(
            "edge(a,pair(b,x)). edge(c,pair(d,y)). kind(x,left). kind(y,right). "
            "mark(a,b,x,left,9). mark(c,d,y,right,9). flag.",
            ["answer(pack(var__(1)),left,var__(2),9)",
             "answer(pack(var__(1)),right,var__(2),9)"],
            ["edge(var__(1),pair(var__(2),var__(3)))", "kind(var__(3),left)",
             "kind(var__(3),right)", "mark(var__(1),var__(2),var__(3),left,9)", "flag"],
            ["answer(pack(a),left,b,9)", "answer(pack(c),right,d,9)"],
            ["answer(pack(c),left,d,9)", "answer(pack(a),right,b,9)"], max_body=3)
        self.compare(task)

    def test_repeated_head_and_literal_variables_reject_mismatched_terms(self):
        task = self.task(
            "edge(a,a). edge(a,b). edge(b,b). node(pair(a,a)). node(pair(a,b)).",
            ["answer(pair(var__(1),var__(1)))"],
            ["edge(var__(1),var__(2))", "edge(var__(2),var__(2))",
             "node(pair(var__(2),var__(2)))"],
            ["answer(pair(a,a))"],
            ["answer(pair(b,b))", "answer(pair(a,b))", "answer(other(a,a))"], max_body=3)
        indices, stats = self.compare(task)
        self.assertEqual(stats["labelled_head_instances"], 2)
        self.assertEqual({entry["head"] for entry in indices[0].classes},
                         {"answer(pair(a,a))", "answer(pair(b,b))"})

    def test_empty_hidden_domains_do_not_remove_valid_shorter_bodies(self):
        task = self.task("p(a). flag.", ["answer(var__(1))"],
                         ["p(var__(1))", "absent(var__(2))",
                          "absent_nested(pair(var__(2),var__(3)))", "flag"],
                         ["answer(a)"], ["answer(z)"], max_body=3)
        indices, stats = self.compare(task)
        self.assertEqual(stats["conceptual_cartesian_witnesses"], 2)
        self.assertTrue(all({"1", "2", "3"} <= row["representative"].keys()
                            for row in indices[0].classes))

    def test_ground_zero_arity_head_and_empty_body(self):
        task = self.task("unrelated(a).", ["success"], [], ["success"], max_body=0)
        indices, stats = self.compare(task)
        self.assertEqual(stats["compressed_witnesses"], 1)
        self.assertEqual(indices[0].coverage(indices[0].body_mask(())), 1)
        self.assertEqual(self.classes(indices[0]), {("success", 0)})

    def test_ground_head_with_hidden_variables_and_no_labels(self):
        task = self.task("p(a). p(b). q(b).", ["success"],
                         ["p(var__(1))", "q(var__(1))"], [], [], max_body=2)
        indices, stats = self.compare(task)
        self.assertEqual(stats["compressed_witnesses"], 0)
        self.assertEqual(indices[0].all_mask, 0)

    def test_small_random_correlated_relations_match_all_conjunctions(self):
        for seed in range(5):
            rng = random.Random(seed)
            facts = [f"domain({value})." for value in "abc"]
            facts += [f"edge({left},{right})." for left in "abc" for right in "abc"
                      if rng.randrange(2)]
            facts += [f"mark({value})." for value in "abc" if rng.randrange(2)]
            task = self.task(" ".join(facts), ["answer(var__(1))"],
                             ["domain(var__(1))", "edge(var__(1),var__(2))",
                              "edge(var__(2),var__(3))", "edge(var__(3),var__(1))",
                              "mark(var__(2))", "mark(var__(3))"],
                             ["answer(a)", "answer(b)"], ["answer(c)"], max_body=3)
            with self.subTest(seed=seed):
                self.compare(task)

    def test_renamed_relation_cache_preserves_slot_orientation(self):
        task = self.task(
            "edge(a,b). edge(b,c). edge(c,a). edge(a,c). node(a). node(b). node(c).",
            ["answer(var__(1))"],
            ["node(var__(1))", "edge(var__(1),var__(2))", "edge(var__(2),var__(1))",
             "edge(var__(2),var__(3))", "edge(var__(3),var__(2))",
             "edge(var__(1),var__(3))", "edge(var__(3),var__(1))"],
            ["answer(a)", "answer(b)"], ["answer(c)"], max_body=3)
        self.compare(task)

    def test_trillion_substitutions_yield_only_64_truth_classes(self):
        # Six independent hidden slots each have 100 domain values. Two literal
        # profiles per slot produce 2**6 classes, without visiting 100**6 rows.
        literals = [f"{predicate}(var__({slot}))"
                    for slot in range(1, 7) for predicate in ("p", "q")]
        task = self.task("p(1..100). q(1).", ["success"], literals, ["success"], max_body=3)
        indices, stats = build_projected_indices(task, seconds=30, max_classes=128)
        self.assertEqual(stats["conceptual_cartesian_witnesses"], 100 ** 6)
        self.assertEqual(stats["materialized_witnesses"], 0)
        self.assertEqual(stats["compressed_witnesses"], 64)
        self.check_representatives(task, indices)
        self.assertEqual(len(self.classes(indices[0])), 64)
        p_mask = sum(1 << i for i, literal in enumerate(task.literals) if literal.name == "p")
        q_slots = [i for i, literal in enumerate(task.literals) if literal.name == "q"]
        expected = {("success", p_mask | sum(1 << slot for i, slot in enumerate(q_slots)
                                            if profile & (1 << i)))
                    for profile in range(64)}
        self.assertEqual(self.classes(indices[0]), expected)

    def test_class_limit_returns_explicit_failure_instead_of_partial_index(self):
        task = self.task("p(1..3). q(1).", ["success"],
                         ["p(var__(1))", "q(var__(1))"], ["success"], max_body=2)
        with self.assertRaises(ResourceLimit) as caught:
            build_projected_indices(task, seconds=30, max_classes=1)
        self.assertTrue(caught.exception.stats)
        indices, stats = build_projected_indices(task, seconds=30, max_classes=2)
        self.assertEqual(stats["compressed_witnesses"], 2)
        self.assertEqual(len(indices[0].classes), 2)

    def test_class_limit_applies_to_all_head_templates_combined(self):
        task = self.task("p(1..3). q(1).", ["first", "second"],
                         ["p(var__(1))", "q(var__(1))"], ["first", "second"], max_body=2)
        with self.assertRaises(ResourceLimit):
            build_projected_indices(task, seconds=30, max_classes=3)
        indices, stats = build_projected_indices(task, seconds=30, max_classes=4)
        self.assertEqual(stats["compressed_witnesses"], 4)
        self.assertEqual(sum(len(index.classes) for index in indices), 4)

    def test_zero_budget_returns_explicit_failure_instead_of_partial_index(self):
        task = self.task("p(a).", ["success"], ["p(var__(1))"], ["success"], max_body=1)
        with self.assertRaises(ResourceLimit):
            build_projected_indices(task, seconds=0, max_classes=100)


if __name__ == "__main__":
    unittest.main()
