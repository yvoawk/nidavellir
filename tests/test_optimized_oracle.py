"""Independent legal-body contracts for reductions in the optimized oracle.

These tests deliberately do not require equality of raw truth profiles. They
compare positive-head coverage and existence of a covered negative with both
the Cartesian reference and direct Clingo evaluation of every legal body.
"""

from itertools import combinations
from pathlib import Path
import tempfile
import unittest
import random

import clingo

from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.task import Task


def model_of(source):
    control = clingo.Control(["2"], logger=lambda *_: None)
    control.add("base", [], source)
    control.ground([("base", [])])
    models = []
    result = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
    if result.unknown or not result.satisfiable or not result.exhausted or len(models) != 1:
        raise AssertionError("Expected one completely checked model")
    return models[0]


class OptimizedOracleContractTests(unittest.TestCase):
    def task(self, background, literals, positives, negatives=(), *, heads=("answer(var__(1))",), bias="", max_body=3):
        temporary = tempfile.TemporaryDirectory(prefix="nidavellir-oracle-contract-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "background.lp").write_text(background)
        (root / "examples.lp").write_text("\n".join(
            [f"positive({atom})." for atom in positives] + [f"negative({atom})." for atom in negatives]))
        program = "\n".join([f"candidate_head({head})." for head in heads]
                              + [f"allowed_literal({literal})." for literal in literals])
        program += f"\nmax_body({max_body}).\n1 {{ head(H) : candidate_head(H) }} 1.\n"
        program += "0 { body(L) : allowed_literal(L) } N :- max_body(N).\n" + bias
        (root / "bias.lp").write_text(program)
        return Task.load(root / "background.lp", root / "examples.lp", root / "bias.lp")

    def compare_legal_bodies(self, task):
        optimized, stats = build_optimized_indices(task, seconds=30, max_classes=100000)
        reference, _ = build_cartesian_indices(task, seconds=30, max_witnesses=100000)
        optimized_by_head = {index.head: index for index in optimized}
        reference_by_head = {index.head: index for index in reference}
        positive_order = sorted(task.positives)
        checked = 0
        for head in task.heads:
            for size in range(min(task.max_body, len(task.literals)) + 1):
                for body in combinations(range(len(task.literals)), size):
                    if not task.accepts(head, body):
                        continue
                    checked += 1
                    actual = optimized_by_head[head]
                    expected = reference_by_head[head]
                    mask = actual.body_mask(body)
                    raw_mask = expected.body_mask(body)
                    rule = task.rule(head, body)
                    predictions = model_of(task.background + "\n" + rule)
                    positive_bits = sum(1 << i for i, atom in enumerate(positive_order) if atom in predictions)
                    with self.subTest(rule=rule):
                        self.assertEqual(actual.coverage(mask), positive_bits)
                        self.assertEqual(actual.coverage(mask), expected.coverage(raw_mask))
                        self.assertEqual(bool(mask & actual.negative_mask), bool(predictions & task.negatives))
                        self.assertEqual(bool(mask & actual.negative_mask), bool(raw_mask & expected.negative_mask))
        self.assertTrue(stats["complete"])
        self.assertGreater(checked, 0)
        return optimized, stats

    def test_carrier_guards_preserve_only_legal_body_semantics(self):
        self.compare_legal_bodies(self.task(
            "base(a). base(b). transition(a,x). transition(b,y). low(x). low(z). high(y).",
            ["base(var__(1))", "transition(var__(1),var__(2))", "low(var__(2))", "high(var__(2))"],
            ["answer(a)"], ["answer(b)"],
            bias=":- body(low(var__(2))), not body(transition(var__(1),var__(2))).\n"
                 ":- body(high(var__(2))), not body(transition(var__(1),var__(2)))."))

    def test_disjunctive_guard_does_not_become_one_mandatory_carrier(self):
        self.compare_legal_bodies(self.task(
            "p(a). p(b). q(a). r(b).",
            ["p(var__(1))", "q(var__(1))", "r(var__(1))"], ["answer(a)"], ["answer(b)"],
            bias=":- body(p(var__(1))), not body(q(var__(1))), not body(r(var__(1)))."))

    def test_guards_are_specific_to_each_head_template(self):
        self.compare_legal_bodies(self.task(
            "p(a). p(b). q(a).",
            ["p(var__(1))", "q(var__(1))"], ["first(a)", "second(b)"], ["first(b)", "second(a)"],
            heads=("first(var__(1))", "second(var__(1))"),
            bias=":- head(first(var__(1))), body(p(var__(1))), not body(q(var__(1)))."))

    def test_functional_relations_with_conflicting_outputs_keep_join(self):
        self.compare_legal_bodies(self.task(
            "r(a,x). s(a,y). r(b,z). s(b,z).",
            ["r(var__(1),var__(2))", "s(var__(1),var__(2))"], ["answer(a)"], ["answer(b)"], max_body=2))

    def test_nonfunctional_carrier_must_not_choose_only_one_output(self):
        self.compare_legal_bodies(self.task(
            "r(a,x). r(a,y). s(y). r(b,x).",
            ["r(var__(1),var__(2))", "s(var__(2))"], ["answer(a)"], ["answer(b)"], max_body=2,
            bias=":- body(s(var__(2))), not body(r(var__(1),var__(2)))."))

    def test_private_and_shared_variables_have_different_join_semantics(self):
        self.compare_legal_bodies(self.task(
            "p(a,x). q(a,y). p(b,x). q(b,x).",
            ["p(var__(1),var__(2))", "q(var__(1),var__(2))", "q(var__(1),var__(3))"],
            ["answer(a)"], ["answer(b)"], max_body=3))

    def test_maximal_profiles_of_different_cardinalities_are_preserved(self):
        self.compare_legal_bodies(self.task(
            "a(x). b(x). c(x). d(y). e(y).",
            ["a(var__(1))", "b(var__(1))", "c(var__(1))", "d(var__(1))", "e(var__(1))"],
            ["success"], heads=("success",), max_body=3))

    def test_negative_profiles_merge_by_equality_not_union(self):
        self.compare_legal_bodies(self.task(
            "p(g1). q(g1). p(g2). q(g2). p(n1). q(n2). p(n3).",
            ["p(var__(1))", "q(var__(1))"], ["answer(g1)", "answer(g2)"],
            ["answer(n1)", "answer(n2)", "answer(n3)"], max_body=2))

    def test_semantically_equal_literals_can_both_be_structurally_required(self):
        self.compare_legal_bodies(self.task(
            "p(a). q(a).",
            ["p(var__(1))", "q(var__(1))"], ["answer(a)"], ["answer(b)"], max_body=2,
            bias=":- not body(p(var__(1))).\n:- not body(q(var__(1)))."))

    def test_empty_hidden_domain_and_ground_empty_body(self):
        self.compare_legal_bodies(self.task(
            "p(a).", ["p(var__(1))", "absent(var__(2))"], ["answer(a)"], ["answer(b)"], max_body=2))
        self.compare_legal_bodies(self.task("unrelated(a).", [], ["success"], heads=("success",), max_body=0))

    def test_conditional_functional_anchors_keep_shared_correlations(self):
        task = self.task(
            "base(a). base(b). base(c). r(a,x). s(a,y). r(b,z). s(b,z). "
            "mark(x). mark(z). link(x,y). link(z,z).",
            ["base(var__(1))", "r(var__(1),var__(2))", "s(var__(1),var__(2))",
             "r(var__(1),var__(3))", "s(var__(1),var__(3))",
             "mark(var__(2))", "link(var__(2),var__(3))"],
            ["answer(a)", "answer(c)"], ["answer(b)"], max_body=3,
            bias="used(2) :- body(mark(var__(2))). used(2) :- body(link(var__(2),var__(3))). "
                 "used(3) :- body(link(var__(2),var__(3))). "
                 ":- used(V), not body(r(var__(1),var__(V))), not body(s(var__(1),var__(V))).")
        _, stats = self.compare_legal_bodies(task)
        self.assertEqual(set(stats["conditional_anchors"][str(task.heads[0])]), {"2", "3"})
        self.assertEqual(stats["direct_label_instances"], 3)
        self.assertEqual(stats["projected_label_instances"], 0)

    def test_conditional_anchors_require_a_bias_proof_and_a_functional_relation(self):
        for extra, bias in (("", ""), ("r(a,y).", ":- body(mark(var__(2))), not body(r(var__(1),var__(2))), not body(s(var__(1),var__(2))).")):
            task = self.task(
                "base(a). base(b). r(a,x). s(b,z). mark(x). mark(y). " + extra,
                ["base(var__(1))", "r(var__(1),var__(2))", "s(var__(1),var__(2))", "mark(var__(2))"],
                ["answer(a)"], ["answer(b)"], bias=bias)
            _, stats = self.compare_legal_bodies(task)
            self.assertEqual(stats["conditional_anchors"][str(task.heads[0])], {})
            self.assertEqual(stats["direct_label_instances"], 0)

    def test_conditional_anchors_support_nested_terms_and_repeated_head_variables(self):
        task = self.task(
            "r(a,box(x,x)). s(a,box(y,y)). r(b,box(z,z)). s(b,box(z,z)). "
            "r(c,box(x,y)). mark(x). mark(z).",
            ["r(var__(1),box(var__(2),var__(2)))", "s(var__(1),box(var__(2),var__(2)))", "mark(var__(2))"],
            ["answer(a,a)"], ["answer(b,b)", "answer(c,c)"], heads=("answer(var__(1),var__(1))",),
            bias=":- body(mark(var__(2))), not body(r(var__(1),box(var__(2),var__(2)))), not body(s(var__(1),box(var__(2),var__(2)))).")
        _, stats = self.compare_legal_bodies(task)
        self.assertEqual(stats["direct_label_instances"], 3)

    def test_conditional_anchor_random_relations_and_resource_limit(self):
        for seed in range(12):
            rng = random.Random(seed)
            facts = [f"base({head})." for head in "abc"]
            facts += [f"{pred}({head},{rng.choice('xyz')})." for pred in ("r", "s")
                      for head in "abc" if rng.random() > .2]
            facts += [f"link({left},{right})." for left in "xyz" for right in "xyz" if rng.random() < .5]
            task = self.task(" ".join(facts),
                ["base(var__(1))", "r(var__(1),var__(2))", "s(var__(1),var__(2))",
                 "r(var__(1),var__(3))", "s(var__(1),var__(3))", "link(var__(2),var__(3))"],
                ["answer(a)", "answer(c)"], ["answer(b)"], max_body=3,
                bias=":- body(link(var__(2),var__(3))), not body(r(var__(1),var__(2))), not body(s(var__(1),var__(2))). "
                     ":- body(link(var__(2),var__(3))), not body(r(var__(1),var__(3))), not body(s(var__(1),var__(3))).")
            with self.subTest(seed=seed):
                self.compare_legal_bodies(task)
                with self.assertRaises(ResourceLimit):
                    build_optimized_indices(task, max_classes=1)

    def test_head_dependent_cover_is_preferred_only_after_proof(self):
        restriction = ":- body(mark(var__(2))), not body(r(var__(1),var__(2))), not body(s(var__(1),var__(2)))."
        for bias in (restriction, ""):
            task = self.task("base(a). base(b). r(a,y). s(a,z). r(b,x). mark(x).",
                ["base(var__(1))", "r(var__(1),var__(2))", "s(var__(1),var__(2))", "mark(var__(2))"],
                ["answer(a)"], ["answer(b)"], bias=bias)
            _, stats = self.compare_legal_bodies(task)
            carriers = stats["conditional_anchors"][str(task.heads[0])]["2"]
            mark = task.literals.index(clingo.parse_term("mark(var__(2))"))
            self.assertEqual(mark in carriers, not bool(bias))


if __name__ == "__main__":
    unittest.main()
