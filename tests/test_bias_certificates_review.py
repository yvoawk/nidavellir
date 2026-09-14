"""Independent exhaustive counterexample search for structural certificates."""

from itertools import combinations, product
import dataclasses
import unittest

from nidavellir.bias_analysis import analyze
from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.search import compile_catalogue, mine_variants
from tests import test_search_optimizations as reference


class BiasCertificateReviewTests(unittest.TestCase):
    task = reference.SearchOptimizationTests.task
    expected = reference.SearchOptimizationTests.expected
    actual = staticmethod(reference.SearchOptimizationTests.actual)

    CASES = {
        "auxiliary_or_with_negation":
            "okay :- body(p(var__(1))). okay :- not body(q(var__(1))). :- not okay.",
        "negated_conjunction":
            "a :- body(p(var__(1))), not body(q(var__(1))). b :- not a. :- b, body(r(var__(1))).",
        "minimum_and_maximum":
            ":- #count { L : body(L) } < 2. :- #count { L : body(L) } > 3.",
        "mixed_sign_cardinality":
            ":- 2 { body(p(var__(1))); not body(q(var__(1))); body(r(var__(1))) }.",
        "both_signs_cardinality":
            ":- not 2 { body(p(var__(1))); not body(p(var__(1))); body(q(var__(1))); body(r(var__(1))) }.",
        "weighted_minimum":
            ":- #sum { 2,p : body(p(var__(1))); 1,q : body(q(var__(1))); 3,r : body(r(var__(1))) } < 3.",
        "auxiliary_disjunction":
            "left | right :- body(p(var__(1))). :- left, not body(q(var__(1))). "
            ":- right, not body(r(var__(1))).",
        "positive_support_cycle":
            "a :- body(p(var__(1))). a :- b. b :- a. :- b, not body(q(var__(1))).",
        "negative_even_cycle":
            "a :- not b. b :- not a. :- a, body(p(var__(1))), not body(q(var__(1))).",
        "negative_odd_cycle_enabled_by_body":
            "a :- not a, not body(p(var__(1))).",
        "choice_support_via_auxiliary":
            "{ chosen } :- body(p(var__(1))). :- body(q(var__(1))), not chosen.",
        "guard_cycle_and_alternative":
            ":- body(p(var__(1))), not body(q(var__(1))). "
            ":- body(q(var__(1))), not body(p(var__(1))). "
            ":- body(r(var__(1))), not 1 { body(q(var__(1))); body(s(var__(1))) }.",
    }

    def fixture(self, constraints, *, identical=False):
        background = (" ".join(f"{p}({x})." for p in "pqrs" for x in "ab") if identical else
                      "p(a). p(c). q(a). q(b). r(b). r(c). s(a). s(b).")
        return self.task(background, [f"{p}(var__(1))" for p in "pqrs"],
                         constraints=constraints, max_body=4)

    def check_certificates(self, task):
        proof = analyze(task).heads[task.heads[0]]
        structural = proof.prefix_filter
        count = len(task.literals)
        legal = {sum(1 << i for i in body)
                 for size in range(min(task.max_body, count) + 1)
                 for body in combinations(range(count), size)
                 if task.accepts(task.heads[0], body)}
        for states in product(range(3), repeat=count):
            included = sum(1 << i for i, state in enumerate(states) if state == 1)
            excluded = sum(1 << i for i, state in enumerate(states) if state == 2)
            completions = [body for body in legal if body & included == included and not body & excluded]
            if completions:
                self.assertTrue(structural.allows(included, excluded), (included, excluded, completions))
            for literal in range(count):
                bit = 1 << literal
                if included & bit and structural.can_delete(literal, included, excluded):
                    for body in completions:
                        self.assertIn(body & ~bit, legal,
                                      ("unsound deletion", literal, included, excluded, body, legal))
        return proof

    def test_every_prefix_and_every_legal_extension(self):
        for name, restrictions in self.CASES.items():
            with self.subTest(name=name):
                self.check_certificates(self.fixture(restrictions))

    def test_pruning_preserves_complete_minimal_catalogue_and_equal_cost_variants(self):
        for name, restrictions in self.CASES.items():
            for identical in (False, True):
                with self.subTest(name=name, identical=identical):
                    task = self.fixture(restrictions, identical=identical)
                    expected = self.expected(task)
                    indices, _ = build_cartesian_indices(task)
                    for order in ("source", "selectivity"):
                        catalogue, stats = compile_catalogue(task, indices, order=order, variants="exhaustive")
                        self.assertTrue(stats["complete"])
                        self.assertEqual(self.actual(catalogue), expected)

    def test_alpha_ties_and_asymmetric_meta_constraints_survive_targeted_pass(self):
        task = self.task("edge(a,x). edge(b,x). edge(c,y). good(x).",
            ["edge(var__(1),var__(2))", "edge(var__(1),var__(3))",
             "good(var__(2))", "good(var__(3))"], max_body=3,
            constraints=":- body(good(var__(2))), body(edge(var__(1),var__(3))), "
                        "not body(edge(var__(1),var__(2))).")
        self.check_certificates(task)
        expected = self.expected(task)
        indices, _ = build_cartesian_indices(task)
        catalogue, _ = compile_catalogue(task, indices, variants="representative")
        variants, stats = mine_variants(task, indices, catalogue, alpha_collect=False)
        self.assertTrue(stats["complete"])
        self.assertEqual(self.actual(variants), expected)

    def test_budget_expiry_never_returns_complete_catalogue(self):
        task = self.fixture(self.CASES["minimum_and_maximum"], identical=True)
        indices, _ = build_cartesian_indices(task)
        with self.assertRaises(ResourceLimit) as stopped:
            compile_catalogue(task, indices, seconds=0)
        self.assertFalse(stopped.exception.stats.get("complete", False))

    def test_conditional_head_support_must_survive_literal_deletion(self):
        for support in (
            "{ head(H) : candidate_head(H) } :- body(q(var__(2))).",
            "ready :- body(q(var__(2))). { head(H) : candidate_head(H) } :- ready.",
            "blocked :- not body(q(var__(2))). { head(H) : candidate_head(H) } :- not blocked.",
            "head(H) :- candidate_head(H), body(q(var__(2))).",
            "ready :- body(q(var__(2))). ready :- again. again :- ready. "
            "{ head(H) : candidate_head(H) } :- ready.",
        ):
            with self.subTest(support=support):
                task = self.task("p(a). q(z).", ["p(var__(1))", "q(var__(2))"], max_body=2,
                                 examples="positive(answer(a)). negative(answer(b)).")
                task = dataclasses.replace(task, bias_source=task.bias_source.replace(
                    "1 <= { head(H): candidate_head(H) } <= 1.", support))
                expected = self.expected(task)
                self.assertEqual(set(expected), {(3, 1)})
                self.check_certificates(task)
                indices, _ = build_cartesian_indices(task)
                catalogue, stats = compile_catalogue(task, indices, variants="exhaustive")
                self.assertTrue(stats["complete"])
                self.assertEqual(self.actual(catalogue), expected)

    def test_support_and_false_alternative_head_are_not_assumed_invariant(self):
        task = self.task("p(a). q(z).", ["p(var__(1))", "q(var__(2))"], max_body=2,
                         heads=("answer(var__(1))", "other(var__(1))"),
                         examples="positive(answer(a)). negative(answer(b)).")
        task = dataclasses.replace(task, bias_source=task.bias_source +
            "head(other(var__(1))) :- not body(q(var__(2))).\n")
        expected = self.expected(task)
        indices, _ = build_cartesian_indices(task)
        catalogue, stats = compile_catalogue(task, indices, variants="exhaustive")
        self.assertTrue(stats["complete"])
        self.assertEqual(self.actual(catalogue), expected)


if __name__ == "__main__":
    unittest.main()
