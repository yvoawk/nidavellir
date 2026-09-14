"""Positive semijoins checked against complete Clingo and fixed-label joins."""

from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.positive_join import has_positive
from nidavellir.relational_coverage import CoverageOracle
from nidavellir.task import Task, variables
from tests.test_relational_coverage import unique_model


class PositiveJoinTests(unittest.TestCase):
    def task(self, background, literals, positives, heads=("answer(var__(1))",)):
        temporary = tempfile.TemporaryDirectory(prefix="nidavellir-positive-join-")
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "background.lp").write_text(background)
        (folder / "examples.lp").write_text("\n".join(f"positive({a})." for a in positives))
        (folder / "bias.lp").write_text(
            "\n".join(f"candidate_head({h})." for h in heads) + "\n" +
            "\n".join(f"allowed_literal({l})." for l in literals) +
            f"\nmax_body({len(literals)}).\n1 {{head(H):candidate_head(H)}} 1.\n" +
            "0 {body(L):allowed_literal(L)} N :- max_body(N).\n")
        return Task.load(folder / "background.lp", folder / "examples.lp", folder / "bias.lp")

    def check_all(self, task):
        oracle = CoverageOracle(task)
        for head in task.heads:
            for size in range(len(task.literals) + 1):
                for body in combinations(range(len(task.literals)), size):
                    used = set().union(*(variables(task.literals[i]) for i in body)) if body else set()
                    if not variables(head) <= used:
                        continue
                    model = unique_model(task.background + "\n" + task.rule(head, body))
                    for possible in range(1 << len(oracle.positives)):
                        labels = [atom for atom, position in oracle.positives.items()
                                  if possible >> position & 1]
                        expected = any(atom in model for atom in labels)
                        fixed = any(oracle.exists(head, body, atom) for atom in labels)
                        with self.subTest(head=str(head), body=body, possible=possible):
                            self.assertEqual(fixed, expected)
                            self.assertEqual(has_positive(oracle, head, body, possible), expected)
        return oracle

    def test_correlated_witnesses_and_disconnected_hidden_variables(self):
        task = self.task("p(a,x). q(a,y). p(b,x). q(b,x). mark(z).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))", "mark(var__(3))"],
                         ["answer(a)", "answer(b)", "answer(missing)"])
        oracle = self.check_all(task)
        body = tuple(i for i, literal in enumerate(task.literals) if literal.name != "mark")
        mask = 1 << oracle.positives[clingo.parse_term("answer(a)")]
        self.assertFalse(has_positive(oracle, task.heads[0], body, mask))

    def test_nested_repeated_head_slots_and_fixed_constants(self):
        self.check_all(self.task("p(a,x). p(b,y). mark(x). mark(y).",
            ["p(var__(1),var__(2))", "mark(var__(2))"],
            ["answer(pair(a,x),a,ok)", "answer(pair(a,x),b,ok)", "answer(pair(b,y),b,bad)"],
            ["answer(pair(var__(1),var__(2)),var__(1),ok)"]))

    def test_multiple_heads_use_global_mask_positions_without_reindexing(self):
        self.check_all(self.task("p(a). p(b). q(b).", ["p(var__(1))", "q(var__(1))"],
                                ["alpha(a)", "alpha(b)", "omega(a)", "omega(b)"],
                                ["alpha(var__(1))", "omega(var__(1))"]))

    def test_small_bk_relation_rejects_many_positives_without_fixed_queries(self):
        labels = [f"answer(a{i})" for i in range(200)]
        task = self.task("p(a0,z). p(unlabelled,z). q(y).",
                         ["p(var__(1),var__(2))", "q(var__(2))"], labels)
        oracle = CoverageOracle(task)
        def no_fixed_query(*_):
            self.fail("The semijoin must not issue one exists query per label")
        oracle.exists = no_fixed_query
        self.assertFalse(has_positive(oracle, task.heads[0], (0, 1), (1 << len(labels)) - 1))
        self.assertLessEqual(oracle.stats["positive_join_matched_atoms"], 3)
        self.assertEqual(oracle.stats.get("positive_join_label_steps", 0), 0)
        self.assertEqual(oracle.stats["exists_queries"], 0)

    def test_small_positive_relation_fixes_head_before_many_bk_outputs(self):
        task = self.task(" ".join(f"p(a{i},x)." for i in range(200)),
                         ["p(var__(1),var__(2))"], ["answer(a199)"])
        oracle = CoverageOracle(task)
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))
        self.assertEqual(oracle.stats["positive_join_label_steps"], 1)
        self.assertEqual(oracle.stats["positive_join_matched_atoms"], 1)
        self.assertEqual(oracle.stats["positive_join_index_builds"], 1)
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), 1))
        self.assertEqual(oracle.stats["positive_join_cache_hits"], 1)
        self.assertEqual(oracle.profile.export()["timings"]["oracle.positive_join"]["calls"], 2)

    def test_predictions_outside_labels_or_possible_mask_do_not_count(self):
        task = self.task("p(a). p(unlabelled).", ["p(var__(1))"], ["answer(a)", "answer(b)"])
        oracle = CoverageOracle(task)
        a = 1 << oracle.positives[clingo.parse_term("answer(a)")]
        b = 1 << oracle.positives[clingo.parse_term("answer(b)")]
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), a))
        self.assertFalse(has_positive(oracle, task.heads[0], (0,), b))
        self.assertFalse(has_positive(oracle, task.heads[0], (0,), 1 << 30))
        self.assertFalse(has_positive(oracle, task.heads[0], (0,), 0))

    def test_compiler_integration_preserves_complete_catalogue_and_ties(self):
        from nidavellir.relational_compile import _Search, compile_catalogue
        from tests.test_relational_compile import exhaustive_catalogue, normalized
        task = self.task("p(a,x). q(a,y). p(b,x). q(b,x). r(a). r(b). r(unlabelled).",
                         ["p(var__(1),var__(2))", "q(var__(1),var__(2))", "r(var__(1))"],
                         ["alpha(a)", "alpha(b)", "omega(a)", "omega(b)"],
                         ["alpha(var__(1))", "omega(var__(1))"])
        expected = exhaustive_catalogue(task)
        def integrated(search, head, body, possible):
            return has_positive(search.oracle, head, body, possible)
        with patch.object(_Search, "has_positive", integrated):
            for order in ("source", "selectivity"):
                for variants in ("exhaustive", "targeted"):
                    oracle = CoverageOracle(task)
                    result, stats = compile_catalogue(task, oracle, order=order, variants=variants)
                    self.assertTrue(stats["complete"])
                    self.assertEqual(normalized(result), expected)
                    self.assertGreater(oracle.stats["positive_join_queries"], 0)

    def test_ground_head_empty_body_and_missing_labels(self):
        self.check_all(self.task("p(a).", ["p(var__(1))"], ["success"], ["success"]))
        task = self.task("p(a).", ["p(var__(1))"], [], ["success"])
        self.assertFalse(has_positive(CoverageOracle(task), task.heads[0], (), 1))

    def test_deadlines_cover_index_building_cached_answers_and_traversal(self):
        labels = [f"answer(a{i})" for i in range(300)]
        task = self.task("p(a0).", ["p(var__(1))"], labels)
        oracle = CoverageOracle(task)
        calls = 0
        def fail_after_two():
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise TimeoutError("test deadline")
        oracle.check = fail_after_two
        with self.assertRaises(TimeoutError):
            has_positive(oracle, task.heads[0], (0,), (1 << len(labels)) - 1)
        self.assertFalse(oracle._positive_join_relations)
        oracle.check = lambda: None
        full = (1 << len(labels)) - 1
        self.assertTrue(has_positive(oracle, task.heads[0], (0,), full))
        oracle.check = lambda: (_ for _ in ()).throw(TimeoutError("cached deadline"))
        with self.assertRaises(TimeoutError):
            has_positive(oracle, task.heads[0], (0,), full)
        oracle.check = fail_after_two
        calls = 0
        with self.assertRaises(TimeoutError):
            has_positive(oracle, task.heads[0], (0,), full & ~1)
        self.assertNotIn((task.heads[0], (0,), full & ~1), oracle._positive_join_results)

    def test_unsafe_heads_and_invalid_masks_are_explicit_errors(self):
        task = self.task("p(a).", ["p(var__(1))"], ["answer(a)"])
        oracle = CoverageOracle(task)
        with self.assertRaises(ValueError):
            has_positive(oracle, task.heads[0], (), 1)
        for possible in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                has_positive(oracle, task.heads[0], (0,), possible)
        with self.assertRaises(ValueError):
            has_positive(oracle, task.heads[0], (-1,), 1)


if __name__ == "__main__":
    unittest.main()
