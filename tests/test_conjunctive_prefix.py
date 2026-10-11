"""Differential checks for mixed cardinality integrity constraints."""
from itertools import product
from pathlib import Path
import tempfile
import unittest

from nidavellir.bias_analysis import analyze
from nidavellir.task import Task


class ConjunctivePrefixTests(unittest.TestCase):
    def test_all_partial_assignments_against_clingo(self):
        restrictions = [
            "used :- body(p). used :- body(q). bound :- body(r). bound :- body(s). :- used, not bound.",
            ":- 2 {body(p); body(q); body(r)}, not body(s).",
            ":- not 2 {body(p); not body(q); body(r)}, body(s).",
            ":- 1 {body(p); body(q)}, 1 {body(r); body(s)}.",
        ]
        for restriction in restrictions:
            with self.subTest(restriction=restriction), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "bk.lp").write_text("p. q. r. s.")
                (root / "ex.lp").write_text("positive(answer).")
                (root / "bias.lp").write_text(
                    "candidate_head(answer). head(answer). max_body(4). "
                    "allowed_literal(p;q;r;s). {body(L)} :- allowed_literal(L). " + restriction)
                task = Task.load(root / "bk.lp", root / "ex.lp", root / "bias.lp")
                proof = analyze(task).heads[task.heads[0]]
                filter_ = proof.prefix_filter
                self.assertTrue(filter_.all_constraints_compiled)
                self.assertTrue(filter_.conjunctions)
                self.assertEqual(filter_.deletion_monotone, 0)
                for states in product((-1, 0, 1), repeat=4):
                    selected = sum(1 << i for i, state in enumerate(states) if state == 1)
                    excluded = sum(1 << i for i, state in enumerate(states) if state == -1)
                    included_ids = [i for i in range(4) if selected >> i & 1]
                    excluded_ids = [i for i in range(4) if excluded >> i & 1]
                    possible = proof.possible_completion(included_ids, excluded_ids)
                    allowed = filter_.allows(selected, excluded)
                    if possible:
                        self.assertTrue(allowed, states)
                    if 0 not in states:
                        self.assertEqual(allowed, possible, states)


if __name__ == "__main__":
    unittest.main()
