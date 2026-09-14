import dataclasses
from pathlib import Path
import tempfile
import unittest

import clingo

from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.oracle_cache import fingerprint, load_cache, save_cache
from nidavellir.task import Task


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "bk.lp").write_text("p(a,x). p(a,y). p(b,z). q(x).")
        (self.root / "examples.lp").write_text("positive(t(a)). negative(t(b)).")
        (self.root / "bias.lp").write_text(
            "candidate_head(t(var__(1))). allowed_literal(p(var__(1),var__(2))). "
            "allowed_literal(q(var__(2))). max_body(2). "
            "1 {head(H):candidate_head(H)} 1. 0 {body(L):allowed_literal(L)} 2.")
        self.task = Task.load(self.root / "bk.lp", self.root / "examples.lp", self.root / "bias.lp")

    def test_round_trip_masks_and_compact_diagnostics(self):
        indices, stats = build_optimized_indices(self.task)
        key = fingerprint(self.task)
        self.assertIsNone(load_cache(self.root, key, self.task))
        save_cache(self.root, key, indices, stats)
        restored, other = load_cache(self.root, key, self.task)
        self.assertTrue(other["cache"]["hit"])
        for a, b in zip(indices, restored):
            self.assertEqual(a.literal_masks, b.literal_masks)
            self.assertEqual(a.positive_heads, b.positive_heads)
            self.assertEqual(a.negative_mask, b.negative_mask)
            self.assertEqual(list(a.classes), list(b.classes))

    def test_mutated_task_or_options_invalidate_cache(self):
        original = fingerprint(self.task)
        variants = [dataclasses.replace(self.task, background=self.task.background + "r.\n"),
                    dataclasses.replace(self.task, bias_source=self.task.bias_source + "% changed\n"),
                    dataclasses.replace(self.task, negatives=frozenset({clingo.parse_term("t(c)")})),
                    dataclasses.replace(self.task, literals=tuple(reversed(self.task.literals)))]
        for task in variants:
            self.assertNotEqual(original, fingerprint(task))
        self.assertNotEqual(original, fingerprint(self.task, backend="projected"))
        self.assertNotEqual(original, fingerprint(self.task, parameters={"body_limit": 1}))

    def test_partial_or_corrupt_cache_rejected(self):
        indices, stats = build_optimized_indices(self.task)
        key = fingerprint(self.task)
        with self.assertRaises(ValueError):
            save_cache(self.root, key, indices, dict(stats, complete=False))
        path = save_cache(self.root, key, indices, stats)
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "interrupted"):
            load_cache(self.root, key, self.task)
        path.write_text(original.replace('"multiplicities"', '"changed"').replace('"encoded_relation_rows":', '"altered_rows":'))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_cache(self.root, key, self.task)


if __name__ == "__main__":
    unittest.main()
