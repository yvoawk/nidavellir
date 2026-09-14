"""Cross-task profile sharing must never reuse old head names or polarities."""

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import clingo

from nidavellir.optimized_oracle import build_optimized_indices
from nidavellir.oracle import ResourceLimit, build_cartesian_indices
from nidavellir.oracle_cache import (fingerprint, profile_fingerprint,
                                            save_profile_cache, load_profile_cache)
from nidavellir.task import Task


class ProfileCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.counter = 0

    def task(self, name="t", positive=("a",), negative=("b", "c"), *,
             background="p(a,x). p(a,y). p(b,z). p(c,w). q(x). q(z).",
             extra="", shape="var__(1)", literal_source=None):
        self.counter += 1
        directory = self.root / str(self.counter)
        directory.mkdir()
        (directory / "bk.lp").write_text(background)
        (directory / "examples.lp").write_text(
            "\n".join(f"positive({name}({args}))." for args in positive) + "\n" +
            "\n".join(f"negative({name}({args}))." for args in negative))
        (directory / "bias.lp").write_text(
            f"candidate_head({name}({shape})). " + (literal_source or
            "allowed_literal(p(var__(1),var__(2))). allowed_literal(q(var__(2))). ") +
            "max_body(2). 1 {head(H):candidate_head(H)} 1. "
            "0 {body(L):allowed_literal(L)} 2. " + extra)
        return Task.load(directory / "bk.lp", directory / "examples.lp", directory / "bias.lp")

    def compare_every_legal_body(self, task, restored):
        fresh, _ = build_optimized_indices(task)
        reference, _ = build_cartesian_indices(task)
        for size in range(len(task.literals) + 1):
            for body in itertools.combinations(range(len(task.literals)), size):
                if not task.accepts(task.heads[0], body):
                    continue
                observations = []
                for indices in (restored, fresh, reference):
                    index, = indices
                    mask = index.body_mask(body)
                    observations.append((index.coverage(mask), bool(mask & index.negative_mask)))
                self.assertEqual(observations[0], observations[1], body)
                self.assertEqual(observations[0], observations[2], body)

    def test_renamed_head_and_reversed_polarities_rebuild_all_masks(self):
        source = self.task()
        target = self.task("u", positive=("b", "c"), negative=("a",), extra="% equivalent bias\n")
        key = profile_fingerprint(source)
        self.assertEqual(key, profile_fingerprint(target))
        self.assertNotEqual(fingerprint(source), fingerprint(target))
        indices, stats = build_optimized_indices(source)
        self.assertIsNone(load_profile_cache(self.root, key, target))
        save_profile_cache(self.root, key, source, indices, stats)
        restored, result = load_profile_cache(self.root, key, target)
        self.assertTrue(result["cache"]["hit"])
        self.assertEqual(result["cache"]["scope"], "unlabelled_profiles")
        self.assertEqual(restored[0].head, target.heads[0])
        self.assertTrue(all(row[0].name == "u" for row in restored[0].classes.rows))
        self.assertEqual(restored[0].positive_mask, indices[0].negative_mask)
        self.assertEqual(restored[0].negative_mask, indices[0].positive_mask)
        self.assertEqual({i for i, _ in restored[0].positive_heads}, {0, 1})
        self.compare_every_legal_body(target, restored)

    def test_same_polarity_renaming_and_equivalent_bias_text_share(self):
        source = self.task()
        target = self.task("renamed", extra="redundant_meta_fact.\n")
        key = profile_fingerprint(source)
        self.assertEqual(key, profile_fingerprint(target))
        indices, stats = build_optimized_indices(source)
        save_profile_cache(self.root, key, source, indices, stats)
        restored, _ = load_profile_cache(self.root, key, target)
        self.compare_every_legal_body(target, restored)

    def test_contract_changes_invalidate_and_cannot_load_with_old_key(self):
        source = self.task()
        key = profile_fingerprint(source)
        variants = [
            self.task(background="p(a,x). p(a,y). p(b,z). p(c,w). q(x)."),
            self.task(negative=("b", "c", "d")),
            self.task(extra=":- body(p(var__(1),var__(2))), not body(q(var__(2)))."),
            self.task(extra=":- body(q(var__(2)))."),
            self.task("t", positive=("a,a",), negative=("b,b", "c,c"), shape="var__(1),var__(1)"),
            dataclasses.replace(source, literals=tuple(reversed(source.literals))),
        ]
        for changed in variants:
            with self.subTest(heads=changed.heads, literals=changed.literals, bias=changed.bias_source):
                self.assertNotEqual(key, profile_fingerprint(changed))
                with self.assertRaisesRegex(ValueError, "contract"):
                    load_profile_cache(self.root, key, changed)

    def test_nested_head_arguments_constants_and_private_variables(self):
        kwargs = {"shape": "pair(var__(1),var__(1)),tag", "positive": ("pair(a,a),tag",),
                  "negative": ("pair(b,b),tag", "pair(c,c),tag"),
                  "literal_source": "allowed_literal(p(var__(1),var__(2))). "}
        source = self.task(**kwargs)
        target = self.task("nested", **dict(kwargs, positive=("pair(b,b),tag",),
                                           negative=("pair(a,a),tag", "pair(c,c),tag")))
        key = profile_fingerprint(source)
        self.assertEqual(key, profile_fingerprint(target))
        indices, stats = build_optimized_indices(source)
        self.assertIn(2, source._reduction_plans[source.heads[0]].private)
        save_profile_cache(self.root, key, source, indices, stats)
        restored, _ = load_profile_cache(self.root, key, target)
        self.compare_every_legal_body(target, restored)

    def test_functionally_eliminated_guard_output_is_reused_exactly(self):
        background = "p(a,x). p(b,y). p(c,z). q(x). q(z)."
        source = self.task(background=background)
        target = self.task("fd", positive=("c",), negative=("a", "b"), background=background)
        key = profile_fingerprint(source)
        self.assertEqual(key, profile_fingerprint(target))
        self.assertTrue(source._reduction_plans[source.heads[0]].functional)
        indices, stats = build_optimized_indices(source)
        save_profile_cache(self.root, key, source, indices, stats)
        restored, _ = load_profile_cache(self.root, key, target)
        self.compare_every_legal_body(target, restored)

    def test_empty_background_and_zero_positive_remapping(self):
        literals = "allowed_literal(p(var__(1),var__(2))). allowed_literal(r(var__(1),var__(2))). "
        source = self.task(background="", literal_source=literals)
        target = self.task("empty", positive=(), negative=("a", "b", "c"),
                           background="", literal_source=literals)
        key = profile_fingerprint(source)
        self.assertEqual(key, profile_fingerprint(target))
        indices, stats = build_optimized_indices(source)
        save_profile_cache(self.root, key, source, indices, stats)
        restored, _ = load_profile_cache(self.root, key, target)
        self.assertEqual(restored[0].positive_mask, 0)
        self.assertEqual(restored[0].positive_heads, ())
        self.compare_every_legal_body(target, restored)

    def test_partial_corrupt_and_structurally_invalid_profiles_rejected(self):
        task = self.task()
        key = profile_fingerprint(task)
        indices, stats = build_optimized_indices(task)
        with self.assertRaisesRegex(ValueError, "complete"):
            save_profile_cache(self.root, key, task, indices, dict(stats, complete=False))
        path = save_profile_cache(self.root, key, task, indices, stats)
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "interrupted"):
            load_profile_cache(self.root, key, task)
        path.write_text(original.replace('"source_head":', '"changed_head":'))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_profile_cache(self.root, key, task)
        records = [json.loads(line) for line in original.splitlines()[:-1]]
        records[1]["arguments"] = ["unlabelled"]
        content = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records)
        path.write_text(content + json.dumps({"type": "complete", "sha256": hashlib.sha256(content.encode()).hexdigest()}) + "\n")
        with self.assertRaisesRegex(ValueError, "outside"):
            load_profile_cache(self.root, key, task)

    def test_deadline_and_unsupported_backends_are_explicit(self):
        task = self.task()
        self.assertIsNone(profile_fingerprint(task, backend="projected"))
        self.assertIsNone(profile_fingerprint(dataclasses.replace(task, heads=(*task.heads, clingo.parse_term("other(var__(1))")))))
        def stop():
            raise ResourceLimit("test deadline")
        with self.assertRaises(ResourceLimit):
            profile_fingerprint(task, check=stop)

    def test_cli_uses_profile_hit_then_exact_task_hit(self):
        self.task(background="p(a,x). p(b,y). p(c,z). q(x).")
        self.task("renamed", background="p(a,x). p(b,y). p(c,z). q(x).")
        scopes = []
        for number, source in enumerate((self.root / "1", self.root / "2", self.root / "2")):
            output = self.root / f"run{number}"
            result = subprocess.run([sys.executable, "-m", "nidavellir",
                "--background", str(source / "bk.lp"), "--examples", str(source / "examples.lp"),
                "--bias", str(source / "bias.lp"), "--output", str(output),
                "--compiler-backend", "bitmask", "--bound-strategy", "off",
                "--bound-seconds", "0", "--oracle-cache", str(self.root / "cache")],
                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            stats = json.loads((output / "oracle.json").read_text())["stats"]
            scopes.append((stats["cache"]["hit"], stats["cache"]["scope"]))
            self.assertGreaterEqual(stats["phase_elapsed_seconds"], stats["elapsed_seconds"])
            self.assertTrue(json.loads((output / "verification.json").read_text())["matched"])
            self.assertEqual(json.loads((output / "status.json").read_text())["optimal_cost"], 3)
        self.assertEqual(scopes, [(False, "exact"), (True, "unlabelled_profiles"), (True, "exact")])


if __name__ == "__main__":
    unittest.main()
