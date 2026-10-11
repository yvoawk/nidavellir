import json
import tempfile
import unittest
import time
import tracemalloc
from pathlib import Path

from nidavellir.__main__ import JsonArray, bound_in_subprocess, write_json
from nidavellir.oracle import MaskIndex
from nidavellir.projected_oracle import build_projected_indices
from nidavellir.task import Task, TaskError, read_ground_model


class StreamingIOTests(unittest.TestCase):
    def test_lazy_record_count_does_not_determine_peak_buffer_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            tracemalloc.start()
            try:
                write_json(path, JsonArray({"i": i, "text": "x" * 2000} for i in range(10000)))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertGreater(path.stat().st_size, 20_000_000)
            self.assertLess(peak, 4_000_000)

    def test_compact_bytes_unicode_and_atomic_failure(self):
        records = [{"name": "étude\n東京", "number": i / 3, "flag": True,
                    "nested": [None, {"x": i}]} for i in range(300)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            write_json(path, {"rows": JsonArray(iter(records))})
            expected = json.dumps({"rows": records}, ensure_ascii=False, separators=(",", ":")) + "\n"
            self.assertEqual(path.read_bytes(), expected.encode())
            with self.assertRaises(TimeoutError):
                write_json(path, {"rows": JsonArray(iter(records))}, deadline=time.monotonic() - 1)
            self.assertEqual(path.read_bytes(), expected.encode())
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_generator_failure_does_not_replace_target(self):
        def rows():
            yield {"x": 1}
            raise ValueError("interrupted input")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            path.write_text("previous\n")
            with self.assertRaisesRegex(ValueError, "interrupted input"):
                write_json(path, JsonArray(rows()))
            self.assertEqual(path.read_text(), "previous\n")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_ground_model_transfer_accepts_only_canonical_ground_atoms(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(folder)
            path = Path(folder) / "model.lp"
            path.write_text("".join(str(atom) + ".\n" for atom in task.atoms))
            self.assertEqual(read_ground_model(path), task.atoms)
            for text in ("p(X).\n", "p(a) :- q(a).\n", "#const n=2.\n", "p(1+2).\n"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(TaskError):
                    read_ground_model(path)

    def test_streams_custom_class_sequence_and_nested_generators(self):
        class Classes:
            def __iter__(self):
                for i in range(3):
                    yield {"truth": hex(i), "values": JsonArray(v for v in (i, i + 1))}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            write_json(path, {"classes": JsonArray(Classes())})
            value = json.loads(path.read_text())
            self.assertEqual(value["classes"], [{"truth": hex(i), "values": [i, i + 1]} for i in range(3)])

    def task(self, folder, bk="p(a).", examples="positive(t(a)).\nnegative(t(b)).\n"):
        folder = Path(folder)
        (folder / "background.lp").write_text(bk)
        (folder / "examples.lp").write_text(examples)
        (folder / "bias.lp").write_text("candidate_head(t(var__(1))). allowed_literal(p(var__(1))).\n"
                                       "max_body(1). head(H) :- candidate_head(H). {body(L)} :- allowed_literal(L).")
        return Task.load(folder / "background.lp", folder / "examples.lp", folder / "bias.lp")

    def test_worker_verifies_streamed_negatives_without_copying_model(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(folder)
            indices, _ = build_projected_indices(task)
            cover, report = bound_in_subprocess(task, indices, 5)
            self.assertIsNotNone(cover)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["verification"]["negative_labels_checked"], 1)
            self.assertEqual(report["verification"]["covered_negatives"], [])

    def test_worker_stream_check_rejects_an_incorrect_negative_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(folder, bk="p(a). p(b).")
            indices, _ = build_projected_indices(task)
            wrong = [MaskIndex(i.head, i.literal_masks, i.all_mask, i.positive_mask,
                               0, i.positive_heads, i.classes) for i in indices]
            with self.assertRaisesRegex(RuntimeError, "failed independent verification"):
                bound_in_subprocess(task, wrong, 5)

    def test_worker_preserves_legacy_normalization_for_constants(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(folder, examples="#const n=b. positive(t(a)). negative(t(n)).")
            indices, _ = build_projected_indices(task)
            cover, report = bound_in_subprocess(task, indices, 5)
            self.assertIsNotNone(cover)
            self.assertEqual(report["status"], "verified")

    def test_preparation_deadline_preserves_original_bias(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(folder)
            indices, _ = build_projected_indices(task)
            cover, report = bound_in_subprocess(task, indices, 1e-12)
            self.assertIsNone(cover)
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(task.max_body, 1)


if __name__ == "__main__":
    unittest.main()
