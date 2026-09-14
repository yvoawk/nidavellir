"""Integration of automatic planning, checkpoints, caches and variant output."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parents[1] / "examples/overlapping_rules"


class ExtendedCLITests(unittest.TestCase):
    def test_version_reports_current_version_and_author_without_inputs(self):
        from nidavellir import __version__
        process = subprocess.run([sys.executable, "-m", "nidavellir", "--version"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout,
                         f"Nidavellir {__version__} - Author: Yvon K. Awuklu - "
                         "Copyright (c) 2026\n")
        self.assertEqual(process.stderr, "")

    def test_bound_background_replay_and_snapshot_preserve_verified_cover(self):
        from nidavellir.__main__ import bound_in_subprocess
        from nidavellir.task import Task
        task = Task.load(FIXTURE / "background.lp", FIXTURE / "examples.lp", FIXTURE / "bias.lp")
        for transfer in ("snapshot", "replay"):
            with self.subTest(transfer=transfer):
                cover, report = bound_in_subprocess(task, [], 5, strategy="relational",
                                                   background_transfer=transfer)
                self.assertEqual(report["status"], "verified")
                self.assertEqual(report["cost"], 4)
                self.assertTrue(report["verification"]["matched"])
                self.assertEqual(len(cover), 2)
                self.assertEqual(report["background_transfer"], transfer)

    def test_expanded_background_uses_native_replay_in_bound_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / "inputs"
            inputs.mkdir()
            for name in ("background.lp", "examples.lp", "bias.lp"):
                text = (FIXTURE / name).read_text()
                if name == "background.lp":
                    text += "\nirrelevant(1..1000).\n"
                (inputs / name).write_text(text)
            process, output, status = self.invoke(folder, "run", "--compiler-backend", "relational", inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(status["optimal_cost"], 4)
            report = json.loads((output / "bound.json").read_text())
            self.assertEqual(report["background_transfer"], "replay")
            self.assertFalse((output / "inputs/background_model.lp").exists())

    def invoke(self, directory, name, *extra, inputs=FIXTURE):
        output = Path(directory) / name
        command = [sys.executable, "-m", "nidavellir", "--background", str(inputs / "background.lp"),
                   "--examples", str(inputs / "examples.lp"), "--bias", str(inputs / "bias.lp"),
                   "--output", str(output), "--bound-seconds", "5", *map(str, extra)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertTrue((output / "status.json").exists(), process.stderr + process.stdout)
        return process, output, json.loads((output / "status.json").read_text())

    def test_auto_selection_and_complete_targeted_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            process, output, status = self.invoke(folder, "run", "--enumerate-optima", "--expand-variants")
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(status["compiler_backend"], "bitmask")
            self.assertEqual(status["optimal_cost"], 4)
            planning = json.loads((output / "planning.json").read_text())
            self.assertEqual(planning["method"], "remaining_hidden_variable_count")
            variants = json.loads((output / "variants.json").read_text())
            self.assertTrue(variants["all_optimal_programs_complete"])
            self.assertEqual(variants["reconstruction"]["variant_scope"], "provided_cost_coverage_pairs")
            self.assertTrue((output / "variant_catalogue.json").exists())

    def test_explicit_relational_compilation_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            first, output, status = self.invoke(folder, "first", "--compiler-backend", "relational",
                                                "--enumerate-optima", "--expand-variants")
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertEqual(status["bound_strategy"], "relational")
            self.assertEqual(status["optimal_cost"], 4)
            self.assertTrue((output / "inputs/background_model.lp").exists())
            checkpoint = output / "bound_checkpoint.json"
            self.assertTrue(checkpoint.exists())
            again, resumed, status = self.invoke(folder, "resumed", "--compiler-backend", "relational",
                "--resume-bound", checkpoint)
            self.assertEqual(again.returncode, 0, again.stderr + again.stdout)
            bound = json.loads((resumed / "bound.json").read_text())
            self.assertTrue(bound["resumed"])
            self.assertEqual(status["optimal_cost"], 4)

    def test_completed_oracle_cache_reloads_same_masks_and_optimum(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache"
            first, output, status = self.invoke(folder, "first", "--oracle-cache", cache)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            original = json.loads((output / "oracle.json").read_text())
            self.assertFalse(original["stats"]["cache"]["hit"])
            again, reused, status = self.invoke(folder, "reused", "--oracle-cache", cache)
            self.assertEqual(again.returncode, 0, again.stderr + again.stdout)
            loaded = json.loads((reused / "oracle.json").read_text())
            self.assertTrue(loaded["stats"]["cache"]["hit"])
            self.assertEqual(loaded["indices"], original["indices"])
            self.assertEqual(status["optimal_cost"], 4)

    def test_incomplete_semantic_enumeration_retains_proven_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = Path(folder) / "inputs"
            inputs.mkdir()
            (inputs / "background.lp").write_text("p(a). p(b). q(b). q(c). r(a). r(c).")
            (inputs / "examples.lp").write_text("positive(t(a)). positive(t(b)). positive(t(c)). negative(t(d)).")
            (inputs / "bias.lp").write_text("candidate_head(t(var__(1))). allowed_literal(p(var__(1))).\n"
                "allowed_literal(q(var__(1))). allowed_literal(r(var__(1))). max_body(1).\n"
                "head(H) :- candidate_head(H). 0 {body(L):allowed_literal(L)} 1.")
            process, output, status = self.invoke(folder, "run", "--enumerate-optima", "--max-optima", 1,
                                                  "--expand-variants", inputs=inputs)
            self.assertEqual(process.returncode, 3, process.stderr + process.stdout)
            self.assertEqual(status["status"], "enumeration_incomplete")
            self.assertEqual(status["optimal_cost"], 4)
            self.assertTrue((output / "learned.lp").exists())
            self.assertFalse(json.loads((output / "variants.json").read_text())["all_optimal_programs_complete"])

    def test_timed_out_targeted_reconstruction_keeps_optimum(self):
        with tempfile.TemporaryDirectory() as folder:
            process, output, status = self.invoke(folder, "run", "--enumerate-optima", "--expand-variants",
                                                  "--variant-seconds", "1e-12")
            self.assertEqual(process.returncode, 3, process.stderr + process.stdout)
            self.assertEqual(status["optimal_cost"], 4)
            self.assertEqual(status["status"], "enumeration_incomplete")
            self.assertTrue((output / "learned.lp").exists())


if __name__ == "__main__":
    unittest.main()
