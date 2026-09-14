"""Exercise the public command, persisted proofs, and incomplete outcomes."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parents[1] / "examples"
ILASP = shutil.which("ILASP") or str(Path(sys.executable).parent / "ILASP")


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="nidavellir-cli-test-")
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)

    def run_compiler(self, *options, fixture="overlapping_rules", background=None):
        source = FIXTURES / fixture
        output = self.directory / "run"
        command = [sys.executable, "-m", "nidavellir",
                   "--background", str(background or source / "background.lp"),
                   "--examples", str(source / "examples.lp"),
                   "--bias", str(source / "bias.lp"),
                   "--output", str(output), *map(str, options)]
        process = subprocess.run(command, cwd=REPOSITORY, text=True,
                                 capture_output=True, timeout=45)
        self.assertTrue((output / "status.json").is_file(),
                        f"No persisted status: {process.stdout}\n{process.stderr}")
        status = self.read(output, "status.json")
        self.assertEqual(json.loads(process.stdout), status,
                         "The printed status must match the persisted result")
        self.assertTrue((output / "report.md").is_file())
        return process, output, status

    @staticmethod
    def read(output, name):
        return json.loads((output / name).read_text())

    def assert_verified(self, output):
        self.assertTrue((output / "learned.lp").is_file())
        self.assertTrue((output / "predictions.lp").is_file())
        check = self.read(output, "verification.json")
        self.assertTrue(check["matched"], check)
        self.assertEqual(check["missing_positives"], [])
        self.assertEqual(check["covered_negatives"], [])

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_internal_multirule_optimum_with_requested_ilasp_verification(self):
        process, output, status = self.run_compiler(
            "--verify-ilasp", "--ilasp-binary", ILASP)
        self.assertEqual(process.returncode, 0, status)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["optimal_cost"], 4)
        self.assertEqual(status["learned_rule_count"], 2)
        self.assert_verified(output)
        selection = self.read(output, "solver.json")
        self.assertEqual(selection["backend"], "internal")
        self.assertTrue(selection["verification"]["matches"])
        self.assertEqual(selection["verification"]["optimum_cost"], 4)
        bound = self.read(output, "bound.json")
        self.assertEqual(bound["status"], "verified")
        self.assertEqual(bound["cost"], 4)
        self.assertEqual(len(bound["rules"]), 2)
        for name in ("inputs/background.lp", "inputs/examples.lp", "inputs/bias.lp",
                     "manifest.json", "oracle.json", "catalogue.json",
                     "compiled_set_cover.lp", "compiled_ilasp.las",
                     "first_feasible.lp", "trace.jsonl"):
            self.assertTrue((output / name).is_file(), name)

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_ilasp_primary_optimizer_on_temporal_join_fixture(self):
        process, output, status = self.run_compiler(
            "--optimizer", "ilasp", "--ilasp-binary", ILASP,
            fixture="temporal_rise")
        self.assertEqual(process.returncode, 0, status)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["optimal_cost"], 5)
        self.assertEqual(status["learned_rule_count"], 1)
        self.assertEqual(self.read(output, "solver.json")["backend"], "ilasp")
        self.assert_verified(output)

    def test_disabled_bound_retains_complete_original_search(self):
        process, output, status = self.run_compiler("--bound-seconds", 0)
        self.assertEqual(process.returncode, 0, status)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["optimal_cost"], 4)
        self.assertEqual(self.read(output, "bound.json")["status"], "disabled")
        self.assertFalse((output / "first_feasible.lp").exists())
        self.assert_verified(output)

    def test_hard_bound_deadline_falls_back_without_claiming_infeasibility(self):
        process, output, status = self.run_compiler("--bound-seconds", 1e-12)
        self.assertEqual(process.returncode, 0, status)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["optimal_cost"], 4)
        bound = self.read(output, "bound.json")
        self.assertEqual(bound["status"], "timeout")
        self.assertIsNone(bound["cost"])
        self.assertEqual(self.read(output, "catalogue.json")["stats"]["effective_body_limit"], 2)
        self.assert_verified(output)

    def test_oracle_resource_limit_is_incomplete_not_unsatisfiable(self):
        process, output, status = self.run_compiler("--oracle-backend", "cartesian", "--max-witnesses", 1)
        self.assertEqual(process.returncode, 3, status)
        self.assertEqual(status["status"], "incomplete")
        self.assertEqual(status["stage"], "oracle")
        self.assertIsNone(status["optimal_cost"])
        self.assertEqual(self.read(output, "resource_limit.json")["stage"], "oracle")
        self.assertFalse((output / "learned.lp").exists())
        self.assertFalse((output / "solver.json").exists())

    def test_compilation_timeout_preserves_verified_multirule_bound(self):
        process, output, status = self.run_compiler("--compile-seconds", 1e-12)
        self.assertEqual(process.returncode, 3, status)
        self.assertEqual(status["status"], "feasible")
        self.assertEqual(status["stage"], "compilation")
        self.assertEqual(status["hypothesis_cost"], 4)
        self.assertIsNone(status["optimal_cost"])
        self.assertEqual(self.read(output, "bound.json")["status"], "verified")
        self.assertEqual(self.read(output, "resource_limit.json")["stage"], "compilation")
        self.assertTrue((output / "first_feasible.lp").is_file())
        self.assertFalse((output / "solver.json").exists())
        self.assert_verified(output)

    def test_unavailable_requested_verifier_keeps_internal_proof_explicit(self):
        missing_binary = self.directory / "missing-ILASP"
        process, output, status = self.run_compiler(
            "--verify-ilasp", "--ilasp-binary", missing_binary)
        self.assertEqual(process.returncode, 3, status)
        self.assertEqual(status["status"], "verification_incomplete")
        self.assertEqual(status["optimal_cost"], 4)
        solver = self.read(output, "solver.json")
        self.assertEqual(solver["status"], "complete")
        self.assertEqual(solver["verification"]["status"], "error")
        self.assertIsNone(solver["verification"]["matches"])
        self.assert_verified(output)

    def test_unavailable_verifier_on_unsatisfiable_task_is_explicit(self):
        background = self.directory / "uncoverable-background.lp"
        background.write_text("p(d). q(e). r(d).\n")
        process, output, status = self.run_compiler(
            "--bound-seconds", 0, "--verify-ilasp",
            "--ilasp-binary", self.directory / "missing-ILASP",
            background=background)
        self.assertEqual(process.returncode, 3, status)
        self.assertEqual(status["status"], "verification_incomplete")
        self.assertEqual(status["selection_status"], "unsatisfiable")
        self.assertEqual(status["ilasp_verification"], "error")
        self.assertIsNone(status["optimal_cost"])
        selection = self.read(output, "solver.json")
        self.assertEqual(selection["status"], "unsatisfiable")
        self.assertEqual(selection["verification"]["status"], "error")
        self.assertIsNone(selection["verification"]["matches"])
        self.assertFalse((output / "learned.lp").exists())

    def test_ilasp_primary_cannot_request_internal_optimizer_verification(self):
        source = FIXTURES / "overlapping_rules"
        output = self.directory / "invalid-options"
        command = [sys.executable, "-m", "nidavellir",
                   "--background", str(source / "background.lp"),
                   "--examples", str(source / "examples.lp"),
                   "--bias", str(source / "bias.lp"),
                   "--output", str(output),
                   "--optimizer", "ilasp", "--verify-ilasp"]
        process = subprocess.run(command, cwd=REPOSITORY, text=True,
                                 capture_output=True, timeout=45)
        self.assertEqual(process.returncode, 2)
        self.assertIn("--verify-ilasp", process.stderr)
        self.assertIn("internal", process.stderr)
        self.assertFalse(output.exists(), "Invalid options must fail before starting a run")

    def test_rejected_target_in_background_persists_invalid_input(self):
        background = self.directory / "invalid-background.lp"
        background.write_text("p(a). t(a).\n")
        process, output, status = self.run_compiler(background=background)
        self.assertEqual(process.returncode, 2, status)
        self.assertEqual(status["status"], "invalid_input")
        self.assertEqual(status["stage"], "inputs")
        self.assertIsNone(status["optimal_cost"])
        self.assertIn("target", status["reason"].lower())
        self.assertFalse((output / "learned.lp").exists())


if __name__ == "__main__":
    unittest.main()
