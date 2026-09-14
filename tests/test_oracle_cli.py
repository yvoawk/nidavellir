"""Check backend selection and the distinct public resource limits."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "overlapping_rules"


class OracleCommandLineTests(unittest.TestCase):
    def invoke(self, output, *options):
        command = [sys.executable, "-m", "nidavellir",
                   "--background", str(FIXTURE / "background.lp"),
                   "--examples", str(FIXTURE / "examples.lp"),
                   "--bias", str(FIXTURE / "bias.lp"),
                   "--output", str(output), "--bound-seconds", "0", *map(str, options)]
        return subprocess.run(command, cwd=REPOSITORY, text=True,
                              capture_output=True, timeout=45)

    def test_default_optimized_oracle_ignores_cartesian_resource_limit(self):
        with tempfile.TemporaryDirectory(prefix="nidavellir-projected-cli-") as directory:
            output = Path(directory) / "run"
            result = self.invoke(output, "--max-witnesses", 1)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["backend"], "guarded_reduced_bitmasks")
            self.assertEqual(status["optimal_cost"], 4)
            oracle = json.loads((output / "oracle.json").read_text())
            self.assertEqual(oracle["stats"]["backend"], status["backend"])
            self.assertTrue(all(c["multiplicity"] is None
                                for index in oracle["indices"] for c in index["classes"]))

    def test_projected_reference_remains_explicitly_available(self):
        with tempfile.TemporaryDirectory(prefix="nidavellir-projected-cli-") as directory:
            output = Path(directory) / "run"
            result = self.invoke(output, "--oracle-backend", "projected")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["backend"], "projected_truth_vector_bitmasks")
            self.assertEqual(status["optimal_cost"], 4)

    def test_cartesian_reference_is_explicit_and_keeps_same_optimum(self):
        with tempfile.TemporaryDirectory(prefix="nidavellir-cartesian-cli-") as directory:
            output = Path(directory) / "run"
            result = self.invoke(output, "--oracle-backend", "cartesian", "--max-classes", 1)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["backend"], "cartesian_reference_bitmasks")
            self.assertEqual(status["optimal_cost"], 4)
            oracle = json.loads((output / "oracle.json").read_text())
            self.assertEqual(oracle["stats"]["backend"], status["backend"])
            self.assertTrue(all(c["multiplicity"] >= 1
                                for index in oracle["indices"] for c in index["classes"]))

    def test_projection_class_limit_persists_incomplete_status(self):
        with tempfile.TemporaryDirectory(prefix="nidavellir-class-limit-cli-") as directory:
            output = Path(directory) / "run"
            result = self.invoke(output, "--max-classes", 1)
            self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "incomplete")
            self.assertEqual(status["stage"], "oracle")
            self.assertIsNone(status["optimal_cost"])
            self.assertTrue((output / "resource_limit.json").is_file())
            self.assertFalse((output / "learned.lp").exists())


if __name__ == "__main__":
    unittest.main()
