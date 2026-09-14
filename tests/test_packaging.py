"""Installed distribution and entry points work outside the repository."""
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nidavellir import __version__


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT / "examples" / "overlapping_rules"


class PackagingTests(unittest.TestCase):
    def test_distribution_metadata_matches_runtime_version(self):
        self.assertEqual(version("nidavellir"), __version__)

    def test_console_and_module_entry_points_work_from_arbitrary_cwd(self):
        console = Path(sys.executable).with_name("nidavellir")
        self.assertTrue(console.is_file(), "Install the package before running its test suite")
        expected = f"Nidavellir {__version__} - Author: Yvon K. Awuklu - Copyright (c) 2026\n"
        with tempfile.TemporaryDirectory() as directory:
            for command in ([str(console), "--version"],
                            [sys.executable, "-m", "nidavellir", "--version"]):
                process = subprocess.run(command, cwd=directory, text=True,
                                         capture_output=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stdout, expected)
                self.assertEqual(process.stderr, "")

    def test_installed_console_runs_example_from_arbitrary_cwd(self):
        console = Path(sys.executable).with_name("nidavellir")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            cwd = temporary / "unrelated"
            cwd.mkdir()
            output = temporary / "result"
            process = subprocess.run([
                str(console),
                "--background", str(EXAMPLE / "background.lp"),
                "--examples", str(EXAMPLE / "examples.lp"),
                "--bias", str(EXAMPLE / "bias.lp"),
                "--output", str(output),
                "--no-verbose",
            ], cwd=cwd, text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            status = json.loads(process.stdout)
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["optimal_cost"], 4)
            self.assertEqual(status["learned_rule_count"], 2)
            verification = json.loads((output / "verification.json").read_text())
            self.assertTrue(verification["matched"], verification)


if __name__ == "__main__":
    unittest.main()
