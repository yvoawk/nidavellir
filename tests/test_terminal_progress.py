"""Terminal progress stays readable while stdout remains machine-readable."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from nidavellir.terminal_progress import TerminalProgress
from tests import test_cli_extended as cli
from tests import test_per_target_bounds as bounds
from tests import test_target_execution as targets


class TerminalProgressTests(unittest.TestCase):
    invoke = cli.ExtendedCLITests.invoke

    def test_reporter_formats_stages_and_search_counters(self):
        stream = io.StringIO()
        progress = TerminalProgress(stream=stream, interval=0)
        progress.emit({"event": "status", "active_target": "onset/1",
                       "stage": "compilation", "wall_seconds": 1.25})
        progress.emit({"nodes": 4000, "legal_safe_rules": 3,
                       "wall_seconds": 2.75})
        self.assertEqual(stream.getvalue().splitlines(), [
            "[Nidavellir] target=onset/1 stage=compilation elapsed=1.2s",
            "[Nidavellir] target=onset/1 stage=compilation elapsed=2.8s nodes=4000 safe_rules=3",
        ])

    def test_disabled_reporter_is_silent(self):
        stream = io.StringIO()
        TerminalProgress(False, stream=stream).emit(
            {"event": "status", "stage": "compilation", "wall_seconds": 1})
        self.assertEqual(stream.getvalue(), "")

    def test_heartbeat_keeps_a_blocking_phase_visible(self):
        stream = io.StringIO()
        progress = TerminalProgress(stream=stream, heartbeat_interval=0.01)
        progress.emit({"event": "status", "stage": "bound", "wall_seconds": 0})
        progress.start(time.monotonic())
        deadline = time.monotonic() + 0.5
        while len(stream.getvalue().splitlines()) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        progress.close()
        lines = stream.getvalue().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all("stage=bound" in line for line in lines))

    def test_single_target_progress_uses_stderr_and_preserves_json_stdout(self):
        with tempfile.TemporaryDirectory() as folder:
            process, _, status = self.invoke(folder, "run")
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(json.loads(process.stdout), status)
            self.assertIn("[Nidavellir] stage=planning", process.stderr)
            self.assertIn("stage=compilation", process.stderr)
            self.assertIn("status=complete", process.stderr)

    def test_quiet_option_and_sequential_target_relay(self):
        with tempfile.TemporaryDirectory() as folder:
            task = bounds.PerTargetBoundTests().sample()
            inputs = targets.TargetExecutionTests.inputs(self, folder, task)
            process, _, status = self.invoke(folder, "quiet", "--no-verbose", inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(process.stderr, "")
            self.assertEqual(json.loads(process.stdout), status)

        with tempfile.TemporaryDirectory() as folder:
            task = bounds.PerTargetBoundTests().sample()
            inputs = targets.TargetExecutionTests.inputs(self, folder, task)
            process, _, status = self.invoke(folder, "visible", inputs=inputs)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(json.loads(process.stdout), status)
            self.assertIn("target=first/1", process.stderr)
            self.assertIn("target=second/1", process.stderr)
            self.assertIn("stage=compilation", process.stderr)


if __name__ == "__main__":
    unittest.main()
