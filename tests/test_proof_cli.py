"""Exercise direct-proof routing, dynamic handoff and persisted guarantees."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ProgramProofCLITests(unittest.TestCase):
    def invoke(self, *flags, heads=("t",), empty_heads=()):
        temp = tempfile.TemporaryDirectory(prefix="nidavellir-proof-cli-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        # q covers b,c; p is unsafe. A different legal literal covers a,b safely.
        (root / "background.lp").write_text("p(a). p(b). q(b). q(c). good(a). good(b). p(n).\n")
        (root / "examples.lp").write_text("".join(
            ("" if head in empty_heads else
             f"positive({head}(a)). positive({head}(b)). positive({head}(c)). ")
            + f"negative({head}(n)).\n" for head in heads))
        (root / "bias.lp").write_text(
            "".join(f"candidate_head({head}(var__(1))).\n" for head in heads)
            + "allowed_literal(p(var__(1))). allowed_literal(q(var__(1))).\n"
              "allowed_literal(good(var__(1))). max_body(2).\n"
              "1 { head(H): candidate_head(H) } 1.\n"
              "1 { body(L): allowed_literal(L) } 2.\n#show head/1.\n#show body/1.\n")
        output = root / "run"
        command = [sys.executable, "-m", "nidavellir", "--background", str(root / "background.lp"),
                   "--examples", str(root / "examples.lp"), "--bias", str(root / "bias.lp"),
                   "--output", str(output), "--compiler-backend", "relational", "--no-verbose", *flags]
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        self.assertTrue((output / "status.json").exists(), result.stderr + result.stdout)
        return result, output, json.loads((output / "status.json").read_text())

    def assert_optimum(self, process, output, status, expected=4):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(status["status"], "complete", status)
        self.assertEqual(status["optimal_cost"], expected)
        verification = json.loads((output / "verification.json").read_text())
        self.assertTrue(verification["matched"])
        self.assertFalse(verification["missing_positives"])
        self.assertFalse(verification["covered_negatives"])

    def test_auto_strict_representative_proves_original_language(self):
        process, output, status = self.invoke()
        self.assert_optimum(process, output, status)
        proof = json.loads((output / "program_proof.json").read_text())
        self.assertTrue(proof["optimality_proven"])
        self.assertEqual(status["optimality_source"], "strict_improvement_unsat")
        self.assertFalse(status["catalogue_complete"])
        self.assertFalse((output / "catalogue.json").exists())

    def test_dynamic_first_cover_hands_off_to_direct_proof(self):
        process, output, status = self.invoke("--bound-seconds", "0")
        self.assert_optimum(process, output, status)
        certificate = json.loads((output / "dynamic_incumbent.json").read_text())
        self.assertTrue(certificate["verification"]["matched"])
        self.assertGreaterEqual(certificate["cost"], status["optimal_cost"])
        self.assertTrue(json.loads((output / "program_proof.json").read_text())["optimality_proven"])
        bound = json.loads((output / "bound.json").read_text())
        self.assertEqual(bound["status"], "verified")
        self.assertEqual(bound["cost"], status["optimal_cost"])
        self.assertTrue(bound["verification"]["matched"])
        self.assertEqual(bound["rules"], (output / "learned.lp").read_text().splitlines())

    def test_requested_catalogue_and_all_optima_keep_their_contract(self):
        for flags in [("--proof-strategy", "catalogue"),
                      ("--proof-strategy", "catalogue", "--no-dynamic-program-bounds"),
                      ("--enumerate-optima", "--expand-variants")]:
            with self.subTest(flags=flags):
                process, output, status = self.invoke(*flags)
                self.assert_optimum(process, output, status)
                self.assertTrue((output / "catalogue.json").exists())
                self.assertFalse((output / "program_proof.json").exists())
                if "--enumerate-optima" in flags:
                    self.assertTrue(status["enumeration_complete"])
                    self.assertTrue(status["variant_expansion_complete"])

    def test_proof_timeout_keeps_verified_incumbent_without_optimum(self):
        process, output, status = self.invoke("--compile-seconds", "0.000000000001")
        self.assertEqual(process.returncode, 3, process.stdout + process.stderr)
        self.assertEqual(status["status"], "feasible", status)
        self.assertIsNone(status["optimal_cost"])
        self.assertTrue(status["semantic_verification"])
        self.assertFalse(json.loads((output / "program_proof.json").read_text())["optimality_proven"])
        solver = json.loads((output / "solver.json").read_text())
        self.assertFalse(solver["optimality_proven"])
        self.assertIn("unfinished", solver["scope"])
        self.assertNotIn("proved absence", solver["scope"])
        self.assertNotIn("exhaustive rejection", solver["scope"])
        self.assertIsNone(status["optimality_source"])

    def test_native_unlimited_mode_records_real_absent_deadlines(self):
        process, output, status = self.invoke("--no-time-limit")
        self.assert_optimum(process, output, status)
        for key in ("oracle_seconds", "compile_seconds", "solver_seconds"):
            self.assertIsNone(status["arguments"][key])
        self.assertEqual(status["arguments"]["bound_seconds"], 30)

    def test_independent_target_proofs_compose(self):
        process, output, status = self.invoke(heads=("first", "second"))
        self.assert_optimum(process, output, status, expected=8)
        self.assertEqual(status["target_count"], 2)
        plan = json.loads((output / "targets.json").read_text())["targets"]
        for target in plan:
            proof = json.loads((output / target["output"] / "program_proof.json").read_text())
            self.assertTrue(proof["optimality_proven"])

    def test_unlimited_empty_positive_tasks_keep_all_supported_routes(self):
        routes = [(), ("--proof-strategy", "catalogue"), ("--allow-partial-coverage",)]
        for flags in routes:
            with self.subTest(flags=flags):
                process, output, status = self.invoke("--no-time-limit", *flags, empty_heads=("t",))
                self.assert_optimum(process, output, status, expected=0)
                self.assertFalse((output / "learned.lp").read_text().strip())
                for key in ("oracle_seconds", "compile_seconds", "solver_seconds"):
                    self.assertIsNone(status["arguments"][key])
                if not flags:
                    proof = json.loads((output / "program_proof.json").read_text())
                    self.assertEqual(status["optimality_source"], "nonnegative_additive_rule_cost")
                    self.assertEqual(proof["proof"], "nonnegative_additive_rule_cost")
                else:
                    self.assertFalse((output / "program_proof.json").exists())
                    self.assertTrue((output / "catalogue.json").exists())

    def test_unlimited_composition_preserves_dynamic_and_empty_target_bounds(self):
        process, output, status = self.invoke("--no-time-limit", "--bound-seconds", "0",
                                               heads=("first", "second"), empty_heads=("second",))
        self.assert_optimum(process, output, status, expected=4)
        self.assertEqual(status["target_count"], 2)
        bound = json.loads((output / "bound.json").read_text())
        self.assertEqual(bound["status"], "verified")
        self.assertEqual(bound["cost"], 4)
        self.assertEqual(bound["target_costs"], {"first/1": 4, "second/1": 0})
        plan = json.loads((output / "targets.json").read_text())["targets"]
        for target in plan:
            folder = output / target["output"]
            child = json.loads((folder / "status.json").read_text())
            self.assertTrue(child["arguments"]["no_time_limit"])
            self.assertEqual(child["arguments"]["bound_seconds"], 0)
            for key in ("oracle_seconds", "compile_seconds", "solver_seconds"):
                self.assertIsNone(child["arguments"][key])
            certificate = json.loads((folder / "bound.json").read_text())
            self.assertEqual(certificate["status"], "verified")
            self.assertEqual(certificate["cost"], child["optimal_cost"])
            self.assertTrue(certificate["verification"]["matched"])

    def test_separate_children_preserve_catalogue_and_disabled_dynamic_flags(self):
        process, output, status = self.invoke("--no-time-limit", "--bound-seconds", "0",
            "--proof-strategy", "catalogue", "--no-dynamic-program-bounds", heads=("first", "second"))
        self.assert_optimum(process, output, status, expected=8)
        plan = json.loads((output / "targets.json").read_text())["targets"]
        for target in plan:
            folder = output / target["output"]
            child = json.loads((folder / "status.json").read_text())
            self.assertEqual(child["arguments"]["proof_strategy"], "catalogue")
            self.assertFalse(child["arguments"]["dynamic_program_bounds"])
            self.assertEqual(child["arguments"]["bound_seconds"], 0)
            self.assertIsNone(child["arguments"]["compile_seconds"])
            self.assertTrue((folder / "catalogue.json").exists())
            self.assertFalse((folder / "program_proof.json").exists())


if __name__ == "__main__":
    unittest.main()
