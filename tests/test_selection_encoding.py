"""Tight ILASP ceilings and one compact hard coverage goal preserve selection."""

import itertools
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import clingo

from nidavellir.solvers import feasible_cover_upper_bound, selection_task, solve


ILASP = shutil.which("ILASP") or str(Path(sys.executable).parent / "ILASP")


class SelectionEncodingTests(unittest.TestCase):
    def test_greedy_certificate_is_feasible_and_never_below_optimum(self):
        randomizer = random.Random(907)
        for trial in range(60):
            count = randomizer.randrange(5)
            candidates = [{"id": str(i), "cost": randomizer.randrange(1, 8),
                           "coverage": randomizer.randrange(1 << count)}
                          for i in range(randomizer.randrange(8))]
            costs = []
            for flags in itertools.product((False, True), repeat=len(candidates)):
                covered = cost = 0
                for included, candidate in zip(flags, candidates):
                    if included:
                        covered |= candidate["coverage"]
                        cost += candidate["cost"]
                if covered == (1 << count) - 1:
                    costs.append(cost)
            with self.subTest(trial=trial):
                certificate = feasible_cover_upper_bound(candidates, count)
                if not costs:
                    self.assertIsNone(certificate)
                    continue
                cost, identifiers, covered = certificate
                chosen = [candidate for candidate in candidates if candidate["id"] in identifiers]
                self.assertEqual(cost, sum(candidate["cost"] for candidate in chosen))
                self.assertEqual(covered, (1 << count) - 1)
                self.assertGreaterEqual(cost, min(costs))
                self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_compact_goal_matches_all_subsets_in_one_unique_model(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 0b011},
                      {"id": "b", "cost": 2, "coverage": 0b110},
                      {"id": "c", "cost": 5, "coverage": 0b111}]
        for count, catalogue in [(3, candidates), (0, []), (1, [])]:
            program = selection_task(catalogue, count)
            background = "\n".join(line for line in program.splitlines()
                                   if not line.startswith("#") and " ~ " not in line)
            for flags in itertools.product((False, True), repeat=len(catalogue)):
                picks = "\n".join(f"covered(P) :- covers({i},P)."
                                  for i, included in enumerate(flags) if included)
                control = clingo.Control(["--models=0"], logger=lambda *_: None)
                control.add("base", [], background + "\n" + picks)
                control.ground([("base", [])])
                models = []
                answer = control.solve(on_model=lambda model: models.append(set(model.symbols(atoms=True))))
                self.assertTrue(answer.exhausted)
                self.assertEqual(len(models), 1)
                covered = 0
                for included, candidate in zip(flags, catalogue):
                    if included:
                        covered |= candidate["coverage"]
                self.assertEqual(clingo.Function("all_covered") in models[0], covered == (1 << count) - 1)

    def test_single_short_example_and_verified_multirule_ceiling(self):
        candidates = [{"id": "a", "cost": 2, "coverage": 3},
                      {"id": "b", "cost": 2, "coverage": 6},
                      {"id": "c", "cost": 5, "coverage": 7}]
        program = selection_task(candidates, 3)
        self.assertIn("#max_penalty(4).", program)
        self.assertEqual(program.count("#pos("), 1)
        self.assertIn("#pos(all, {all_covered}, {}, {}).", program)
        large = selection_task([{"id": "all", "cost": 2, "coverage": (1 << 1000) - 1}], 1000)
        self.assertEqual([line for line in large.splitlines() if line.startswith("#pos")],
                         ["#pos(all, {all_covered}, {}, {})."])

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_ilasp_penalty_is_inclusive(self):
        candidate = [{"id": "required", "cost": 3, "coverage": 1}]
        program = selection_task(candidate, 1)
        with tempfile.TemporaryDirectory() as directory:
            for ceiling in (2, 3, 4):
                path = Path(directory) / f"ceiling{ceiling}.las"
                path.write_text(re.sub(r"#max_penalty\(\d+\)", f"#max_penalty({ceiling})", program))
                result = subprocess.run([ILASP, "--version=4", "--small-constraints", str(path)],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("UNSATISFIABLE" in result.stdout, ceiling < 3)
                self.assertEqual("covers(0," in result.stdout, ceiling >= 3)

    @unittest.skipUnless(Path(ILASP).is_file(), "ILASP binary unavailable")
    def test_real_ilasp_compact_goal_preserves_empty_unsat_and_overlap(self):
        cases = [([], 0, 0), ([], 1, None),
                 ([{"id": "a", "cost": 2, "coverage": 3},
                   {"id": "b", "cost": 2, "coverage": 6},
                   {"id": "expensive", "cost": 5, "coverage": 7}], 3, 4),
                 ([{"id": "same_expensive", "cost": 5, "coverage": 1},
                   {"id": "same_cheap", "cost": 2, "coverage": 1},
                   {"id": "irrelevant", "cost": 1, "coverage": 0}], 1, 2)]
        for candidates, count, optimum in cases:
            with self.subTest(count=count, candidates=candidates):
                result = solve(candidates, count, backend="ilasp", ilasp_binary=ILASP, timeout=5)
                self.assertEqual(result["status"], "unsatisfiable" if optimum is None else "complete", result)
                self.assertEqual(result["optimum_cost"], optimum)
                self.assertEqual(result["stats"]["example_count"], 1)
                if optimum is not None:
                    self.assertEqual(result["stats"]["feasible_cover_cost"], optimum)
                else:
                    self.assertEqual(result["stats"]["max_penalty_policy"], "all_candidate_costs")

    def test_decode_requires_known_id_and_the_same_single_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ILASP"
            for hypothesis, expected in [
                ("covered(V7) :- covers(0, V7).", "complete"),
                ("covered(P) :- covers(0, Q).", "error"),
                ("covered(P) :- covers(1, P).", "error"),
                ("covered(P) :- covers(0, P).\ncovered(P) :- covers(0, P).", "error"),
                ("pick(0).", "error"),
            ]:
                executable.write_text(f"#!{sys.executable}\nprint('%% Total : 0s')\nprint({hypothesis!r})\n")
                executable.chmod(0o700)
                with self.subTest(hypothesis=hypothesis):
                    result = solve([{"id": "a", "cost": 1, "coverage": 1}], 1,
                                   backend="ilasp", ilasp_binary=executable, timeout=2)
                    self.assertEqual(result["status"], expected, result)
                    self.assertEqual(result["optimum_cost"], 1 if expected == "complete" else None)


if __name__ == "__main__":
    unittest.main()
