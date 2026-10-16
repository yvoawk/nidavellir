"""Exact strict-improvement search over whole positive-rule programs.

The finite abstraction contains every legal program cheaper than a verified
incumbent. Actual grounded ASP bias rules are copied with fresh atom numbers,
not translated through a restricted catalogue of source-level bias patterns.
Positive join cores and real negative witnesses refine this abstraction. Only
an exhausted UNSAT solve proves that no cheaper program exists; a deadline
always retains the caller's independently verified incumbent without a proof.
"""
from __future__ import annotations

from math import isfinite
import time

import clingo

from .bias_analysis import _HeadSolver
from .oracle import ResourceLimit, match
from .relational_coverage import CoverageOracle
from .verification import verify_rules


class UnsupportedProgramProof(ValueError):
    """A future bias extension cannot be copied exactly by this backend."""


class _ProgramSearch:
    """Finite program relaxation with exact, independently activated bias copies."""

    def __init__(self, task, checkers, upper, check):
        self.task, self.check, self.upper = task, check, upper
        self.control = clingo.Control(["1", "--warn=none"])
        self.heads = tuple(task.heads)
        # Empty-body, ground-headed facts cost one. No stronger minimum is
        # assumed, including when some other heads contain variables.
        self.slots = max(0, upper - 1)
        self.active, self.gates, self.selected = [], [], []
        self.claims, self.positive_cores, self.negative_vectors = {}, set(), set()
        self.solve_seconds = 0.0
        with self.control.backend() as backend:
            for slot in range(self.slots):
                check()
                active = backend.add_atom()
                self.active.append(active)
                backend.add_rule([active], [], choice=True)
                if slot:
                    backend.add_rule([], [active, -self.active[slot - 1]])
                gates = [backend.add_atom() for _ in self.heads]
                selected = [backend.add_atom() for _ in task.literals]
                self.gates.append(gates)
                self.selected.append(selected)
                backend.add_rule(gates, [active], choice=True)
                backend.add_rule([], [active, *(-gate for gate in gates)])
                backend.add_weight_rule([], 2, [(gate, 1) for gate in gates])
                for head_index, head in enumerate(self.heads):
                    check()
                    solver, gate = checkers[head], gates[head_index]
                    observer = solver.ground_observer
                    if observer.external_atoms:
                        raise UnsupportedProgramProof("External bias atoms cannot be copied exactly")
                    all_atoms = set()
                    for _, heads, body in observer.rules:
                        all_atoms.update(heads)
                        all_atoms.update(abs(atom) for atom in body)
                    for _, heads, _, body in observer.weights:
                        all_atoms.update(heads)
                        all_atoms.update(abs(atom) for atom, _ in body)
                    atoms = {atom: backend.add_atom() for atom in sorted(all_atoms)}
                    def mapped(atom):
                        return atoms[atom] if atom > 0 else -atoms[-atom]
                    for choice, heads, body in observer.rules:
                        backend.add_rule([atoms[atom] for atom in heads],
                                         [gate, *(mapped(atom) for atom in body)], choice=choice)
                    for choice, heads, lower, body in observer.weights:
                        # A weighted body's gate must make it false when the
                        # copy is inactive, even for zero/negative thresholds.
                        total = sum(max(0, weight) for _, weight in body)
                        gate_weight = max(1, total - lower + 1)
                        backend.add_weight_rule([atoms[atom] for atom in heads],
                            lower + gate_weight,
                            [(mapped(atom), weight) for atom, weight in body] + [(gate, gate_weight)],
                            choice=choice)
                    for i, atom in enumerate(solver.body_atoms):
                        if atom is not None:
                            backend.add_rule([selected[i]], [gate, atoms[atom]])
            # Strict lexicographic ordering removes only rule permutations and
            # duplicate rules. Positive additive cost makes duplicates useless.
            for slot in range(self.slots - 1):
                left = self.gates[slot] + self.selected[slot]
                right = self.gates[slot + 1] + self.selected[slot + 1]
                prefix = backend.add_atom()
                backend.add_rule([prefix], [self.active[slot + 1]])
                for a, b in zip(left, right):
                    backend.add_rule([], [prefix, -a, b])
                    following = backend.add_atom()
                    backend.add_rule([following], [prefix, a, b])
                    backend.add_rule([following], [prefix, -a, -b])
                    prefix = following
                backend.add_rule([], [prefix])
            self._bound(backend, upper)

    def _bound(self, backend, upper):
        terms = [(atom, 1) for atom in self.active]
        terms.extend((atom, 1) for row in self.selected for atom in row)
        backend.add_weight_rule([], upper, terms)

    def tighten(self, upper):
        if not 0 < upper < self.upper:
            raise ValueError("Program bound must strictly decrease and remain positive")
        with self.control.backend() as backend:
            self._bound(backend, upper)
        self.upper = upper

    def positive(self, label):
        if label in self.claims:
            return False
        claims = {}
        with self.control.backend() as backend:
            for head_index, head in enumerate(self.heads):
                if match(head, label) is None:
                    continue
                for slot in range(self.slots):
                    atom = backend.add_atom()
                    claims[slot, head_index] = atom
                    backend.add_rule([atom], [self.gates[slot][head_index]], choice=True)
            backend.add_rule([], [-atom for atom in claims.values()])
        self.claims[label] = claims
        return True

    def positive_core(self, head_index, label, body):
        key = head_index, label, tuple(sorted(body))
        if key in self.positive_cores:
            return False
        self.positive_cores.add(key)
        with self.control.backend() as backend:
            for slot in range(self.slots):
                claim = self.claims[label].get((slot, head_index))
                if claim is not None:
                    backend.add_rule([], [claim, *(self.selected[slot][i] for i in key[2])])
        return True

    def negative(self, head_index, truth):
        key = head_index, truth
        if key in self.negative_vectors:
            return False
        self.negative_vectors.add(key)
        with self.control.backend() as backend:
            for slot in range(self.slots):
                backend.add_rule([], [self.gates[slot][head_index],
                    *(-atom for i, atom in enumerate(self.selected[slot]) if not truth >> i & 1)])
        return True

    def next(self):
        self.check()
        found = []
        def on_model(model):
            rules = {}
            for slot, active in enumerate(self.active):
                if not model.is_true(active):
                    continue
                heads = [i for i, gate in enumerate(self.gates[slot]) if model.is_true(gate)]
                if len(heads) != 1:
                    raise AssertionError("An active program slot must have one head template")
                rules[slot] = heads[0], tuple(i for i, atom in enumerate(self.selected[slot])
                                            if model.is_true(atom))
            claims = [(slot, head, label) for label, entries in self.claims.items()
                      for (slot, head), atom in entries.items() if model.is_true(atom)]
            found.append((rules, claims))
        started = time.monotonic()
        try:
            with self.control.solve(on_model=on_model, async_=True) as handle:
                while not handle.wait(0.01):
                    self.check()
                result = handle.get()
        finally:
            self.solve_seconds += time.monotonic() - started
        self.check()
        if result.unsatisfiable and result.exhausted and not found:
            return None
        if result.unknown or not result.satisfiable or len(found) != 1:
            raise RuntimeError(f"Program relaxation did not return a definite result: {result}")
        return found[0]


def prove_program_optimality(task, incumbent, *, seconds=None, verify=None, check=None,
                             progress=None, initial_positives=24, analysis=None, oracle=None):
    """Return ``(best_cover, report)`` for a caller-verified strict incumbent.

    Components use the existing ``head/body/cost/rule/coverage`` dictionary
    format. Their syntax, bias legality and independent full-program semantics
    are rechecked before searching. A deadline during this recheck preserves
    the caller's incumbent with ``incumbent_revalidated=False`` and never
    asserts optimality. No positive label may be left uncovered, every supplied
    negative remains forbidden, and unlabelled outputs remain unconstrained.

    ``seconds=None`` is unlimited. An external ``check`` can impose the shared
    caller deadline. A shared oracle is restored to its previous callback on
    exit. The report contains no complete-catalogue or all-optima claim.
    """
    if seconds is not None and (not isfinite(seconds) or seconds < 0):
        raise ValueError("Program proof seconds must be finite and nonnegative, or None")
    if type(initial_positives) is not int or initial_positives < 1:
        raise ValueError("Program proof initial_positives must be a positive integer")
    cover = [dict(row, body=list(row["body"])) for row in incumbent]
    for row in cover:
        head, body = clingo.parse_term(row["head"]), tuple(row["body"])
        if (head not in task.heads or type(row["cost"]) is not int
                or row["cost"] != len(body) + 1 or len(body) != len(set(body))
                or any(type(i) is not int or i < 0 or i >= len(task.literals) for i in body)
                or row["rule"] != task.rule(head, body)):
            raise ValueError("Program incumbent has inconsistent rule metadata")
    started = time.monotonic()
    deadline = None if seconds is None else started + seconds
    upper = sum(row["cost"] for row in cover)
    report = {"method": "whole_program_strict_improvement_cegis", "status": "timeout",
              "scope": "one minimum-cost representative over the full actual bias and all supplied hard labels",
              "optimality_proven": False, "catalogue_complete": False,
              "incumbent_revalidated": False, "initial_cost": upper, "cost": upper,
              "optimal_cost": None, "proved_unsatisfiable_through_cost": None,
              "rounds": 0, "positive_cores": 0, "negative_witnesses": 0,
              "active_positive_labels": 0, "improvements": [], "diagnostics": [],
              "ground_seconds": 0.0, "solve_seconds": 0.0, "verification_seconds": 0.0,
              "elapsed_seconds": 0.0}
    def checked():
        if check is not None:
            check()
        if deadline is not None and time.monotonic() >= deadline:
            raise ResourceLimit("Whole-program cost proof time budget reached")
    def emit():
        report["elapsed_seconds"] = time.monotonic() - started
        if search is not None:
            report.update(positive_cores=len(search.positive_cores),
                          negative_witnesses=len(search.negative_vectors),
                          solve_seconds=search.solve_seconds)
        if progress is not None:
            progress(dict(report))
    verification = verify or (lambda rules: verify_rules(task, rules))
    def verified(rules):
        verification_started = time.monotonic()
        try:
            return verification(rules)
        finally:
            report["verification_seconds"] += time.monotonic() - verification_started
    search, previous_oracle_check = None, None
    try:
        checked()
        analysis = analysis or getattr(task, "_bias_analysis", None)
        checkers = {}
        ground_started = time.monotonic()
        for head in task.heads:
            checked()
            checkers[head] = (analysis.heads[head]._checker if analysis is not None
                              else _HeadSolver(task, head, report["diagnostics"]))
        report["ground_seconds"] += time.monotonic() - ground_started
        for row in cover:
            head, body = clingo.parse_term(row["head"]), frozenset(row["body"])
            if checkers[head].find(body, set(range(len(task.literals))) - body, check=checked) is None:
                raise ValueError("Program incumbent violates the actual bias")
        checked()
        initial_verification = verified([row["rule"] for row in cover])
        if not initial_verification["matched"]:
            raise ValueError("Program incumbent failed independent full-program verification")
        report["incumbent_revalidated"] = True
        report["verification"] = {key: value for key, value in initial_verification.items() if key != "predictions"}
        checked()
        if upper == 0:
            report.update(status="complete", optimality_proven=True, optimal_cost=0,
                          proved_unsatisfiable_through_cost=-1, proof="nonnegative_additive_rule_cost")
            return cover, report
        if oracle is None:
            oracle = CoverageOracle(task, checked)
        else:
            if oracle.task is not task:
                raise ValueError("Shared program-proof oracle belongs to another task")
            previous_oracle_check = oracle.check
            oracle.check = checked
        ground_started = time.monotonic()
        search = _ProgramSearch(task, checkers, upper, checked)
        report["ground_seconds"] += time.monotonic() - ground_started
        report["program_slots"] = search.slots
        positives = sorted(task.positives)
        full = (1 << len(positives)) - 1
        def add_positives(labels):
            for label in labels:
                checked()
                if not search.positive(label):
                    continue
                # Exact singleton failures are cheap, reusable positive cores;
                # they never assert that separate successful joins agree.
                for head_index, head in enumerate(task.heads):
                    if match(head, label) is None:
                        continue
                    for i in range(len(task.literals)):
                        if not oracle.exists(head, (i,), label):
                            search.positive_core(head_index, label, (i,))
            report["active_positive_labels"] = len(search.claims)
        add_positives(positives[:initial_positives])
        emit()
        while True:
            checked()
            if report["rounds"] and report["rounds"] % 25 == 0:
                emit()
            proposal = search.next()
            report["rounds"] += 1
            if proposal is None:
                report.update(status="complete", optimality_proven=True, optimal_cost=upper,
                              proved_unsatisfiable_through_cost=upper - 1,
                              proof="exhausted_unsat_of_full_program_strict_improvement_relaxation")
                break
            rules, claims = proposal
            refined = False
            for slot, head_index, label in claims:
                selected_head, body = rules[slot]
                if selected_head != head_index:
                    raise AssertionError("Positive claim does not match selected head")
                head = task.heads[head_index]
                if not oracle.exists(head, body, label):
                    core = oracle.core(head, body, label)
                    if core is None:
                        raise AssertionError("Failed positive claim did not produce a sound core")
                    search.positive_core(head_index, label, core)
                    refined = True
            if refined:
                continue
            for head_index, body in rules.values():
                witness = oracle.negative(task.heads[head_index], body)
                if witness is not None:
                    search.negative(head_index, witness["truth"])
                    refined = True
            if refined:
                continue
            proposed, covered = [], 0
            for head_index, body in rules.values():
                head = task.heads[head_index]
                coverage = oracle.coverage(head, body)
                covered |= coverage
                proposed.append({"head": str(head), "body": list(body), "cost": len(body) + 1,
                                 "coverage": coverage, "rule": task.rule(head, body)})
            if covered != full:
                missing = [label for i, label in enumerate(positives) if not covered >> i & 1]
                if any(label in search.claims for label in missing):
                    raise AssertionError("Verified positive claims left an active label uncovered")
                add_positives(missing[:initial_positives])
                continue
            checked()
            certificate = verified([row["rule"] for row in proposed])
            if not certificate["matched"]:
                raise AssertionError("Program proof candidate failed independent full-program verification")
            proposed_cost = sum(row["cost"] for row in proposed)
            if proposed_cost >= upper:
                raise AssertionError("Program proposal did not strictly improve its incumbent")
            # Publish only after independent verification. If the deadline was
            # crossed during verification the better feasible result is retained,
            # but the next checked() prevents publishing an optimum proof.
            cover, upper = proposed, proposed_cost
            report.update(cost=upper, verification={key: value for key, value in certificate.items() if key != "predictions"})
            report["improvements"].append({"cost": upper, "round": report["rounds"],
                                           "elapsed_seconds": time.monotonic() - started})
            emit()
            checked()
            if upper == 0:
                report.update(status="complete", optimality_proven=True, optimal_cost=0,
                              proved_unsatisfiable_through_cost=-1, proof="nonnegative_additive_rule_cost")
                break
            search.tighten(upper)
    except ResourceLimit as error:
        report["reason"] = str(error)
    finally:
        if search is not None:
            report.update(positive_cores=len(search.positive_cores),
                          negative_witnesses=len(search.negative_vectors))
        if oracle is not None:
            report["oracle"] = dict(oracle.stats)
            if previous_oracle_check is not None:
                oracle.check = previous_oracle_check
        report["rules"] = [row["rule"] for row in cover]
        emit()
    return cover, report
