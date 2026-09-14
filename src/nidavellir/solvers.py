"""Exact selection from a finite, negative-safe rule coverage catalogue.

Candidate costs are positive integers and coverage is a bitmask over the
positive examples. This layer verifies optimization only: callers must establish
that each mask is correct and each candidate covers no negative example.

ILASP receives a compact explicit hypothesis space of weighted coverage rules,
not a new clinical bias. Both backends therefore optimize the same catalogue.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


def _catalogue(candidates: Sequence[Mapping[str, Any]], positive_count: int):
    if type(positive_count) is not int or positive_count < 0:
        raise ValueError("positive_count must be a nonnegative integer")
    full = (1 << positive_count) - 1
    result = []
    identifiers = set()
    for item in candidates:
        identifier, cost, coverage = item["id"], item["cost"], item["coverage"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Candidate ids must be unique nonempty strings")
        if type(cost) is not int or cost < 1:
            raise ValueError("Candidate costs must be positive integers")
        if type(coverage) is not int or coverage < 0 or coverage & ~full:
            raise ValueError("Candidate coverage must contain only known positive indices")
        identifiers.add(identifier)
        result.append({"id": identifier, "cost": cost, "coverage": coverage})
    if sum(item["cost"] for item in result) >= 2**31 - 1:
        raise ValueError("Total catalogue cost exceeds the supported ASP integer range")
    return result


def _indices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def set_cover_program(candidates: Sequence[Mapping[str, Any]], positive_count: int) -> str:
    """Build the exact internal objective; the candidate id prevents cost merging."""
    return _set_cover_encoding(candidates, positive_count)[0]


def _set_cover_encoding(candidates, positive_count):
    candidates = _catalogue(candidates, positive_count)
    certificate = feasible_cover_upper_bound(candidates, positive_count)
    lines = ["% Weighted set cover over negative-safe candidate rules."]
    lines += [f"candidate({i},{c['cost']})." for i, c in enumerate(candidates)]
    lines += [f"positive({p})." for p in range(positive_count)]
    lines += [f"covers({i},{p})." for i, c in enumerate(candidates)
              for p in _indices(c["coverage"])]
    lines += ["{ selected(C) : candidate(C,_) }.",
              "covered(P) :- selected(C), covers(C,P).",
              ":- positive(P), not covered(P).",
              "#minimize { Cost,C : selected(C), candidate(C,Cost) }.",
              "#show selected/1."]
    if certificate is not None:
        # A verified feasible cover certifies optimum <= U independently of
        # either solver. The candidate id keeps equal weights additive.
        lines += [f":- #sum {{ Cost,C : selected(C), candidate(C,Cost) }} > {certificate[0]}."]
    return "\n".join(lines) + "\n", {
        "feasible_cover_cost": certificate[0] if certificate is not None else None,
        "feasible_cover": list(certificate[1]) if certificate is not None else None,
        "upper_bound_applied": certificate is not None,
        "upper_bound_policy": "greedy_feasible_candidate_cover" if certificate is not None else None,
    }


def feasible_cover_upper_bound(candidates: Sequence[Mapping[str, Any]], positive_count: int):
    """Return a verified greedy cover certificate, or None if no cover exists.

    The certificate is (cost, candidate ids, coverage). It is a feasible upper
    bound, not an optimality claim. Costs are positive, so removing redundant
    rules can only improve this ceiling without excluding a global optimum.
    """
    candidates = _catalogue(candidates, positive_count)
    full = (1 << positive_count) - 1
    uncovered = full
    chosen = []
    while uncovered:
        eligible = [candidate for candidate in candidates if candidate["coverage"] & uncovered]
        if not eligible:
            return None
        candidate = min(eligible, key=lambda item: (
            Fraction(item["cost"], (item["coverage"] & uncovered).bit_count()),
            item["cost"], -(item["coverage"] & uncovered).bit_count(), item["id"]))
        chosen.append(candidate)
        uncovered &= ~candidate["coverage"]
    # The remaining cover stays complete after every removal. Iterating once
    # suffices: later removals cannot make a formerly necessary rule redundant.
    for candidate in tuple(reversed(chosen)):
        coverage = 0
        for other in chosen:
            if other is not candidate:
                coverage |= other["coverage"]
        if coverage == full:
            chosen.remove(candidate)
    coverage = 0
    for candidate in chosen:
        coverage |= candidate["coverage"]
    if coverage != full:
        raise RuntimeError("Greedy ILASP ceiling lacks complete positive coverage")
    return sum(item["cost"] for item in chosen), tuple(item["id"] for item in chosen), coverage


def _selection_encoding(candidates, positive_count):
    candidates = _catalogue(candidates, positive_count)
    certificate = feasible_cover_upper_bound(candidates, positive_count)
    # ILASP's max_penalty is inclusive. If no cover exists, the sum of every
    # cost still permits all hypotheses and cannot manufacture an UNSAT proof.
    ceiling = certificate[0] if certificate is not None else sum(c["cost"] for c in candidates)
    policy = "greedy_feasible_candidate_cover" if certificate is not None else "all_candidate_costs"
    minimum = min((c["cost"] for c in candidates), default=None)
    singleton = bool(positive_count and certificate is not None and minimum is not None
                     and 2 * minimum > ceiling)
    full = (1 << positive_count) - 1
    retained = [(i, candidate) for i, candidate in enumerate(candidates)
                if not singleton or candidate["coverage"] == full and candidate["cost"] <= ceiling]
    # Every optimum has cost <= the independently feasible U. Under the
    # strict inequality two rules already cost too much; with nonempty
    # positives an optimum must therefore be one complete-cover candidate.
    # Do not renumber: ILASP output is decoded against the original catalogue.
    if singleton and not retained:
        raise RuntimeError("Singleton cover certificate has no retained feasible candidate")
    lines = ["% ILASP optimum check: finite negative-safe coverage catalogue only."]
    lines += [f"% Inclusive penalty ceiling: {policy}."]
    # Derive coverage directly: hidden pick facts multiply the interpretations
    # considered by ILASP's conflict analysis. Each distinct covers index still
    # identifies exactly one candidate with exactly its original additive cost.
    lines += [f"{c['cost']} ~ covered(P) :- covers({i},P)." for i, c in retained]
    lines += [f"#max_penalty({ceiling})."]
    lines += [f"positive({p})." for p in range(positive_count)]
    lines += [f"covers({i},{p})." for i, c in retained
              for p in _indices(c["coverage"])]
    # For each fixed selection this stratified auxiliary BK has one model.
    # The nullary goal enforces the conjunction without a giant interpretation
    # (parser stack) or a separately analysed ILASP example for every positive.
    lines += ["uncovered :- positive(P), not covered(P).",
              "all_covered :- not uncovered.",
              "#pos(all, {all_covered}, {}, {})."]
    return "\n".join(lines) + "\n", {
        "max_penalty": ceiling, "max_penalty_policy": policy,
        "feasible_cover": list(certificate[1]) if certificate is not None else None,
        "feasible_cover_cost": certificate[0] if certificate is not None else None,
        "example_count": 1, "hypothesis_encoding": "direct_coverage_rules",
        "singleton_reduction": {
            "applied": singleton, "minimum_rule_cost": minimum,
            "feasible_upper_bound": certificate[0] if certificate is not None else None,
            "certificate_source": "verified greedy coverage; independent of the internal optimizer result",
            "criterion": "nonempty positives and 2 * minimum_rule_cost > feasible_upper_bound",
            "scope": "Preserves every optimum of the original catalogue; not just an internally selected hypothesis",
            "original_candidate_count": len(candidates), "retained_candidate_count": len(retained),
            "eliminated_candidate_count": len(candidates) - len(retained),
            "retained_original_indices": [i for i, _ in retained] if singleton else None,
            "retained_candidate_ids": [c["id"] for _, c in retained] if singleton else None,
        },
    }


def selection_task(candidates: Sequence[Mapping[str, Any]], positive_count: int) -> str:
    """Build ILASP's explicit weighted hypothesis space on this same catalogue."""
    return _selection_encoding(candidates, positive_count)[0]


def _base(backend, program, started):
    return {"backend": backend, "status": "error", "selected": [],
            "cost": None, "optimum_cost": None, "program": program,
            "stdout": "", "stderr": "", "stats": {},
            "elapsed_seconds": time.perf_counter() - started}


def _selection(candidates, selected_indices, positive_count):
    coverage = 0
    cost = 0
    for index in selected_indices:
        coverage |= candidates[index]["coverage"]
        cost += candidates[index]["cost"]
    return coverage == (1 << positive_count) - 1, cost


def _internal(candidates, positive_count, timeout):
    started = time.perf_counter()
    program, encoding_stats = _set_cover_encoding(candidates, positive_count)
    result = _base("internal", program, started)
    result["stats"].update(encoding_stats)
    messages = []
    try:
        import clingo
        control = clingo.Control(["--opt-mode=opt", "--models=0"],
                                 logger=lambda code, message: messages.append(message))
        control.add("base", [], program)
        grounding_started = time.perf_counter()
        control.ground([("base", [])])
        result["stats"]["grounding_seconds"] = time.perf_counter() - grounding_started
        remaining = None if timeout is None else timeout - (time.perf_counter() - started)
        if remaining is not None and remaining <= 0:
            result["status"] = "timeout"
        else:
            best = None
            model_count = 0

            def model_found(model):
                nonlocal best, model_count
                selected = sorted(atom.arguments[0].number for atom in model.symbols(shown=True)
                                  if atom.name == "selected" and len(atom.arguments) == 1)
                valid, cost = _selection(candidates, selected, positive_count)
                if not valid:
                    raise RuntimeError("Internal solver emitted incomplete positive coverage")
                if best is None or cost < best[0]:
                    best = cost, selected
                model_count += 1

            with control.solve(on_model=model_found, async_=True) as handle:
                finished = handle.wait(remaining)
                if not finished:
                    handle.cancel()
                answer = handle.get()
            if best is not None:
                result["cost"] = best[0]
                result["selected"] = [candidates[i]["id"] for i in best[1]]
            if answer.exhausted and answer.satisfiable:
                result.update(status="complete", optimum_cost=best[0])
            elif answer.unsatisfiable and answer.exhausted:
                result["status"] = "unsatisfiable"
            else:
                result["status"] = "timeout" if not finished or answer.interrupted else "error"
            result["stdout"] = str(answer) + "\n"
            result["stats"].update(models=model_count, exhausted=answer.exhausted,
                                   interrupted=answer.interrupted,
                                   clingo=control.statistics)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["stderr"] = "\n".join(messages)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _ilasp_path(binary):
    if binary is not None:
        return os.fspath(binary)
    found = shutil.which("ILASP")
    if found:
        return found
    colocated = Path(sys.executable).parent / "ILASP"
    return str(colocated) if colocated.is_file() else "ILASP"


def _ilasp(candidates, positive_count, timeout, binary):
    started = time.perf_counter()
    program, encoding_stats = _selection_encoding(candidates, positive_count)
    result = _base("ilasp", program, started)
    result["stats"].update(encoding_stats)
    try:
        with tempfile.TemporaryDirectory(prefix="nidavellir-rule-selection-") as directory:
            task = Path(directory) / "selection.las"
            task.write_text(program, encoding="utf-8")
            command = [_ilasp_path(binary), "--version=4", "--small-constraints", str(task)]
            result["command"] = [*command[:-1], "selection.las"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=(os.name == "posix"))
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # ILASP launches child solvers: cancel the task's process group, too.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                result.update(status="timeout", stdout=stdout, stderr=stderr)
                result["stats"]["returncode"] = process.returncode
                return result
            result.update(stdout=stdout, stderr=stderr)
            result["stats"]["returncode"] = process.returncode
            if process.returncode != 0:
                result["error"] = f"ILASP exited with status {process.returncode}"
                return result
            # ILASP 4 standard output has no 'OPTIMUM FOUND' token. Its successful
            # completed optimization prints a Total footer plus the final hypothesis.
            # Require both this completion evidence and valid full coverage; never
            # infer optimality from a partial/timed-out hypothesis.
            if not re.search(r"^%%\s+Total\s*:", stdout, re.MULTILINE):
                result["error"] = "ILASP output has no completed-run summary"
                return result
            if re.search(r"^UNSATISFIABLE\s*$", stdout, re.MULTILINE):
                if encoding_stats["feasible_cover_cost"] is not None:
                    result["error"] = "ILASP UNSAT contradicts the verified feasible-cover certificate"
                    return result
                result["status"] = "unsatisfiable"
                return result
            hypothesis = [line.strip() for line in stdout.splitlines()
                          if line.strip() and not line.lstrip().startswith("%")]
            # ILASP may rename the single variable or insert whitespace. The
            # backreference checks that head and body use that same variable.
            matches = [re.fullmatch(
                r"covered\(\s*([A-Z][A-Za-z0-9_]*)\s*\)\s*:-\s*"
                r"covers\(\s*(\d+)\s*,\s*\1\s*\)\s*\.", line)
                for line in hypothesis]
            if any(match is None for match in matches):
                result["error"] = "Unexpected ILASP hypothesis/output format"
                return result
            selected = [int(match.group(2)) for match in matches]
            if len(set(selected)) != len(selected) or any(i >= len(candidates) for i in selected):
                result["error"] = "ILASP returned duplicate or unknown candidate ids"
                return result
            reduced_indices = encoding_stats["singleton_reduction"]["retained_original_indices"]
            if reduced_indices is not None and not set(selected) <= set(reduced_indices):
                result["error"] = "ILASP returned a candidate excluded by the singleton certificate"
                return result
            valid, cost = _selection(candidates, selected, positive_count)
            if not valid or cost > encoding_stats["max_penalty"]:
                result["error"] = "ILASP returned incomplete coverage or exceeded the verified cost ceiling"
                return result
            result.update(status="complete", selected=[candidates[i]["id"] for i in sorted(selected)],
                          cost=cost, optimum_cost=cost)
            return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _enumeration_limits(timeout, maximum, name):
    if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("enumeration timeout must be a finite positive number, or None")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError(name + " must be a positive integer, or None")


def optimal_cover_program(candidates, positive_count, optimum_cost):
    """Enumerate a proved cost using fixed sum guards, without cost grounding."""
    candidates = _catalogue(candidates, positive_count)
    if type(optimum_cost) is not int or optimum_cost < 0 or optimum_cost >= 2**31 - 1:
        raise ValueError("optimum_cost must be a supported nonnegative integer")
    program = set_cover_program(candidates, positive_count)
    objective = "#minimize { Cost,C : selected(C), candidate(C,Cost) }."
    total = "#sum { Cost,C : selected(C), candidate(C,Cost) }"
    return program.replace(objective, f":- {total} < {optimum_cost}.\n:- {total} > {optimum_cost}.")


def enumerate_optimal_covers(candidates, positive_count, optimum_cost, *,
                            backend="auto", timeout=None, max_optima=None):
    """Enumerate every cover at an already proved optimum.

    The caller supplies the cost proof; this function does not establish that
    smaller costs are impossible. A resource limit preserves the covers already
    found, with ``complete=False``. Equality in the singleton bound deliberately
    takes the general multi-rule path.
    """
    started = time.perf_counter()
    _enumeration_limits(timeout, max_optima, "max_optima")
    candidates = _catalogue(candidates, positive_count)
    if backend not in {"auto", "clingo"}:
        raise ValueError("enumeration backend must be 'auto' or 'clingo'")
    program = optimal_cover_program(candidates, positive_count, optimum_cost)
    minimum = min((candidate["cost"] for candidate in candidates), default=None)
    result = {"status": "error", "complete": False, "backend": None,
              "optimum_cost": optimum_cost, "semantic_hypotheses": [], "hypothesis_count": 0,
              "minimum_rule_cost": minimum, "maximum_rules_at_optimum": optimum_cost // minimum if minimum else 0,
              "program": program, "stats": {}, "elapsed_seconds": 0.0,
              "scope": "All semantic candidate subsets at the caller-proved optimum cost"}
    deadline = math.inf if timeout is None else started + timeout

    def finish(status):
        result.update(status=status, complete=status == "complete",
                      hypothesis_count=len(result["semantic_hypotheses"]),
                      elapsed_seconds=time.perf_counter() - started)
        result["semantic_hypotheses"].sort()
        return result

    if time.perf_counter() >= deadline:
        return finish("timeout")
    if positive_count == 0 and optimum_cost == 0:
        result.update(backend="empty_program_proof", semantic_hypotheses=[[]])
        return finish("complete")
    if backend == "auto" and minimum is not None and 2 * minimum > optimum_cost:
        result["backend"] = "singleton_proof"
        full = (1 << positive_count) - 1
        for candidate in sorted(candidates, key=lambda item: item["id"]):
            if time.perf_counter() >= deadline:
                return finish("timeout")
            if candidate["cost"] != optimum_cost or candidate["coverage"] != full:
                continue
            if max_optima is not None and len(result["semantic_hypotheses"]) >= max_optima:
                return finish("limit")
            result["semantic_hypotheses"].append([candidate["id"]])
        if not result["semantic_hypotheses"]:
            result["error"] = "No cover at supplied optimum cost; the caller's cost proof is inconsistent"
            return finish("error")
        return finish("complete")
    result["backend"] = "clingo_guarded_sum"
    messages = []
    try:
        import clingo

        control = clingo.Control(["0"], logger=lambda _, message: messages.append(message))
        control.add("base", [], program)
        grounding_started = time.perf_counter()
        control.ground([("base", [])])
        result["stats"]["grounding_seconds"] = time.perf_counter() - grounding_started
        remaining = None if timeout is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0:
            return finish("timeout")
        stopped = None

        def found(model):
            nonlocal stopped
            if time.perf_counter() >= deadline:
                stopped = "timeout"
                return False
            chosen = sorted(atom.arguments[0].number for atom in model.symbols(shown=True)
                            if atom.name == "selected" and len(atom.arguments) == 1)
            valid, cost = _selection(candidates, chosen, positive_count)
            if not valid or cost != optimum_cost:
                raise RuntimeError("Enumeration returned an invalid cover or cost")
            if max_optima is not None and len(result["semantic_hypotheses"]) >= max_optima:
                stopped = "limit"
                return False
            result["semantic_hypotheses"].append(sorted(candidates[i]["id"] for i in chosen))
            return True

        with control.solve(on_model=found, async_=True) as handle:
            finished = handle.wait(remaining)
            if not finished:
                handle.cancel()
            answer = handle.get()
        result["stats"].update(exhausted=answer.exhausted, interrupted=answer.interrupted,
                               clingo=control.statistics)
        result["stderr"] = "\n".join(messages)
        if stopped:
            return finish(stopped)
        if not finished or answer.interrupted:
            return finish("timeout")
        if answer.exhausted and answer.satisfiable:
            return finish("complete")
        result["error"] = "No cover at supplied optimum cost, or enumeration did not finish with a proof"
        return finish("error")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["stderr"] = "\n".join(messages)
        return finish("error")


def expand_optimal_programs(candidates, semantic_hypotheses, *, timeout=None, max_programs=None):
    """Expand semantic optima using one representative per alpha class.

    A caller must establish each variant's coverage and cost. This routine checks
    identifiers and product bookkeeping, not ASP semantics. ``complete`` refers
    only to the supplied semantic hypotheses; the caller combines it with the
    semantic-enumeration completion status.
    """
    started = time.perf_counter()
    _enumeration_limits(timeout, max_programs, "max_programs")
    deadline = math.inf if timeout is None else started + timeout
    by_id = {}
    for candidate in candidates:
        identifier = candidate["id"]
        if identifier in by_id or not isinstance(identifier, str):
            raise ValueError("Variant expansion requires unique string candidate ids")
        if type(candidate["cost"]) is not int or candidate["cost"] <= 0:
            raise ValueError("Variant expansion requires positive integer costs")
        by_id[identifier] = candidate
    result = {"status": "running", "complete": False, "programs": [], "program_count": 0,
              "semantic_hypothesis_count": len(semantic_hypotheses), "expanded_combinations": 0,
              "elapsed_seconds": 0.0, "scope": "Syntactic products of the supplied semantic hypotheses"}
    seen = set()

    def finish(status):
        result.update(status=status, complete=status == "complete", program_count=len(result["programs"]),
                      elapsed_seconds=time.perf_counter() - started)
        result["programs"].sort(key=lambda item: (item["cost"], item["rules"], item["selected"]))
        return result

    for hypothesis in semantic_hypotheses:
        if time.perf_counter() >= deadline:
            return finish("timeout")
        if len(set(hypothesis)) != len(hypothesis) or any(identifier not in by_id for identifier in hypothesis):
            raise ValueError("Semantic hypothesis contains duplicate or unknown candidate ids")
        chosen = [by_id[identifier] for identifier in sorted(hypothesis)]
        variants = []
        for candidate in chosen:
            alpha_values = candidate.get("alpha_variants")
            if alpha_values is not None:
                if (not isinstance(alpha_values, (list, tuple)) or not alpha_values
                        or any(not isinstance(group, (list, tuple)) or not group for group in alpha_values)):
                    raise ValueError("Each alpha variant class must contain nonempty rule strings")
                if any(any(not isinstance(rule, str) or not rule.strip() for rule in group)
                       for group in alpha_values):
                    raise ValueError("Each alpha variant class must contain nonempty rule strings")
                values = [min(group) for group in alpha_values]
            else:
                values = candidate.get("variants")
            if values is None:
                values = [candidate["rule"]] if "rule" in candidate else []
            if (not isinstance(values, (list, tuple)) or not values
                    or any(not isinstance(rule, str) or not rule.strip() for rule in values)):
                raise ValueError("Each expanded candidate requires nonempty rule strings in variants or rule")
            variants.append(sorted(set(values)))
        cost = sum(candidate["cost"] for candidate in chosen)
        for rules in product(*variants):
            if time.perf_counter() >= deadline:
                return finish("timeout")
            if len(set(rules)) != len(rules):
                raise ValueError("Different selected candidates expand to the same rule; set cost would change")
            result["expanded_combinations"] += 1
            key = tuple(sorted(rules))
            if key in seen:
                continue
            if max_programs is not None and len(result["programs"]) >= max_programs:
                return finish("limit")
            seen.add(key)
            result["programs"].append({"selected": sorted(hypothesis), "rules": list(key), "cost": cost})
    return finish("complete")


def solve(candidates: Sequence[Mapping[str, Any]], positive_count: int, *,
          backend: str = "internal", ilasp_binary: str | Path | None = None,
          timeout: float | None = None, verify_ilasp: bool = False,
          enumerate_optima: bool = False, enumeration_backend: str = "auto",
          enumeration_timeout: float | None = None, max_optima: int | None = None,
          expand_variants: bool = False, max_programs: int | None = None) -> dict[str, Any]:
    """Return one minimum-cost cover, with explicit completion/proof status.

    ``timeout`` is a separate budget for selection and optional ILASP verification.
    Internal grounding is timed but cannot be interrupted by the Python solve
    handle; any overrun prevents a subsequent solve. An interrupted search can
    return an incumbent ``cost``, but never an ``optimum_cost``.

    Verification checks cost and coverage, allowing different tied optimal covers.
    A failed/unavailable verification is reported separately and does not invalidate
    a proven internal optimum. An actual solver disagreement is a top-level error.

    Optional ``enumeration`` and ``variant_expansion`` have independent completion
    statuses. Their timeout never erases a proved selection optimum; callers must
    inspect them before claiming that all optimal programs have been returned.
    ``enumeration_timeout`` applies separately to these two optional phases.
    """
    started = time.perf_counter()
    original_candidates = list(candidates)
    _enumeration_limits(enumeration_timeout, max_optima, "max_optima")
    _enumeration_limits(enumeration_timeout, max_programs, "max_programs")
    if enumeration_backend not in {"auto", "clingo"}:
        raise ValueError("enumeration_backend must be 'auto' or 'clingo'")
    if expand_variants and not enumerate_optima:
        raise ValueError("expand_variants requires enumerate_optima")
    candidates = _catalogue(original_candidates, positive_count)
    if backend not in {"internal", "ilasp"}:
        raise ValueError("backend must be 'internal' or 'ilasp'")
    if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a finite positive number, or None")
    if verify_ilasp and backend != "internal":
        raise ValueError("verify_ilasp applies to the internal weighted set-cover backend")
    result = (_internal(candidates, positive_count, timeout) if backend == "internal" else
              _ilasp(candidates, positive_count, timeout, ilasp_binary))
    result["stats"].update(candidate_count=len(candidates), positive_count=positive_count)
    result["scope"] = "finite negative-safe coverage catalogue only"
    if verify_ilasp:
        if result["status"] not in {"complete", "unsatisfiable"}:
            result["verification"] = {"status": "skipped", "matches": None,
                                      "reason": "Internal selection did not finish with a proof"}
        else:
            check = _ilasp(candidates, positive_count, timeout, ilasp_binary)
            check["scope"] = result["scope"]
            check["matches"] = None
            if check["status"] in {"complete", "unsatisfiable"}:
                check["matches"] = (check["status"] == result["status"] and
                                      check["optimum_cost"] == result["optimum_cost"])
                if not check["matches"]:
                    result["selection_status"] = result["status"]
                    result["status"] = "error"
                    result["error"] = "Internal and ILASP selection results disagree"
            result["verification"] = check
    if enumerate_optima:
        if result["status"] == "complete":
            enumeration = enumerate_optimal_covers(candidates, positive_count, result["optimum_cost"],
                                                   backend=enumeration_backend, timeout=enumeration_timeout,
                                                   max_optima=max_optima)
            result["enumeration"] = enumeration
            if expand_variants:
                try:
                    expansion = expand_optimal_programs(original_candidates, enumeration["semantic_hypotheses"],
                                                        timeout=enumeration_timeout, max_programs=max_programs)
                    expansion["semantic_enumeration_complete"] = enumeration["complete"]
                    expansion["all_optimal_programs_complete"] = expansion["complete"] and enumeration["complete"]
                    result["variant_expansion"] = expansion
                except (ValueError, KeyError, TypeError) as error:
                    result["variant_expansion"] = {"status": "error", "complete": False,
                                                    "all_optimal_programs_complete": False, "error": str(error),
                                                    "programs": [], "program_count": 0}
        elif result["status"] == "unsatisfiable":
            result["enumeration"] = {"status": "complete", "complete": True,
                                     "backend": "unsatisfiable_proof", "semantic_hypotheses": [],
                                     "hypothesis_count": 0, "elapsed_seconds": 0.0}
            if expand_variants:
                result["variant_expansion"] = {"status": "complete", "complete": True,
                    "all_optimal_programs_complete": True, "programs": [], "program_count": 0,
                    "semantic_enumeration_complete": True, "elapsed_seconds": 0.0}
        else:
            result["enumeration"] = {"status": "skipped", "complete": False,
                                     "semantic_hypotheses": [], "hypothesis_count": 0,
                                     "reason": "Selection did not finish with a consistent optimum proof"}
            if expand_variants:
                result["variant_expansion"] = {"status": "skipped", "complete": False,
                    "all_optimal_programs_complete": False, "programs": [], "program_count": 0,
                    "reason": "Selection did not finish with a consistent optimum proof"}
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
