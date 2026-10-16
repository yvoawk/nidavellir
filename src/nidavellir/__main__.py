"""Compile a finite positive-rule ILP task into an optimal rule program."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import clingo

from . import __version__
from .oracle import ResourceLimit, build_cartesian_indices
from .projected_oracle import build_projected_indices
from .search import compile_catalogue
from .solvers import expand_optimal_programs, solve, selection_task, set_cover_program
from .task import Task, TaskError
from .verification import verify_rules
from .labels import CompactNegativeSet
from .terminal_progress import TerminalProgress


from .json_stream import JsonArray, write_json


class _ProofFinished(Exception):
    """Internal control flow after publishing a direct program-proof result."""


class _IncumbentReady(Exception):
    """A verified dynamic cover permits switching from enumeration to proof."""


def recover_target_cover(task, cover, bound):
    """Compose complete verified target certificates after worker timeout.

    A missing global worker checkpoint must not discard a cover whose target
    certificates already cover every supplied positive. Recheck the combined
    rules against the original BK before promoting the global incumbent.
    """
    if cover is not None or not bound.get("target_bounds"):
        return cover, bound
    from .target_bounds import costs_from_certificates, positive_scopes, key_text
    certificates = bound["target_bounds"]
    costs_from_certificates(task, certificates)
    scopes = positive_scopes(task)
    if any(mask and key_text(key) not in certificates for key, mask in scopes.items()):
        return cover, bound
    recovered = []
    for key, mask in scopes.items():
        if mask:
            recovered.extend(dict(row, coverage=int(row["coverage"], 16))
                             for row in certificates[key_text(key)]["cover"])
    verification = verify_rules(task, [row["rule"] for row in recovered])
    if not verification["matched"]:
        raise AssertionError("Composed target incumbents failed independent verification")
    updated = dict(bound, status="verified", discovery_status=bound["status"],
                   cost=sum(row["cost"] for row in recovered),
                   rules=[row["rule"] for row in recovered], verification=verification,
                   incumbent_source="verified_target_certificates")
    return recovered, updated


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def index_json(index, *, representatives=False):
    result = {"head": str(index.head), "literal_masks": JsonArray(hex(m) for m in index.literal_masks),
              "all_mask": hex(index.all_mask), "positive_mask": hex(index.positive_mask),
              "negative_mask": hex(index.negative_mask),
              "positive_heads": JsonArray((n, hex(m)) for n, m in index.positive_heads)}
    if representatives:
        result["classes"] = JsonArray(index.classes)
    return result


def bound_in_subprocess(task, indices, seconds, snapshot_dir=None, *, strategy="bitmask",
                        fingerprint=None, checkpoint_path=None, resume_path=None,
                        background_transfer="auto", per_target_bounds=False):
    if not seconds:
        return None, {"status": "disabled", "cost": None, "elapsed_seconds": 0}
    started = time.monotonic()
    if background_transfer not in {"auto", "snapshot", "replay"}:
        raise ValueError("background_transfer must be auto, snapshot or replay")
    # Large derived relations can be much cheaper to recreate natively than
    # to stringify and reparse atom by atom. Display is disabled in _model,
    # while its complete true-atom set is still retrieved. This heuristic
    # affects only transfer cost; the parent enforces the same hard deadline.
    if background_transfer == "auto":
        background_transfer = ("replay" if len(task.atoms) > 10 * max(1, task.background.count("\n"))
                               else "snapshot")
    # CWD is not assumed to be the repository root.
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="nidavellir-bound-") as folder:
        folder = Path(folder)
        try:
            if snapshot_dir is None:
                snapshot_dir = folder / "inputs"
                snapshot_dir.mkdir()
                (snapshot_dir / "background.lp").write_text(task.background)
                task.write_examples(snapshot_dir / "examples.lp")
            snapshot_dir = Path(snapshot_dir).resolve()
            compact = {}
            if strategy == "relational" and isinstance(task.negatives, CompactNegativeSet):
                def check_compact_export():
                    if time.monotonic() >= started + seconds:
                        raise TimeoutError("Compact label transfer exhausted the bound budget")
                compact_path = snapshot_dir / "negative_labels.jsonl"
                examples_hash = file_digest(snapshot_dir / "examples.lp")
                compact_hash = task.negatives.dump(compact_path, source_sha256=examples_hash,
                                                   check=check_compact_export)
                compact = {"compact_negative_file": str(compact_path),
                           "compact_negative_sha256": compact_hash,
                           "examples_sha256": examples_hash}
            if (strategy == "relational" and background_transfer == "snapshot"
                    and not (snapshot_dir / "background_model.lp").exists()):
                with (snapshot_dir / "background_model.lp").open("w", buffering=1024 * 1024) as stream:
                    for atom in task.atoms:
                        if time.monotonic() >= started + seconds:
                            raise TimeoutError("Ground-model transfer exhausted the bound budget")
                        stream.write(str(atom) + ".\n")
            data = {"task": {**compact, "background_file": str(snapshot_dir / "background.lp"),
                             "background_model_file": str(snapshot_dir / "background_model.lp"),
                             "examples_file": str(snapshot_dir / "examples.lp"),
                             "explicit_examples": task.examples_source is None,
                             "background_transfer": background_transfer,
                             "bias_source": task.bias_source, "max_body": task.max_body,
                             **{name: JsonArray(map(str, getattr(task, name)))
                                for name in ("heads", "literals", "positives")}},
                    "indices": JsonArray(index_json(i) for i in indices), "seconds": seconds,
                    "strategy": strategy, "fingerprint": fingerprint,
                    "per_target_bounds": per_target_bounds,
                    "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
                    "resume_path": str(resume_path.resolve()) if resume_path else None}
            payload_path = folder / "payload.json"
            write_json(payload_path, data, deadline=started + seconds)
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Preparation exceeded its wall-clock budget")
        except TimeoutError:
            return None, {"status": "timeout", "cost": None, "elapsed_seconds": time.monotonic() - started,
                          "reason": "Input preparation exhausted the bound budget; original bias retained"}
        process = subprocess.Popen([sys.executable, "-m", "nidavellir.bound_worker", str(payload_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, env=env)
        try:
            output, errors = process.communicate(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            if isinstance(error, KeyboardInterrupt):
                raise
            if checkpoint_path and checkpoint_path.exists():
                state = json.loads(checkpoint_path.read_text())
                target_certificates = (state.get("target_bounds", {})
                    if per_target_bounds and fingerprint and state.get("fingerprint") == fingerprint else {})
                if target_certificates:
                    from .target_bounds import costs_from_certificates
                    costs_from_certificates(task, target_certificates)
                incumbent = state.get("incumbent")
                verification = state.get("_verification", {})
                if (fingerprint and state.get("fingerprint") == fingerprint and incumbent
                        and verification.get("matched")
                        and verification.get("rules") == [c["rule"] for c in incumbent]):
                    cover = [dict(c, coverage=int(c["coverage"], 16)) for c in incumbent]
                    return cover, {"status": "verified", "cost": sum(c["cost"] for c in cover),
                                   "verification": verification, "rules": verification["rules"],
                                   "elapsed_seconds": time.monotonic() - started,
                                   "target_bounds": target_certificates,
                                   "reason": "Worker deadline reached; independently verified checkpoint incumbent retained"}
                if target_certificates:
                    return None, {"status": "timeout", "cost": None,
                                  "target_bounds": target_certificates,
                                  "elapsed_seconds": time.monotonic() - started,
                                  "reason": "Global worker deadline reached; verified per-target incumbents retained"}
            return None, {"status": "timeout", "cost": None, "elapsed_seconds": time.monotonic() - started,
                          "reason": "Hard wall-clock limit; original bias retained"}
    if process.returncode:
        raise RuntimeError("Bound worker failed: " + errors.decode(errors="replace"))
    cover, report = json.loads(output)
    if cover is not None:
        for c in cover:
            c["coverage"] = int(c["coverage"], 16)
    report["elapsed_seconds"] = time.monotonic() - started
    report["background_transfer"] = background_transfer if strategy == "relational" else "not_needed"
    report["timeout_policy"] = "isolated subprocess, wall-clock including preparation, startup and verification"
    return cover, report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version",
                   version=(f"Nidavellir {__version__} - Author: Yvon K. Awuklu - "
                            "Copyright (c) 2026"))
    for flag in ("background", "examples", "bias", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--optimizer", choices=("internal", "ilasp"), default="internal")
    p.add_argument("--proof-strategy", choices=("auto", "catalogue", "cegis"), default="auto",
                   help="auto proves strict relational representative optima by searching cheaper programs; catalogue retains the full selection pipeline")
    p.add_argument("--dynamic-program-bounds", action=argparse.BooleanOptionalAction, default=True,
                   help="revise verified cost bounds from exact rules discovered during strict relational compilation")
    p.add_argument("--allow-partial-coverage", action="store_true",
                   help="maximize covered positives, then minimize rule cost; negatives stay strict (default: require every positive)")
    p.add_argument("--internal-opt-strategy", choices=("usc,oll", "bb,lin"), default="usc,oll",
                   help="internal Clingo optimization strategy; bb,lin reproduces the previous default")
    p.add_argument("--verify-ilasp", action="store_true")
    p.add_argument("--ilasp-binary")
    p.add_argument("--enumerate-optima", action="store_true")
    p.add_argument("--enumeration-backend", choices=("auto", "clingo"), default="auto")
    p.add_argument("--enumeration-seconds", type=float,
                   help="separate enumeration budget; defaults to --solver-seconds")
    p.add_argument("--variant-seconds", type=float,
                   help="joint targeted reconstruction/expansion budget; defaults to --solver-seconds")
    p.add_argument("--max-optima", type=int)
    p.add_argument("--expand-variants", action="store_true")
    p.add_argument("--raw-alpha-variants", action="store_true",
                   help="Retain raw alpha spellings during collection (reference ablation)")
    p.add_argument("--positive-join-backend", choices=("indexed", "reference"), default="indexed",
                   help="Positive existence joins during catalogue compilation; bound worker uses its default")
    p.add_argument("--positive-core-pruning", action=argparse.BooleanOptionalAction, default=False,
                   help="Learn globally verified positive-failure cores and prune their extensions (opt-in)")
    p.add_argument("--bias-core-pruning", action=argparse.BooleanOptionalAction, default=False,
                   help="Reuse signed UNSAT cores of the exact bias in the parent compiler (opt-in)")
    p.add_argument("--no-positive-residual-cache", action="store_true",
                   help="Disable query-local reuse of completed failed positive joins (parent oracle)")
    p.add_argument("--no-positive-lookahead", action="store_true",
                   help="Disable early correlated BK support filters in positive joins (parent oracle)")
    p.add_argument("--no-positive-residual-normalization", action="store_true",
                   help="Keep verified fully bound literals in positive residual-cache keys (parent oracle)")
    p.add_argument("--no-positive-pair-pruning", action="store_true",
                   help="Disable exact empty-pair BK proofs in the indexed positive join")
    p.add_argument("--binding-backend", choices=("indexed", "reference"), default="indexed",
                   help="Fixed-label and negative joins during compilation; bound worker uses its default")
    p.add_argument("--negative-join-backend", choices=("labels", "bindings"), default="labels",
                   help="Join explicit negative-label tuples early, or retain BK-only traversal for ablation")
    p.add_argument("--negative-label-limit", type=int, default=65536,
                   help="Maximum materialized negative-label rows; compact bucket joins can exceed this count; 0 disables both (parent oracle)")
    p.add_argument("--no-exact-program-bound", action="store_true",
                   help="Use only atomic majorants for the verified program bound (reference ablation)")
    p.add_argument("--bound-positive-probes", type=int, default=24,
                   help="Exact positive counterexample probes before expensive negative queries; 0 disables")
    p.add_argument("--no-ground-prefix-filter", action="store_true",
                   help="Disable necessary ground prefix checks; exact completion remains unless separately disabled")
    p.add_argument("--no-prefix-completion", action="store_true",
                   help="Skip exact relational prefix feasibility queries; final candidate legality stays exact")
    p.add_argument("--target-mode", choices=("separate", "grouped"), default="separate",
                   help="Separate sequential target calls (default), or one grouped compilation/selection")
    p.add_argument("--target-signature", help="learn one declared predicate/arity while retaining its complete rule-level bias")
    p.add_argument("--per-target-bounds", action=argparse.BooleanOptionalAction, default=True,
                   help="Use verified target bounds in bitmask or relational compilation (default); budget is per call")
    p.add_argument("--max-programs", type=int)
    p.add_argument("--bound-seconds", type=float, default=30)
    p.add_argument("--bound-strategy", choices=("auto", "relational", "bitmask", "off"), default="auto")
    p.add_argument("--resume-bound", type=Path, help="compatible relational-bound checkpoint to resume")
    p.add_argument("--compiler-backend", choices=("auto", "bitmask", "relational"), default="auto")
    p.add_argument("--oracle-cache", type=Path,
                   help="directory for complete exact caches and reusable unlabelled optimized profiles")
    p.add_argument("--oracle-seconds", type=float, default=300)
    p.add_argument("--oracle-backend", choices=("optimized", "projected", "cartesian"), default="optimized",
                   help="optimized truth-vector classes (default), or projected/Cartesian reference oracles")
    p.add_argument("--compile-seconds", type=float, default=300)
    p.add_argument("--solver-seconds", type=float, default=300)
    p.add_argument("--max-classes", type=int, default=1_000_000,
                   help="maximum truth-vector classes for the projected oracle (default: 1000000)")
    p.add_argument("--max-witnesses", type=int, default=1_000_000,
                   help="maximum raw witnesses for --oracle-backend cartesian only (default: 1000000)")
    p.add_argument("--no-time-limit", action="store_true",
                   help="remove relational oracle, compilation, proof and selection deadlines; optional initial bound discovery retains its heuristic budget")
    p.add_argument("--no-verbose", dest="verbose", action="store_false", default=True,
                   help="disable the default compact progress on stderr")
    return p


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "stage": "inputs", "version": __version__,
              "backend": {"optimized": "guarded_reduced_bitmasks", "projected": "projected_truth_vector_bitmasks",
                          "cartesian": "cartesian_reference_bitmasks"}[args.oracle_backend], "optimal_cost": None,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    partial = args.allow_partial_coverage
    status.update(coverage_mode="partial" if partial else "strict",
                  objective="lexicographic_missed_positives_then_rule_cost" if partial else "minimum_cost_complete_cover")
    coverage_report = None
    cover = None
    task = None
    bound = {"status": "disabled", "cost": None}
    proof_enabled = False
    compilation_deadline = None
    terminal = TerminalProgress(args.verbose, target=args.target_signature)
    terminal.start(started)

    def update(**values):
        status.update(values, elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", status)
        if values:
            trace({"event": "status", **values})

    def trace(values):
        event = dict(values, wall_seconds=time.monotonic() - started)
        with (output / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        terminal.emit(event)

    def emit_rules(rules, cost, optimal):
        verification = verify_rules(task, rules, allow_partial_coverage=partial)
        write_json(output / "verification.json", verification)
        if not verification["matched"]:
            raise AssertionError("Learned rules failed independent original-BK verification")
        if partial:
            if coverage_report is None or verification["missing_positives"] != coverage_report["uncoverable_positives"]:
                raise AssertionError("Independent positive coverage disagrees with the compiled maximum")
            coverage_report.update(covered_positives=verification["covered_positives"],
                missed_positives=verification["missed_positives"], positive_recall=verification["positive_recall"],
                covered_negatives=len(verification["covered_negatives"]), cost_optimal=optimal,
                hypothesis_cost=cost, optimum_cost=cost if optimal else None)
            write_json(output / "coverage.json", coverage_report)
            update(coverage=coverage_report,
                   optimum_objective=[verification["missed_positives"], cost] if optimal else None)
        (output / "learned.lp").write_text("\n".join(rules) + "\n")
        (output / "predictions.lp").write_text("\n".join(a + "." for a in verification["predictions"]) + "\n")
        update(hypothesis_cost=cost, optimal_cost=cost if optimal else None,
               learned_rule_count=len(rules), semantic_verification=True)

    def accept_dynamic_incumbent(certificate):
        nonlocal cover, bound
        rows = [dict(row, coverage=int(row["coverage"], 16)
                     if isinstance(row["coverage"], str) else row["coverage"])
                for row in certificate["cover"]]
        if cover is not None and sum(row["cost"] for row in cover) <= certificate["cost"]:
            return
        cover = rows
        bound = dict(bound, status="verified", cost=certificate["cost"],
                     rules=certificate["rules"], verification=certificate["verification"],
                     incumbent_source="dynamic_compilation")
        write_json(output / "dynamic_incumbent.json", certificate)
        write_json(output / "bound.json", bound)
        if proof_enabled:
            raise _IncumbentReady()

    def finish_with_program_proof(oracle):
        nonlocal cover, bound
        from .program_proof import prove_program_optimality
        update(stage="program_proof", proof_strategy="cegis")
        remaining = (None if compilation_deadline is None else
                     max(0, compilation_deadline - time.monotonic()))
        cover, proof = prove_program_optimality(task, cover, seconds=remaining,
                                              oracle=oracle, progress=trace)
        write_json(output / "program_proof.json", proof)
        cost = sum(row["cost"] for row in cover)
        optimal = proof.get("optimality_proven") is True
        bound = dict(bound, cost=cost, rules=[row["rule"] for row in cover])
        emit_rules([row["rule"] for row in cover], cost, optimal)
        bound.update(status="verified", verification=json.loads((output / "verification.json").read_text()))
        write_json(output / "bound.json", bound)
        write_json(output / "solver.json", {
            "backend": "program_cegis", "status": "complete" if optimal else "timeout",
            "cost": cost, "optimum_cost": cost if optimal else None,
            "rules": [row["rule"] for row in cover], "selected": [],
            "scope": ("original finite rule language; proved absence of strictly cheaper programs" if optimal
                      else "original finite rule language; strict-improvement proof unfinished"),
            "catalogue_complete": False, "proof_file": "program_proof.json",
            "optimality_proven": optimal})
        update(status="complete" if optimal else "feasible", stage="finished" if optimal else "program_proof",
               optimality_source=("nonnegative_additive_rule_cost" if cost == 0 else
                                  "strict_improvement_unsat") if optimal else None,
               catalogue_complete=False, enumeration_complete=not args.enumerate_optima,
               variant_expansion_complete=not args.expand_variants,
               ilasp_verification="not_requested")
        if not optimal:
            update(reason=proof.get("reason", "Program proof deadline reached; verified incumbent retained"))
        raise _ProofFinished()

    update()
    try:
        task = Task.load(args.background, args.examples, args.bias)
        if args.target_signature:
            from .target_execution import select_signature
            task = select_signature(task, args.target_signature)
        task._bias_core_pruning = args.bias_core_pruning
        update(target_mode=args.target_mode,
               target_signature=args.target_signature,
               bound_budget_scope="per target call" if args.target_mode == "separate" or args.target_signature else "shared grouped call")
        snap = output / "inputs"
        snap.mkdir()
        sources = {"background.lp": task.background, "bias.lp": task.bias_source}
        for name, text in sources.items():
            (snap / name).write_text(text)
        task.write_examples(snap / "examples.lp")
        manifest = {"version": __version__, "clingo_version": clingo.__version__,
                    "python": sys.version, "task": task.stats(),
                    "arguments": status["arguments"],
                    "literal_templates": list(map(str, task.literals)),
                    "examples": {"snapshot": "inputs/examples.lp", "positive_count": len(task.positives),
                                 "negative_count": len(task.negatives),
                                 "scope": "Complete labels in the hashed snapshot; not duplicated in this manifest"},
                    "sha256": {n: file_digest(snap / n) for n in ("background.lp", "examples.lp", "bias.lp")},
                    "input_files_sha256": {n: file_digest(getattr(args, n)) for n in ("background", "examples", "bias")},
                    "engine_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob("*.py")}}
        write_json(output / "manifest.json", manifest)
        update(stage="planning", input_seconds=time.monotonic() - started)
        planning_started = time.monotonic()
        def planning_check():
            if args.oracle_seconds is not None and time.monotonic() >= planning_started + args.oracle_seconds:
                raise ResourceLimit("Backend planning time budget reached")
        if args.compiler_backend == "auto":
            from .planning import choose_backend
            compiler_backend, planning = choose_backend(task, check=planning_check)
        else:
            compiler_backend = args.compiler_backend
            planning = {"method": "explicit_user_choice", "chosen_backend": compiler_backend}
        planning["elapsed_seconds"] = time.monotonic() - planning_started
        write_json(output / "planning.json", planning)
        strategy = args.bound_strategy
        if partial:
            if args.resume_bound:
                raise TaskError("--allow-partial-coverage cannot resume a strict full-cover bound")
            strategy = "off"
        if strategy == "auto":
            strategy = "bitmask" if compiler_backend == "bitmask" else "relational"
        eligible = (compiler_backend == "relational" and not partial
                    and not args.enumerate_optima and not args.expand_variants
                    and args.optimizer == "internal" and not args.verify_ilasp)
        if args.proof_strategy == "cegis" and not eligible:
            raise TaskError("CEGIS proof requires strict relational representative execution with the internal optimizer and no ILASP verification")
        proof_enabled = eligible and args.proof_strategy != "catalogue"
        update(proof_strategy="cegis" if proof_enabled else "catalogue")
        if compiler_backend == "relational" and strategy == "bitmask":
            raise TaskError("--bound-strategy bitmask requires --compiler-backend bitmask")
        if args.resume_bound and strategy != "relational":
            raise TaskError("--resume-bound requires the relational bound strategy")
        if args.resume_bound and not args.resume_bound.is_file():
            raise TaskError("Bound checkpoint does not exist")
        update(compiler_backend=compiler_backend, bound_strategy=strategy,
               backend="relational_positive_bitmasks" if compiler_backend == "relational" else status["backend"])
        fingerprint = hashlib.sha256(json.dumps({"snapshots": manifest["sha256"],
                         "engine": manifest["engine_sha256"], "clingo": clingo.__version__}, sort_keys=True).encode()).hexdigest()
        bound = {"status": "disabled", "cost": None, "elapsed_seconds": 0}
        if partial:
            bound["reason"] = "Partial coverage compiles the full language without full-cover cost bounds"
        if not task.positives:
            cover, bound = [], {"status": "verified", "cost": 0, "rules": [], "elapsed_seconds": 0}
        elif strategy == "relational":
            update(stage="bound")
            cover, bound = bound_in_subprocess(task, [], args.bound_seconds, snap, strategy="relational",
                fingerprint=fingerprint, checkpoint_path=output / "bound_checkpoint.json", resume_path=args.resume_bound,
                per_target_bounds=args.per_target_bounds)
        if not partial:
            cover, bound = recover_target_cover(task, cover, bound)
        write_json(output / "bound.json", bound)
        if cover is not None:
            (output / "first_feasible.lp").write_text("\n".join(c["rule"] for c in cover) + "\n")
        update(stage="oracle")
        indices, relational_oracle = [], None
        oracle_started = time.monotonic()
        def oracle_check():
            if args.oracle_seconds is not None and time.monotonic() >= oracle_started + args.oracle_seconds:
                raise ResourceLimit("Oracle initialization time budget reached")
        if not task.positives:
            oracle_stats = {"status": "not_needed", "reason": "Empty positive set; empty program is optimal"}
        elif compiler_backend == "relational":
            from .relational_coverage import CoverageOracle
            relational_oracle = CoverageOracle(task, check=oracle_check,
                positive_core_pruning=args.positive_core_pruning,
                positive_join_backend=args.positive_join_backend,
                binding_backend=args.binding_backend,
                negative_join_backend=args.negative_join_backend,
                negative_label_limit=args.negative_label_limit,
                positive_pair_pruning=not args.no_positive_pair_pruning,
                positive_residual_cache=not args.no_positive_residual_cache,
                positive_lookahead=not args.no_positive_lookahead,
                positive_residual_normalization=not args.no_positive_residual_normalization)
            relational_oracle.check = lambda: None  # compilation installs its own deadline
            oracle_stats = {"backend": "relational_positive_bitmasks", "initialized": True,
                            "scope": "lazy conjunctive queries; no full witness catalogue",
                            "initialization_seconds": time.monotonic() - oracle_started}
        else:
            cached, key, profile_key = None, None, None
            if args.oracle_cache:
                from .oracle_cache import (fingerprint as cache_fingerprint, load_cache, save_cache,
                                           profile_fingerprint, load_profile_cache, save_profile_cache)
                key = cache_fingerprint(task, backend=args.oracle_backend)
                cached = load_cache(args.oracle_cache, key, task)
                oracle_check()
                if cached is None:
                    profile_key = profile_fingerprint(task, backend=args.oracle_backend, check=oracle_check)
                    if profile_key is not None:
                        cached = load_profile_cache(args.oracle_cache, profile_key, task, check=oracle_check)
            if cached is not None:
                indices, oracle_stats = cached
                if oracle_stats.get("cache", {}).get("scope") == "unlabelled_profiles":
                    # Preserve the exact per-task layer too; its masks have just
                    # been rebuilt from this task's current names and labels.
                    save_cache(args.oracle_cache, key, indices, oracle_stats)
            else:
                remaining = max(0, args.oracle_seconds - (time.monotonic() - oracle_started))
                if args.oracle_backend == "optimized":
                    from .optimized_oracle import build_optimized_indices
                    indices, oracle_stats = build_optimized_indices(task, seconds=remaining,
                        max_classes=args.max_classes, progress=trace)
                elif args.oracle_backend == "projected":
                    indices, oracle_stats = build_projected_indices(task, seconds=remaining,
                        max_classes=args.max_classes, progress=trace)
                else:
                    indices, oracle_stats = build_cartesian_indices(task, seconds=remaining,
                        max_witnesses=args.max_witnesses, progress=trace)
                if args.oracle_cache:
                    path = save_cache(args.oracle_cache, key, indices, oracle_stats)
                    oracle_stats["cache"] = {"hit": False, "scope": "exact", "path": str(path), "fingerprint": key}
                    if profile_key is not None:
                        profile_path = save_profile_cache(args.oracle_cache, profile_key, task,
                                                          indices, oracle_stats, check=oracle_check)
                        oracle_stats["cache"]["profile_path"] = str(profile_path)
            oracle_stats["phase_elapsed_seconds"] = time.monotonic() - oracle_started
        update(backend=oracle_stats.get("backend", status["backend"]))
        write_json(output / "oracle.json", {"stats": oracle_stats,
                   "indices": JsonArray(index_json(i, representatives=True) for i in indices)})
        if task.positives and strategy == "bitmask":
            update(stage="bound")
            cover, bound = bound_in_subprocess(task, indices, args.bound_seconds, snap,
                per_target_bounds=args.per_target_bounds, fingerprint=fingerprint,
                checkpoint_path=output / "bound_checkpoint.json" if args.per_target_bounds else None)
        write_json(output / "bound.json", bound)
        if cover is not None:
            (output / "first_feasible.lp").write_text("\n".join(c["rule"] for c in cover) + "\n")
        update(stage="compilation")
        compilation_deadline = None if args.compile_seconds is None else time.monotonic() + args.compile_seconds
        if proof_enabled and cover is not None:
            finish_with_program_proof(relational_oracle)
        from .target_bounds import costs_from_certificates
        target_costs = (costs_from_certificates(task, bound.get("target_bounds", {}))
                        if args.per_target_bounds and not partial else None)
        if compiler_backend == "relational" and task.positives:
            from .relational_compile import compile_catalogue as compile_relational
            try:
                candidates, compilation = compile_relational(task, relational_oracle, bound=bound["cost"],
                    seconds=args.compile_seconds, progress=trace, variants="representative",
                    alpha_collect=not args.raw_alpha_variants,
                    exact_program_bound=not args.no_exact_program_bound,
                    bound_positive_probes=args.bound_positive_probes,
                    ground_prefix_filter=not args.no_ground_prefix_filter,
                    prefix_completion=not args.no_prefix_completion,
                    target_bounds=target_costs,
                    dynamic_program_bound=args.dynamic_program_bounds and not partial,
                    on_incumbent=(accept_dynamic_incumbent
                                  if args.dynamic_program_bounds and not partial else None),
                    program_bound=(cover is not None and bound.get("status") == "verified"
                                   and bound["cost"] is not None and bound["cost"] > 0))
            except _IncumbentReady:
                finish_with_program_proof(relational_oracle)
        else:
            candidates, compilation = compile_catalogue(task, indices, bound=bound["cost"],
                seconds=args.compile_seconds, progress=trace, variants="representative",
                alpha_collect=not args.raw_alpha_variants, target_bounds=target_costs)
        serial = [dict(c, coverage=hex(c["coverage"])) for c in candidates]
        write_json(output / "catalogue.json", {"stats": compilation, "candidates": serial})
        if not compilation.get("complete"):
            raise ResourceLimit("Catalogue compilation did not finish with a completeness proof", compilation)
        if partial:
            from .solvers import _partial_catalogue
            _, quantified = _partial_catalogue(candidates, len(task.positives))
            labels = list(map(str, sorted(task.positives)))
            coverage_report = dict(quantified,
                uncoverable_positives=sorted(labels[i] for i in quantified["uncoverable_positive_indices"]),
                scope="complete supplied finite language; strict explicit negatives; training labels only",
                catalogue_complete=True, cost_optimal=False,
                covered_positives=None, missed_positives=None, positive_recall=None)
            write_json(output / "coverage.json", coverage_report)
            update(coverage=coverage_report)
        (output / "compiled_set_cover.lp").write_text(set_cover_program(candidates, len(task.positives),
                                                    allow_partial_coverage=partial))
        (output / "compiled_ilasp.las").write_text(selection_task(candidates, len(task.positives),
                                                 allow_partial_coverage=partial))
        update(stage="selection", proof_strategy="catalogue", catalogue_complete=True)
        selected = solve(candidates, len(task.positives), backend=args.optimizer,
                         ilasp_binary=args.ilasp_binary, timeout=args.solver_seconds,
                         internal_opt_strategy=args.internal_opt_strategy,
                         verify_ilasp=args.verify_ilasp,
                         enumerate_optima=args.enumerate_optima, enumeration_backend=args.enumeration_backend,
                         enumeration_timeout=args.enumeration_seconds or args.solver_seconds,
                         max_optima=args.max_optima, expand_variants=False,
                         allow_partial_coverage=partial)
        if selected["status"] == "complete":
            chosen = [c for c in candidates if c["id"] in selected["selected"]]
            emit_rules([c["rule"] for c in chosen], selected["optimum_cost"], True)
        if args.expand_variants and selected["status"] == "complete":
            update(stage="variants")
            variant_started = time.monotonic()
            variant_seconds = args.variant_seconds or args.solver_seconds
            hypotheses = selected.get("enumeration", {}).get("semantic_hypotheses", [])
            required = set().union(*(set(h) for h in hypotheses)) if hypotheses else set()
            targets = [c for c in candidates if c["id"] in required]
            try:
                if compiler_backend == "relational" and task.positives:
                    from .relational_compile import mine_variants
                    reconstructed, reconstruction = mine_variants(task, relational_oracle, targets,
                        seconds=variant_seconds, progress=trace, alpha_collect=not args.raw_alpha_variants,
                        ground_prefix_filter=not args.no_ground_prefix_filter,
                        prefix_completion=not args.no_prefix_completion)
                else:
                    from .search import mine_variants
                    reconstructed, reconstruction = mine_variants(task, indices, targets,
                        seconds=variant_seconds, progress=trace, alpha_collect=not args.raw_alpha_variants)
                write_json(output / "variant_catalogue.json", {"stats": reconstruction,
                           "candidates": JsonArray(dict(c, coverage=hex(c["coverage"])) for c in reconstructed)})
                remaining = variant_seconds - (time.monotonic() - variant_started)
                if remaining <= 0:
                    expansion = {"status": "timeout", "complete": False, "programs": [],
                                 "reason": "Targeted reconstruction exhausted the joint variant budget"}
                else:
                    expansion = expand_optimal_programs(reconstructed, hypotheses,
                        timeout=remaining, max_programs=args.max_programs)
                expansion["reconstruction"] = reconstruction
            except ResourceLimit as error:
                expansion = {"status": "timeout", "complete": False, "programs": [],
                             "reason": str(error), "reconstruction": dict(error.stats, complete=False)}
            semantic_complete = selected.get("enumeration", {}).get("complete", False)
            expansion.update(semantic_enumeration_complete=semantic_complete,
                             all_optimal_programs_complete=expansion["complete"] and semantic_complete,
                             elapsed_seconds=time.monotonic() - variant_started,
                             scope="targeted reconstruction for the selected optimal semantic signatures")
            selected["variant_expansion"] = expansion
        elif args.expand_variants and selected["status"] == "unsatisfiable":
            selected["variant_expansion"] = {"status": "complete", "complete": True, "programs": [],
                "semantic_enumeration_complete": True, "all_optimal_programs_complete": True,
                "scope": "No optimal programs for a proven unsatisfiable task"}
        write_json(output / "solver.json", selected)
        if "enumeration" in selected:
            write_json(output / "enumeration.json", selected["enumeration"])
        if "variant_expansion" in selected:
            write_json(output / "variants.json", selected["variant_expansion"])
        if selected["status"] == "complete":
            verified = selected.get("verification", {}).get("matches", True)
            enumeration_complete = (not args.enumerate_optima or selected.get("enumeration", {}).get("complete", False))
            variants_complete = (not args.expand_variants or selected.get("variant_expansion", {}).get("all_optimal_programs_complete", False))
            final_status = ("verification_incomplete" if verified is not True else
                            "enumeration_incomplete" if not (enumeration_complete and variants_complete) else "complete")
            update(status=final_status, stage="finished", enumeration_complete=enumeration_complete,
                   variant_expansion_complete=variants_complete,
                   ilasp_verification=selected.get("verification", {}).get("status", "not_requested"))
        elif selected["status"] == "unsatisfiable":
            if cover is not None:
                raise AssertionError("Complete catalogue contradicts verified feasible bound")
            verified = selected.get("verification", {}).get("matches", True)
            update(status="unsatisfiable" if verified is True else "verification_incomplete", stage="finished",
                   selection_status="unsatisfiable",
                   ilasp_verification=selected.get("verification", {}).get("status", "not_requested"))
        elif selected["status"] == "timeout":
            chosen = [c for c in candidates if c["id"] in selected["selected"]]
            if selected.get("cost") is not None and (cover is None or selected["cost"] <= bound["cost"]):
                emit_rules([c["rule"] for c in chosen], selected["cost"], False)
                update(status="feasible", stage="selection")
            elif cover is not None:
                emit_rules([c["rule"] for c in cover], bound["cost"], False)
                update(status="feasible", stage="selection")
            else:
                update(status="incomplete", reason="Selection time budget reached")
        else:
            update(status="error", reason=selected.get("error") or selected.get("stderr") or "Solver error")
    except _ProofFinished:
        pass
    except ResourceLimit as error:
        write_json(output / "resource_limit.json", {"stage": status["stage"], "reason": str(error), "stats": error.stats})
        certificate = error.stats.get("dynamic_incumbent")
        if certificate is not None and not partial:
            proof_enabled = False  # No search time remains after this deadline.
            accept_dynamic_incumbent(certificate)
        if cover is not None and not partial:
            emit_rules([c["rule"] for c in cover], sum(c["cost"] for c in cover), False)
            update(status="feasible", reason=str(error))
        else:
            update(status="incomplete", reason=str(error))
    except KeyboardInterrupt:
        update(status="interrupted", reason="User interruption")
    except (TaskError, ValueError, RuntimeError, OSError, AssertionError) as error:
        update(status="invalid_input" if isinstance(error, TaskError) else "error", reason=str(error))
    finally:
        update()
        terminal.close()
        lines = ["# Nidavellir execution", "", f"Status: **{status['status']}**.",
                 f"Stage: `{status['stage']}`. Elapsed: {status['elapsed_seconds']:.3f} s.",
                 f"Oracle backend: `{status['backend']}`.",
                 f"Proof strategy: `{status.get('proof_strategy', 'not_started')}`.",
                 f"Proven optimum: {status['optimal_cost']}.", "",
                 ("Positives are maximized before minimizing rule cost; supplied negatives remain hard."
                  if partial else "Inputs are ground hard labels. Unlabelled outputs are unconstrained."),
                 "ILASP verification, when requested, checks selection from the compiled catalogue.",
                 "`verification.json` checks learned clauses independently in the original BK."]
        if (output / "program_proof.json").exists():
            lines += ["", "`program_proof.json` records strict-improvement search and its proof state; "
                      "a complete catalogue is not claimed."]
        if coverage_report is not None:
            lines += ["", f"Maximum positive coverage: {coverage_report['maximum_covered_positives']}/{coverage_report['positive_count']}.",
                      f"Uncoverable positives in this language: {coverage_report['minimum_missed_positives']}.",
                      "`coverage.json` lists the uncovered labels and separates coverage and cost proofs."]
        if "reason" in status:
            lines += ["", status["reason"]]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] in ("complete", "unsatisfiable") else 2 if status["status"] in ("error", "invalid_input") else 3


def main():
    p = parser()
    args = p.parse_args()
    if args.bound_positive_probes < 0:
        p.error("bound-positive-probes must be nonnegative")
    if args.negative_label_limit < 0:
        p.error("negative-label-limit must be nonnegative")
    for key in ("bound_seconds", "oracle_seconds", "compile_seconds", "solver_seconds"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key != "bound_seconds" and value == 0):
            p.error(f"{key} must be finite and {'nonnegative' if key == 'bound_seconds' else 'positive'}")
    if args.max_witnesses < 1:
        p.error("max-witnesses must be positive")
    if args.max_classes < 1:
        p.error("max-classes must be positive")
    if args.enumeration_seconds is not None and (not math.isfinite(args.enumeration_seconds) or args.enumeration_seconds <= 0):
        p.error("enumeration-seconds must be finite and positive")
    if args.variant_seconds is not None and (not math.isfinite(args.variant_seconds) or args.variant_seconds <= 0):
        p.error("variant-seconds must be finite and positive")
    for name in ("max_optima", "max_programs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(name.replace("_", "-") + " must be positive")
    if args.expand_variants and not args.enumerate_optima:
        p.error("--expand-variants requires --enumerate-optima")
    if args.max_programs is not None and not args.expand_variants:
        p.error("--max-programs requires --expand-variants")
    if args.max_optima is not None and not args.enumerate_optima:
        p.error("--max-optima requires --enumerate-optima")
    if args.verify_ilasp and args.optimizer != "internal":
        p.error("--verify-ilasp verifies the internal optimizer; use --optimizer internal")
    if args.allow_partial_coverage and args.resume_bound:
        p.error("--allow-partial-coverage cannot resume a strict full-cover bound")
    if args.proof_strategy == "cegis" and (args.allow_partial_coverage or args.enumerate_optima
            or args.expand_variants or args.optimizer != "internal" or args.verify_ilasp
            or args.compiler_backend == "bitmask"):
        p.error("--proof-strategy cegis requires strict relational representative optimization without ILASP")
    if args.no_time_limit:
        if args.compiler_backend != "relational" or args.enumerate_optima or args.expand_variants:
            p.error("--no-time-limit currently requires relational representative execution")
        args.oracle_seconds = args.compile_seconds = args.solver_seconds = None
    if args.output.exists():
        p.error("output directory already exists; use a new directory")
    if args.target_mode == "separate" and not args.target_signature:
        from .target_execution import dispatch_separate
        return dispatch_separate(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
