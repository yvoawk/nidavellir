"""Sequential isolated learning of independent target signatures.

The full rule-level bias is preserved. Only head enumeration and labels are
restricted; target independence is checked against all original signatures.
Costs add, and complete optimal program sets compose by Cartesian product.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from . import __version__
from .task import Task, TaskError, _background, _bias, _examples, signature
from .target_bounds import key_text, target_view
from .terminal_progress import TerminalProgress


def select_signature(task, text):
    for key in {signature(h) for h in task.heads}:
        if key_text(key) == text:
            return target_view(task, key)
    raise TaskError(f"Unknown target signature: {text}")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def prepare(args, output, bias):
    """Partition labels without grounding the BK or rewriting bias metadata."""
    from .__main__ import file_digest, write_json
    source, heads, literals, max_body, diagnostics = bias
    keys = sorted({signature(h) for h in heads})
    background, _, messages = _background(args.background, set(keys), evaluate=False)
    examples, positives, negatives, example_messages = _examples(args.examples, set(keys))
    task = Task(background, frozenset(), source, examples, heads, literals, max_body,
                positives, negatives)
    snap = output / "inputs"
    snap.mkdir()
    (snap / "background.lp").write_text(background)
    (snap / "bias.lp").write_text(source)
    task.write_examples(snap / "examples.lp")
    plan = []
    for position, key in enumerate(keys):
        identifier = f"{position:03d}_{re.sub(r'[^a-zA-Z0-9_]', '_', key[0])}_{key[1]}"
        view = target_view(task, key)
        path = snap / "targets" / identifier / "examples.lp"
        path.parent.mkdir(parents=True)
        view.write_examples(path)
        plan.append({"signature": key_text(key), "id": identifier,
                     "head_templates": list(map(str, view.heads)),
                     "positive_count": len(view.positives), "negative_count": len(view.negatives),
                     "examples": str(path.relative_to(output)), "examples_sha256": file_digest(path),
                     "output": f"targets/{identifier}", "status": "pending"})
    manifest = {"version": __version__, "target_mode": "separate", "targets": plan,
                "positive_count": len(positives), "negative_count": len(negatives),
                "diagnostics": {"bias": diagnostics, "background": messages, "examples": example_messages},
                "sha256": {p.name: file_digest(p) for p in snap.glob("*.lp")},
                "input_files_sha256": {n: file_digest(getattr(args, n)) for n in ("background", "bias", "examples")},
                "engine_sha256": {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
                "budget_scope": "Per target call; composition has an additional enumeration/variant budget",
                "decomposition": "Disjoint predicate/arity signatures, absent from BK and learned bodies; additive rule costs"}
    write_json(output / "manifest.json", manifest)
    return plan


def child_command(args, output, target):
    from .__main__ import parser
    command = [sys.executable, "-m", "nidavellir",
               "--background", str(output / "inputs/background.lp"),
               "--bias", str(output / "inputs/bias.lp"),
               "--examples", str(output / target["examples"]),
               "--output", str(output / target["output"]),
               "--target-mode", "grouped", "--target-signature", target["signature"]]
    excluded = {"help", "version", "background", "bias", "examples", "output",
                "target_mode", "target_signature"}
    for action in parser()._actions:
        if action.dest in excluded:
            continue
        value = getattr(args, action.dest)
        if isinstance(action, argparse.BooleanOptionalAction):
            command.append(action.option_strings[0 if value else 1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                command.append(action.option_strings[0])
        elif value is not None:
            command.extend((action.option_strings[0], str(value)))
    return command


def launch(command, logs, env, started_callback, verbose=True):
    stderr_path = logs.with_suffix(".stderr.log")
    with logs.with_suffix(".stdout.log").open("w") as stdout, stderr_path.open("w") as stderr:
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=subprocess.PIPE,
                                 text=True, bufsize=1)

        def relay_stderr():
            for line in child.stderr:
                stderr.write(line)
                stderr.flush()
                if verbose:
                    sys.stderr.write(line)
                    sys.stderr.flush()

        relay = threading.Thread(target=relay_stderr, name="nidavellir-child-stderr", daemon=True)
        relay.start()
        try:
            started_callback(child.pid)
            code = child.wait()
            relay.join()
            return code
        except KeyboardInterrupt:
            # Let the child's normal handler dispose of its own bound worker.
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            child.wait()
            relay.join()
            raise


def product_report(parts, *, variants, cost, limit, seconds):
    """Bound the global product too; reaching exactly the cap is complete."""
    started = time.monotonic()
    field = "programs" if variants else "semantic_hypotheses"
    local_complete = all(p.get("all_optimal_programs_complete" if variants else "complete", False)
                         for _, p in parts)
    values, seen = [], set()
    result_status = "complete" if local_complete else "incomplete"
    for combination in itertools.product(*(p.get(field, []) for _, p in parts)):
        if time.monotonic() - started >= seconds:
            result_status = "timeout"
            break
        if variants:
            rules = sorted(rule for program in combination for rule in program["rules"])
            key = tuple(rules)
            value = {"rules": rules, "cost": cost,
                     "selected": [f"{name}:{c}" for (name, _), p in zip(parts, combination)
                                  for c in p.get("selected", [])]}
        else:
            value = [f"{name}:{c}" for (name, _), hypothesis in zip(parts, combination) for c in hypothesis]
            key = tuple(value)
        if key in seen:
            continue
        if limit is not None and len(values) >= limit:
            result_status = "limit"
            break
        seen.add(key)
        values.append(value)
    complete = result_status == "complete"
    result = {"status": result_status, "complete": complete, "backend": "target_cartesian_product",
              "optimum_cost": cost, field: values, "local_enumerations_complete": local_complete,
              "elapsed_seconds": time.monotonic() - started,
              "program_count" if variants else "hypothesis_count": len(values)}
    if variants:
        result["all_optimal_programs_complete"] = complete
    return result


def compose(args, output, plan):
    """Publish global proofs only when the independent results justify them."""
    from .__main__ import write_json
    states = [t.get("result", {}) for t in plan]
    folders = [output / t["output"] for t in plan]
    verified = all(s.get("semantic_verification") is True and
                   read_json(p / "verification.json").get("matched") is True and
                   (p / "learned.lp").exists() and (p / "predictions.lp").exists()
                   for s, p in zip(states, folders))
    optimal = verified and all(s.get("optimal_cost") is not None for s in states)
    cost = sum(s["optimal_cost"] for s in states) if optimal else None
    invalid = any(s.get("status") == "invalid_input" for s in states)
    failed = any(s.get("status") == "error" for s in states)
    unsat = any(s.get("status") == "unsatisfiable" for s in states)
    final = ("invalid_input" if invalid else "error" if failed else "unsatisfiable" if unsat
             else "complete" if optimal else "feasible" if verified else "incomplete")
    result = {"status": final, "optimal_cost": cost, "semantic_verification": verified,
              "completed_targets": sum(s.get("status") in ("complete", "unsatisfiable") for s in states),
              "target_count": len(plan)}
    solvers = [read_json(p / "solver.json") for p in folders]
    solver = {"backend": "sequential_targets", "status": final, "optimum_cost": cost,
              "selected": [f"{t['signature']}:{c}" for t, s in zip(plan, solvers) for c in s.get("selected", [])],
              "targets": {t["signature"]: t["output"] + "/solver.json" for t in plan},
              "scope": "Additive composition of independently solved target signatures"}
    if verified:
        for name in ("learned.lp", "predictions.lp"):
            with (output / name).open("w") as stream:
                for folder in folders:
                    with (folder / name).open() as source:
                        shutil.copyfileobj(source, stream)
                    stream.write("\n")
        result.update(hypothesis_cost=sum(s["hypothesis_cost"] for s in states),
                      learned_rule_count=sum(s["learned_rule_count"] for s in states))
        solver["cost"] = result["hypothesis_cost"]
        write_json(output / "verification.json", {
            "matched": True, "missing_positives": [], "covered_negatives": [],
            "predictions_file": "predictions.lp",
            "scope": "Composition of independently verified disjoint target programs in the same original BK",
            "targets": {t["signature"]: t["output"] + "/verification.json" for t in plan}})
    checks = [s.get("verification", {}) for s in solvers]
    if args.verify_ilasp:
        matches = all(c.get("matches") is True for c in checks)
        solver["verification"] = {"matches": matches, "status": "complete" if matches else "incomplete",
                                  "scope": "ILASP selection verification per target; additive composition",
                                  "targets": {t["signature"]: c.get("status", "unavailable")
                                              for t, c in zip(plan, checks)}}
        result["ilasp_verification"] = solver["verification"]["status"]
        if optimal and not matches:
            result["status"] = "verification_incomplete"
    for variants, requested, filename, cap, seconds, status_key in (
        (False, args.enumerate_optima, "enumeration.json", args.max_optima,
         args.enumeration_seconds or args.solver_seconds, "enumeration_complete"),
        (True, args.expand_variants, "variants.json", args.max_programs,
         args.variant_seconds or args.solver_seconds, "variant_expansion_complete")):
        if not requested:
            continue
        parts = [(t["signature"], read_json(p / filename)) for t, p in zip(plan, folders)]
        if not optimal:
            parts = [(t["signature"], {"complete": final == "unsatisfiable",
                                       "all_optimal_programs_complete": final == "unsatisfiable"}) for t in plan]
        expansion = product_report(parts, variants=variants, cost=cost, limit=cap, seconds=seconds)
        write_json(output / filename, expansion)
        solver["variant_expansion" if variants else "enumeration"] = expansion
        result[status_key] = expansion["complete"]
        if optimal and not expansion["complete"] and result["status"] == "complete":
            result["status"] = "enumeration_incomplete"
    bounds = {t["signature"]: read_json(p / "bound.json").get("cost") for t, p in zip(plan, folders)}
    bound_cost = sum(bounds.values()) if all(v is not None for v in bounds.values()) else None
    write_json(output / "bound.json", {"status": "verified" if bound_cost is not None else "incomplete",
        "cost": bound_cost, "target_costs": bounds, "scope": "Sum of local feasible bounds; certificates remain in target directories"})
    write_json(output / "solver.json", solver)
    return result


def dispatch_separate(args):
    from .__main__ import run, write_json
    try:
        bias = _bias(args.bias)
    except (TaskError, OSError):
        return run(args)  # Retain the normal input-error artifacts.
    if len({signature(h) for h in bias[1]}) == 1:
        return run(args)  # Preserve the flat single-target API and artifact layout.
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = []
    status = {"status": "running", "stage": "inputs", "version": __version__,
              "target_mode": "separate", "backend": "sequential_targets", "optimal_cost": None,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "bound_budget_scope": "per target call", "fresh_process_per_target": True}
    terminal = TerminalProgress(args.verbose)

    def update(**values):
        status.update(values, elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", status)
        write_json(output / "targets.json", {"targets": plan})
        event = dict(values, event="status", wall_seconds=status["elapsed_seconds"])
        with (output / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if values:
            terminal.emit(event)

    update()
    try:
        if args.resume_bound:
            raise TaskError("--resume-bound accepts one target checkpoint; use a single-target call or --target-mode grouped")
        plan = prepare(args, output, bias)
        (output / "logs").mkdir()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        for target in plan:
            target.update(status="running", started_seconds=time.monotonic() - started)
            command = child_command(args, output, target)
            target["command"] = command
            update(stage="target:" + target["signature"], active_target=target["signature"])
            def record_pid(pid):
                target["pid"] = pid
                update()
            code = launch(command, output / "logs" / target["id"], env, record_pid, args.verbose)
            state = read_json(output / target["output"] / "status.json")
            expected_code = (0 if state.get("status") in ("complete", "unsatisfiable") else
                             2 if state.get("status") in ("error", "invalid_input") else 3)
            if not state or state.get("status") == "running" or code != expected_code:
                state = dict(state, status="error", reason=f"Target process exited without a consistent final status (code {code})")
            target.update(result=state, status=state["status"], exit_code=code,
                          finished_seconds=time.monotonic() - started)
            update()
        update(stage="composition", active_target=None)
        update(**compose(args, output, plan), stage="finished")
    except KeyboardInterrupt:
        update(status="interrupted", reason="User interruption; completed target artifacts retained")
    except (TaskError, ValueError, RuntimeError, OSError, AssertionError) as error:
        update(status="invalid_input" if isinstance(error, TaskError) else "error", reason=str(error))
    finally:
        update()
        lines = ["# Sequential target execution", "", f"Status: **{status['status']}**.",
                 f"Elapsed: {status['elapsed_seconds']:.3f} s. Proven optimum: {status['optimal_cost']}.", "",
                 "Each predicate/arity is learned in a fresh sequential process with its own phase budgets.",
                 "The complete original bias is retained; same-signature head templates stay together.",
                 "Catalogues, local mask positions, certificates and ILASP checks remain in each target directory.",
                 "Global optima are additive; enumerations are Cartesian products with separate composition budgets.",
                 "Completed target artifacts survive failures of another target.", ""]
        lines += [f"- [{t['signature']}]({t['output']}/report.md): {t['status']}" for t in plan]
        if status.get("reason"):
            lines += ["", status["reason"]]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] in ("complete", "unsatisfiable") else 2 if status["status"] in ("error", "invalid_input") else 3
