"""Run prepared cases sequentially, recording complete and limited executions.

Each task folder contains background.lp, bias.lp and examples.lp.  The engine
is always invoked in a fresh subprocess; no learned rules or checkpoints are
reused.  An external deadline also covers input grounding and native calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time


ENGINE = Path(__file__).resolve().parents[1]
REPOSITORY = ENGINE.parents[1]
INPUT_NAMES = ("background.lp", "bias.lp", "examples.lp")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def engine_hashes():
    return {p.name: digest(p) for p in sorted(ENGINE.glob("*.py"))}


def input_hashes(folder):
    # Include provenance and other prepared artifacts, not only the entry files.
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def discover(case_folder):
    tasks = {}
    for examples in sorted(case_folder.rglob("examples.lp")):
        folder = examples.parent
        if all((folder / name).is_file() for name in INPUT_NAMES):
            relative = folder.relative_to(case_folder)
            name = str(relative) if relative.parts else case_folder.name
            tasks[name] = folder
    return tasks


def process_table():
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="],
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ps returned a nonzero exit code")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, parent, rss_kib = map(int, fields)
            rows[pid] = (parent, rss_kib)
    return rows


def process_family(root_pid, rows, previous=()):
    family = {root_pid, *(pid for pid in previous if pid in rows)}
    while True:
        enlarged = family | {pid for pid, (parent, _) in rows.items() if parent in family}
        if enlarged == family:
            return family
        family = enlarged


def kill_processes(process, family):
    # The bound worker starts its own process group.  Kill observed descendants
    # explicitly as well as the compiler's group so those workers do not linger.
    try:
        family = process_family(process.pid, process_table(), family)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError):
        pass
    for pid in sorted(family - {process.pid}, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def leading_stats(path):
    """Read the first stats object without loading potentially huge mask arrays."""
    if not path.exists():
        return None
    decoder = json.JSONDecoder()
    prefix = ""
    with path.open() as stream:
        while chunk := stream.read(65536):
            prefix += chunk
            match = re.match(r'\s*\{\s*"stats"\s*:\s*', prefix)
            if match is None:
                return None
            try:
                return decoder.raw_decode(prefix, match.end())[0]
            except json.JSONDecodeError:
                continue
    return None


def phase_times(trace_path, compiler_status, subprocess_seconds):
    """Stage wall times, including an unfinished final stage after a hard stop."""
    events = []
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "status" and "stage" in event:
                events.append(event)
    phases = {}
    phase, since = "inputs", 0.0
    for event in events:
        if event["stage"] != phase:
            ended = event["wall_seconds"]
            if phase != "finished":
                phases[phase] = phases.get(phase, 0) + ended - since
            phase, since = event["stage"], ended
    status = compiler_status or {}
    end = (status.get("elapsed_seconds", subprocess_seconds)
           if status.get("status") not in (None, "running") else subprocess_seconds)
    if phase != "finished":
        phases[phase] = phases.get(phase, 0) + max(0, end - since)
    return {name: round(value, 6) for name, value in phases.items()}


def command_for(args, inputs, output):
    command = [args.python, "-m", "nidavellir",
               "--background", str(inputs / "background.lp"),
               "--bias", str(inputs / "bias.lp"),
               "--examples", str(inputs / "examples.lp"),
               "--output", str(output), "--compiler-backend", args.compiler_backend,
               "--oracle-backend", args.oracle_backend, "--bound-strategy", args.bound_strategy,
               "--optimizer", args.optimizer]
    for name in ("bound_seconds", "oracle_seconds", "compile_seconds", "solver_seconds", "max_classes"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.verify_ilasp:
        command.append("--verify-ilasp")
    if args.ilasp_binary:
        command += ["--ilasp-binary", args.ilasp_binary]
    if getattr(args, "oracle_cache", None):
        command += ["--oracle-cache", str(args.oracle_cache)]
    command += ["--target-mode", args.target_mode]
    if getattr(args, "target_signature", None):
        command += ["--target-signature", args.target_signature]
    if getattr(args, "positive_core_pruning", None) is not None:
        command.append("--positive-core-pruning" if args.positive_core_pruning else "--no-positive-core-pruning")
    if getattr(args, "bias_core_pruning", None) is not None:
        command.append("--bias-core-pruning" if args.bias_core_pruning else "--no-bias-core-pruning")
    command.append("--per-target-bounds" if args.per_target_bounds else "--no-per-target-bounds")
    if getattr(args, "no_prefix_completion", False):
        command.append("--no-prefix-completion")
    for flag in ("enumerate_optima", "expand_variants"):
        if getattr(args, flag, False):
            command.append("--" + flag.replace("_", "-"))
    return command


def run_task(args, name, inputs, destination, frozen_engine, frozen_inputs, monitor_error):
    destination.mkdir(parents=True, exist_ok=False)
    compiler_output = destination / "compiler"
    command = command_for(args, inputs, compiler_output)
    record = {"task": name, "case": args.case, "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "cwd": str(REPOSITORY),
              "cold_start": not bool(args.oracle_cache), "fresh_process": True,
              "oracle_cache_directory": str(args.oracle_cache) if args.oracle_cache else None,
              "engine_sha256": frozen_engine, "input_sha256": frozen_inputs,
              "wall_limit_seconds": args.wall_seconds, "rss_limit_mib": args.max_rss_mb,
              "rss_monitor": {"available": monitor_error is None, "error": monitor_error,
                              "method": "sampled sum of RSS for observed compiler descendants",
                              "units": "MiB", "poll_seconds": args.poll_seconds,
                              "shared_pages_may_be_counted_more_than_once": True},
              "peak_rss_mib": None, "subprocess_wall_seconds": None}
    write_json(destination / "benchmark.json", record)
    if engine_hashes() != frozen_engine or input_hashes(inputs) != frozen_inputs:
        record.update(status="inputs_changed", reason="Engine or prepared inputs changed before execution")
        write_json(destination / "benchmark.json", record)
        return record
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + env.get("PYTHONPATH", "")
    family = set()
    peak_kib = None
    terminal_reason = None
    last_stage, last_progress = None, started
    with (destination / "stdout.log").open("w") as stdout, (destination / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
        except OSError as error:
            record.update(status="error", reason=str(error), exit_code=None,
                          finished_utc=datetime.now(timezone.utc).isoformat(),
                          subprocess_wall_seconds=round(time.monotonic() - started, 6))
            write_json(destination / "benchmark.json", record)
            return record
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if monitor_error is None:
                    try:
                        rows = process_table()
                        family = process_family(process.pid, rows, family)
                        rss_kib = sum(rows[pid][1] for pid in family if pid in rows)
                        peak_kib = max(peak_kib or 0, rss_kib)
                        if args.max_rss_mb and rss_kib > args.max_rss_mb * 1024:
                            terminal_reason = "rss_limit"
                    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                        monitor_error = str(error)
                        record["rss_monitor"].update(available=False, error=monitor_error)
                        if args.max_rss_mb:
                            terminal_reason = "rss_monitor_failed"
                if elapsed >= args.wall_seconds:
                    terminal_reason = terminal_reason or "wall_limit"
                state = read_json(compiler_output / "status.json") or {}
                if state.get("stage") != last_stage or time.monotonic() - last_progress >= 30:
                    last_stage, last_progress = state.get("stage"), time.monotonic()
                    print(json.dumps({"task": name, "elapsed_seconds": round(elapsed, 3),
                                      "stage": last_stage, "peak_rss_mib": None if peak_kib is None else round(peak_kib / 1024, 3)},
                                     ensure_ascii=False), flush=True)
                if terminal_reason:
                    kill_processes(process, family)
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            terminal_reason = "interrupted"
            kill_processes(process, family)
        except BaseException:
            kill_processes(process, family)
            raise
    elapsed = time.monotonic() - started
    compiler_status = read_json(compiler_output / "status.json")
    resource_limit = read_json(compiler_output / "resource_limit.json")
    engine_unchanged = engine_hashes() == frozen_engine
    inputs_unchanged = input_hashes(inputs) == frozen_inputs
    status = ("incomplete" if terminal_reason else (compiler_status or {}).get("status", "error"))
    if status == "running":
        status = "error"
    if not engine_unchanged or not inputs_unchanged:
        status = "inputs_changed"
    record.update(status=status, termination_reason=terminal_reason,
                  finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=process.returncode,
                  subprocess_wall_seconds=round(elapsed, 6), compiler_status=compiler_status,
                  engine_unchanged=engine_unchanged, inputs_unchanged=inputs_unchanged,
                  phase_wall_seconds=phase_times(compiler_output / "trace.jsonl", compiler_status, elapsed),
                  peak_rss_mib=None if peak_kib is None else round(peak_kib / 1024, 3),
                  resource_limit=resource_limit, oracle_stats=leading_stats(compiler_output / "oracle.json"),
                  bound=read_json(compiler_output / "bound.json"),
                  compilation_stats=leading_stats(compiler_output / "catalogue.json"),
                  solver=read_json(compiler_output / "solver.json"))
    write_json(destination / "benchmark.json", record)
    print(json.dumps({"task": name, "status": status, "subprocess_wall_seconds": record["subprocess_wall_seconds"],
                      "termination_reason": terminal_reason}, ensure_ascii=False), flush=True)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True,
                   help="Name of the prepared input collection below --inputs-root")
    p.add_argument("--inputs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+", help="Task folder names relative to the case folder; default: all")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--optimizer", choices=("internal", "ilasp"), default="internal")
    p.add_argument("--compiler-backend", choices=("auto", "bitmask", "relational"), default="auto")
    p.add_argument("--oracle-backend", choices=("optimized", "projected", "cartesian"), default="optimized")
    p.add_argument("--bound-strategy", choices=("auto", "relational", "bitmask", "off"), default="auto")
    p.add_argument("--no-prefix-completion", action="store_true",
                   help="Ablate relational prefix feasibility; retain exact final candidate legality")
    p.add_argument("--target-mode", choices=("separate", "grouped"), default="separate")
    p.add_argument("--target-signature", help="Restrict each prepared task to this predicate/arity")
    p.add_argument("--positive-core-pruning", action=argparse.BooleanOptionalAction, default=None,
                   help="Ablate global positive-failure cores; omitted uses the compiler default")
    p.add_argument("--bias-core-pruning", action=argparse.BooleanOptionalAction, default=None,
                   help="Ablate signed bias UNSAT cores; omitted uses the compiler default")
    p.add_argument("--per-target-bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enumerate-optima", action="store_true")
    p.add_argument("--expand-variants", action="store_true")
    p.add_argument("--oracle-cache", type=Path,
                   help="Shared complete oracle cache; hits and source timing remain explicit in oracle stats")
    p.add_argument("--verify-ilasp", action="store_true")
    p.add_argument("--ilasp-binary")
    p.add_argument("--bound-seconds", type=float, default=30)
    p.add_argument("--oracle-seconds", type=float, default=300)
    p.add_argument("--compile-seconds", type=float, default=300)
    p.add_argument("--solver-seconds", type=float, default=120)
    p.add_argument("--max-classes", type=int, default=1_000_000)
    p.add_argument("--wall-seconds", type=float, default=900)
    p.add_argument("--max-rss-mb", type=float, default=4096,
                   help="Summed sampled process RSS in MiB; 0 explicitly disables enforcement")
    p.add_argument("--poll-seconds", type=float, default=0.5)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ("bound_seconds", "oracle_seconds", "compile_seconds", "solver_seconds", "wall_seconds", "max_rss_mb", "poll_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (value == 0 and name not in ("bound_seconds", "max_rss_mb")):
            p.error(name + " must be finite and positive (bound/RSS limits may be zero)")
    if args.max_classes < 1:
        p.error("--max-classes must be positive")
    if args.verify_ilasp and args.optimizer != "internal":
        p.error("--verify-ilasp requires --optimizer internal")
    args.output = args.output.resolve()
    if args.oracle_cache:
        args.oracle_cache = args.oracle_cache.resolve()
    tasks = discover(args.inputs_root.resolve() / args.case)
    if args.targets:
        unknown = set(args.targets) - tasks.keys()
        if unknown:
            p.error("Unknown task folders: " + ", ".join(sorted(unknown)))
        tasks = {name: tasks[name] for name in args.targets}
    if not tasks:
        p.error("No complete task folders found under --inputs-root/--case")
    if args.output.exists():
        p.error("Output already exists; choose a fresh directory")
    monitor_error = None
    try:
        process_table()
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        monitor_error = str(error)
    if monitor_error and args.max_rss_mb:
        p.error("RSS monitoring unavailable: " + monitor_error + "; run with process-list permission or explicitly set --max-rss-mb 0")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_engine = engine_hashes()
    frozen_inputs = {name: input_hashes(folder) for name, folder in tasks.items()}
    (args.output / "engine").mkdir()
    for name in frozen_engine:
        (args.output / "engine" / name).write_bytes((ENGINE / name).read_bytes())
    summary = {"case": args.case, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "python": sys.version, "platform": platform.platform(), "engine_sha256": frozen_engine,
               "harness_sha256": digest(Path(__file__)), "rss_monitor_error": monitor_error,
               "timing_policy": "Sequential fresh compiler subprocesses; no checkpoint or learned-rule reuse. Optional complete oracle cache is explicit; hits are reported. Input preparation is excluded.",
               "tasks": {}, "task_order": list(tasks)}
    started = time.monotonic()
    write_json(args.output / "summary.json", summary)
    for name, folder in tasks.items():
        result = run_task(args, name, folder, args.output / name, frozen_engine, frozen_inputs[name], monitor_error)
        summary["tasks"][name] = result
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["subprocess_seconds_total"] = sum(r["subprocess_wall_seconds"] or 0 for r in summary["tasks"].values())
        write_json(args.output / "summary.json", summary)
        if result.get("termination_reason") == "interrupted" or result["status"] == "inputs_changed":
            break
    summary.update(status="complete" if len(summary["tasks"]) == len(tasks) and all(
        r["status"] == "complete" for r in summary["tasks"].values()) else "incomplete",
        finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
    write_json(args.output / "summary.json", summary)
    return 0 if summary["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
