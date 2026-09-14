"""Content-addressed, complete-only oracle caches with streamed diagnostics."""
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import time
from heapq import merge

import clingo

from .oracle import MaskIndex, match
from .optimized_oracle import CompactClasses
from .labels import canonical_text, canonical_records


SCHEMA = 1
PROFILE_SCHEMA = 1


def fingerprint(task, *, backend="optimized", parameters=None):
    digest = hashlib.sha256()
    for text in (task.background, task.bias_source,
                 json.dumps({"backend": backend, "parameters": parameters or {}, "schema": SCHEMA,
                             "clingo_version": clingo.__version__}, sort_keys=True),
                 "\n".join(map(str, task.heads)), "\n".join(map(str, task.literals))):
        data = text.encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    # Hash the actual labels, independent of source layout and set iteration.
    # Compact labels stream canonical text without interning outer symbols.
    for polarity, labels in ((b"positive", task.positives), (b"negative", task.negatives)):
        digest.update(polarity)
        for label in canonical_text(labels):
            data = label.encode()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def save_cache(directory, key, indices, stats):
    if not stats.get("complete"):
        raise ValueError("An incomplete oracle cannot be cached as complete")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (key + ".jsonl")
    temporary = directory / (key + f".{os.getpid()}.tmp")
    checksum = hashlib.sha256()
    with temporary.open("w") as stream:
        def write(record):
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            checksum.update(line.encode())
            stream.write(line)
        write({"type": "header", "schema": SCHEMA, "fingerprint": key, "stats": stats})
        for index in indices:
            compact = isinstance(index.classes, CompactClasses)
            write({"type": "index", "head": str(index.head),
                   "literal_masks": [hex(m) for m in index.literal_masks],
                   "all_mask": hex(index.all_mask), "positive_mask": hex(index.positive_mask),
                   "negative_mask": hex(index.negative_mask),
                   "positive_heads": [(i, hex(mask)) for i, mask in index.positive_heads],
                   "class_count": len(index.classes), "format": "core" if compact else "raw",
                   "slots": index.classes.slots if compact else []})
            if compact:
                for head, truth, values in index.classes.rows:
                    write({"type": "class", "head": str(head), "truth": hex(truth),
                           "values": list(map(str, values))})
            else:
                for entry in index.classes:
                    write({"type": "class", "value": entry})
        stream.write(json.dumps({"type": "complete", "sha256": checksum.hexdigest()}) + "\n")
    temporary.replace(destination)
    return destination


def load_cache(directory, key, task):
    started = time.monotonic()
    path = Path(directory) / (key + ".jsonl")
    if not path.exists():
        return None
    checksum = hashlib.sha256()
    indices, current, expected_count, stats = [], None, None, None
    complete = False

    def validate():
        if current is None:
            return
        if len(current.classes) != expected_count or current.all_mask != (1 << expected_count) - 1:
            raise ValueError("Cached class count does not match its masks")
        if len(current.literal_masks) != len(task.literals) or current.head not in task.heads:
            raise ValueError("Cached index does not match task templates")
        if current.positive_mask & current.negative_mask or current.positive_mask | current.negative_mask != current.all_mask:
            raise ValueError("Cached label masks do not partition the witnesses")
        if any(mask < 0 or mask & ~current.all_mask for mask in current.literal_masks):
            raise ValueError("Cached literal mask outside witness universe")
        union = 0
        for i, mask in current.positive_heads:
            if not 0 <= i < len(task.positives) or mask < 0 or mask & ~current.positive_mask:
                raise ValueError("Invalid cached positive-head mapping")
            union |= mask
        if union != current.positive_mask:
            raise ValueError("Cached positive-head mapping is incomplete")

    with path.open() as stream:
        for number, line in enumerate(stream):
            item = json.loads(line)
            if complete:
                raise ValueError("Unexpected content after cache completion marker")
            if item.get("type") == "complete":
                if item.get("sha256") != checksum.hexdigest():
                    raise ValueError("Oracle cache checksum mismatch")
                validate()
                complete = True
                continue
            checksum.update(line.encode())
            kind = item.get("type")
            if number == 0:
                if (kind != "header" or item.get("schema") != SCHEMA
                        or item.get("fingerprint") != key or not item.get("stats", {}).get("complete")):
                    raise ValueError("Incompatible or incomplete oracle cache")
                stats = item["stats"]
            elif kind == "index":
                validate()
                classes = CompactClasses(item["slots"]) if item["format"] == "core" else []
                current = MaskIndex(clingo.parse_term(item["head"]),
                                    tuple(int(m, 16) for m in item["literal_masks"]),
                                    int(item["all_mask"], 16), int(item["positive_mask"], 16),
                                    int(item["negative_mask"], 16),
                                    tuple((i, int(m, 16)) for i, m in item["positive_heads"]), classes)
                expected_count = item["class_count"]
                indices.append(current)
            elif kind == "class" and current is not None:
                if isinstance(current.classes, CompactClasses):
                    current.classes.rows.append((clingo.parse_term(item["head"]), int(item["truth"], 16),
                                                 tuple(clingo.parse_term(s) for s in item["values"])))
                else:
                    current.classes.append(item["value"])
            else:
                raise ValueError("Malformed oracle cache record")
    if not complete or stats is None:
        raise ValueError("Oracle cache was interrupted before completion")
    return indices, dict(stats, elapsed_seconds=time.monotonic() - started,
                         cache={"hit": True, "scope": "exact", "path": str(path),
                                "fingerprint": key, "source_elapsed_seconds": stats.get("elapsed_seconds")})


def _profile_context(task, *, backend="optimized", parameters=None, check=lambda: None):
    """Prove the effective oracle contract before ignoring names or polarities.

    The first implementation requires one head template. Its name never occurs
    in the target-free BK; its arguments and polarity are retained in the key.
    Different legal grammars may share profiles only if their proved guards,
    unusable literals and effective elimination plan are exactly the same.
    """
    if backend != "optimized" or len(task.heads) != 1:
        return None
    from .bias_analysis import analyze
    from .reduction import plan_reduction
    from .witness_space import WitnessSpace

    check()
    head, = task.heads
    analysis = getattr(task, "_bias_analysis", None)
    if analysis is None:
        analysis = analyze(task, check=check)
        task._bias_analysis = analysis
    head_analysis = analysis.heads[head]
    if not head_analysis.legal_body_exists:
        return None
    plans = getattr(task, "_reduction_plans", {})
    plan = plans.get(head)
    if plan is None:
        plan = plan_reduction(task, head, head_analysis, WitnessSpace(task, check), check)
        task._reduction_plans = dict(plans)
        task._reduction_plans[head] = plan
    contract = {
        "schema": PROFILE_SCHEMA, "backend": backend, "parameters": parameters or {},
        "clingo_version": clingo.__version__,
        "head_arguments": list(map(str, head.arguments)), "head_positive": head.positive,
        "literals": list(map(str, task.literals)),
        "guards": [sorted(guard) for guard in plan.guards],
        "unusable": sorted(plan.unusable), "private": sorted(plan.private),
        "functional": {str(i): sorted(slots) for i, slots in plan.functional.items()},
        "eliminated": sorted(plan.eliminated),
    }
    digest = hashlib.sha256()

    def add(value):
        data = value.encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    add(task.background)
    add(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    # Sort symbol handles, not copies of millions of label strings. Explicit
    # labels are disjoint by Task's contract. The one-head restriction makes
    # their order independent of the renamed predicate and of label polarity.
    labelled = merge(canonical_records(task.positives), canonical_records(task.negatives),
                     key=lambda atom: atom.order_key)
    for position, label in enumerate(labelled):
        if position % 256 == 0:
            check()
        fixed = match(head, label)
        if fixed is not None:
            add(json.dumps([(v, str(value)) for v, value in sorted(fixed.items())],
                           separators=(",", ":")))
    for path in sorted(Path(__file__).parent.glob("*.py")):
        check()
        add(path.name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), head, plan, analysis


def profile_fingerprint(task, *, backend="optimized", parameters=None, check=lambda: None):
    """Key for reusable, unlabelled profiles, or None outside the safe scope."""
    context = _profile_context(task, backend=backend, parameters=parameters, check=check)
    return None if context is None else context[0]


def save_profile_cache(directory, key, task, indices, stats, *, check=lambda: None):
    """Save complete optimized profiles without positive/negative assignments."""
    context = _profile_context(task, check=check)
    if context is None or context[0] != key:
        raise ValueError("Profile cache key does not match the proved task contract")
    _, head, _, _ = context
    if not stats.get("complete") or len(indices) != 1:
        raise ValueError("Only a complete one-head oracle can be cached as reusable profiles")
    index, = indices
    if index.head != head or not isinstance(index.classes, CompactClasses):
        raise ValueError("Reusable profiles require optimized compact classes for the expected head")
    if (index.all_mask != (1 << len(index.classes)) - 1
            or len(index.literal_masks) != len(task.literals)):
        raise ValueError("Profile class count does not match its masks")
    directory = Path(directory) / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (key + ".jsonl")
    temporary = directory / (key + f".{os.getpid()}.tmp")
    checksum = hashlib.sha256()
    try:
        with temporary.open("w") as stream:
            def write(record):
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                checksum.update(line.encode())
                stream.write(line)
            write({"type": "header", "schema": PROFILE_SCHEMA, "fingerprint": key,
                   "complete": True, "source_elapsed_seconds": stats.get("elapsed_seconds"),
                   "source_head": str(head), "class_count": len(index.classes),
                   "slots": list(index.classes.slots)})
            full_truth = (1 << len(task.literals)) - 1
            for position, (label, truth, values) in enumerate(index.classes.rows):
                if position % 256 == 0:
                    check()
                if (match(head, label) is None or label not in task.positives and label not in task.negatives
                        or truth < 0 or truth & ~full_truth or len(values) != len(index.classes.slots)):
                    raise ValueError("Invalid optimized profile row")
                # A profile belongs to a ground binding, never to a polarity.
                write({"type": "class", "arguments": list(map(str, label.arguments)),
                       "truth": hex(truth), "values": list(map(str, values))})
            stream.write(json.dumps({"type": "complete", "sha256": checksum.hexdigest()}) + "\n")
        check()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_profile_cache(directory, key, task, *, check=lambda: None):
    """Remap head names and derive every label mask from the current task."""
    started = time.monotonic()
    context = _profile_context(task, check=check)
    if context is None or context[0] != key:
        raise ValueError("Profile cache key does not match the proved task contract")
    _, head, plan, analysis = context
    path = Path(directory) / "profiles" / (key + ".jsonl")
    if not path.exists():
        return None
    from .task import variables
    expected_slots = tuple(sorted(set().union(variables(head), *(variables(literal)
                                   for literal in task.literals)) - plan.eliminated))
    positives = {label: i for i, label in enumerate(sorted(task.positives))}
    checksum = hashlib.sha256()
    header = None
    classes = CompactClasses(expected_slots)
    buffers = [bytearray() for _ in task.literals]
    label_buffers = {}
    complete = False
    full_truth = (1 << len(task.literals)) - 1
    with path.open() as stream:
        for number, line in enumerate(stream):
            if number % 256 == 0:
                check()
            item = json.loads(line)
            if complete:
                raise ValueError("Unexpected content after profile cache completion marker")
            if item.get("type") == "complete":
                if header is None or item.get("sha256") != checksum.hexdigest():
                    raise ValueError("Profile cache checksum mismatch")
                if len(classes) != header["class_count"]:
                    raise ValueError("Profile cache class count mismatch")
                complete = True
                continue
            checksum.update(line.encode())
            if number == 0:
                if (item.get("type") != "header" or item.get("schema") != PROFILE_SCHEMA
                        or item.get("fingerprint") != key or item.get("complete") is not True
                        or tuple(item.get("slots", ())) != expected_slots
                        or type(item.get("class_count")) is not int or item["class_count"] < 0):
                    raise ValueError("Incompatible or incomplete profile cache")
                header = item
                continue
            if item.get("type") != "class" or header is None:
                raise ValueError("Malformed profile cache record")
            label = clingo.Function(head.name, [clingo.parse_term(s) for s in item["arguments"]], head.positive)
            values = tuple(clingo.parse_term(s) for s in item["values"])
            truth = int(item["truth"], 16)
            if (match(head, label) is None or label not in positives and label not in task.negatives
                    or len(values) != len(expected_slots) or truth < 0 or truth & ~full_truth):
                raise ValueError("Cached profile is outside the current task contract")
            position = len(classes)
            if position >= header["class_count"]:
                raise ValueError("Too many cached profiles")
            classes.rows.append((label, truth, values))
            byte, offset = divmod(position, 8)
            if offset == 0:
                for buffer in buffers:
                    buffer.append(0)
            bits = truth
            while bits:
                bit = bits & -bits
                buffers[bit.bit_length() - 1][byte] |= 1 << offset
                bits ^= bit
            # Lists of positions avoid repeatedly copying growing big integers.
            label_buffers.setdefault(label, []).append(position)
    if not complete:
        raise ValueError("Profile cache was interrupted before completion")
    positive_buffer = bytearray((len(classes) + 7) // 8)
    positive_heads = []
    for number, (label, positions) in enumerate(label_buffers.items()):
        if number % 256 == 0:
            check()
        if label not in positives:
            continue
        buffer = bytearray((positions[-1] + 8) // 8)
        for position in positions:
            byte, offset = divmod(position, 8)
            buffer[byte] |= 1 << offset
            positive_buffer[byte] |= 1 << offset
        positive_heads.append((positives[label], int.from_bytes(buffer, "little")))
    positive_heads.sort()
    positive_mask = int.from_bytes(positive_buffer, "little")
    all_mask = (1 << len(classes)) - 1
    index = MaskIndex(head, tuple(int.from_bytes(buffer, "little") for buffer in buffers),
                      all_mask, positive_mask, all_mask ^ positive_mask, tuple(positive_heads), classes)
    check()
    stats = {"backend": "guarded_reduced_bitmasks", "complete": True,
             "compressed_witnesses": len(classes), "labelled_head_instances": len(label_buffers),
             "reductions": {str(head): plan.stats}, "bias_analysis": analysis.stats,
             "elapsed_seconds": time.monotonic() - started,
             "cache": {"hit": True, "scope": "unlabelled_profiles", "path": str(path),
                       "fingerprint": key, "source_head": header["source_head"],
                       "source_elapsed_seconds": header["source_elapsed_seconds"]}}
    return [index], stats
