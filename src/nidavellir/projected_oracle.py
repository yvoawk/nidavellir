"""Construct exact truth classes without expanding complete witness products.

Each model chooses one value per hidden slot. Per-literal relations determine
its truth atoms. Clingo projects enumeration onto those truth atoms, so each
returned model represents one full truth vector, not one raw substitution.
Only local relation products may be enumerated to encode a sparse complement.
"""
from itertools import product
from math import isfinite, prod
import time

import clingo

from .oracle import MaskIndex, ResourceLimit, match
from .witness_space import WitnessSpace


def build_projected_indices(task, *, seconds=300, max_classes=1_000_000, progress=None):
    if not isfinite(seconds) or seconds < 0 or type(max_classes) is not int or max_classes < 1:
        raise ValueError("Nonnegative finite seconds and positive integer max_classes are required")
    started = time.monotonic()
    deadline = started + seconds
    stats = {"backend": "projected_truth_vector_bitmasks", "conceptual_cartesian_witnesses": 0,
             "materialized_witnesses": 0, "compressed_witnesses": 0,
             "labelled_head_instances": 0, "positive_relations": 0,
             "complement_relations": 0, "encoded_relation_rows": 0,
             "local_complement_tuples_examined": 0,
             "multiplicities": "not_counted", "complete": False}

    def check():
        stats["elapsed_seconds"] = time.monotonic() - started
        if time.monotonic() >= deadline:
            raise ResourceLimit("Projected oracle time budget reached", dict(stats))

    check()
    space = WitnessSpace(task, check)
    positives = {a: i for i, a in enumerate(sorted(task.positives))}
    labels = sorted(task.positives | task.negatives)
    indices = []
    for head in task.heads:
        buffers = [bytearray() for _ in task.literals]
        classes = []
        head_masks = {}
        for label in labels:
            check()
            fixed = match(head, label)
            if fixed is None:
                continue
            stats["labelled_head_instances"] += 1
            stats["current_head"] = str(label)
            domains = space.domains(fixed)
            stats["conceptual_cartesian_witnesses"] += prod(map(len, domains.values()))
            control = clingo.Control(["0", "--project=project", "--warn=none"])
            domain_ids = {v: tuple(space.intern(s) for s in values) for v, values in domains.items()}
            assignments = {}
            truths = []
            with control.backend() as backend:
                for v, values in domain_ids.items():
                    check()
                    atoms = []
                    for value in values:
                        atom = backend.add_atom(clingo.Function("rc_value", [clingo.Number(v), clingo.Number(value)]))
                        assignments[v, value] = atom
                        atoms.append(atom)
                    backend.add_rule(atoms, [], choice=True)
                    backend.add_rule([], [-a for a in atoms])  # at least one
                    backend.add_weight_rule([], 2, [(a, 1) for a in atoms])  # at most one
                for i, literal in enumerate(task.literals):
                    check()
                    truth = backend.add_atom(clingo.Function("rc_truth", [clingo.Number(i)]))
                    truths.append(truth)
                    slots, rows = space.relation(i, fixed, domains)
                    local_count = prod(len(domain_ids[v]) for v in slots)
                    if len(rows) > local_count // 2:
                        # Dense tables (e.g. inequality) are much smaller as a
                        # list of forbidden tuples. This default negation is in
                        # the oracle encoding, never in a learned rule or BK.
                        stats["complement_relations"] += 1
                        bad = backend.add_atom()
                        backend.add_rule([truth], [-bad])
                        source = product(*(domain_ids[v] for v in slots))
                        for n, row in enumerate(source):
                            if n % 1024 == 0:
                                check()
                            stats["local_complement_tuples_examined"] += 1
                            if row not in rows:
                                backend.add_rule([bad], [assignments[v, x] for v, x in zip(slots, row)])
                                stats["encoded_relation_rows"] += 1
                    else:
                        stats["positive_relations"] += 1
                        for n, row in enumerate(sorted(rows)):
                            if n % 1024 == 0:
                                check()
                            backend.add_rule([truth], [assignments[v, x] for v, x in zip(slots, row)])
                            stats["encoded_relation_rows"] += 1
                backend.add_project(truths)
            check()
            seen = set()
            first_position = len(classes)
            with control.solve(yield_=True, async_=True) as handle:
                handle.resume()
                while True:
                    stats["elapsed_seconds"] = time.monotonic() - started
                    remaining = max(0, deadline - time.monotonic())
                    if not remaining or not handle.wait(remaining):
                        handle.cancel()
                        stats["elapsed_seconds"] = time.monotonic() - started
                        raise ResourceLimit("Projected model enumeration time budget reached", dict(stats))
                    model = handle.model()
                    if model is None:
                        result = handle.get()
                        break
                    truth = sum(1 << i for i, atom in enumerate(truths) if model.is_true(atom))
                    if truth in seen:
                        raise AssertionError("Projected enumeration repeated a truth vector")
                    if stats["compressed_witnesses"] >= max_classes:
                        handle.cancel()
                        raise ResourceLimit("Compressed witness class limit reached", dict(stats))
                    seen.add(truth)
                    binding = dict(fixed)
                    for atom in model.symbols(atoms=True):
                        if atom.name == "rc_value":
                            v, x = (arg.number for arg in atom.arguments)
                            binding[v] = space.symbols[x]
                    position = len(classes)
                    classes.append({"head": str(label), "truth": hex(truth), "multiplicity": None,
                                    "representative": {str(v): str(x) for v, x in sorted(binding.items())}})
                    byte, offset = divmod(position, 8)
                    if offset == 0:
                        for buffer in buffers:
                            buffer.append(0)
                    bits = truth
                    while bits:
                        b = bits & -bits
                        buffers[b.bit_length() - 1][byte] |= 1 << offset
                        bits ^= b
                    stats["compressed_witnesses"] += 1
                    if progress and stats["compressed_witnesses"] % 1000 == 0:
                        progress(dict(stats, elapsed_seconds=time.monotonic() - started))
                    handle.resume()
            if result.unknown or not result.exhausted:
                raise ResourceLimit("Projected enumeration did not finish with a completeness proof", dict(stats))
            if not seen:
                raise AssertionError("Nonempty completed slot domains must admit a witness")
            head_masks[label] = ((1 << (len(classes) - first_position)) - 1) << first_position
            if progress:
                progress(dict(stats, elapsed_seconds=time.monotonic() - started))
        positive_mask = negative_mask = 0
        positive_heads = []
        for label, mask in head_masks.items():
            if label in positives:
                positive_mask |= mask
                positive_heads.append((positives[label], mask))
            else:
                negative_mask |= mask
        indices.append(MaskIndex(head, tuple(int.from_bytes(b, "little") for b in buffers), (1 << len(classes)) - 1,
                                 positive_mask, negative_mask, tuple(positive_heads), classes))
    check()
    stats["relation_caches"] = dict(space.stats)
    stats["complete"] = True
    return indices, stats
