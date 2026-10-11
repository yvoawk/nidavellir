"""Exact bitmask oracle for legal positive bodies, with proved reductions.

Unlike the raw projected reference, profiles describe guarded literals after
existential elimination. Coverage is preserved for every legal body, including
its negative-witness test. Class diagnostics store compact core assignments;
they are not claimed to be full raw witnesses.
"""
from itertools import chain, product
from math import isfinite, prod
import time

import clingo

from .bias_analysis import analyze
from .conditional_anchors import plan_anchors, restrict_domains, direct_profiles
from .oracle import MaskIndex, ResourceLimit, match
from .reduction import guarded_relation, plan_reduction
from .witness_space import WitnessSpace


class CompactClasses:
    """Keep interned symbols and integer profiles; render diagnostics on demand."""
    def __init__(self, slots):
        self.slots = tuple(slots)
        self.rows = []

    def append(self, head, truth, binding):
        self.rows.append((head, truth, tuple(binding[v] for v in self.slots)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        head, truth, values = self.rows[position]
        return {"head": str(head), "truth": hex(truth), "multiplicity": None,
                "core_assignment": {str(v): str(x) for v, x in zip(self.slots, values)},
                "semantics": "guarded literals; eliminated variables existentially quantified"}

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def build_optimized_indices(task, *, seconds=300, max_classes=1_000_000, progress=None):
    if not isfinite(seconds) or seconds < 0 or type(max_classes) is not int or max_classes < 1:
        raise ValueError("Nonnegative finite seconds and positive integer max_classes are required")
    started = time.monotonic()
    deadline = started + seconds
    stats = {"backend": "guarded_reduced_bitmasks", "materialized_witnesses": 0,
             "compressed_witnesses": 0, "labelled_head_instances": 0,
             "encoded_relation_rows": 0, "local_complement_tuples_examined": 0,
             "complete": False, "reductions": {}, "conditional_anchors": {},
             "direct_assignments_examined": 0, "direct_label_instances": 0,
             "projected_label_instances": 0}

    def check():
        stats["elapsed_seconds"] = time.monotonic() - started
        if time.monotonic() >= deadline:
            raise ResourceLimit("Optimized oracle time budget reached", dict(stats))

    check()
    analysis = getattr(task, "_bias_analysis", None)
    if analysis is None:
        analysis = analyze(task, seconds=seconds, check=check)
        task._bias_analysis = analysis
    stats["bias_analysis"] = analysis.stats
    space = WitnessSpace(task, check)
    positives = {a: i for i, a in enumerate(sorted(task.positives))}
    indices = []
    for head in task.heads:
        head_analysis = analysis.heads[head]
        if not head_analysis.legal_body_exists:
            continue
        plan = getattr(task, "_reduction_plans", {}).get(head)
        if plan is None:
            plan = plan_reduction(task, head, head_analysis, space, check)
        stats["reductions"][str(head)] = plan.stats
        anchors = plan_anchors(task, head, head_analysis, space, plan, check)
        stats["conditional_anchors"][str(head)] = {
            str(v): [i for i, _, _ in carriers] for v, carriers in anchors.items()}
        retained = sorted(space.all_vars - plan.eliminated)
        buffers = [bytearray() for _ in task.literals]
        classes = CompactClasses(retained)
        positive_heads = []
        positive_mask = negative_mask = 0
        # Do not allocate a union plus a sorted copy of all explicit negatives.
        for label in chain(sorted(task.positives), task.negatives):
            check()
            fixed = match(head, label)
            if fixed is None:
                continue
            stats["labelled_head_instances"] += 1
            stats["current_head"] = str(label)
            domains = restrict_domains(space.domains(fixed), fixed, anchors, space)
            domain_ids = {v: tuple(space.intern(s) for s in values)
                          for v, values in domains.items() if v not in plan.eliminated}
            first_position = len(classes)
            if anchors and prod(len(values) for values in domain_ids.values()) <= 65536:
                stats["direct_label_instances"] += 1
                profiles = direct_profiles(task, space, plan, fixed, domains, check, stats)
            else:
                stats["projected_label_instances"] += 1
                profiles = _projected_profiles(task, space, plan, fixed, domains, domain_ids,
                                               check, deadline, stats)
            try:
                for truth, binding in profiles:
                    check()
                    if stats["compressed_witnesses"] >= max_classes:
                        raise ResourceLimit("Optimized class limit reached", dict(stats))
                    position = len(classes)
                    classes.append(label, truth, binding)
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
                        progress(dict(stats))
            finally:
                profiles.close()
            mask = ((1 << (len(classes) - first_position)) - 1) << first_position
            if label in positives:
                positive_mask |= mask
                positive_heads.append((positives[label], mask))
            else:
                negative_mask |= mask
            if progress:
                progress(dict(stats))
        indices.append(MaskIndex(head, tuple(int.from_bytes(b, "little") for b in buffers),
                                 (1 << len(classes)) - 1, positive_mask, negative_mask,
                                 tuple(positive_heads), classes))
    check()
    stats.update(complete=True, relation_caches=dict(space.stats))
    return indices, stats


def _projected_profiles(task, space, plan, fixed, domains, domain_ids, check, deadline, stats):
    control = clingo.Control(["0", "--project=project", "--warn=none"])
    assignments, truths = {}, []
    with control.backend() as backend:
        for v, values in domain_ids.items():
            check()
            atoms = []
            for value in values:
                atom = backend.add_atom(clingo.Function("rc_value", [clingo.Number(v), clingo.Number(value)]))
                assignments[v, value] = atom
                atoms.append(atom)
            backend.add_rule(atoms, [], choice=True)
            backend.add_rule([], [-a for a in atoms])
            backend.add_weight_rule([], 2, [(a, 1) for a in atoms])
        for i in range(len(task.literals)):
            check()
            truth = backend.add_atom(clingo.Function("rc_truth", [clingo.Number(i)]))
            truths.append(truth)
            slots, rows = guarded_relation(space, plan, i, fixed, domains, check)
            local_count = prod(len(domain_ids[v]) for v in slots)
            if len(rows) > local_count // 2:
                bad = backend.add_atom()
                backend.add_rule([truth], [-bad])
                for n, row in enumerate(product(*(domain_ids[v] for v in slots))):
                    if n % 1024 == 0:
                        check()
                    stats["local_complement_tuples_examined"] += 1
                    if row not in rows:
                        backend.add_rule([bad], [assignments[v, x] for v, x in zip(slots, row)])
                        stats["encoded_relation_rows"] += 1
            else:
                for n, row in enumerate(rows):
                    if n % 1024 == 0:
                        check()
                    backend.add_rule([truth], [assignments[v, x] for v, x in zip(slots, row)])
                    stats["encoded_relation_rows"] += 1
        backend.add_project(truths)
    with control.solve(yield_=True, async_=True) as handle:
        handle.resume()
        while True:
            check()
            if not handle.wait(max(0, deadline - time.monotonic())):
                handle.cancel()
                check()
                raise ResourceLimit("Optimized enumeration interrupted", dict(stats))
            model = handle.model()
            if model is None:
                result = handle.get()
                break
            truth = sum(1 << i for i, atom in enumerate(truths) if model.is_true(atom))
            binding = dict(fixed)
            for symbol in model.symbols(atoms=True):
                if symbol.name == "rc_value":
                    v, x = (arg.number for arg in symbol.arguments)
                    binding[v] = space.symbols[x]
            yield truth, binding
            handle.resume()
    if result.unknown or not result.exhausted:
        raise ResourceLimit("Optimized oracle lacks completeness proof", dict(stats))
