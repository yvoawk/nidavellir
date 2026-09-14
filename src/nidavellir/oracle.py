"""Generic finite witnesses, exact truth-vector compression and bitmask joins.

Only labelled output atoms are materialised: unlabelled atoms are unconstrained.
Truth is always membership in the positive background model computed by Clingo.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
import time

import clingo

from .task import variables, instantiate


class ResourceLimit(RuntimeError):
    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = stats or {}


def match(pattern, atom, fixed=None):
    binding = dict(fixed or {})

    def visit(p, a):
        if p.type == clingo.SymbolType.Function and p.name == "var__":
            v = p.arguments[0].number
            if v in binding:
                return binding[v] == a
            binding[v] = a
            return True
        if p.type != a.type:
            return False
        if p.type != clingo.SymbolType.Function:
            return p == a
        return (p.name == a.name and p.positive == a.positive
                and len(p.arguments) == len(a.arguments)
                and all(visit(x, y) for x, y in zip(p.arguments, a.arguments)))

    return binding if visit(pattern, atom) else None


@dataclass
class MaskIndex:
    head: object
    literal_masks: tuple
    all_mask: int
    positive_mask: int
    negative_mask: int
    positive_heads: tuple  # (global positive index, witness mask)
    classes: list

    def coverage(self, mask):
        return sum(1 << i for i, h in self.positive_heads if h & mask)

    def body_mask(self, body):
        result = self.all_mask
        for i in body:
            result &= self.literal_masks[i]
        return result


def build_indices(task, *, seconds=300, max_witnesses=1_000_000, progress=None):
    started = time.monotonic()
    deadline = started + seconds
    stats = {"raw_witnesses": 0, "compressed_witnesses": 0,
             "labelled_head_instances": 0, "backend": "cartesian_reference_bitmasks"}

    def check():
        stats["elapsed_seconds"] = time.monotonic() - started
        if time.monotonic() >= deadline:
            raise ResourceLimit("Oracle time budget reached", dict(stats))

    relations = {}
    for a in task.atoms:
        relations.setdefault((a.name, len(a.arguments)), []).append(a)
    # Argument indexes restrict matches using known head values (entity, time,
    # etc.), without assuming particular argument positions or predicate names.
    argument_indexes = {}
    projections = {}

    def rows(literal, fixed):
        sig = (literal.name, len(literal.arguments))
        candidates = relations.get(sig, ())
        for pos, arg in enumerate(literal.arguments):
            if not variables(arg) <= fixed.keys():
                continue
            key = sig, pos
            if key not in argument_indexes:
                table = {}
                for row in relations.get(sig, ()):
                    table.setdefault(row.arguments[pos], []).append(row)
                argument_indexes[key] = table
            selected = argument_indexes[key].get(instantiate(arg, fixed), ())
            if len(selected) < len(candidates):
                candidates = selected
        return candidates

    positives = {a: i for i, a in enumerate(sorted(task.positives))}
    labels = sorted(task.positives | task.negatives)
    all_vars = set().union(*(variables(l) for l in task.literals))
    indices = []
    for head in task.heads:
        masks = [0] * len(task.literals)
        classes = []
        head_masks = {}
        for label in labels:
            check()
            fixed = match(head, label)
            if fixed is None:
                continue
            stats["labelled_head_instances"] += 1
            hidden = sorted(all_vars - fixed.keys())
            domains = {v: set() for v in hidden}
            for literal in task.literals:
                check()
                needed = variables(literal) - fixed.keys()
                if not needed:
                    continue
                # If every argument is a distinct unbound slot, the domain
                # union is simply a column projection. Reuse it across renamed
                # templates instead of traversing a dense relation repeatedly.
                simple = []
                for arg in literal.arguments:
                    if (arg.type != clingo.SymbolType.Function or arg.name != "var__"
                            or arg.arguments[0].number in fixed):
                        break
                    simple.append(arg.arguments[0].number)
                if len(simple) == len(literal.arguments) and len(set(simple)) == len(simple):
                    sig = literal.name, len(literal.arguments)
                    for pos, v in enumerate(simple):
                        key = sig, pos
                        if key not in projections:
                            values_at_position = set()
                            for n, row in enumerate(relations.get(sig, ())):
                                if n % 4096 == 0:
                                    check()
                                values_at_position.add(row.arguments[pos])
                            projections[key] = values_at_position
                        domains[v].update(projections[key])
                    continue
                for row in rows(literal, fixed):
                    binding = match(literal, row, fixed)
                    if binding is not None:
                        for v in needed:
                            domains[v].add(binding[v])
            # A variable unused by a body must not erase that body's witnesses.
            # If a slot has no possible positive relation match, any completion
            # works; every literal using it is false in the fixed BK model.
            values = [sorted(domains[v]) or [clingo.Function("__rc_unused")] for v in hidden]
            count = prod(map(len, values))
            if stats["raw_witnesses"] + count > max_witnesses:
                raise ResourceLimit("Raw witness limit would be exceeded; no incomplete index is used",
                                    dict(stats, next_head=str(label), next_head_raw_witnesses=count))
            vectors = {}
            for row in product(*values):
                check()
                binding = dict(fixed)
                binding.update(zip(hidden, row))
                truth = sum(1 << i for i, literal in enumerate(task.literals)
                            if instantiate(literal, binding) in task.atoms)
                stats["raw_witnesses"] += 1
                if truth in vectors:
                    classes[vectors[truth]]["multiplicity"] += 1
                    continue
                position = len(classes)
                vectors[truth] = position
                bit = 1 << position
                head_masks[label] = head_masks.get(label, 0) | bit
                classes.append({"head": str(label), "truth": hex(truth), "multiplicity": 1,
                                "representative": {str(v): str(x) for v, x in sorted(binding.items())}})
                remaining = truth
                while remaining:
                    b = remaining & -remaining
                    masks[b.bit_length() - 1] |= bit
                    remaining ^= b
            if progress:
                progress(dict(stats, current_head=str(label)))
        pos_mask = neg_mask = 0
        pos_heads = []
        for label, mask in head_masks.items():
            if label in positives:
                pos_mask |= mask
                pos_heads.append((positives[label], mask))
            else:
                neg_mask |= mask
        index = MaskIndex(head, tuple(masks), (1 << len(classes)) - 1,
                          pos_mask, neg_mask, tuple(pos_heads), classes)
        indices.append(index)
        stats["compressed_witnesses"] += len(classes)
    check()
    return indices, stats


# Retained as an independent exhaustive reference for equivalence checks.
build_cartesian_indices = build_indices
