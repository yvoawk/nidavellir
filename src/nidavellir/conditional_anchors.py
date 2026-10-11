"""Proved, head-determined domains for conditionally used hidden variables.

For each restricted variable, every legal body using it must select at least
one verified functional carrier whose other variables occur in the head.
The union of those carriers' outputs therefore contains every needed value.
An unused variable may take any original domain value. No predicate names,
labels, missing-value imputation, or independence assumptions enter the proof.
"""
from itertools import product

from .reduction import guarded_relation
from .task import variables


def plan_anchors(task, head, analysis, space, reduction, check):
    head_vars = variables(head)
    hidden = space.all_vars - head_vars - reduction.eliminated
    anchors, global_domains = {}, None
    for variable in sorted(hidden):
        candidates = []
        uses = []
        for i, literal in enumerate(task.literals):
            check()
            slots = variables(literal)
            if variable not in slots or i in analysis.unusable_literals:
                continue
            uses.append(i)
            if slots - head_vars != {variable}:
                continue
            if global_domains is None:
                global_domains = space.domains({})
            columns, rows = space.relation(i, {}, global_domains)
            output = columns.index(variable)
            inputs = tuple(v for v in columns if v != variable)
            positions = tuple(columns.index(v) for v in inputs)
            mapping = {}
            for n, row in enumerate(rows):
                if n % 1024 == 0:
                    check()
                key = tuple(row[p] for p in positions)
                previous = mapping.setdefault(key, row[output])
                if previous != row[output]:
                    break
            else:
                candidates.append((i, inputs, mapping))
        if not candidates:
            continue
        # Constant carriers may add every value back to every labelled head.
        # Prefer a head-dependent cover when the bias proves it sufficient;
        # otherwise retain all candidates and repeat the exact proof.
        head_dependent = [carrier for carrier in candidates if carrier[1]]
        choices = [head_dependent, candidates] if head_dependent and len(head_dependent) < len(candidates) else [candidates]
        for selected in choices:
            excluded = frozenset(i for i, _, _ in selected)
            if all(i in excluded or not analysis.possible_completion((i,), excluded, check=check)
                   for i in uses):
                anchors[variable] = tuple(selected)
                break
    check()
    return anchors


def restrict_domains(domains, fixed, anchors, space):
    result = dict(domains)
    for variable, candidates in anchors.items():
        values = set()
        for _, inputs, mapping in candidates:
            value = mapping.get(tuple(space.intern(fixed[v]) for v in inputs))
            if value is not None:
                values.add(space.symbols[value])
        # If none of the carriers fires, no true legal body can use this
        # variable. Keep one real completion for bodies that do not use it.
        result[variable] = tuple(sorted(values)) or domains[variable][:1]
    return result


def direct_profiles(task, space, reduction, fixed, domains, check, stats):
    """Enumerate a bounded assignment product with shared-variable joins.

    Group relations by their columns; one lookup produces all their truth bits.
    This preserves correlation even when two carriers give different outputs.
    The caller limits the product and retains the projected solver fallback.
    """
    hidden = tuple(v for v in sorted(domains) if v not in reduction.eliminated)
    groups = {}
    for i in range(len(task.literals)):
        check()
        slots, rows = guarded_relation(space, reduction, i, fixed, domains, check)
        table = groups.setdefault(slots, {})
        bit = 1 << i
        for n, row in enumerate(rows):
            if n % 1024 == 0:
                check()
            table[row] = table.get(row, 0) | bit
    tables = [(tuple(hidden.index(v) for v in slots), table) for slots, table in groups.items()]
    values = [tuple(space.intern(x) for x in domains[v]) for v in hidden]
    seen = set()
    for n, row in enumerate(product(*values)):
        if n % 256 == 0:
            check()
        stats["direct_assignments_examined"] += 1
        truth = 0
        for positions, table in tables:
            truth |= table.get(tuple(row[p] for p in positions), 0)
        if truth in seen:
            continue
        seen.add(truth)
        binding = dict(fixed)
        binding.update((v, space.symbols[x]) for v, x in zip(hidden, row))
        yield truth, binding
    check()
