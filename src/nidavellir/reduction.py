"""Proved relational reductions for positive, legal hypothesis bodies.

No predicate names have special meaning. A variable is projected away only
when it is private to one template, or a mandatory carrier functionally
determines it. Functional dependencies are checked on the actual finite BK.
"""
from dataclasses import dataclass
from itertools import combinations

from .task import variables


@dataclass
class ReductionPlan:
    guards: tuple
    unusable: frozenset
    private: frozenset
    functional: dict  # carrier index -> eliminated slots
    eliminated: frozenset
    stats: dict


def _functional(slots, rows, outputs, check=lambda: None):
    inputs = tuple(i for i, v in enumerate(slots) if v not in outputs)
    outgoing = tuple(i for i, v in enumerate(slots) if v in outputs)
    mapping = {}
    for n, row in enumerate(rows):
        if n % 2048 == 0:
            check()
        key = tuple(row[i] for i in inputs)
        value = tuple(row[i] for i in outgoing)
        previous = mapping.setdefault(key, value)
        if previous != value:
            return False
    return True


def plan_reduction(task, head, analysis, space, check):
    head_vars = variables(head)
    occurrences = {}
    for i, literal in enumerate(task.literals):
        if i in analysis.unusable_literals:
            continue
        for v in variables(literal):
            occurrences.setdefault(v, set()).add(i)
    private = {v for v, uses in occurrences.items() if v not in head_vars and len(uses) == 1}
    unused = space.all_vars - set(occurrences) - head_vars
    eliminated = set(private) | unused
    protected = set(head_vars)
    functional = {}
    global_domains = None
    attempts = 0
    for carrier, literal in enumerate(task.literals):
        check()
        if carrier in analysis.unusable_literals:
            continue
        slots = variables(literal)
        possible = sorted(v for v in slots - eliminated - protected
                          if all(i == carrier or carrier in analysis.guards[i]
                                 for i in occurrences.get(v, ())))
        if not possible:
            continue
        # Never create a circular chain of eliminated determinants. Conservative
        # fallback leaves such variables in the exact finite oracle.
        if (slots - set(possible)) & (eliminated - private):
            continue
        if global_domains is None:
            global_domains = space.domains({})
        columns, rows = space.relation(carrier, {}, global_domains)
        if private & set(columns):
            retained = tuple(v for v in columns if v not in private)
            positions = tuple(columns.index(v) for v in retained)
            rows = frozenset(tuple(row[i] for i in positions) for row in rows)
            columns = retained
        selected = None
        # Prefer the largest jointly determined output group. Individual FDs
        # alone do NOT justify eliminating both sides of a bijection.
        for size in range(len(possible), 0, -1):
            for group in combinations(possible, size):
                check()
                attempts += 1
                if _functional(columns, rows, set(group), check):
                    selected = frozenset(group)
                    break
            if selected is not None:
                break
        if selected:
            functional[carrier] = selected
            eliminated.update(selected)
            protected.update(slots - selected - private)
    return ReductionPlan(analysis.guards, analysis.unusable_literals, frozenset(private),
                         functional, frozenset(eliminated),
                         {"private_variables": sorted(private),
                          "unusable_variables": sorted(unused),
                          "functional_outputs": {str(i): sorted(v) for i, v in functional.items()},
                          "functional_checks": attempts,
                          "eliminated_variables": sorted(eliminated)})


def join_relations(left, right, check):
    """Indexed natural join of interned finite relations, retaining correlation."""
    ls, lr = left
    rs, rr = right
    columns = tuple(sorted(set(ls) | set(rs)))
    if not lr or not rr:
        return columns, frozenset()
    common = tuple(v for v in ls if v in rs)
    li, ri = tuple(ls.index(v) for v in common), tuple(rs.index(v) for v in common)
    table = {}
    for n, row in enumerate(rr):
        if n % 2048 == 0:
            check()
        table.setdefault(tuple(row[i] for i in ri), []).append(row)
    positions = tuple((0, ls.index(v)) if v in ls else (1, rs.index(v)) for v in columns)
    result = set()
    examined = 0
    for n, lrow in enumerate(lr):
        if n % 256 == 0:
            check()
        for rrow in table.get(tuple(lrow[i] for i in li), ()):
            examined += 1
            if examined % 2048 == 0:
                check()
            result.add(tuple((lrow if side == 0 else rrow)[i] for side, i in positions))
    return columns, frozenset(result)


def guarded_relation(space, plan, literal, fixed, domains, check):
    """Truth of L and its proved guards, projected onto retained variables."""
    if literal in plan.unusable:
        return (), frozenset()
    required = {literal, *plan.guards[literal]}
    relations = [space.relation(i, fixed, domains) for i in required]
    output = set().union(*(set(slots) for slots, _ in relations)) - plan.eliminated

    def project(relation, keep):
        slots, rows = relation
        columns = tuple(v for v in slots if v in keep)
        if slots == columns:
            return relation
        positions = tuple(slots.index(v) for v in columns)
        result = set()
        for n, row in enumerate(rows):
            if n % 2048 == 0:
                check()
            result.add(tuple(row[i] for i in positions))
        return columns, frozenset(result)

    # Existential liveness: discard a column as soon as neither the result nor
    # another relation needs it. In particular, independent private guards do
    # not create a large Cartesian intermediate before their projection.
    relations = [project(relation, output | set().union(*(set(other[0])
                 for j, other in enumerate(relations) if j != i)))
                 for i, relation in enumerate(relations)]
    relations.sort(key=lambda r: len(r[1]))
    result = ((), frozenset({()}))
    for position, relation in enumerate(relations):
        result = join_relations(result, relation, check)
        if not result[1]:
            break
        needed = output | set().union(*(set(other[0]) for other in relations[position + 1:]))
        result = project(result, needed)
    return project(result, output)
