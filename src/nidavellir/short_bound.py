"""Seed a feasible-program pool with exact short, legal, negative-safe rules.

This bounded phase changes neither the bias nor the later search grammar. It
tries effective body lengths zero, one and two, sharing the caller's oracle
and pool. Atomic masks only reject impossible positive coverage; correlated
joins and all supplied negatives are checked before a component is stored.
"""
from itertools import combinations
import math
import time

from .oracle import match
from .positive_join import has_positive
from .task import variables


class _ShortPhaseExpired(Exception):
    """Only expiration of this optional phase; never the caller's deadline."""


def seed_short_components(task, oracle, analysis, pool, *, seconds, check, on_component):
    """Add safe components to ``pool[coverage]`` and report the phase outcome.

    Pool entries must already be legal, negative-safe candidates for this task
    using the global positive index in ``oracle.positives``. A candidate must
    add positive coverage beyond the current union; this phase does not improve
    already covered portions of the pool. ``on_component(candidate)`` runs
    after each new component is stored.

    The local budget includes preprocessing and callbacks. Its expiration
    returns ``budget_exhausted`` with all completed pool entries preserved.
    Exceptions from the caller's check, original oracle check or callback are
    propagated. Finding a full union returns ``cover_found`` immediately: the
    caller must still choose and independently verify a complete program.
    ``complete`` concerns bodies of length <= min(2, task.max_body) potentially
    useful for adding coverage. It is not an exhaustive short-rule catalogue,
    a minimum rule-cost certificate, global infeasibility or optimality.
    """
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Short phase seconds must be finite and nonnegative')
    started = time.monotonic()
    deadline = started + seconds
    original_check = oracle.check
    maximum = min(2, task.max_body)
    full = (1 << len(task.positives)) - 1
    report = {
        'method': 'progressive_short_safe_components', 'status': 'running',
        'scope': 'Short bodies potentially adding uncovered positives; not an exhaustive catalogue, minimum rule-cost certificate or global optimum/infeasibility proof',
        'maximum_body_literals': maximum, 'completed_body_levels': [],
        'seconds_limit': seconds, 'elapsed_seconds': 0.0,
        'heads_considered': 0, 'generated_bodies': 0, 'duplicate_closed_bodies': 0,
        'guard_closures_above_limit': 0, 'guard_closures_unavailable': 0,
        'pruned_head_unsafe': 0, 'pruned_illegal': 0, 'pruned_no_atomic_positive': 0,
        'pruned_no_joint_positive': 0, 'pruned_already_covered': 0, 'rejected_negative': 0,
        'atomic_queries': 0, 'atomic_cache_hits': 0, 'joint_positive_queries': 0,
        'negative_queries': 0, 'coverage_queries': 0, 'safe_rules': 0,
        'components_added': 0, 'components_replaced': 0,
    }

    def phase_check():
        # Caller errors have priority and must never become a local timeout.
        check()
        if original_check is not check:
            original_check()
        if time.monotonic() >= deadline:
            raise _ShortPhaseExpired()

    def closure(proof, literal):
        selected, pending = {literal}, [literal]
        while pending:
            phase_check()
            for guard in proof.guards[pending.pop()]:
                if guard not in selected:
                    selected.add(guard)
                    pending.append(guard)
        return frozenset(selected)

    covered = 0
    for coverage, candidate in pool.items():
        if type(coverage) is not int or coverage < 0 or coverage & ~full or candidate['coverage'] != coverage:
            raise ValueError('Short phase pool does not use the current positive coverage domain')
        covered |= coverage
    contexts, masks = [], {}
    oracle.check = phase_check
    try:
        phase_check()
        if covered == full:
            report['status'] = 'cover_found'
            return report
        literal_variables = tuple(variables(literal) for literal in task.literals)
        for head in task.heads:
            phase_check()
            proof = analysis.heads[head]
            if not proof.legal_body_exists:
                continue
            positives = 0
            for number, (label, position) in enumerate(oracle.positives.items()):
                if number % 256 == 0:
                    phase_check()
                if match(head, label) is not None:
                    positives |= 1 << position
            if not positives:
                continue
            closures = {}
            for literal in range(len(task.literals)):
                phase_check()
                closed = closure(proof, literal)
                if len(closed) > maximum:
                    report['guard_closures_above_limit'] += 1
                elif closed & proof.unusable_literals:
                    report['guard_closures_unavailable'] += 1
                else:
                    closures[literal] = closed
            contexts.append((head, proof, variables(head), positives, closures, set()))
        report['heads_considered'] = len(contexts)
        for level in range(maximum + 1):
            for head, proof, head_variables, positives, closures, seen in contexts:
                for generators in combinations(closures, level):
                    phase_check()
                    closed = frozenset().union(*(closures[i] for i in generators))
                    if len(closed) != level:
                        continue
                    body = tuple(sorted(closed))
                    if body in seen:
                        report['duplicate_closed_bodies'] += 1
                        continue
                    seen.add(body)
                    report['generated_bodies'] += 1
                    used = set().union(*(literal_variables[i] for i in body))
                    if not head_variables <= used:
                        report['pruned_head_unsafe'] += 1
                        continue
                    if not proof.accepts(body, check=phase_check):
                        report['pruned_illegal'] += 1
                        continue
                    possible = positives
                    for literal in body:
                        phase_check()
                        key = head, literal
                        if key not in masks:
                            report['atomic_queries'] += 1
                            masks[key] = oracle.atomic_coverage(head, literal)
                        else:
                            report['atomic_cache_hits'] += 1
                        possible &= masks[key]
                        if not possible:
                            break
                    if not possible:
                        report['pruned_no_atomic_positive'] += 1
                        continue
                    remaining_possible = possible & ~covered
                    if not remaining_possible:
                        report['pruned_already_covered'] += 1
                        continue
                    report['joint_positive_queries'] += 1
                    if not has_positive(oracle, head, body, remaining_possible):
                        report['pruned_no_joint_positive'] += 1
                        continue
                    report['negative_queries'] += 1
                    if oracle.negative(head, body) is not None:
                        report['rejected_negative'] += 1
                        continue
                    report['coverage_queries'] += 1
                    coverage = oracle.coverage(head, body)
                    phase_check()
                    if not coverage or coverage & ~possible:
                        raise AssertionError('Exact positive coverage disagrees with its atomic upper bound')
                    if not coverage & ~covered:
                        raise AssertionError('A short component must add a genuinely covered positive')
                    report['safe_rules'] += 1
                    candidate = {'head': str(head), 'body': list(body), 'cost': level + 1,
                                 'coverage': coverage, 'rule': task.rule(head, body)}
                    previous = pool.get(coverage)
                    if previous is None or (candidate['cost'], candidate['rule']) < (previous['cost'], previous['rule']):
                        pool[coverage] = candidate
                        report['components_added' if previous is None else 'components_replaced'] += 1
                        covered |= coverage
                        on_component(candidate)
                        phase_check()
                        if covered == full:
                            report['status'] = 'cover_found'
                            return report
            report['completed_body_levels'].append(level)
        report['status'] = 'complete'
    except _ShortPhaseExpired:
        report['status'] = 'budget_exhausted'
    finally:
        oracle.check = original_check
        report['elapsed_seconds'] = time.monotonic() - started
        report['pool_components'] = len(pool)
        report['covered_positives'] = covered.bit_count()
        report['total_positives'] = len(task.positives)
    return report
