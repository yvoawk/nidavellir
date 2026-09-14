"""Exact existential joins between a rule body and supplied positive labels.

Positive labels are an indexed virtual relation on the variables of the head.
They can be joined before or after a selective BK literal.  Every newly bound
head variable immediately filters this relation; no unlabelled rule outputs or
Cartesian products of variable domains are materialized.
"""

from collections import OrderedDict
from dataclasses import dataclass

from .oracle import match
from .profiling import timed
from .join_profile import profile_join, measured_rows
from .indexed_join import matched_relation, incompatible_pairs
from .positive_support import filter_supported
from .task import instantiate, variables


RESIDUAL_CACHE_LIMIT = 8192


def _positions(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass(frozen=True)
class _PositiveRelation:
    slots: tuple
    rows: dict
    indexes: dict
    mask: int

    @classmethod
    def build(cls, oracle, head):
        slots = tuple(sorted(variables(head)))
        rows, indexes, mask = {}, {slot: {} for slot in slots}, 0
        for number, (label, position) in enumerate(oracle.positives.items()):
            if number % 256 == 0:
                oracle.check()
            binding = match(head, label)
            if binding is None:
                continue
            bit = 1 << position
            mask |= bit
            row = tuple(binding[slot] for slot in slots)
            if oracle.positive_join_backend == "indexed":
                row = tuple(oracle.space.intern(value) for value in row)
            rows[position] = row
            for slot, value in zip(slots, row):
                column = indexes[slot]
                column[value] = column.get(value, 0) | bit
        # Publish only a complete relation: a deadline cannot cache a prefix.
        oracle.check()
        return cls(slots, rows, indexes, mask)

    def compatible(self, possible, binding):
        for slot in self.slots:
            if slot in binding:
                possible &= self.indexes[slot].get(binding[slot], 0)
                if not possible:
                    break
        return possible


@timed("oracle.positive_join")
@profile_join
def has_positive(oracle, head, body, possible, metrics):
    """Whether this head-safe body covers a positive in the global bitmask.

    ``possible`` uses the positions in ``oracle.positives``, including gaps
    belonging to other head templates.  Singleton coverage masks may provide
    this upper bound, but the final answer requires one correlated binding for
    all literals and one matching supplied label.  Labels and BK are immutable
    for the oracle lifetime, as for its existing coverage/exists caches.
    """
    return _has_positive(oracle, head, body, possible, metrics, require_head_safe=True)


@timed("oracle.positive_core_probe")
@profile_join(name="positive_core_probe")
def pattern_has_positive(oracle, head, body, possible, metrics):
    """Existence of body AND a matching label; not necessarily a safe rule.

    The label relation supplies head slots missing from a partial body. This
    operation only proves a conjunction's support, never ASP rule legality.
    """
    return _has_positive(oracle, head, body, possible, metrics, require_head_safe=False)


def _has_positive(oracle, head, body, possible, metrics, *, require_head_safe):
    oracle.check()
    if type(possible) is not int or possible < 0:
        raise ValueError("Possible positives must be a nonnegative integer bitmask")
    body = tuple(sorted(set(body)))
    if any(type(i) is not int or i < 0 or i >= len(oracle.task.literals) for i in body):
        raise ValueError("Body indices must identify existing literal templates")
    used = set().union(*(oracle.space._literal_vars[i] for i in body)) if body else set()
    head_vars = oracle._head_vars(head)
    if require_head_safe and not head_vars <= used:
        raise ValueError("Positive join requires a head-safe body")
    stats = oracle.stats
    indexed = oracle.positive_join_backend == "indexed"
    stats["positive_join_queries"] = stats.get("positive_join_queries", 0) + 1
    if not possible:
        return False
    if not hasattr(oracle, "_positive_join_relations"):
        oracle._positive_join_relations = {}
        oracle._positive_join_results = OrderedDict()
    relation = oracle._positive_join_relations.get(head)
    if relation is None:
        relation = _PositiveRelation.build(oracle, head)
        oracle._positive_join_relations[head] = relation
        stats["positive_join_index_builds"] = stats.get("positive_join_index_builds", 0) + 1
    possible &= relation.mask
    if not possible:
        return False
    key = head, body, possible
    cache = oracle._positive_join_results
    if key in cache:
        metrics["cache"] = "hit"
        cache.move_to_end(key)
        stats["positive_join_cache_hits"] = stats.get("positive_join_cache_hits", 0) + 1
        return cache[key]
    stats["positive_join_cache_misses"] = stats.get("positive_join_cache_misses", 0) + 1
    incompatible = False
    if indexed and oracle.positive_pair_pruning:
        with oracle.profile.measure("oracle.positive_pairs"):
            incompatible = incompatible_pairs(oracle.space, body)
    if incompatible:
        oracle.check()
        oracle._put(cache, key, False, limit=8192)
        stats["positive_join_incompatible_pairs"] = stats.get("positive_join_incompatible_pairs", 0) + 1
        return False

    # A suffix cannot observe bindings of variables that have disappeared
    # from both its literals and the head. Only completed failures are reused;
    # the cache belongs to this head/body query and never outlives it.
    failed, live_slots = OrderedDict(), OrderedDict()

    def walk(remaining, binding, compatible):
        oracle.check()
        metrics["nodes"] += 1
        stats["positive_join_nodes"] = stats.get("positive_join_nodes", 0) + 1
        compatible = relation.compatible(compatible, binding)
        if not compatible:
            return False
        if oracle.positive_residual_normalization:
            # A true fully bound literal no longer constrains the residual
            # query. Verify it BEFORE dropping its variables from the key.
            # In particular, a failed literal must not cache a failure for
            # the normalized suffix, which may succeed under another binding.
            unbound = []
            for i in remaining:
                if not oracle.space._literal_vars[i] <= binding.keys():
                    unbound.append(i)
                    continue
                stats['positive_join_normalization_checks'] = stats.get('positive_join_normalization_checks', 0) + 1
                if indexed:
                    slots, table = matched_relation(oracle.space, i)
                    found = tuple(binding[slot] for slot in slots) in table.members
                else:
                    found = instantiate(oracle.task.literals[i], binding) in oracle.task.atoms
                oracle.check()
                if not found:
                    stats['positive_join_normalization_failures'] = stats.get('positive_join_normalization_failures', 0) + 1
                    return False
            removed = len(remaining) - len(unbound)
            if removed:
                stats['positive_join_normalized_states'] = stats.get('positive_join_normalized_states', 0) + 1
                stats['positive_join_normalized_literals'] = stats.get('positive_join_normalized_literals', 0) + removed
                metrics['normalized_literals'] = metrics.get('normalized_literals', 0) + removed
                remaining = tuple(unbound)
            oracle.check()
            if not remaining:
                return True
        state = None
        if oracle.positive_residual_cache and RESIDUAL_CACHE_LIMIT:
            slots = live_slots.get(remaining)
            if slots is None:
                slots = tuple(sorted(head_vars.union(*(oracle.space._literal_vars[i] for i in remaining))))
                oracle._put(live_slots, remaining, slots, limit=128)
            state = remaining, tuple((slot, binding[slot]) for slot in slots if slot in binding), compatible
            if state in failed:
                failed.move_to_end(state)
                stats['positive_join_residual_hits'] = stats.get('positive_join_residual_hits', 0) + 1
                metrics['residual_hits'] = metrics.get('residual_hits', 0) + 1
                return False
            stats['positive_join_residual_misses'] = stats.get('positive_join_residual_misses', 0) + 1
        result = walk_uncached(remaining, binding, compatible)
        if not result and state is not None:
            # With normalization disabled, fully bound literals remain in
            # state. With it enabled, all removed literals were proved true.
            oracle.check()
            if len(failed) >= RESIDUAL_CACHE_LIMIT:
                failed.popitem(last=False)
                stats['positive_join_residual_evictions'] = stats.get('positive_join_residual_evictions', 0) + 1
            failed[state] = True
            stats['positive_join_residual_stores'] = stats.get('positive_join_residual_stores', 0) + 1
            stats['positive_join_residual_peak_entries'] = max(
                stats.get('positive_join_residual_peak_entries', 0), len(failed))
        return result

    def walk_uncached(remaining, binding, compatible):
        best, best_rows, unbound = None, None, []
        for i in remaining:
            literal = oracle.task.literals[i]
            if oracle.space._literal_vars[i] <= binding.keys():
                if indexed:
                    slots, table = matched_relation(oracle.space, i)
                    found = tuple(binding[slot] for slot in slots) in table.members
                else:
                    found = instantiate(literal, binding) in oracle.task.atoms
                if not found:
                    return False
                continue
            unbound.append(i)
            rows = measured_rows(oracle, i, binding, metrics)
            if not rows:
                return False
            if best_rows is None or len(rows) < len(best_rows):
                best, best_rows = i, rows
        remaining = tuple(unbound)
        if not remaining:
            return True
        # A label is a row containing all head variables.  Choosing it binds
        # them jointly, preserving repeated/nested terms and fixed constants.
        # Choosing BK first can reject thousands of labels in one small join.
        if not head_vars <= binding.keys() and compatible.bit_count() <= len(best_rows):
            metrics["label_steps"] += 1
            stats["positive_join_label_steps"] = stats.get("positive_join_label_steps", 0) + 1
            for number, position in enumerate(_positions(compatible)):
                if number % 256 == 0:
                    oracle.check()
                extended = dict(binding)
                extended.update(zip(relation.slots, relation.rows[position]))
                if walk(remaining, extended, 1 << position):
                    return True
            return False
        stats["positive_join_literal_steps"] = stats.get("positive_join_literal_steps", 0) + 1
        metrics["literal_steps"] += 1
        rest = tuple(i for i in remaining if i != best)
        literal = oracle.task.literals[best]
        best_rows = filter_supported(oracle, best, best_rows, rest, binding, metrics)
        for number, atom in enumerate(best_rows):
            metrics["relations"][best]["visited_rows"] += 1
            if number % 256 == 0:
                oracle.check()
            if indexed:
                slots, _ = matched_relation(oracle.space, best)
                extended = dict(binding)
                extended.update(zip(slots, atom))
            else:
                extended = match(literal, atom, binding)
            if extended is not None:
                stats["positive_join_matched_atoms"] = stats.get("positive_join_matched_atoms", 0) + 1
                if walk(rest, extended, compatible):
                    return True
        return False

    try:
        result = walk(body, {}, possible)
    finally:
        # Recursive closures may wait for cyclic GC. Release their retained
        # states immediately, also when the query was interrupted.
        failed.clear()
        live_slots.clear()
    # Do not cache a partial answer if the deadline was crossed on the last
    # membership operation; ordinary search code may replace oracle.check.
    oracle.check()
    oracle._put(cache, key, result, limit=8192)
    return result
