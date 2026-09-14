"""Necessary correlated BK supports before expanding a positive join branch.

Every surviving row must still pass the full residual query. Independent
supports may disagree on hidden variables, so they never prove a conjunction.
Projections belong to complete immutable BK tables, not to labels or queries.
"""
from .indexed_join import matched_relation


MIN_ROWS = 8
MAX_PROJECT_ROWS = 262144


def filter_supported(oracle, chosen, rows, remaining, binding, metrics):
    """Remove rows lacking support in another residual literal.

    ``remaining`` excludes ``chosen``. Keys contain all columns of the other
    literal bound now or supplied by a chosen row, in their original order.
    This preserves joint values, including already fixed hidden variables.
    Thresholds skip work conservatively; no missing projection means false.
    """
    if (not oracle.positive_lookahead or oracle.positive_join_backend != 'indexed'
            or len(rows) < MIN_ROWS or not remaining):
        return rows
    oracle.check()
    stats = oracle.stats

    def count(name, amount=1):
        key = 'positive_lookahead_' + name
        stats[key] = stats.get(key, 0) + amount

    with oracle.profile.measure('oracle.positive_lookahead'):
        slots, _ = matched_relation(oracle.space, chosen)
        positions = {slot: i for i, slot in enumerate(slots)}
        new_slots = set(slots) - binding.keys()
        supports = []
        for index in remaining:
            oracle.check()
            other_slots, table = matched_relation(oracle.space, index)
            if not new_slots.intersection(other_slots):
                continue
            columns = tuple(i for i, slot in enumerate(other_slots)
                            if slot in binding or slot in positions)
            if columns == tuple(range(len(other_slots))):
                support = table.members
                count('membership_supports')
            else:
                cached = columns in table.projections
                if not cached and len(table.rows) > MAX_PROJECT_ROWS:
                    count('skipped_large_projections')
                    continue
                support = table.project(columns)
                count('projection_hits' if cached else 'projection_builds')
            # The table handles constants, nesting and repeated occurrences
            # before these columns are projected. Never split correlated keys.
            layout = tuple((False, binding[other_slots[i]]) if other_slots[i] in binding
                           else (True, positions[other_slots[i]]) for i in columns)
            supports.append((support, layout))
        if not supports:
            return rows
        supports.sort(key=lambda item: len(item[0]))
        count('queries')
        kept, tested = [], 0
        for number, row in enumerate(rows):
            if number % 256 == 0:
                oracle.check()
            for support, layout in supports:
                tested += 1
                key = tuple(row[value] if from_row else value for from_row, value in layout)
                if key not in support:
                    break
            else:
                kept.append(row)
        oracle.check()
        removed = len(rows) - len(kept)
        count('rows_scanned', len(rows))
        count('rows_pruned', removed)
        count('constraint_tests', tested)
        metrics['support_rows_scanned'] = metrics.get('support_rows_scanned', 0) + len(rows)
        metrics['support_rows_pruned'] = metrics.get('support_rows_pruned', 0) + removed
        metrics['support_tests'] = metrics.get('support_tests', 0) + tested
        return tuple(kept)
