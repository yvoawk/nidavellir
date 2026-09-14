"""Bounded exact negative-label tables for selective correlated joins."""
from .indexed_join import JoinRelation
from .labels import CompactNegativeSet
from .oracle import match


def negative_relation(oracle, head):
    """Return a complete matched table, or None for the ordinary BK traversal.

    The budgets count all retained negative-label tables in this oracle.
    Large compact label sets can retain bucket keys instead of expanded rows.
    Exceeding a budget disables this join-order optimization, never safety.
    Compact records do not intern the outer negative predicate symbols.
    """
    oracle.check()
    if oracle.negative_join_backend == "bindings" or not oracle.negative_label_limit:
        return None
    space = oracle.space
    structure, slots = space._view(head, {})
    if not hasattr(oracle, "_negative_join_relations"):
        oracle._negative_join_relations = {}
        oracle._negative_join_rows = 0
    if structure in oracle._negative_join_relations:
        return slots, oracle._negative_join_relations[structure]
    # All supplied labels are an upper bound on the matched distinct rows.
    # This conservative capacity check may choose fallback unnecessarily, but
    # never constructs a table larger than the remaining storage allowance.
    if len(oracle._negative_join_relations) >= 64:
        oracle.stats["negative_label_fallbacks"] = oracle.stats.get("negative_label_fallbacks", 0) + 1
        return None
    labels = oracle.task.negatives
    if oracle._negative_join_rows + len(labels) > oracle.negative_label_limit:
        if isinstance(labels, CompactNegativeSet):
            from .compact_negative_join import compact_relation
            relation = compact_relation(oracle, head, slots)
            if relation is not None:
                oracle._negative_join_relations[structure] = relation
                oracle.stats["negative_label_tables"] = len(oracle._negative_join_relations)
                return slots, relation
        oracle.stats["negative_label_fallbacks"] = oracle.stats.get("negative_label_fallbacks", 0) + 1
        return None
    records = labels.iter_records() if isinstance(labels, CompactNegativeSet) else iter(labels)
    rows = set()
    for number, label in enumerate(records):
        if number % 256 == 0:
            oracle.check()
        fixed = match(head, label)
        if fixed is not None:
            rows.add(tuple(space.intern(fixed[slot]) for slot in slots))
    relation = JoinRelation(rows, space.check)
    oracle.check()
    oracle._negative_join_relations[structure] = relation
    oracle._negative_join_rows += len(rows)
    oracle.stats["negative_label_tables"] = len(oracle._negative_join_relations)
    oracle.stats["negative_label_rows"] = oracle._negative_join_rows
    return slots, relation
