"""Exact matched integer relations shared across alpha-renamed literals.

All template constraints are checked before publishing a table. Subsequent
joins equate its columns, so repeated and hidden variables remain correlated.
These tables describe complete BK relations, never partially answered queries.
"""
from collections import OrderedDict

from .oracle import match
from .labels import AtomRecord, CompactNegativeSet
from .task import instantiate


class JoinRelation:
    def __init__(self, rows, check):
        self.rows = tuple(rows)
        self.members = frozenset(rows)
        self.check = check
        self.indexes = {}
        self.filters = OrderedDict()
        self.filter_rows = 0
        self.projections = OrderedDict()
        self.projection_rows = 0

    def project(self, columns):
        self.check()
        if not self.rows:
            return frozenset()
        if columns == tuple(range(len(self.rows[0]))):
            return self.members
        if columns not in self.projections:
            rows = set()
            for number, row in enumerate(self.rows):
                if number % 256 == 0:
                    self.check()
                rows.add(tuple(row[i] for i in columns))
            result = frozenset(rows)
            self.check()
            if len(result) > 262144:
                return result
            while self.projections and (len(self.projections) >= 64 or
                    self.projection_rows + len(result) > 262144):
                _, removed = self.projections.popitem(last=False)
                self.projection_rows -= len(removed)
            self.projections[columns] = result
            self.projection_rows += len(result)
        self.projections.move_to_end(columns)
        return self.projections[columns]

    def select(self, fixed, stats):
        if not fixed:
            return self.rows
        self.check()
        key = tuple(fixed)
        if len(fixed) == (len(self.rows[0]) if self.rows else 0):
            row = tuple(value for _, value in fixed)
            return (row,) if row in self.members else ()
        if key in self.filters:
            self.filters.move_to_end(key)
            stats["integer_filter_hits"] = stats.get("integer_filter_hits", 0) + 1
            return self.filters[key]
        candidates = self.rows
        for column, value in fixed:
            index = self.indexes.get(column)
            if index is None:
                index = {}
                for number, row in enumerate(self.rows):
                    if number % 256 == 0:
                        self.check()
                    index.setdefault(row[column], []).append(row)
                self.check()
                self.indexes[column] = index
                stats["integer_index_builds"] = stats.get("integer_index_builds", 0) + 1
                stats["integer_index_row_references"] = stats.get("integer_index_row_references", 0) + len(self.rows)
            selected = index.get(value, ())
            if len(selected) < len(candidates):
                candidates = selected
            if not candidates:
                return ()
        if len(fixed) == 1:
            return candidates
        rows = []
        for number, row in enumerate(candidates):
            if number % 256 == 0:
                self.check()
            if all(row[column] == value for column, value in fixed):
                rows.append(row)
        self.check()
        rows = tuple(rows)
        stats["integer_filter_scanned"] = stats.get("integer_filter_scanned", 0) + len(candidates)
        if len(rows) <= 8192:
            while self.filters and (len(self.filters) >= 256 or self.filter_rows + len(rows) > 8192):
                _, removed = self.filters.popitem(last=False)
                self.filter_rows -= len(removed)
            self.filters[key] = rows
            self.filter_rows += len(rows)
        return rows


def matched_relation(space, index):
    space.check()
    if not hasattr(space, "_integer_relations"):
        space._integer_relations = {}
        space._integer_templates = {}
    if index in space._integer_templates:
        space.stats["integer_relation_hits"] = space.stats.get("integer_relation_hits", 0) + 1
        return space._integer_templates[index]
    literal = space.task.literals[index]
    structure, slots = space._view(literal, {})
    table = space._integer_relations.get(structure)
    if table is None:
        rows = set()
        simple = space._simple(literal, {})
        for number, atom in enumerate(space._rows(literal, {})):
            if number % 256 == 0:
                space.check()
            if simple is not None:
                rows.add(tuple(space.intern(value) for value in atom.arguments))
            else:
                binding = match(literal, atom)
                if binding is not None:
                    rows.add(tuple(space.intern(binding[slot]) for slot in slots))
        table = JoinRelation(rows, space.check)
        space.check()
        space._integer_relations[structure] = table
        space.stats["integer_relation_builds"] = space.stats.get("integer_relation_builds", 0) + 1
        space.stats["integer_relation_rows"] = space.stats.get("integer_relation_rows", 0) + len(rows)
    space._integer_templates[index] = slots, table
    return slots, table


def incompatible_pairs(space, body):
    """A completed empty pair join is a sound rejection of every extension.

    Keys refer to complete immutable tables and the correlated common-column
    positions. Renaming slots does not change the proof. Missing cache entries
    always execute the actual query; an interrupted projection publishes none.
    """
    if not hasattr(space, "_integer_pairs"):
        space._integer_pairs = {}
    relations = []
    for i in body:
        slots, table = matched_relation(space, i)
        if not table.rows:
            return True
        for other_slots, other in relations:
            common = tuple(sorted(set(slots) & set(other_slots)))
            if not common:
                continue
            left = tuple(slots.index(slot) for slot in common)
            right = tuple(other_slots.index(slot) for slot in common)
            key = table, left, other, right
            if key not in space._integer_pairs:
                answer = table.project(left).isdisjoint(other.project(right))
                space.check()
                space._integer_pairs[key] = answer
                space.stats["integer_pair_queries"] = space.stats.get("integer_pair_queries", 0) + 1
            else:
                space.stats["integer_pair_hits"] = space.stats.get("integer_pair_hits", 0) + 1
            if space._integer_pairs[key]:
                return True
        relations.append((slots, table))
    return False


def bindings(oracle, body, fixed=None, *, negative_head=None):
    """Yield the same complete bindings as the Symbol matcher, using integer columns."""
    body = tuple(sorted(set(body)))
    space = oracle.space
    head_vars = oracle._head_vars(negative_head) if negative_head is not None else None
    tables = {}
    negative_table = None
    negative_table_checked = False

    def table_for(index):
        if index not in tables:
            tables[index] = matched_relation(space, index)
        return tables[index]

    def walk(remaining, binding, head_checked=False):
        nonlocal negative_table, negative_table_checked
        oracle.check()
        if head_vars is not None and not head_checked and head_vars <= binding.keys():
            symbolic = {slot: space.symbols[binding[slot]] for slot in head_vars}
            record = AtomRecord(negative_head.name, tuple(
                instantiate(argument, symbolic) for argument in negative_head.arguments))
            oracle.stats["negative_head_checks"] += 1
            labelled = (record in oracle.task.negatives if isinstance(oracle.task.negatives, CompactNegativeSet)
                        else record.to_symbol() in oracle.task.negatives)
            oracle.check()
            if not labelled:
                oracle.stats["negative_head_prunes"] += 1
                return
            head_checked = True
        negative_rows = None
        if head_vars is not None and not head_checked:
            if not negative_table_checked:
                from .negative_join import negative_relation
                negative_table = negative_relation(oracle, negative_head)
                negative_table_checked = True
            if negative_table is not None:
                negative_slots, labelled = negative_table
                negative_rows = labelled.select(tuple((j, binding[slot])
                    for j, slot in enumerate(negative_slots) if slot in binding), space.stats)
                if not negative_rows:
                    oracle.stats["negative_label_prunes"] = oracle.stats.get("negative_label_prunes", 0) + 1
                    return
        unbound = []
        best, best_rows = None, None
        for i in remaining:
            slots, table = table_for(i)
            if all(slot in binding for slot in slots):
                if tuple(binding[slot] for slot in slots) not in table.members:
                    return
                continue
            rows = table.select(tuple((j, binding[slot]) for j, slot in enumerate(slots)
                                      if slot in binding), space.stats)
            if not rows:
                return
            unbound.append(i)
            if best_rows is None or len(rows) < len(best_rows):
                best, best_rows = i, rows
        if not unbound:
            oracle.check()
            yield {slot: space.symbols[value] for slot, value in binding.items()}
            return
        # A matching label binds every head slot jointly. This can precede BK
        # joins when it is selective, but all hidden-variable joins remain.
        # Only explicitly supplied negative tuples occur in this relation.
        if negative_rows is not None and len(negative_rows) <= len(best_rows):
            oracle.stats["negative_label_steps"] = oracle.stats.get("negative_label_steps", 0) + 1
            for number, row in enumerate(negative_rows):
                if number % 256 == 0:
                    oracle.check()
                extended = dict(binding)
                extended.update(zip(negative_slots, row))
                yield from walk(tuple(unbound), extended, True)
            return
        rest = tuple(i for i in unbound if i != best)
        slots, _ = table_for(best)
        for number, row in enumerate(best_rows):
            if number % 256 == 0:
                oracle.check()
            extended = dict(binding)
            extended.update(zip(slots, row))
            oracle.stats["matched_atoms"] += 1
            yield from walk(rest, extended, head_checked)

    yield from walk(body, {slot: space.intern(value) for slot, value in (fixed or {}).items()})
