"""Lazy exact joins over compact label buckets, without expanding their tuples.

Distinct direct head variables and ground arguments map directly to columns.
Other templates keep the general matched-label/BK join implementation.
"""
from bisect import bisect_left

from .indexed_join import JoinRelation
from .labels import _ids
from .task import variables


COMPACT_BUCKET_LIMIT = 65536


class _Rows:
    def __init__(self, relation, groups, count):
        self.relation, self.groups, self.count = relation, tuple(groups), count

    def __len__(self):
        return self.count

    def __iter__(self):
        relation = self.relation
        table, space = relation.table, relation.space
        number = 0
        relation.check()
        for key, bucket, fixed_pivot in self.groups:
            for value in _ids(bucket) if fixed_pivot is None else (fixed_pivot,):
                if number % 256 == 0:
                    relation.check()
                number += 1
                row = key[:table.pivot] + (value,) + key[table.pivot:]
                yield tuple(space.intern(table.values[column][row[column]])
                            for column in relation.columns)
        relation.check()


class CompactNegativeRelation:
    def __init__(self, table, columns, constants, space, check):
        self.table, self.columns, self.constants = table, columns, constants
        self.space, self.check = space, check
        # Index references point to bucket keys, never to expanded label rows.
        self.keys = JoinRelation(table.buckets, check)
        self.rows = self._select((), {})

    def select(self, fixed, stats):
        self.check()
        stats['negative_compact_selections'] = stats.get('negative_compact_selections', 0) + 1
        return self._select(fixed, stats) if fixed else self.rows

    def _select(self, fixed, stats):
        self.check()
        table = self.table
        constraints = dict(self.constants)
        constraints.update((self.columns[column], self.space.symbols[value])
                           for column, value in fixed)
        ids = {column: table.maps[column].get(value, -1)
               for column, value in constraints.items()}
        if -1 in ids.values():
            return _Rows(self, (), 0)
        pivot = table.pivot
        key_fixed = tuple(sorted((column if column < pivot else column - 1, value)
                                 for column, value in ids.items() if column != pivot))
        if len(key_fixed) == len(table.values) - 1:
            key = tuple(value for _, value in key_fixed)
            candidates = (key,) if key in table.buckets else ()
        else:
            candidates = self.keys.select(key_fixed, stats)
        fixed_pivot = ids.get(pivot)
        groups, count = [], 0
        for number, key in enumerate(candidates):
            if number % 256 == 0:
                self.check()
            bucket = table.buckets[key]
            if fixed_pivot is None:
                size = bucket.bit_count() if type(bucket) is int else len(bucket)
            elif type(bucket) is int:
                size = (bucket >> fixed_pivot) & 1
            else:
                position = bisect_left(bucket, fixed_pivot)
                size = int(position < len(bucket) and bucket[position] == fixed_pivot)
            if size:
                groups.append((key, bucket, fixed_pivot))
                count += size
        self.check()
        return _Rows(self, groups, count)


def compact_relation(oracle, head, slots):
    """Return a complete lazy relation, or None when its shape/storage is unsuitable."""
    table = oracle.task.negatives._tables.get((head.name, len(head.arguments)))
    if table is None or not table.values:
        return None
    columns, constants = {}, {}
    for column, argument in enumerate(head.arguments):
        used = variables(argument)
        if not used:
            constants[column] = argument
        elif argument.name == 'var__' and len(argument.arguments) == 1:
            slot, = used
            if slot in columns:
                return None
            columns[slot] = column
        else:
            return None
    retained = getattr(oracle, '_negative_compact_buckets', 0)
    if retained + len(table.buckets) > COMPACT_BUCKET_LIMIT:
        return None
    result = CompactNegativeRelation(table, tuple(columns[slot] for slot in slots),
                                     constants, oracle.space, oracle._check)
    oracle.check()
    oracle._negative_compact_buckets = retained + len(table.buckets)
    oracle.stats['negative_compact_buckets'] = oracle._negative_compact_buckets
    oracle.stats['negative_compact_tables'] = oracle.stats.get('negative_compact_tables', 0) + 1
    return result
