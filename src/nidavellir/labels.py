"""Exact negative labels without interning their outer predicate symbols.

Each bucket stores only supplied tuples. Neither a column product nor a
closed-world domain is inferred. Bitmap/sparse choices change storage only.
"""
from collections import OrderedDict
from collections.abc import Set
from dataclasses import dataclass
from heapq import merge
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import clingo


class NeedsNormalization(ValueError):
    pass


@dataclass(frozen=True)
class AtomRecord:
    name: str
    arguments: tuple
    type = clingo.SymbolType.Function
    positive = True

    def __str__(self):
        return self.name + ('(' + ','.join(map(str, self.arguments)) + ')' if self.arguments else '')

    def to_symbol(self):
        return clingo.Function(self.name, self.arguments)

    @property
    def order_key(self):
        # Positive predicate symbols: arity, name, lexicographic arguments.
        return len(self.arguments), self.name, self.arguments


def parse_atom(text, cache):
    """Parse arguments separately; never parse a composite wrapper or tuple."""
    match = re.fullmatch(r'([a-z][A-Za-z0-9_]*)(?:\((.*)\))?', text.strip(), re.S)
    if match is None:
        raise NeedsNormalization()
    name, inside = match.groups()
    if inside is None:
        return AtomRecord(name, ())
    parts, start, depth, quoted, escaped = [], 0, 0, False, False
    for position, char in enumerate(inside):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                raise NeedsNormalization()
        elif char == ',' and depth == 0:
            parts.append(inside[start:position].strip())
            start = position + 1
    if quoted or depth:
        raise NeedsNormalization()
    parts.append(inside[start:].strip())
    if any(not part for part in parts):
        raise NeedsNormalization()
    arguments = []
    for part in parts:
        value = cache.get(part)
        if value is None:
            try:
                value = clingo.parse_term(part)
            except RuntimeError:
                raise NeedsNormalization() from None
            cache[part] = value
            if len(cache) > 8192:
                cache.popitem(last=False)
        arguments.append(value)
    return AtomRecord(name, tuple(arguments))


def _ids(bucket):
    if type(bucket) is int:
        while bucket:
            bit = bucket & -bucket
            yield bit.bit_length() - 1
            bucket ^= bit
    else:
        yield from bucket


def _insert(bucket, value):
    if bucket is None:
        return 1 << value if value <= 4095 else (value,)
    if type(bucket) is int:
        if value <= 4095 or value <= 64 * max(1, bucket.bit_count()):
            return bucket | (1 << value)
        bucket = set(_ids(bucket))
    elif type(bucket) is tuple:
        if value in bucket:
            return bucket
        bucket = set(bucket)
    bucket.add(value)
    highest = max(bucket)
    if highest <= 4095 or highest <= 64 * len(bucket):
        return sum(1 << i for i in bucket)
    return bucket


class _TableBuilder:
    def __init__(self, name, arity):
        self.name, self.arity = name, arity
        self.values = [[] for _ in range(arity)]
        self.maps = [{} for _ in range(arity)]
        self.buffer, self.buckets, self.pivot = [], {}, None
        self.zero = False

    def add(self, arguments):
        if not self.arity:
            self.zero = True
            return
        row = []
        for column, value in enumerate(arguments):
            identifier = self.maps[column].get(value)
            if identifier is None:
                identifier = len(self.values[column])
                self.values[column].append(value)
                self.maps[column][value] = identifier
            row.append(identifier)
        row = tuple(row)
        if self.pivot is None:
            self.buffer.append(row)
            if len(self.buffer) >= 512:
                self.choose_pivot()
        else:
            self.insert(row)

    def choose_pivot(self):
        # Every buffered tuple is subsequently inserted. This trial block
        # chooses a storage layout, never a subset of the labels.
        def cost(pivot):
            groups = {}
            for row in self.buffer:
                groups.setdefault(row[:pivot] + row[pivot + 1:], set()).add(row[pivot])
            return sum(72 + min(8 * len(values), (max(values) + 8) // 8)
                       for values in groups.values())
        self.pivot = min(range(self.arity), key=lambda p: (cost(p), -p))
        for row in self.buffer:
            self.insert(row)
        self.buffer.clear()

    def insert(self, row):
        key = row[:self.pivot] + row[self.pivot + 1:]
        self.buckets[key] = _insert(self.buckets.get(key), row[self.pivot])

    def freeze(self):
        if self.arity and self.pivot is None:
            self.choose_pivot()
        values = tuple(tuple(column) for column in self.values)
        maps = tuple(MappingProxyType(dict(column)) for column in self.maps)
        buckets = {key: value if type(value) is int else tuple(sorted(value))
                   for key, value in self.buckets.items()}
        return _Table(self.name, values, maps, self.pivot, MappingProxyType(buckets), self.zero)


@dataclass(frozen=True)
class _Table:
    name: str
    values: tuple
    maps: tuple
    pivot: int | None
    buckets: object
    zero: bool = False

    def contains(self, arguments):
        if not self.values:
            return self.zero and not arguments
        row = tuple(mapping.get(arg, -1) for mapping, arg in zip(self.maps, arguments))
        if -1 in row:
            return False
        value = row[self.pivot]
        bucket = self.buckets.get(row[:self.pivot] + row[self.pivot + 1:])
        return bool(bucket is not None and ((bucket >> value) & 1 if type(bucket) is int else value in bucket))

    def count(self):
        return int(self.zero) + sum(v.bit_count() if type(v) is int else len(v) for v in self.buckets.values())

    def arguments(self):
        if self.zero:
            yield ()
            return
        pivot = self.pivot
        groups = {}
        for key, bucket in self.buckets.items():
            args = tuple(self.values[i][v] for i, v in zip(
                (i for i in range(len(self.values)) if i != pivot), key))
            groups.setdefault(args[:pivot], []).append((args, bucket))

        def rows(args, bucket):
            for value in sorted((self.values[pivot][i] for i in _ids(bucket))):
                yield args[:pivot] + (value,) + args[pivot:]

        for prefix in sorted(groups):
            yield from merge(*(rows(args, bucket) for args, bucket in groups[prefix]))


class NegativeBuilder:
    def __init__(self):
        self.tables = {}

    def add(self, record):
        key = record.name, len(record.arguments)
        if key not in self.tables:
            self.tables[key] = _TableBuilder(*key)
        self.tables[key].add(record.arguments)

    def freeze(self):
        result = CompactNegativeSet([table.freeze() for table in self.tables.values()])
        self.tables.clear()
        return result


class CompactNegativeSet(Set):
    """Immutable exact set; Symbol iteration is compatibility-only.

    Internal IO must use iter_text/iter_records, which do not intern outer
    negative symbols. Ordinary reference-oracle iteration can still do so.
    """
    def __init__(self, tables):
        self._tables = MappingProxyType({(table.name, len(table.values)): table for table in tables})
        self._count = sum(table.count() for table in self._tables.values())

    def __contains__(self, atom):
        if not hasattr(atom, 'type') or atom.type != clingo.SymbolType.Function or not atom.positive:
            return False
        arguments = atom.arguments
        table = self._tables.get((atom.name, len(arguments)))
        return table is not None and table.contains(arguments)

    def __len__(self):
        return self._count

    def iter_records(self):
        for key in sorted(self._tables, key=lambda k: (k[1], k[0])):
            table = self._tables[key]
            for arguments in table.arguments():
                yield AtomRecord(table.name, arguments)

    def iter_text(self):
        return (str(record) for record in self.iter_records())

    def __iter__(self):
        return (record.to_symbol() for record in self.iter_records())

    @classmethod
    def _from_iterable(cls, values):
        return frozenset(values)

    def __and__(self, other):
        return frozenset(value for value in other if value in self)

    __rand__ = __and__

    def stats(self):
        return {'negative_labels': len(self), 'negative_storage': 'exact_tuple_buckets',
                'argument_symbols': sum(sum(map(len, t.values)) for t in self._tables.values()),
                'buckets': sum(len(t.buckets) for t in self._tables.values()),
                'bitmap_buckets': sum(type(v) is int for t in self._tables.values() for v in t.buckets.values()),
                'pivots': {f'{t.name}/{len(t.values)}': t.pivot for t in self._tables.values()}}

    def dump(self, path, *, source_sha256, check=lambda: None):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        digest = hashlib.sha256()
        with temporary.open('w') as stream:
            def write(value):
                check()
                line = json.dumps(value, separators=(',', ':')) + '\n'
                stream.write(line)
                digest.update(line.encode())
            write({'type': 'header', 'schema': 1, 'source_sha256': source_sha256,
                   'count': len(self), 'tables': len(self._tables)})
            for key in sorted(self._tables, key=lambda k: (k[1], k[0])):
                table = self._tables[key]
                write({'type': 'table', 'name': table.name, 'pivot': table.pivot,
                       'values': [[str(v) for v in column] for column in table.values],
                       'zero': table.zero, 'buckets': len(table.buckets)})
                for key, bucket in sorted(table.buckets.items()):
                    write({'type': 'bucket', 'key': key,
                           **({'bits': hex(bucket)} if type(bucket) is int else {'ids': bucket})})
            footer = json.dumps({'type': 'complete', 'sha256': digest.hexdigest()}) + '\n'
            stream.write(footer)
            digest.update(footer.encode())
        temporary.replace(path)
        return digest.hexdigest()

    @classmethod
    def load(cls, path, *, source_sha256, artifact_sha256=None):
        # Validate the entire immutable artifact before exposing any labels.
        digest = hashlib.sha256()
        artifact_digest = hashlib.sha256()
        header = None
        complete = False
        tables = []
        current = None
        buckets = None
        def finish_table():
            if current is not None:
                if len(buckets) != current['buckets']:
                    raise ValueError('Incomplete compact negative table')
                values = tuple(tuple(clingo.parse_term(v) for v in column) for column in current['values'])
                if any(len(set(column)) != len(column) for column in values):
                    raise ValueError('Duplicate compact argument value')
                tables.append(_Table(current['name'], values,
                              tuple(MappingProxyType({v: i for i, v in enumerate(column)}) for column in values),
                              current['pivot'], MappingProxyType(dict(buckets)), current['zero']))
        with Path(path).open() as stream:
            for line in stream:
                artifact_digest.update(line.encode())
                item = json.loads(line)
                if complete:
                    raise ValueError('Data after compact negative completion')
                if item.get('type') == 'complete':
                    if header is None or item.get('sha256') != digest.hexdigest():
                        raise ValueError('Compact negative checksum mismatch')
                    finish_table()
                    complete = True
                    continue
                digest.update(line.encode())
                if header is None:
                    if item.get('type') != 'header' or item.get('schema') != 1 or item.get('source_sha256') != source_sha256:
                        raise ValueError('Incompatible compact negative artifact')
                    header = item
                elif item.get('type') == 'table':
                    finish_table()
                    current, buckets = item, {}
                    arity = len(item['values'])
                    if (not re.fullmatch(r'[a-z][A-Za-z0-9_]*', item['name']) or
                            (arity and not (type(item['pivot']) is int and 0 <= item['pivot'] < arity)) or
                            (not arity and item['pivot'] is not None) or type(item['zero']) is not bool or
                            item['zero'] != (not arity) or type(item['buckets']) is not int or item['buckets'] < 0):
                        raise ValueError('Invalid compact negative table')
                elif item.get('type') == 'bucket' and current is not None:
                    key = tuple(item['key'])
                    arity, pivot = len(current['values']), current['pivot']
                    dimensions = [len(v) for i, v in enumerate(current['values']) if i != pivot]
                    if (len(key) != arity - 1 or any(type(v) is not int or v < 0 or v >= n for v, n in zip(key, dimensions))
                            or key in buckets or ('bits' in item) == ('ids' in item)):
                        raise ValueError('Invalid compact negative bucket')
                    if 'bits' in item:
                        value = int(item['bits'], 16)
                        if value <= 0 or value.bit_length() > len(current['values'][pivot]):
                            raise ValueError('Compact negative bitmap outside domain')
                    else:
                        value = tuple(item['ids'])
                        if (not value or any(type(i) is not int or i < 0 or i >= len(current['values'][pivot]) for i in value)
                                or tuple(sorted(set(value))) != value):
                            raise ValueError('Invalid compact negative sparse IDs')
                    buckets[key] = value
                else:
                    raise ValueError('Invalid compact negative record')
        if not complete:
            raise ValueError('Interrupted compact negative artifact')
        if artifact_sha256 is not None and artifact_digest.hexdigest() != artifact_sha256:
            raise ValueError('Compact negative artifact fingerprint mismatch')
        result = cls(tables)
        if len(tables) != header['tables'] or len(result._tables) != len(tables) or len(result) != header['count']:
            raise ValueError('Compact negative count mismatch')
        return result


def canonical_text(labels):
    if isinstance(labels, CompactNegativeSet):
        return labels.iter_text()
    return map(str, sorted(labels))


def canonical_records(labels):
    if isinstance(labels, CompactNegativeSet):
        return labels.iter_records()
    return (AtomRecord(atom.name, tuple(atom.arguments)) for atom in sorted(labels))
