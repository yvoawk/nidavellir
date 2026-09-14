"""Finite matched relations used to construct exact witness quotients.

This module never enumerates a product of hidden-variable domains.  It keeps
the same domain convention as the Cartesian reference oracle: each hidden
slot receives the union of values in matching positive BK atoms.  Matching
respects constants, nested constructor terms and repeated variable slots.
"""
from collections import OrderedDict

import clingo

from .task import instantiate, variables


def _match(pattern, atom, fixed):
    binding = dict(fixed)

    def visit(p, a):
        if p.type == clingo.SymbolType.Function and p.name == "var__":
            slot = p.arguments[0].number
            if slot in binding:
                return binding[slot] == a
            binding[slot] = a
            return True
        if p.type != a.type:
            return False
        if p.type != clingo.SymbolType.Function:
            return p == a
        return (p.name == a.name and p.positive == a.positive
                and len(p.arguments) == len(a.arguments)
                and all(visit(x, y) for x, y in zip(p.arguments, a.arguments)))

    return binding if visit(pattern, atom) else None


class WitnessSpace:
    """Indexed BK relations with caches shared across variable renamings.

    ``symbols[id]`` and ``ids[symbol]`` are inverse, append-only mappings for
    the lifetime of this space.  ``relation`` columns follow sorted slot IDs;
    its row entries are interned symbol IDs.  An empty column list represents
    a fully fixed atom, with rows ``{()}`` if true and ``{}`` otherwise.

    Relation and projection caches are bounded by entry count.  The underlying
    predicate/argument indexes are retained because they index the fixed BK.
    """

    def __init__(self, task, check):
        self.task = task
        self.check = check
        self.symbols = []
        self.ids = {}
        self.stats = {"relation_cache_hits": 0, "relation_cache_misses": 0,
                      "projection_cache_hits": 0, "projection_cache_misses": 0,
                      "matched_rows": 0}
        self.all_vars = set().union(*(variables(l) for l in task.literals))
        self._literal_vars = tuple(variables(l) for l in task.literals)
        self._relations = {}
        for n, atom in enumerate(task.atoms):
            if n % 4096 == 0:
                check()
            self._relations.setdefault((atom.name, len(atom.arguments)), []).append(atom)
        self._argument_indexes = {}
        self._filtered_rows = OrderedDict()
        self._filtered_row_count = 0
        self._column_projections = {}
        self._projection_cache = OrderedDict()
        self._relation_cache = OrderedDict()

    def intern(self, symbol):
        found = self.ids.get(symbol)
        if found is None:
            found = len(self.symbols)
            self.ids[symbol] = found
            self.symbols.append(symbol)
        return found

    @staticmethod
    def _cache_get(cache, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _cache_put(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _view(self, literal, fixed):
        """Canonical structure and columns in first-occurrence slot order.

        Fixed values are substituted before canonicalising.  Consequently
        cache sharing across templates is valid even if their slot numbers,
        head-variable roles or fixed/unfixed occurrences differ.
        """
        slots = []
        positions = {}

        def visit(term):
            if term.type != clingo.SymbolType.Function:
                return ("constant", term)
            if term.name == "var__":
                slot = term.arguments[0].number
                if slot in fixed:
                    # A fixed symbol is data: do not reinterpret a nested
                    # constructor named var__ as a template placeholder.
                    return ("fixed", fixed[slot])
                if slot not in positions:
                    positions[slot] = len(slots)
                    slots.append(slot)
                return ("slot", positions[slot])
            return ("function", term.name, term.positive,
                    tuple(visit(arg) for arg in term.arguments))

        return visit(literal), tuple(slots)

    def _rows(self, literal, fixed):
        signature = literal.name, len(literal.arguments)
        candidates = self._relations.get(signature, ())
        conditions = []
        for position, argument in enumerate(literal.arguments):
            if not variables(argument) <= fixed.keys():
                continue
            value = instantiate(argument, fixed)
            conditions.append((position, value))
            key = signature, position
            if key not in self._argument_indexes:
                table = {}
                for n, atom in enumerate(self._relations.get(signature, ())):
                    if n % 4096 == 0:
                        self.check()
                    table.setdefault(atom.arguments[position], []).append(atom)
                self.check()
                self._argument_indexes[key] = table
            selected = self._argument_indexes[key].get(value, ())
            if len(selected) < len(candidates):
                candidates = selected
            if not candidates:
                return ()
        if len(conditions) < 2:
            return candidates
        # A mono-column index is only a starting point. Apply every known
        # argument equality before planning or walking a join. Repeated and
        # nested unbound slots remain checked by the ordinary matcher.
        key = signature, tuple(conditions)
        cached = self._cache_get(self._filtered_rows, key)
        if cached is not None:
            self.stats['filtered_rows_cache_hits'] = self.stats.get('filtered_rows_cache_hits', 0) + 1
            return cached
        rows = []
        for n, atom in enumerate(candidates):
            if n % 256 == 0:
                self.check()
            arguments = atom.arguments
            if all(arguments[position] == value for position, value in conditions):
                rows.append(atom)
        self.check()
        rows = tuple(rows)
        self.stats['filtered_rows_scanned'] = self.stats.get('filtered_rows_scanned', 0) + len(candidates)
        # Bound both entry count and retained row references; large results
        # are returned but never kept by this cache.
        if len(rows) <= 32768:
            while self._filtered_rows and (len(self._filtered_rows) >= 512 or
                    self._filtered_row_count + len(rows) > 32768):
                _, removed = self._filtered_rows.popitem(last=False)
                self._filtered_row_count -= len(removed)
            self._filtered_rows[key] = rows
            self._filtered_row_count += len(rows)
        return rows

    @staticmethod
    def _simple(literal, fixed):
        slots = []
        for argument in literal.arguments:
            if (argument.type != clingo.SymbolType.Function
                    or argument.name != "var__"
                    or argument.arguments[0].number in fixed):
                return None
            slots.append(argument.arguments[0].number)
        return tuple(slots) if len(set(slots)) == len(slots) else None

    def _projections(self, index, fixed):
        literal = self.task.literals[index]
        key, slots = self._view(literal, fixed)
        cached = self._cache_get(self._projection_cache, key)
        if cached is not None:
            self.stats["projection_cache_hits"] += 1
            return slots, cached
        self.stats["projection_cache_misses"] += 1
        simple = self._simple(literal, fixed)
        if simple is not None:
            signature = literal.name, len(literal.arguments)
            columns = []
            for position in range(len(simple)):
                projection_key = signature, position
                if projection_key not in self._column_projections:
                    values = set()
                    for n, atom in enumerate(self._relations.get(signature, ())):
                        if n % 4096 == 0:
                            self.check()
                        values.add(atom.arguments[position])
                    self._column_projections[projection_key] = frozenset(values)
                columns.append(self._column_projections[projection_key])
            result = tuple(columns)
        else:
            columns = [set() for _ in slots]
            for n, atom in enumerate(self._rows(literal, fixed)):
                if n % 4096 == 0:
                    self.check()
                binding = _match(literal, atom, fixed)
                if binding is not None:
                    for values, slot in zip(columns, slots):
                        values.add(binding[slot])
            result = tuple(frozenset(column) for column in columns)
        self._cache_put(self._projection_cache, key, result, 4096)
        return slots, result

    def domains(self, fixed):
        """Return every hidden slot's exact reference-oracle domain."""
        values = {slot: set() for slot in sorted(self.all_vars - fixed.keys())}
        for index, literal_vars in enumerate(self._literal_vars):
            self.check()
            if not literal_vars - fixed.keys():
                continue
            slots, columns = self._projections(index, fixed)
            for slot, column in zip(slots, columns):
                values[slot].update(column)
        result = {}
        for slot, column in values.items():
            self.check()
            result[slot] = tuple(sorted(column)) or (clingo.Function("__rc_unused"),)
            for n, symbol in enumerate(result[slot]):
                if n % 4096 == 0:
                    self.check()
                self.intern(symbol)
        return result

    def relation(self, literal_index, fixed, domains):
        """Return the positive matched relation, restricted to given domains."""
        self.check()
        literal = self.task.literals[literal_index]
        key, first_slots = self._view(literal, fixed)
        slots = tuple(sorted(first_slots))
        if not slots:
            truth = instantiate(literal, fixed) in self.task.atoms
            return slots, frozenset({()}) if truth else frozenset()
        # Store columns in canonical first-occurrence order.  Any renaming
        # then reads the same rows, only permuting columns when necessary.
        cached = self._cache_get(self._relation_cache, key)
        if cached is None:
            self.stats["relation_cache_misses"] += 1
            rows = set()
            if self._simple(literal, fixed) is not None:
                signature = literal.name, len(literal.arguments)
                for n, atom in enumerate(self._relations.get(signature, ())):
                    if n % 4096 == 0:
                        self.check()
                    rows.add(tuple(self.intern(value) for value in atom.arguments))
            else:
                for n, atom in enumerate(self._rows(literal, fixed)):
                    if n % 4096 == 0:
                        self.check()
                    binding = _match(literal, atom, fixed)
                    if binding is not None:
                        rows.add(tuple(self.intern(binding[slot]) for slot in first_slots))
            rows = frozenset(rows)
            column_ids = [set() for _ in first_slots]
            for n, row in enumerate(rows):
                if n % 4096 == 0:
                    self.check()
                for column, value in zip(column_ids, row):
                    column.add(value)
            cached = rows, tuple(frozenset(column) for column in column_ids)
            self.stats["matched_rows"] += len(rows)
            self._cache_put(self._relation_cache, key, cached, 256)
        else:
            self.stats["relation_cache_hits"] += 1
        rows, columns = cached
        allowed = tuple(frozenset(self.intern(value) for value in domains[slot])
                        for slot in first_slots)
        unrestricted = all(column <= domain for column, domain in zip(columns, allowed))
        positions = tuple(first_slots.index(slot) for slot in slots)
        if unrestricted and positions == tuple(range(len(slots))):
            return slots, rows
        result = set()
        for n, row in enumerate(rows):
            if n % 4096 == 0:
                self.check()
            if unrestricted or all(value in domain for value, domain in zip(row, allowed)):
                result.add(tuple(row[position] for position in positions))
        return slots, frozenset(result)
