"""Lazy exact joins for finding bounds before building a full bitmask oracle.

These queries use the same positive BK atoms and templates as both reference
oracles. No clinical names or gold rules appear in the query mechanism.
"""
from collections import OrderedDict

import clingo

from .labels import AtomRecord, CompactNegativeSet
from .oracle import match
from .profiling import SearchProfile, timed
from .join_profile import profile_negative
from .task import instantiate, variables
from .witness_space import WitnessSpace


class CoverageOracle:
    def __init__(self, task, check=lambda: None, *, max_negative_witnesses=256,
                 positive_join_backend="indexed", positive_pair_pruning=True, binding_backend="indexed",
                 negative_join_backend="labels", negative_label_limit=65536, positive_residual_cache=True,
                 positive_lookahead=True, positive_residual_normalization=True, positive_core_pruning=False):
        if type(max_negative_witnesses) is not int or max_negative_witnesses < 0:
            raise ValueError("Negative-witness cache size must be a nonnegative integer")
        if type(positive_core_pruning) is not bool:
            raise ValueError("positive_core_pruning must be a boolean")
        self.task = task
        if positive_join_backend not in {"indexed", "reference"}:
            raise ValueError("positive_join_backend must be indexed or reference")
        if type(positive_pair_pruning) is not bool:
            raise ValueError("positive_pair_pruning must be a boolean")
        self.positive_join_backend = positive_join_backend
        self.positive_pair_pruning = positive_pair_pruning
        if type(positive_residual_cache) is not bool:
            raise ValueError("positive_residual_cache must be a boolean")
        self.positive_residual_cache = positive_residual_cache
        if type(positive_lookahead) is not bool:
            raise ValueError("positive_lookahead must be a boolean")
        self.positive_lookahead = positive_lookahead
        if type(positive_residual_normalization) is not bool:
            raise ValueError("positive_residual_normalization must be a boolean")
        self.positive_residual_normalization = positive_residual_normalization
        if binding_backend not in {"indexed", "reference"}:
            raise ValueError("binding_backend must be indexed or reference")
        self.binding_backend = binding_backend
        if negative_join_backend not in {"labels", "bindings"}:
            raise ValueError("negative_join_backend must be labels or bindings")
        if type(negative_label_limit) is not int or negative_label_limit < 0:
            raise ValueError("negative_label_limit must be a nonnegative integer")
        self.negative_join_backend = negative_join_backend
        self.negative_label_limit = negative_label_limit
        self.check = check
        self.profile = SearchProfile()
        with self.profile.measure("oracle.index"):
            self.space = WitnessSpace(task, self._check)
        self.head_variables = {head: variables(head) for head in task.heads}
        self.positives = {a: i for i, a in enumerate(sorted(task.positives))}
        self.stats = {"exists_queries": 0, "negative_queries": 0, "coverage_queries": 0,
                      "cache_hits": 0, "matched_atoms": 0, "safe_subsumption_hits": 0,
                      "negative_witness_cache_hits": 0, "negative_witnesses_cached": 0,
                      "negative_head_checks": 0, "negative_head_prunes": 0,
                      "positive_join_matched_atoms": 0, "positive_join_nodes": 0,
                      "positive_join_label_steps": 0, "positive_join_literal_steps": 0}
        self._exists_cache = OrderedDict()
        self._coverage_cache = OrderedDict()
        self._safe = {}
        self._negative_witnesses = {}
        self._atomic_projections = OrderedDict()
        self._atomic_head_bindings = {}
        self._atomic_label_groups = {}
        self.max_negative_witnesses = max_negative_witnesses
        self.positive_cores = None
        if positive_core_pruning:
            from .positive_cores import PositiveCores
            self.positive_cores = PositiveCores(self)
            self.profile.diagnostics['positive_cores'] = self.positive_cores

    def _check(self):
        self.check()

    def _head_vars(self, head):
        if head not in self.head_variables:
            self.head_variables[head] = variables(head)
        return self.head_variables[head]

    @timed("oracle.atomic_coverage")
    def atomic_coverage(self, head, literal_index):
        """Project one real BK relation onto the shared head variables.

        This is exact for a singleton, including repeated/nested variables.
        ANDs of such masks remain only upper bounds for multi-literal bodies:
        they do not assert that hidden-variable witnesses agree.
        """
        self.check()
        literal = self.task.literals[literal_index]
        structure, columns = self.space._view(literal, {})
        shared = tuple(slot for slot in columns if slot in self._head_vars(head))
        positions = tuple(columns.index(slot) for slot in shared)
        group_key = head, shared
        groups = self._atomic_label_groups.get(group_key)
        if groups is None:
            fixed_labels = self._atomic_head_bindings.get(head)
            if fixed_labels is None:
                fixed_labels = []
                for number, (label, position) in enumerate(self.positives.items()):
                    if number % 256 == 0:
                        self.check()
                    fixed = match(head, label)
                    if fixed is not None:
                        fixed_labels.append((position, fixed))
                self._atomic_head_bindings[head] = fixed_labels
            groups = {}
            for number, (position, fixed) in enumerate(fixed_labels):
                if number % 256 == 0:
                    self.check()
                values = tuple(fixed[slot] for slot in shared)
                groups[values] = groups.get(values, 0) | (1 << position)
            self._atomic_label_groups[group_key] = groups
        if not groups:
            return 0
        key = structure, positions
        cached = self._atomic_projections.get(key)
        if cached is None:
            self.stats["atomic_projection_builds"] = self.stats.get("atomic_projection_builds", 0) + 1
            rows = set()
            if not shared:
                if self.has_binding((literal_index,)):
                    rows.add(())
                cached = "complete", frozenset(rows)
            else:
                candidates = self.space._rows(literal, {})
                if len(candidates) <= len(groups):
                    for number, atom in enumerate(candidates):
                        if number % 256 == 0:
                            self.check()
                        binding = match(literal, atom)
                        if binding is not None:
                            rows.add(tuple(binding[slot] for slot in shared))
                    cached = "complete", frozenset(rows)
                else:
                    # Large BK relations (e.g. all temporal comparisons) are
                    # cheaper queried through the already indexed head keys.
                    # This cache contains exact answers, not a partial relation
                    # that could mistake an unqueried key for a negative.
                    cached = "answers", {}
            self._put(self._atomic_projections, key, cached, limit=256)
        else:
            self._atomic_projections.move_to_end(key)
            self.stats["atomic_projection_cache_hits"] = self.stats.get("atomic_projection_cache_hits", 0) + 1
        kind, projected = cached
        mask = 0
        for number, (values, positions_mask) in enumerate(groups.items()):
            if number % 256 == 0:
                self.check()
            if kind == "complete":
                found = values in projected
            else:
                if values not in projected:
                    projected[values] = self.has_binding((literal_index,), dict(zip(shared, values)))
                found = projected[values]
            if found:
                mask |= positions_mask
        return mask

    @staticmethod
    def _put(cache, key, value, limit=8192):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def bindings(self, body, fixed=None):
        """Yield satisfying bindings, expanding only matching positive atoms."""
        yield from self._bindings(body, fixed)

    def _bindings(self, body, fixed=None, *, negative_head=None):
        """Join BK atoms, optionally rejecting a fixed nonnegative head early.

        Once every head variable is bound, later positive joins cannot change
        that head. Rejected heads therefore have no negative completion. The
        accepted branches retain every variable and their real full witness.
        """
        if self.binding_backend == "indexed":
            from .indexed_join import bindings
            yield from bindings(self, body, fixed, negative_head=negative_head)
            return
        body = tuple(sorted(set(body)))
        head_vars = self._head_vars(negative_head) if negative_head is not None else None

        def walk(remaining, binding, head_checked=False):
            self.check()
            if head_vars is not None and not head_checked and head_vars <= binding.keys():
                record = AtomRecord(negative_head.name, tuple(
                    instantiate(argument, binding) for argument in negative_head.arguments))
                self.stats["negative_head_checks"] += 1
                if isinstance(self.task.negatives, CompactNegativeSet):
                    labelled = record in self.task.negatives
                else:
                    labelled = record.to_symbol() in self.task.negatives
                # Never turn an interrupted membership/filter into a cached
                # safety proof, including the last rejected branch.
                self.check()
                if not labelled:
                    self.stats["negative_head_prunes"] += 1
                    return
                head_checked = True
            # Ground filters need one set membership, not another column index
            # or a traversal of an already bound relation.
            unbound = []
            for i in remaining:
                literal = self.task.literals[i]
                if self.space._literal_vars[i] <= binding.keys():
                    if instantiate(literal, binding) not in self.task.atoms:
                        return
                else:
                    unbound.append(i)
            remaining = tuple(unbound)
            if not remaining:
                yield binding
                return
            # Filter by every already bound argument, then join the smallest
            # relation. Correlated variables are never existentialized apart.
            best = min(remaining, key=lambda i: len(self.space._rows(self.task.literals[i], binding)))
            rest = tuple(i for i in remaining if i != best)
            literal = self.task.literals[best]
            for n, atom in enumerate(self.space._rows(literal, binding)):
                if n % 256 == 0:
                    self.check()
                extended = match(literal, atom, binding)
                if extended is not None:
                    self.stats["matched_atoms"] += 1
                    yield from walk(rest, extended, head_checked)

        yield from walk(body, dict(fixed or {}))

    @timed("oracle.binding_exists")
    def has_binding(self, body, fixed=None):
        """Consume a join while keeping profiling outside its generator."""
        result = next(self.bindings(body, fixed), None) is not None
        self.check()
        return result

    @timed("oracle.exists")
    def exists(self, head, body, label):
        self.check()
        self.stats["exists_queries"] += 1
        key = head, tuple(sorted(body)), label
        if key in self._exists_cache:
            self.stats["cache_hits"] += 1
            return self._exists_cache[key]
        fixed = match(head, label)
        found = fixed is not None and self.has_binding(body, fixed)
        self.check()
        self._put(self._exists_cache, key, found)
        return found

    @timed("oracle.core")
    def core(self, head, body, label):
        """Return a subset whose impossibility at label is checked, or None."""
        if self.exists(head, body, label):
            return None
        current = tuple(sorted(body))
        for literal in tuple(current):
            smaller = tuple(i for i in current if i != literal)
            if not self.exists(head, smaller, label):
                current = smaller
        return current

    @timed("oracle.negative")
    @profile_negative
    def negative(self, head, body):
        """Return a real negative witness/profile, or prove no negative exists.

        Any body contained in the returned raw truth profile covers this same
        supplied negative. This is a sound CEGIS cut, including unused slots:
        they receive an arbitrary explicit completion and truths are checked
        against the real BK, never inferred from separate existential tests.
        """
        self.check()
        self.stats["negative_queries"] += 1
        selected = frozenset(body)
        used = set().union(*(self.space._literal_vars[i] for i in body)) if body else set()
        if not self._head_vars(head) <= used:
            raise ValueError("Negative query requires a head-safe body")
        if any(safe <= selected for safe in self._safe.get(head, ())):
            self.stats["safe_subsumption_hits"] += 1
            return None
        selected_bits = sum(1 << i for i in selected)
        for witness in self._negative_witnesses.get(head, ()):
            if selected_bits & witness["truth"] == selected_bits:
                self.stats["negative_witness_cache_hits"] += 1
                return dict(witness, binding=dict(witness["binding"]))
        for binding in self._bindings(body, negative_head=head):
            atom = instantiate(head, binding)
            completed = dict(binding)
            for slot in self.space.all_vars - completed.keys():
                completed[slot] = clingo.Function("__rc_unused")
            truth = sum(1 << i for i, literal in enumerate(self.task.literals)
                        if instantiate(literal, completed) in self.task.atoms)
            self.check()
            witness = {"head": atom, "binding": completed, "truth": truth}
            if self.max_negative_witnesses:
                previous = self._negative_witnesses.setdefault(head, [])
                # A profile with more true literals invalidates every body
                # invalidated by its subset profile. Keep a bounded antichain
                # of real witnesses, never an invented union of profiles.
                if not any(truth & old["truth"] == truth for old in previous):
                    previous[:] = [old for old in previous if old["truth"] & truth != old["truth"]]
                    previous.append(witness)
                    del previous[:-self.max_negative_witnesses]
                self.stats["negative_witnesses_cached"] = sum(map(len, self._negative_witnesses.values()))
            return dict(witness, binding=dict(completed))
        self.check()
        safe = self._safe.setdefault(head, [])
        safe[:] = [previous for previous in safe if not selected <= previous]
        safe.append(selected)
        return None

    @timed("oracle.coverage")
    def coverage(self, head, body):
        """Return the exact bitmask of covered positive labels."""
        self.check()
        self.stats["coverage_queries"] += 1
        key = head, tuple(sorted(body))
        if key in self._coverage_cache:
            self.stats["cache_hits"] += 1
            return self._coverage_cache[key]
        used = set().union(*(self.space._literal_vars[i] for i in body)) if body else set()
        if not self._head_vars(head) <= used:
            raise ValueError("Coverage query requires a head-safe body")
        result = 0
        # Coverage concerns only supplied positives. Fixing each labelled
        # head before the existential join avoids enumerating unlabelled
        # outputs and duplicate substitutions of hidden variables.
        for atom, position in self.positives.items():
            if self.exists(head, body, atom):
                result |= 1 << position
        self._put(self._coverage_cache, key, result)
        return result

    @timed("oracle.predictions")
    def predictions(self, head, body):
        used = set().union(*(self.space._literal_vars[i] for i in body)) if body else set()
        if not self._head_vars(head) <= used:
            raise ValueError("Prediction query requires a head-safe body")
        return frozenset(instantiate(head, binding) for binding in self.bindings(body))
