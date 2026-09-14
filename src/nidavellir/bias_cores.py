"""Bounded signed assumption cores from one immutable head legality problem.

A core requires both present and absent body atoms. Adding a body literal can
repair an illegal rule, so positive-only subset pruning is not sound here.
Temporary guard-query gates must never enter this bank.
"""
from collections import OrderedDict
from itertools import combinations
from math import comb


MAX_CORES = 256


class BiasCores:
    def __init__(self, stats, *, max_body=None):
        self.stats = stats
        self.max_body = max_body
        self.cores = OrderedDict()
        self.by_present = {}

    def count(self, name, amount=1):
        key = 'bias_core_' + name
        self.stats[key] = self.stats.get(key, 0) + amount

    @staticmethod
    def positive_key(core):
        return frozenset(literal for literal in core if literal > 0)

    def match(self, present, assumptions):
        if not self.cores:
            return False
        self.count('lookups')
        present = frozenset(present)
        comparisons = 0
        for size, groups in self.by_present.items():
            if size > len(present):
                continue
            # A 7-literal query has only seven possible 6-literal subsets,
            # even if hundreds of cached cores share its first literal.
            # For larger bodies, scan at most the bounded cached key set.
            if size == len(present):
                keys = (present,)
            elif size == 0:
                keys = (frozenset(),)
            elif comb(len(present), size) < len(groups):
                keys = (frozenset(items) for items in combinations(present, size))
            else:
                keys = (key for key in groups if key <= present)
            for key in keys:
                for core in groups.get(key, ()):
                    comparisons += 1
                    if core <= assumptions:
                        self.count('comparisons', comparisons)
                        self.count('hits')
                        return True
        self.count('comparisons', comparisons)
        return False

    def remove(self, core):
        del self.cores[core]
        key = self.positive_key(core)
        groups = self.by_present[len(key)]
        bucket = groups[key]
        del bucket[core]
        if not bucket:
            del groups[key]
        if not groups:
            del self.by_present[len(key)]

    def store(self, core, assumptions):
        """Caller must have completed UNSAT, with no temporary gate active."""
        core = frozenset(core)
        if not core <= assumptions:
            # A solver literal outside the explicit body assumptions has no
            # transferable certificate here. Omitting it would weaken proof.
            self.count('unmapped')
            return
        if core and self.max_body is not None and sum(literal > 0 for literal in core) >= self.max_body:
            # These requirements already fill every body slot. No different
            # body can contain them; keep the existing exact-answer cache.
            # Admitting them evicts reusable cores during max-length traversal.
            self.count('full_body_skips')
            return
        if MAX_CORES < 1 or any(old <= core for old in self.cores):
            return
        for old in tuple(self.cores):
            if core <= old:
                self.remove(old)
                self.count('subsumed')
        while len(self.cores) >= MAX_CORES:
            self.remove(next(iter(self.cores)))
            self.count('evictions')
        self.cores[core] = None
        key = self.positive_key(core)
        self.by_present.setdefault(len(key), {}).setdefault(key, {})[core] = None
        self.count('stored')
        self.stats['bias_core_peak_entries'] = max(self.stats.get('bias_core_peak_entries', 0), len(self.cores))

    def export(self):
        return dict(max_cores=MAX_CORES, max_body=self.max_body, minimality_guaranteed=False,
                    scope='Signed body assumptions for this exact grounded head problem; no temporary gates',
                    cores=[sorted(core) for core in self.cores])
