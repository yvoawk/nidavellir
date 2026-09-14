"""Bounded reuse of conjunctions with NO supplied positive for one exact head.

No pilot, restricted mask, bias illegality or interrupted query is a proof.
Only a completed full-positive query may populate the antichain. Omitting a
core or stopping minimization affects performance only, not completeness.
"""
from collections import OrderedDict
from itertools import combinations
import time

from .positive_join import pattern_has_positive


MAX_CORES_PER_HEAD = 128
MAX_RECENT_ATTEMPTS = 256
MAX_DEFERRED_PROBES = 1024
MAX_SMALL_CORE_SIZE = 4
MAX_SMALL_PROBES = 128
MIN_QUERY_NODES = 4096
MAX_PROBE_SECONDS = 0.05
MAX_LEARN_SECONDS = 0.25
LEARNING_FRACTION = 0.1


class _LearningExpired(Exception):
    pass


class _ProbeExpired(Exception):
    pass


class PositiveCores:
    def __init__(self, oracle):
        self.oracle = oracle
        self.full = (1 << len(oracle.positives)) - 1
        self.cores = {}
        self.attempted = OrderedDict()
        self.deferred = OrderedDict()
        self.credit = MAX_LEARN_SECONDS

    def count(self, name, amount=1):
        key = 'positive_core_' + name
        stats = self.oracle.stats
        stats[key] = stats.get(key, 0) + amount

    def match(self, head, selected):
        self.oracle.check()
        bank = self.cores.get(head)
        if not bank:
            return None
        self.count('lookups')
        for position, core in enumerate(bank):
            if position % 32 == 0:
                self.oracle.check()
            if core & selected == core:
                self.count('hits')
                return core
        return None

    def _store(self, head, body):
        """Caller supplies a completed global proof, never a partial failure."""
        self.oracle.check()
        bits = sum(1 << i for i in body)
        bank = self.cores.setdefault(head, OrderedDict())
        if any(core & bits == core for core in bank):
            return
        removed = [core for core in bank if core & bits == bits]
        for core in removed:
            del bank[core]
        self.count('subsumed', len(removed))
        if MAX_CORES_PER_HEAD < 1:
            return
        while len(bank) >= MAX_CORES_PER_HEAD:
            bank.popitem(last=False)
            self.count('evictions')
        bank[bits] = tuple(body)
        self.count('stored')
        self.oracle.stats['positive_core_peak_entries'] = max(
            self.oracle.stats.get('positive_core_peak_entries', 0),
            sum(len(values) for values in self.cores.values()))

    def _probe(self, head, body):
        self.oracle.check()
        key = head, body
        if key in self.deferred:
            self.deferred.move_to_end(key)
            self.count('deferred_probes')
            return None  # Unknown, never a negative answer.
        previous = self.oracle.check
        deadline = time.monotonic() + MAX_PROBE_SECONDS

        def check():
            previous()  # Caller cancellation must never become a local timeout.
            if time.monotonic() >= deadline:
                raise _ProbeExpired()

        self.count('probes')
        self.oracle.check = check
        try:
            result = pattern_has_positive(self.oracle, head, body, self.full)
            check()
            return result
        except _ProbeExpired:
            self.count('probe_timeouts')
            self.oracle._put(self.deferred, key, None, limit=MAX_DEFERRED_PROBES)
            return None
        finally:
            self.oracle.check = previous

    def learn(self, head, body, query_seconds):
        """Try to generalize an expensive failed query, using all positives.

        The caller's failure may concern only a restricted mask. Deletions are
        independently checked globally; without such a proof nothing is saved.
        Retained cores need not be minimal if a probe or learning budget expires.
        """
        self.oracle.check()
        body = tuple(sorted(set(body)))
        key = head, body
        if key in self.attempted:
            self.attempted.move_to_end(key)
            self.count('repeated_attempts')
            return
        self.oracle._put(self.attempted, key, True, limit=MAX_RECENT_ATTEMPTS)
        self.credit = min(1.0, self.credit + max(0.0, query_seconds) * LEARNING_FRACTION)
        allowance = min(MAX_LEARN_SECONDS, self.credit)
        if allowance <= 0:
            return
        self.count('attempts')
        started = time.monotonic()
        deadline = started + allowance
        previous = self.oracle.check
        current, certified = body, None

        def check():
            previous()
            if time.monotonic() >= deadline:
                raise _LearningExpired()

        self.oracle.check = check
        try:
            with self.oracle.profile.measure('oracle.positive_core_learning'):
                # Small queries avoid repeating most of an expensive failed
                # join before irrelevant literals can be removed. No schema
                # or previously mined clinical conjunction is preloaded.
                probes = 0
                for size in range(1, min(MAX_SMALL_CORE_SIZE, len(body)) + 1):
                    for smaller in combinations(body, size):
                        check()
                        if probes >= MAX_SMALL_PROBES:
                            break
                        probes += 1
                        if self._probe(head, smaller) is False:
                            certified = smaller
                            break
                    if certified is not None or probes >= MAX_SMALL_PROBES:
                        break
                if certified is None:
                    for literal in body:
                        check()
                        smaller = tuple(i for i in current if i != literal)
                        if self._probe(head, smaller) is False:
                            current = certified = smaller
                    if certified is None and self._probe(head, current) is False:
                        certified = current
        except _LearningExpired:
            self.count('learning_timeouts')
        finally:
            self.oracle.check = previous
            elapsed = time.monotonic() - started
            self.credit = max(0.0, self.credit - elapsed)
            self.count('learning_seconds', elapsed)
        if certified is not None:
            self._store(head, certified)

    def export(self):
        return dict(scope='No supplied positive matches this exact head and conjunction; per immutable oracle',
            minimality_guaranteed=False, max_cores_per_head=MAX_CORES_PER_HEAD,
            max_recent_attempts=MAX_RECENT_ATTEMPTS, min_query_nodes=MIN_QUERY_NODES,
            max_deferred_probes=MAX_DEFERRED_PROBES, max_small_core_size=MAX_SMALL_CORE_SIZE,
            max_small_probes=MAX_SMALL_PROBES,
            max_probe_seconds=MAX_PROBE_SECONDS, max_learn_seconds=MAX_LEARN_SECONDS,
            learning_fraction=LEARNING_FRACTION,
            cores=[dict(head=str(head), body=list(body))
                   for head, bank in self.cores.items() for body in bank.values()])
