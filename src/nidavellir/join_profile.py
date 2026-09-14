"""Bounded slow-query diagnostics; no query result is inferred from this data."""
from collections import defaultdict
from functools import wraps
import heapq
import time


class JoinProfile:
    def __init__(self, limit=64):
        self.limit = limit
        self.slowest = []
        self.calls = 0
        self.seconds = 0.0
        self.groups = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self.literals = {}

    def record(self, task, head, body, possible, seconds, outcome, metrics):
        self.calls += 1
        self.seconds += seconds
        group = self.groups[f"length={len(body)},result={outcome},cache={metrics['cache']}"]
        group["calls"] += 1
        group["seconds"] += seconds
        for index, values in metrics["relations"].items():
            total = self.literals.setdefault(index, dict(literal=str(task.literals[index]),
                calls=0, seconds=0.0, candidate_rows=0, max_rows=0, visited_rows=0))
            for key in ("calls", "seconds", "candidate_rows", "visited_rows"):
                total[key] += values[key]
            total["max_rows"] = max(total["max_rows"], values["max_rows"])
        if not self.limit or (len(self.slowest) == self.limit and seconds <= self.slowest[0][0]):
            return
        row = dict(head=str(head), body=list(body), rule=task.rule(head, body),
                   possible_hex=None if possible is None else hex(possible), seconds=seconds, outcome=outcome, **metrics)
        row["relations"] = {str(k): dict(v) for k, v in metrics["relations"].items()}
        entry = seconds, self.calls, row
        if len(self.slowest) < self.limit:
            heapq.heappush(self.slowest, entry)
        else:
            heapq.heapreplace(self.slowest, entry)

    def export(self):
        return dict(calls=self.calls, seconds=self.seconds, retained_calls=len(self.slowest),
                    retention_limit=self.limit, groups={k: dict(v) for k, v in self.groups.items()},
                    literals={str(k): dict(v) for k, v in self.literals.items()},
                    slowest_calls=[row for _, _, row in sorted(self.slowest, reverse=True)],
                    scope="Exact totals by length/result/cache and literal; bounded slowest individual calls, not all bodies")


def profile_join(function=None, *, name="positive_join"):
    if function is None:
        return lambda wrapped: profile_join(wrapped, name=name)
    @wraps(function)
    def wrapped(oracle, head, body, possible):
        body = tuple(body)
        metrics = dict(cache="miss", label_steps=0, literal_steps=0, nodes=0,
                       relations={}, index_deltas={})
        before = dict(oracle.space.stats)
        started = time.perf_counter()
        outcome = "interrupted"
        try:
            answer = function(oracle, head, body, possible, metrics)
            outcome = str(answer).lower()
            return answer
        finally:
            seconds = time.perf_counter() - started
            metrics["index_deltas"] = {k: v - before.get(k, 0) for k, v in oracle.space.stats.items()
                                       if v != before.get(k, 0)}
            # Invalid inputs are errors, not query records with malformed IDs.
            if (type(possible) is int and possible >= 0 and all(type(i) is int and
                    0 <= i < len(oracle.task.literals) for i in body)):
                profile = oracle.profile.diagnostics.setdefault(name, JoinProfile())
                profile.record(oracle.task, head, body, possible, seconds, outcome, metrics)
    return wrapped


def measured_rows(oracle, literal_index, binding, metrics):
    started = time.perf_counter()
    entry = metrics["relations"].setdefault(literal_index, dict(calls=0, seconds=0.0,
                             candidate_rows=0, max_rows=0, visited_rows=0))
    entry["calls"] += 1
    try:
        if oracle.positive_join_backend == "indexed":
            from .indexed_join import matched_relation
            slots, table = matched_relation(oracle.space, literal_index)
            rows = table.select(tuple((j, binding[slot]) for j, slot in enumerate(slots)
                                      if slot in binding), oracle.space.stats)
        else:
            rows = oracle.space._rows(oracle.task.literals[literal_index], binding)
        entry["candidate_rows"] += len(rows)
        entry["max_rows"] = max(entry["max_rows"], len(rows))
        return rows
    finally:
        entry["seconds"] += time.perf_counter() - started


def profile_negative(function):
    """Keep bounded slow negative bodies, including the query interrupted at a deadline."""
    @wraps(function)
    def wrapped(oracle, head, body):
        body = tuple(body)
        before = dict(oracle.stats)
        indexes = dict(oracle.space.stats)
        started = time.perf_counter()
        outcome = "interrupted"
        try:
            result = function(oracle, head, body)
            outcome = "safe" if result is None else "witness"
            return result
        finally:
            if all(type(i) is int and 0 <= i < len(oracle.task.literals) for i in body):
                cache_hit = any(oracle.stats.get(k,0) > before.get(k,0) for k in
                                ("safe_subsumption_hits","negative_witness_cache_hits"))
                metrics = dict(cache="hit" if cache_hit else "miss", relations={},
                    oracle_deltas={k:v-before.get(k,0) for k,v in oracle.stats.items() if v != before.get(k,0)},
                    index_deltas={k:v-indexes.get(k,0) for k,v in oracle.space.stats.items() if v != indexes.get(k,0)})
                profile = oracle.profile.diagnostics.setdefault("negative_join", JoinProfile())
                profile.record(oracle.task, head, body, None, time.perf_counter()-started, outcome, metrics)
    return wrapped
