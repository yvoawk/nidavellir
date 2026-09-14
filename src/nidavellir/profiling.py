"""Nested wall-clock measurements, adapted from experiment 11.

Inclusive timings must not be summed across nested spans. Exclusive timings
charge each elapsed interval once. Measurements survive exceptions/timeouts.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time


class SearchProfile:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.spans = defaultdict(lambda: {"calls": 0, "seconds": 0.0,
                                          "exclusive_seconds": 0.0})
        self.stack = []
        self.diagnostics = {}

    @contextmanager
    def measure(self, name):
        frame = [self.clock(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            duration = self.clock() - frame[0]
            assert self.stack.pop() is frame
            if self.stack:
                self.stack[-1][1] += duration
            span = self.spans[name]
            span["calls"] += 1
            span["seconds"] += duration
            span["exclusive_seconds"] += duration - frame[1]

    def export(self):
        return {"timings": {name: dict(values) for name, values in sorted(self.spans.items())},
                "diagnostics": {name: value.export() for name, value in self.diagnostics.items()},
                "timing_semantics": "Wall seconds; seconds includes nested calls, exclusive_seconds excludes them."}


def timed(name):
    """Instrument an ordinary method on an object exposing ``profile``.

    Do not apply to generators: their work runs during iteration, after the
    decorated call has returned. Instrument the query that consumes them.
    """
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with self.profile.measure(name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate
