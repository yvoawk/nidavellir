"""Compact human-readable progress reporting for command-line runs."""
import sys
import threading
import time


class TerminalProgress:
    """Render selected trace fields on stderr without changing JSON stdout."""

    def __init__(self, enabled=True, *, target=None, stream=None, interval=1.0,
                 heartbeat_interval=10.0):
        self.enabled = enabled
        self.target = target
        self.stream = stream if stream is not None else sys.stderr
        self.interval = interval
        self.heartbeat_interval = heartbeat_interval
        self.stage = None
        self.last_write = None
        self.last_line = None
        self.started = None
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def start(self, started=None):
        """Start elapsed-time heartbeats for phases without trace callbacks."""
        if not self.enabled or self.thread is not None:
            return
        self.started = time.monotonic() if started is None else started

        def heartbeat():
            while not self.stop_event.wait(self.heartbeat_interval):
                self.emit({"wall_seconds": time.monotonic() - self.started}, force=True)

        self.thread = threading.Thread(target=heartbeat, name="nidavellir-terminal-progress", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def emit(self, event, *, force=False):
        if not self.enabled or not event:
            return
        with self.lock:
            if "active_target" in event:
                self.target = event["active_target"]
            if event.get("target_signature"):
                self.target = event["target_signature"]
            if event.get("stage"):
                self.stage = event["stage"]

            visible = {"active_target", "target_signature", "stage", "phase", "current_head",
                       "nodes", "legal_safe_rules", "compressed_witnesses", "raw_witnesses",
                       "hypothesis_cost", "optimal_cost", "learned_rule_count", "status"}
            if not force and not any(event.get(key) is not None for key in visible):
                return

            status_event = event.get("event") == "status"
            now = time.monotonic()
            if (not force and not status_event and self.last_write is not None
                    and now - self.last_write < self.interval):
                return

            fields = ["[Nidavellir]"]
            if self.target:
                fields.append(f"target={self.target}")
            if self.stage:
                fields.append(f"stage={self.stage}")
            phase = event.get("phase")
            if phase and phase != self.stage:
                fields.append(f"phase={phase}")
            elapsed = event.get("wall_seconds", event.get("elapsed_seconds"))
            if isinstance(elapsed, (int, float)):
                fields.append(f"elapsed={elapsed:.1f}s")
            for key, label in (
                ("current_head", "head"),
                ("nodes", "nodes"),
                ("legal_safe_rules", "safe_rules"),
                ("compressed_witnesses", "witnesses"),
                ("raw_witnesses", "raw_witnesses"),
                ("hypothesis_cost", "cost"),
                ("optimal_cost", "optimum"),
                ("learned_rule_count", "rules"),
                ("status", "status"),
            ):
                value = event.get(key)
                if value is not None:
                    fields.append(f"{label}={value}")
            line = " ".join(fields)
            if not force and line == self.last_line:
                return
            print(line, file=self.stream, flush=True)
            self.last_line = line
            self.last_write = now
