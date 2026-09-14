"""Per-signature incumbents over shared bitmask indices and global label slots."""
from math import isfinite
import time

from .oracle import ResourceLimit
from .search import _Traversal, _single_rule_cegis, candidate, pool_cover
from .target_bounds import collect_verified_target_bounds, key_text, positive_scopes
from .task import signature


def discover_target_bounds(task, indices, seconds, verify, *, verify_target=None,
                           checkpoint=None, fingerprint=None):
    """One worker/deadline, no reindexing or separately loaded per-target tasks.

    A bounded single-rule attempt uses all positives of each signature. When
    overlapping templates or several rules are necessary, a shared component
    search supplies the fallback. Only independently verified certificates are
    published; a partial set of certificates never implies global feasibility.
    """
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Bound seconds must be finite and nonnegative")
    started = time.monotonic()
    deadline = started + seconds
    report = {"status": "disabled" if seconds == 0 else "not_found", "cost": None,
              "method": "bitmask_per_signature_cegis_then_shared_components",
              "target_bounds": {}, "target_attempts": {}, "elapsed_seconds": 0,
              "safe_components": 0, "budget_scope": "one global worker deadline"}
    if seconds == 0:
        return None, report
    scopes = positive_scopes(task)
    pool, cover, traversal = {}, None, None
    full = (1 << len(task.positives)) - 1
    uncovered = full

    def check():
        if time.monotonic() >= deadline:
            raise ResourceLimit("Per-target bitmask bound budget reached")

    def emit():
        check()
        if checkpoint:
            incumbent = ([dict(row, coverage=hex(row["coverage"])) for row in cover]
                         if report["status"] == "verified" else None)
            checkpoint({"fingerprint": fingerprint, "target_bounds": report["target_bounds"],
                        "incumbent": incumbent, "strategy": "bitmask_per_target"})

    class Found(Exception):
        pass

    try:
        traversal = _Traversal(task, indices, check, lambda: deadline - time.monotonic())

        def save(index, body, mask):
            nonlocal uncovered, cover
            body, mask = traversal.simplify(index, body)
            row = candidate(task, index, body, index.coverage(mask))
            previous = pool.get(row["coverage"])
            if previous is not None and (previous["cost"], previous["rule"]) <= (row["cost"], row["rule"]):
                return
            pool[row["coverage"]] = row
            uncovered &= ~row["coverage"]
            report["safe_components"] = len(pool)
            collect_verified_target_bounds(task, pool, report["target_bounds"], traversal.analysis,
                                           check, verify_target=verify_target)
            emit()
            if not uncovered:
                # Every positive signature is covered by its verified program.
                cover = [dict(row, coverage=int(row["coverage"], 16))
                         for key in sorted(report["target_bounds"])
                         for row in report["target_bounds"][key]["cover"]]
                raise Found

        try:
            if not full:
                cover = []
            else:
                keys = [key for key in sorted(scopes) if scopes[key]]
                single_deadline = min(deadline, time.monotonic() + min(5., seconds * .25))
                class SinglePhaseFinished(Exception):
                    pass
                for position, key in enumerate(keys):
                    check()
                    remaining = single_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    target_deadline = time.monotonic() + remaining / (len(keys) - position)
                    def single_check():
                        check()
                        if time.monotonic() >= target_deadline:
                            raise SinglePhaseFinished
                    attempt = {}
                    report["target_attempts"][key_text(key)] = attempt
                    attempt_started = time.monotonic()
                    try:
                        found = _single_rule_cegis(task, [i for i in indices if signature(i.head) == key],
                                                  single_check, attempt, required_coverage=scopes[key])
                        if found is not None:
                            save(*found)
                    except SinglePhaseFinished:
                        attempt["sub_budget_reached"] = True
                    finally:
                        attempt["elapsed_seconds"] = time.monotonic() - attempt_started
                for size in range(task.max_body + 1):
                    traversal.walk(size, save, required_coverage=lambda: uncovered)
        except Found:
            pass
        if cover is not None:
            check()
            verification = verify([row["rule"] for row in cover])
            check()
            if not verification["matched"]:
                raise AssertionError("Target-bound union failed original-BK verification")
            report.update(status="verified", cost=sum(row["cost"] for row in cover),
                          rules=[row["rule"] for row in cover], verification=verification)
            emit()
            remaining = deadline - time.monotonic()
            if remaining > 0 and pool:
                exact, pool_report = pool_cover(list(pool.values()), len(task.positives), seconds=remaining)
                report["pool_selection"] = pool_report
                if exact is not None and sum(row["cost"] for row in exact) < report["cost"]:
                    verification = verify([row["rule"] for row in exact])
                    check()
                    if not verification["matched"]:
                        raise AssertionError("Optimized target-bound pool failed verification")
                    cover = exact
                    report.update(cost=sum(row["cost"] for row in cover),
                                  rules=[row["rule"] for row in cover], verification=verification)
                    emit()
    except ResourceLimit:
        if report["status"] != "verified":
            report["status"] = "timeout"
            cover = None
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if traversal is not None:
            report["search"] = traversal.stats
    return cover, report
