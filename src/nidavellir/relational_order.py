"""Bounded negative selectivity for ordering an otherwise unchanged grammar.

Only completed existential queries contribute a rejection score. Sampling and
local timeout affect order alone: this module proves no safety, removes no
literal, and never supplies coverage masks to the compiler.
"""

from math import isfinite
import time

from .labels import AtomRecord
from .oracle import match
from .search import _grouped_order


class _OrderTimeout(Exception):
    """Local heuristic budget only; global deadline exceptions pass through."""


def order_literals(task, oracle, head, proof, positive_masks, check, *,
                   seconds=3.0, sample_size=32, max_records=4096):
    """Return ``(permutation, stats)`` using a bounded prefix of negatives.

    ``positive_masks`` use the oracle's global positive-label positions. The
    retention of a mandatory closure is an intersection of these masks (an
    upper bound when witnesses differ). Negative rejection is queried exactly
    for the entire closure. Both quantities are heuristics, never proofs.

    Complete scores follow the bitmask search's ``keep * reject + .25 *
    (1 - keep)``. An unscored literal keeps the positive-only second term.
    At most ``sample_size`` compatible negative outer symbols are constructed;
    incompatible compact records remain records. A scan cap also bounds work
    when a head matches few labels. No uniform/random population claim is made.

    The caller supplies its global ``check`` and a *local* seconds allowance.
    Existing oracle checks remain active and are restored even on exceptions.
    Local expiry returns an order; exceptions from either global check escape.
    No state in this module is shared across concurrent oracle operations.
    """
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Ordering seconds must be finite and nonnegative")
    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("Ordering sample size must be a nonnegative integer")
    if type(max_records) is not int or max_records < 0:
        raise ValueError("Ordering record limit must be a nonnegative integer")
    masks = tuple(positive_masks)
    if len(masks) != len(task.literals) or any(type(mask) is not int or mask < 0 for mask in masks):
        raise ValueError("One nonnegative integer positive mask is required per literal")
    started = time.monotonic()
    deadline = started + seconds
    original_check = oracle.check

    def local_check():
        # Global expiry must not be mistaken for an optional heuristic timeout.
        check()
        if original_check != check:
            original_check()
        if time.monotonic() >= deadline:
            raise _OrderTimeout()

    stats = {"sample_strategy": "prefix_compatible", "sample_limit": sample_size,
             "record_limit": max_records, "sample_size": 0, "scanned_records": 0,
             "negative_population": len(task.negatives),
             "compatible_negative_population": None,
             "population_known": False, "sample_exhausted": False,
             "scan_limit_reached": False, "completed_literals": 0,
             "completed_literal_indices": [], "negative_queries": 0,
             "completed_negative_queries": 0, "local_timeout": False,
             "negative_scores": [None] * len(masks),
             "positive_population": None, "positive_population_known": False,
             "positive_retention_basis": "closure_mask_intersection",
             "score_formula": "keep * reject + 0.25 * (1 - keep)",
             "fallback_formula": "0.25 * (1 - keep)", "ordering_only": True}
    closures = tuple(tuple(sorted({i, *proof.guards[i]})) for i in range(len(masks)))
    # A local timeout during head filtering still has a cheap defined fallback.
    all_positive = (1 << len(oracle.positives)) - 1
    denominator = max(1, len(oracle.positives))
    samples = []
    oracle.check = local_check
    try:
        matching = 0
        for label, position in oracle.positives.items():
            local_check()
            if match(head, label) is not None:
                matching |= 1 << position
        all_positive = matching
        denominator = max(1, matching.bit_count())
        stats.update(positive_population=matching.bit_count(), positive_population_known=True)
        iterator = (task.negatives.iter_records() if hasattr(task.negatives, "iter_records")
                    else iter(task.negatives))
        while len(samples) < sample_size and stats["scanned_records"] < max_records:
            local_check()
            try:
                record = next(iterator)
            except StopIteration:
                stats["sample_exhausted"] = True
                break
            stats["scanned_records"] += 1
            local_check()
            if match(head, record) is not None:
                samples.append(record.to_symbol() if isinstance(record, AtomRecord) else record)
        stats["sample_size"] = len(samples)
        if stats["scanned_records"] == stats["negative_population"]:
            stats["sample_exhausted"] = True
        if stats["sample_exhausted"]:
            stats.update(compatible_negative_population=len(samples), population_known=True)
        stats["scan_limit_reached"] = (stats["scanned_records"] >= max_records
                                        and not stats["sample_exhausted"])
        if samples:
            completed = {}
            for literal, closure in enumerate(closures):
                local_check()
                # Cyclic guards can make several literals share one closure.
                if closure not in completed:
                    rejected = 0
                    for label in samples:
                        local_check()
                        stats["negative_queries"] += 1
                        covered = oracle.exists(head, closure, label)
                        local_check()
                        stats["completed_negative_queries"] += 1
                        rejected += not covered
                    completed[closure] = rejected / len(samples)
                stats["negative_scores"][literal] = completed[closure]
                stats["completed_literals"] += 1
                stats["completed_literal_indices"].append(literal)
    except _OrderTimeout:
        stats["local_timeout"] = True
        stats["sample_size"] = len(samples)
    finally:
        oracle.check = original_check

    retention, scores = [], []
    for literal, closure in enumerate(closures):
        check()
        possible = all_positive
        for guard in closure:
            possible &= masks[guard]
        keep = possible.bit_count() / denominator
        retention.append(keep)
        rejection = stats["negative_scores"][literal]
        scores.append(.25 * (1 - keep) + (keep * rejection if rejection is not None else 0))
    permutation, groups = _grouped_order(proof, scores)
    check()
    stats.update(head_groups=groups, positive_retention=retention, scores=scores,
                 elapsed_seconds=time.monotonic() - started)
    return permutation, stats
