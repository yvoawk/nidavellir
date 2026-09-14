"""Find a verified multi-rule bound before constructing the bitmask oracle.

Each component is searched in a fresh copy of the exact ASP bias. A training
positive only orients this heuristic: the final compiler retains the full
original grammar. Positive cores forbid unsatisfiable conjunction supersets;
real negative witnesses forbid bodies contained in their complete truth vector.
No clinical predicates or manually supplied target rules enter the search.
"""
from __future__ import annotations

import hashlib
import json
from math import isfinite
import random
import time

import clingo

from .bias_analysis import _HeadSolver, analyze
from .labels import canonical_text
from .oracle import ResourceLimit, match
from .relational_coverage import CoverageOracle
from .search import greedy_cover, pool_cover
from .short_bound import seed_short_components


def _fingerprint(task):
    # CLI callers supply archived input/engine hashes. The direct-API fallback
    # also includes every label, streaming canonical records without interning
    # outer negative symbols or retaining a second list of their strings.
    data = {"version": 2, "clingo": clingo.__version__, "bk": task.background,
            "bias": task.bias_source, "max_body": task.max_body,
            "heads": list(map(str, task.heads)), "literals": list(map(str, task.literals))}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
    for name, labels in (("positive", task.positives), ("negative", task.negatives)):
        digest.update(name.encode())
        for text in canonical_text(labels):
            encoded = text.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _dump_candidate(candidate):
    return dict(candidate, coverage=hex(candidate["coverage"]))


def _load_candidate(candidate):
    return dict(candidate, coverage=int(candidate["coverage"], 16))


class ComponentSearch:
    """First-model search covering any of a set of pilot positives.

    Anonymous claims are an OR, not an AND: a positive core only invalidates
    its own pilot claim. Negative counterexamples constrain every proposal.
    The returned body may be checked against any compatible pilot, even if
    Clingo happened to choose another claim in its first model.
    """
    def __init__(self, task, head, pilot, *, positive_cores=(), negative_witnesses=(), pilot_cores=(),
                 max_body=None):
        if max_body is not None and (type(max_body) is not int or max_body < 0):
            raise ValueError("Component body cap must be a nonnegative integer or None")
        self.max_body = max_body
        self.head = head
        self.pilots = (pilot,) if isinstance(pilot, clingo.Symbol) else tuple(dict.fromkeys(pilot))
        if not self.pilots or any(match(head, label) is None for label in self.pilots):
            raise ValueError("Component pilots must be nonempty and match the head template")
        self.pilot = self.pilots[0]  # Compatibility for callers using a single pilot.
        self.diagnostics = []
        self.solver = _HeadSolver(task, head, self.diagnostics)
        self._positive_cores = {label: [] for label in self.pilots}
        with self.solver.control.backend() as backend:
            if max_body is not None:
                # Count actual selected literals, including mandatory guards.
                # A new solver per level prevents obsolete caps accumulating.
                backend.add_weight_rule([], max_body + 1,
                    [(atom, 1) for atom in self.solver.body_atoms if atom is not None])
            self.claims = {label: backend.add_atom() for label in self.pilots}
            for claim in self.claims.values():
                backend.add_rule([claim], [], choice=True)
            backend.add_rule([], [-claim for claim in self.claims.values()])
            for witness in negative_witnesses:
                truth = int(witness["truth"], 16)
                backend.add_rule([], [-atom for i, atom in enumerate(self.solver.body_atoms)
                                      if atom is not None and not truth >> i & 1])
            # Restored/bootstrap cuts are committed in the same backend batch.
            for label, core in [(self.pilot, core) for core in positive_cores] + list(pilot_cores):
                atoms = self._core_constraint(core, label)
                if atoms is not None:
                    backend.add_rule([], atoms)

    def next(self, check):
        return self.solver.find(check=check)

    def compatible_pilots(self, body):
        selected = frozenset(body)
        return tuple(label for label in self.pilots
                     if not any(core <= selected for core in self._positive_cores[label]))

    def _core_constraint(self, core, pilot):
        if pilot is None:
            if len(self.pilots) != 1:
                raise ValueError("A positive core must identify its pilot")
            pilot = self.pilot
        if pilot not in self.claims:
            raise ValueError("Positive core pilot is outside this component search")
        core = frozenset(core)
        if any(type(i) is not int or i < 0 or i >= len(self.solver.body_atoms) for i in core):
            raise ValueError("Positive core has an invalid literal index")
        if core in self._positive_cores[pilot]:
            return
        self._positive_cores[pilot].append(core)
        atoms = [self.solver.body_atoms[i] for i in sorted(core)]
        # A core containing an impossible selection is already satisfied.
        if any(atom is None for atom in atoms):
            return
        return [self.claims[pilot], *atoms]

    def positive_core(self, core, pilot=None):
        atoms = self._core_constraint(core, pilot)
        if atoms is not None:
            with self.solver.control.backend() as backend:
                backend.add_rule([], atoms)

    def negative_witness(self, truth):
        with self.solver.control.backend() as backend:
            backend.add_rule([], [-atom for i, atom in enumerate(self.solver.body_atoms)
                                  if atom is not None and not truth >> i & 1])


def discover_relational_bound(task, seconds, verify, *, state=None, checkpoint=None,
                              progress=None, fingerprint=None, seed=42, initial_positives=24,
                              short_phase_seconds=None, per_target_bounds=False):
    """Return (verified cover or None, report); timeouts never prove infeasibility.

    ``seconds`` is the single caller budget, including analysis, joins and
    validation. Expensive external validation should additionally have a hard
    process deadline in the CLI. Checkpoints contain cuts/components and an
    optional independently verified incumbent, not Clingo's internal state.
    Resumption rejects a changed input/engine fingerprint and rechecks saved
    components. initial_positives samples up to 24 uncovered labels by default;
    only one of them must have a real witness. The value 1 keeps single-pilot
    search available as an ablation. All costs returned are upper bounds, not
    global optimum proofs.

    A bounded first pass tries legal bodies of at most two literals. Its
    default budget is min(5 seconds, one quarter of the remaining search
    budget); short_phase_seconds=0 disables this heuristic for ablation.
    Thereafter CEGIS increases a body-length cap from zero to max_body for
    each component. Only UNSAT for every pilot-compatible head completes a
    level. These levels concern the sampled pilots, not global rule-set cost;
    resumption replays saved cuts and starts again at cap zero.
    """
    if type(initial_positives) is not int or initial_positives < 1:
        raise ValueError("Initial positive sample size must be a positive integer")
    if type(per_target_bounds) is not bool:
        raise ValueError("per_target_bounds must be a boolean")
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Bound seconds must be finite and nonnegative")
    if short_phase_seconds is not None and (not isfinite(short_phase_seconds) or short_phase_seconds < 0):
        raise ValueError("Short phase seconds must be finite and nonnegative")
    started = time.monotonic()
    deadline = started + seconds
    report = {"method": "relational_cegis_before_bitmask_oracle", "status": "disabled" if seconds == 0 else "not_found",
              "cost": None, "elapsed_seconds": 0.0, "examined": 0, "safe_components": 0,
              "positive_cores": 0, "negative_witnesses": 0, "resumed": state is not None,
              "initial_positives": initial_positives, "pilot_strategy": "any_sampled_uncovered",
              "unary_literals_checked": 0, "unary_cores_added": 0, "unary_bootstrap_seconds": 0.0,
              "search_order": "progressive_body_caps", "capped_searches": 0,
              "completed_body_levels": []}
    if per_target_bounds:
        report["target_bounds"] = {}
    if seconds == 0:
        return None, report
    def check():
        if time.monotonic() >= deadline:
            raise ResourceLimit("Relational bound search time budget reached")
    fingerprint = fingerprint or _fingerprint(task)
    if state is not None and (state.get("version") != 1 or state.get("fingerprint") != fingerprint):
        raise ValueError("Bound checkpoint fingerprint does not match current inputs/engine")
    work = json.loads(json.dumps(state)) if state is not None else {
        "version": 1, "fingerprint": fingerprint, "components": [], "positive_cores": [],
        "negative_witnesses": [], "rounds": 0, "incumbent": None}
    report["state"] = work
    if per_target_bounds:
        # Resumed certificates are not trusted until current components have
        # been reconstructed and independently verified again.
        work["target_bounds"] = {}
        report["target_bounds"] = work["target_bounds"]
    cover = None
    oracle = None
    pool = {}
    restoration_complete = not work["components"]
    def emit():
        # A deadline during checkpoint revalidation must not erase the saved
        # components that have not yet been revisited in this process.
        preserved = {} if restoration_complete else {c["coverage"]: c for c in work["components"]}
        preserved.update({hex(c["coverage"]): _dump_candidate(c) for c in pool.values()})
        work["components"] = sorted(preserved.values(), key=lambda c: (c["cost"], c["rule"]))
        report.update(elapsed_seconds=time.monotonic() - started, safe_components=len(pool),
                      positive_cores=len(work["positive_cores"]), negative_witnesses=len(work["negative_witnesses"]))
        if checkpoint:
            checkpoint(work)
        if progress:
            progress({key: value for key, value in report.items() if key != "state"})
    def updated_components(_candidate=None):
        if per_target_bounds:
            from .target_bounds import collect_verified_target_bounds
            collect_verified_target_bounds(task, pool, work["target_bounds"], analysis, check)
        emit()
    try:
        check()
        analysis = getattr(task, "_bias_analysis", None)
        if analysis is None:
            analysis = analyze(task, seconds=max(0, deadline - time.monotonic()), check=check)
            task._bias_analysis = analysis
        if work.get("incumbent") is not None:
            resumed_cover = [_load_candidate(c) for c in work["incumbent"]]
            for c in resumed_cover:
                head, body = clingo.parse_term(c["head"]), tuple(c["body"])
                if (head not in analysis.heads or not analysis.heads[head].accepts(body, check=check)
                        or task.rule(head, body) != c["rule"] or len(body) + 1 != c["cost"]):
                    raise ValueError("Checkpoint incumbent is not a legal reconstructed hypothesis")
            check()
            verification = verify([c["rule"] for c in resumed_cover])
            if not verification["matched"]:
                raise ValueError("Checkpoint incumbent failed independent verification")
            cover = resumed_cover
            report.update(status="verified", cost=sum(c["cost"] for c in cover),
                          rules=[c["rule"] for c in cover], verification=verification,
                          first_verified_seconds=time.monotonic() - started)
        oracle = CoverageOracle(task, check)
        positives = sorted(task.positives)
        full = (1 << len(positives)) - 1
        covered = 0
        for saved in work["components"]:
            check()
            c = _load_candidate(saved)
            head = clingo.parse_term(c["head"])
            body = tuple(c["body"])
            if head not in analysis.heads or not analysis.heads[head].accepts(body, check=check):
                raise ValueError("Checkpoint component no longer satisfies the actual bias")
            if oracle.negative(head, body) is not None:
                raise ValueError("Checkpoint component covers a current negative")
            coverage = oracle.coverage(head, body)
            if coverage != c["coverage"] or task.rule(head, body) != c["rule"] or len(body) + 1 != c["cost"]:
                raise ValueError("Checkpoint component metadata does not match independent reconstruction")
            pool[coverage] = c
            covered |= coverage
        restoration_complete = True
        if per_target_bounds:
            updated_components()
        remaining = max(0, deadline - time.monotonic())
        short_budget = min(5, remaining / 4) if short_phase_seconds is None else min(short_phase_seconds, remaining)
        if covered != full and short_budget:
            report['short_phase'] = seed_short_components(
                task, oracle, analysis, pool, seconds=short_budget,
                check=check, on_component=updated_components)
            covered = 0
            for candidate in pool.values():
                covered |= candidate['coverage']
            # Reuse only actual negative witnesses retained by the oracle.
            # The finite cache is not claimed to exhaust all counterexamples.
            for head, witnesses in oracle._negative_witnesses.items():
                for witness in witnesses:
                    entry = {'template': str(head), 'head': str(witness['head']),
                             'binding': {str(slot): str(value) for slot, value in witness['binding'].items()},
                             'truth': hex(witness['truth'])}
                    if not any(old['template'] == entry['template'] and old['truth'] == entry['truth']
                               for old in work['negative_witnesses']):
                        work['negative_witnesses'].append(entry)
            emit()
        while covered != full:
            check()
            uncovered = [i for i in range(len(positives)) if not covered >> i & 1]
            # Match experiment 11: a component-local seed is reproducible
            # after restoring the same safe pool from a checkpoint.
            rng = random.Random(seed + len(pool))
            sampled = tuple(positives[i] for i in rng.sample(uncovered, min(initial_positives, len(uncovered))))
            heads = [(head, tuple(pilot for pilot in sampled if match(head, pilot) is not None))
                     for head in task.heads if analysis.heads[head].legal_body_exists]
            for head, pilots in heads:
                if not pilots:
                    continue
                # Unary impossibilities remain scoped to their pilot. A body
                # may cover another sampled positive even if this one fails.
                bootstrap_started = time.monotonic()
                completed_bootstraps = work.setdefault("unary_preprocessed", [])
                for pilot in pilots:
                    identity = {"head": str(head), "pilot": str(pilot)}
                    if identity not in completed_bootstraps:
                        known = {tuple(entry["body"]) for entry in work["positive_cores"]
                                 if entry["head"] == str(head) and entry["pilot"] == str(pilot)}
                        with oracle.profile.measure("bound.unary_bootstrap"):
                            for literal in range(len(task.literals)):
                                check()
                                if literal in analysis.heads[head].unusable_literals or (literal,) in known:
                                    continue
                                report["unary_literals_checked"] += 1
                                if not oracle.exists(head, (literal,), pilot):
                                    work["positive_cores"].append(dict(identity, body=[literal]))
                                    report["unary_cores_added"] += 1
                        completed_bootstraps.append(identity)
                report["unary_bootstrap_seconds"] += time.monotonic() - bootstrap_started
            component = None
            # A new pilot set (including after resume) starts at zero. Bootstrap
            # work and all exact cuts are shared across the successive caps.
            for level in range(task.max_body + 1):
                check()
                report["current_body_level"] = level
                searches = []
                for head, pilots in heads:
                    if not pilots:
                        continue
                    check()
                    by_text = {str(pilot): pilot for pilot in pilots}
                    cores = [(by_text[entry["pilot"]], entry["body"]) for entry in work["positive_cores"]
                             if entry["head"] == str(head) and entry["pilot"] in by_text]
                    negatives = [entry for entry in work["negative_witnesses"] if entry["template"] == str(head)]
                    searches.append(ComponentSearch(task, head, pilots, pilot_cores=cores, negative_witnesses=negatives, max_body=level))
                    check()
                    report["capped_searches"] += 1
                while searches and component is None:
                    for search in list(searches):
                        check()
                        body = search.next(check)
                        report["examined"] += 1
                        work["rounds"] += 1
                        if body is None:
                            searches.remove(search)
                            continue
                        body = tuple(sorted(body))
                        valid_pilot = False
                        compatible = search.compatible_pilots(body)
                        if not compatible:
                            raise AssertionError("A proposal must permit at least one sampled positive claim")
                        for pilot in compatible:
                            check()
                            if oracle.exists(search.head, body, pilot):
                                valid_pilot = True
                                break
                            core = oracle.core(search.head, body, pilot)
                            if core is None:
                                raise AssertionError("A failed positive query must produce an impossible core")
                            search.positive_core(core, pilot)
                            entry = {"head": str(search.head), "pilot": str(pilot), "body": list(core)}
                            if entry not in work["positive_cores"]:
                                work["positive_cores"].append(entry)
                        if valid_pilot:
                            negative = oracle.negative(search.head, body)
                            if negative is not None:
                                search.negative_witness(negative["truth"])
                                entry = {"template": str(search.head), "head": str(negative["head"]),
                                         "binding": {str(slot): str(value) for slot, value in negative["binding"].items()},
                                         "truth": hex(negative["truth"])}
                                if not any(old["template"] == entry["template"] and old["truth"] == entry["truth"]
                                           for old in work["negative_witnesses"]):
                                    work["negative_witnesses"].append(entry)
                            else:
                                proof = analysis.heads[search.head]
                                for literal in tuple(reversed(body)):
                                    check()
                                    smaller = tuple(i for i in body if i != literal)
                                    if proof.accepts(smaller, check=check) and oracle.negative(search.head, smaller) is None:
                                        body = smaller
                                coverage = oracle.coverage(search.head, body)
                                if not coverage & ~covered:
                                    raise AssertionError("A safe pilot-covering component must cover a new positive")
                                component = {"head": str(search.head), "body": list(body), "cost": len(body) + 1,
                                             "coverage": coverage, "rule": task.rule(search.head, body)}
                                break
                        if work["rounds"] % 25 == 0:
                            emit()
                if component is not None:
                    break
                # Every search was removed only following explicit UNSAT.
                # An exception/timeout exits before publishing this level.
                report["completed_body_levels"].append({
                    "component": len(pool), "max_body_literals": level,
                    "head_templates": sum(bool(pilots) for _, pilots in heads),
                    "sampled_positives": len(sampled)})
                emit()
            if component is None:
                report["reason"] = "No safe component found for any sampled remaining positive after complete local refinement"
                break
            old = pool.get(component["coverage"])
            if old is None or (component["cost"], component["rule"]) < (old["cost"], old["rule"]):
                pool[component["coverage"]] = component
            covered |= component["coverage"]
            updated_components()
        if covered == full:
            proposed_cover = greedy_cover(list(pool.values()), len(positives))
            check()
            verification = verify([c["rule"] for c in proposed_cover])
            if not verification["matched"]:
                raise AssertionError("Relational bound failed independent full-program verification")
            proposed_cost = sum(c["cost"] for c in proposed_cover)
            if cover is None or proposed_cost < report["cost"]:
                cover = proposed_cover
                report.update(status="verified", cost=proposed_cost,
                              rules=[c["rule"] for c in cover], verification=verification)
                report.setdefault("first_verified_seconds", time.monotonic() - started)
                work["incumbent"] = [_dump_candidate(c) for c in cover]
            emit()
            remaining = deadline - time.monotonic()
            if remaining > 0 and pool:
                exact, result = pool_cover(list(pool.values()), len(positives), seconds=remaining)
                report["pool_selection"] = result
                if exact is not None and sum(c["cost"] for c in exact) < report["cost"]:
                    check()
                    verification = verify([c["rule"] for c in exact])
                    if not verification["matched"]:
                        raise AssertionError("Pool optimum failed independent full-program verification")
                    cover = exact
                    report.update(cost=sum(c["cost"] for c in cover), rules=[c["rule"] for c in cover], verification=verification)
                    work["incumbent"] = [_dump_candidate(c) for c in cover]
    except ResourceLimit:
        if report["status"] != "verified":
            report["status"] = "timeout"
            cover = None
    finally:
        if oracle is not None:
            report["oracle"] = oracle.stats
            report["profile"] = oracle.profile.export()
        emit()
    return cover, report
