"""Exact compilation with proved bias closure and task-relative bitmask pruning.

All pruning preserves minimum-cost covers and equal-cost syntactic variants.
Semantic implications alone never justify deleting a structurally useful
literal. Negative-covering and structurally incomplete prefixes remain open.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import isfinite
import time

import clingo

from .bias_analysis import PrefixFilter, _HeadSolver, analyze
from .alpha_collection import VariantRecords, VariantBucket
from .oracle import ResourceLimit
from .task import variables, signature


def _members(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def candidate(task, index, body, coverage):
    body = tuple(sorted(body))
    return {"head": str(index.head), "body": list(body), "cost": len(body) + 1,
            "coverage": coverage, "rule": task.rule(index.head, body)}


def greedy_cover(candidates, positive_count):
    remaining = (1 << positive_count) - 1
    selected = []
    while remaining:
        useful = [c for c in candidates if c["coverage"] & remaining]
        if not useful:
            return None
        chosen = min(useful, key=lambda c: (Fraction(c["cost"], (c["coverage"] & remaining).bit_count()),
                                            c["cost"], c["rule"]))
        selected.append(chosen)
        remaining &= ~chosen["coverage"]
    for c in list(reversed(selected)):
        covered = 0
        for other in selected:
            if other is not c:
                covered |= other["coverage"]
        if covered == (1 << positive_count) - 1:
            selected.remove(c)
    return selected


def pool_cover(candidates, positive_count, *, seconds=30):
    """Optimize the discovered finite pool only; this is not a global proof."""
    from .solvers import solve
    ordered = sorted(candidates, key=lambda c: (c["cost"], c["rule"]))
    numbered = [dict(c, id=f"p{i}") for i, c in enumerate(ordered)]
    result = solve(numbered, positive_count, timeout=seconds)
    selected = set(result["selected"])
    cover = [c for c in numbered if c["id"] in selected] if result["cost"] is not None else None
    return cover, {"status": result["status"], "cost": result["cost"],
                   "optimum_cost": result["optimum_cost"], "elapsed_seconds": result["elapsed_seconds"],
                   "scope": "discovered_components_only"}


@dataclass
class Preprocessing:
    dead: int
    incompatible: tuple[int, ...]
    implies: tuple[int, ...]
    equivalents: tuple[tuple[int, int], ...]

    @classmethod
    def discover(cls, index, check):
        masks = index.literal_masks
        incompatible, implies = [0] * len(masks), [0] * len(masks)
        equivalents = []
        for a, left in enumerate(masks):
            check()
            for b in range(a + 1, len(masks)):
                right = masks[b]
                common = left & right
                if not common:
                    incompatible[a] |= 1 << b
                    incompatible[b] |= 1 << a
                if common == left:
                    implies[a] |= 1 << b
                if common == right:
                    implies[b] |= 1 << a
                if left == right:
                    equivalents.append((a, b))
        return cls(sum(1 << i for i, mask in enumerate(masks) if not mask),
                   tuple(incompatible), tuple(implies), tuple(equivalents))


def _analysis(task, check, seconds):
    result = getattr(task, "_bias_analysis", None)
    if result is None:
        result = analyze(task, seconds=max(0, seconds), check=check)
        task._bias_analysis = result
    return result


def _grouped_order(proof, scores):
    """Order proved at-most-one groups, placing dependencies first if possible.

    Overlapping cardinalities are partitioned conservatively; the complete
    constraint filter remains authoritative. Guard cycles fall back to closure
    enumeration, so this ordering imposes no extra grammar restriction.
    """
    groups, used = [], 0
    structural = proof.prefix_filter
    if structural is not None:
        for bits in sorted({p for p, n, lo, hi in structural.bounds if not n and hi == 1},
                           key=lambda bits: (-bits.bit_count(), bits)):
            members = tuple(_members(bits & ~used))
            if members:
                groups.append(members)
                used |= sum(1 << i for i in members)
    groups.extend((i,) for i in range(len(scores)) if not used >> i & 1)
    positions = {literal: group for group, members in enumerate(groups) for literal in members}
    dependencies = [{positions[g] for i in members for g in proof.guards[i]} - {group}
                    for group, members in enumerate(groups)]
    priorities = [max(scores[i] for i in members) for members in groups]
    remaining, done, result = set(range(len(groups))), set(), []
    while remaining:
        ready = [group for group in remaining if dependencies[group] <= done]
        chosen = min(ready or remaining, key=lambda group: (-priorities[group], min(groups[group])))
        result.extend(sorted(groups[chosen], key=lambda i: (-scores[i], i)))
        remaining.remove(chosen)
        done.add(chosen)
    return tuple(result), len(groups)


def _negative_head_masks(index, check):
    """Recover head-level scoring masks from optional witness diagnostics."""
    if not index.classes:
        return None
    rows = getattr(index.classes, "rows", None)
    labels = (row[0] for row in rows) if rows is not None else (row["head"] for row in index.classes)
    masks, previous, start, size = {}, None, 0, 0
    def finish(stop):
        if previous is not None:
            mask = ((1 << (stop - start)) - 1) << start
            if mask & index.negative_mask:
                masks[previous] = masks.get(previous, 0) | (mask & index.negative_mask)
    for position, label in enumerate(labels):
        if position % 4096 == 0:
            check()
        if label != previous:
            finish(position)
            previous, start = label, position
        size = position + 1
    finish(size)
    return tuple(masks.values())


@dataclass(frozen=True)
class _TargetPlan:
    """Necessary static restrictions for one exact cost/coverage signature.

    Groups form a disjoint partition into proved at-most-one sets. Unproved
    literals remain singletons. A suffix counts groups, not closure-addition
    steps: one step can add several guards and must not be counted as one
    remaining literal. Ignoring other ASP constraints only overestimates the
    remaining capacity, so the exact legality checker stays authoritative.
    """
    cost: int
    coverage: int
    required_heads: tuple[int, ...]
    eligible: int
    order: tuple[int, ...]
    prefix: tuple[int, ...]
    literal_groups: tuple[int, ...]
    suffix_groups: tuple[int, ...]

    @property
    def length(self):
        return self.cost - 1

    def remaining_capacity(self, start, selected):
        occupied = 0
        for literal in _members(selected):
            occupied |= self.literal_groups[literal]
        return (self.suffix_groups[start] & ~occupied).bit_count()

    @classmethod
    def build(cls, index, proof, closures, unavailable, order, cost, coverage, check):
        """No removed literal can belong to a legal rule for this signature."""
        check()
        length = cost - 1
        known_heads = sum(1 << i for i, _ in index.positive_heads)
        if length < 0 or length > proof._checker.task.max_body or coverage <= 0 or coverage & ~known_heads:
            return None
        required_heads = tuple(sorted((mask for i, mask in index.positive_heads if coverage >> i & 1),
                                      key=int.bit_count))
        eligible = 0
        for literal, mask in enumerate(index.literal_masks):
            check()
            if not unavailable >> literal & 1 and all(mask & head for head in required_heads):
                eligible |= 1 << literal
        # A rule containing L must also contain its complete proved closure.
        # This also handles cyclic guards without assuming a topological order.
        previous = None
        while eligible != previous:
            check()
            previous = eligible
            for literal in _members(previous):
                if closures[literal] & ~eligible or closures[literal].bit_count() > length:
                    eligible &= ~(1 << literal)
        order = tuple(literal for literal in order if eligible >> literal & 1)
        prefix = [0]
        for literal in order:
            prefix.append(prefix[-1] | (1 << literal))
        groups, used = [], 0
        structural = proof.prefix_filter
        if structural is not None:
            for members in sorted({positive for positive, negative, _, upper in structural.bounds
                                   if not negative and upper == 1},
                                  key=lambda members: (-members.bit_count(), members)):
                check()
                members &= eligible & ~used
                if members:
                    groups.append(members)
                    used |= members
        groups.extend(1 << literal for literal in _members(eligible & ~used))
        literal_groups = [0] * len(index.literal_masks)
        for group, members in enumerate(groups):
            for literal in _members(members):
                literal_groups[literal] = 1 << group
        suffix_groups = [0] * (len(order) + 1)
        for position in range(len(order) - 1, -1, -1):
            suffix_groups[position] = suffix_groups[position + 1] | literal_groups[order[position]]
        return cls(cost, coverage, required_heads, eligible, order, tuple(prefix),
                   tuple(literal_groups), tuple(suffix_groups))


class _Traversal:
    def __init__(self, task, indices, check, remaining_seconds, *, order="selectivity", preprocess=True,
                 legal_pruning=True):
        if order not in {"source", "selectivity"}:
            raise ValueError("order must be source or selectivity")
        self.task, self.indices, self.check = task, indices, check
        self.analysis = _analysis(task, check, remaining_seconds())
        self.legal_pruning = legal_pruning
        self.stats = {"nodes": 0, "pruned_no_positive": 0, "pruned_safe_extensions": 0,
                      "pruned_guard_conflict": 0, "pruned_structural_completion": 0,
                      "pruned_incompatible": 0, "pruned_semantic_noop": 0,
                      "pruned_cheaper_cover": 0, "pruned_target": 0,
                      "pruned_unsafe_at_limit": 0,
                      "target_plans": 0, "target_static_rejected_literals": 0,
                      "pruned_target_capacity": 0,
                      "legal_safe_rules": 0, "semantic_implications": 0,
                      "semantic_equivalences": 0, "semantic_incompatibilities": 0,
                      "dead_literals": 0, "head_orders": {}, "order": order,
                      "ground_prefix_checks": 0, "ground_prefix_constraints": 0,
                      "ground_prefix_fallback_heads": 0,
                      "deletion_monotone_literals": 0,
                      "preprocessing": bool(preprocess), "complete": False}
        self.contexts = {}
        self.coverage_cache = {}
        self.coverage_cache_limit = max(16, 4096 // max(1, len(indices)))
        self.stats["coverage_cache_hits"] = 0
        for index in indices:
            check()
            proof = self.analysis.heads[index.head]
            structural = proof.prefix_filter
            if structural is not None:
                self.stats["ground_prefix_constraints"] += structural.compiled_constraints
                self.stats["ground_prefix_fallback_heads"] += int(not structural.all_constraints_compiled)
                self.stats["deletion_monotone_literals"] += structural.deletion_monotone.bit_count()
            closures = tuple((1 << i) | sum(1 << g for g in proof.guards[i])
                             for i in range(len(task.literals)))
            prep = Preprocessing.discover(index, check) if preprocess else None
            if prep:
                self.stats["semantic_implications"] += sum(m.bit_count() for m in prep.implies)
                self.stats["semantic_equivalences"] += len(prep.equivalents)
                self.stats["semantic_incompatibilities"] += sum(m.bit_count() for m in prep.incompatible) // 2
                self.stats["dead_literals"] += prep.dead.bit_count()
            excluded = sum(1 << i for i in proof.unusable_literals)
            if prep:
                excluded |= prep.dead
            scores = []
            negative_heads = _negative_head_masks(index, check) if order == "selectivity" else None
            for i, mask in enumerate(index.literal_masks):
                keep = index.coverage(mask).bit_count() / max(1, len(index.positive_heads))
                reject = (sum(not mask & head for head in negative_heads) / max(1, len(negative_heads))
                          if negative_heads is not None else
                          1 - (mask & index.negative_mask).bit_count() / max(1, index.negative_mask.bit_count()))
                scores.append(keep * reject + .25 * (1 - keep))
            self.stats.setdefault("selectivity_basis", {})[str(index.head)] = (
                "labelled_heads" if negative_heads is not None else "witness_fallback")
            if order == "selectivity":
                permutation, group_count = _grouped_order(proof, scores)
                self.stats.setdefault("head_choice_groups", {})[str(index.head)] = group_count
            else:
                permutation = tuple(range(len(task.literals)))
            prefix = [0]
            for i in permutation:
                prefix.append(prefix[-1] | (1 << i))
            conflicts = []
            for closure in closures:
                combined = 0
                for i in _members(closure):
                    if prep:
                        combined |= prep.incompatible[i]
                    if structural:
                        combined |= structural.incompatible[i]
                conflicts.append(combined)
            residual = structural
            if structural is not None:
                # Pair conflicts and the traversal's body cap already enforce
                # these bounds. Keep every other necessary constraint.
                all_literals = (1 << len(task.literals)) - 1
                residual = PrefixFilter(
                    tuple((p, n) for p, n in structural.nogoods if n or p.bit_count() != 2),
                    tuple((p, n, lo, hi) for p, n, lo, hi in structural.bounds
                          if not (not n and lo is None and
                                  (hi == 1 or (p == all_literals and hi >= task.max_body)))),
                    structural.incompatible, structural.all_constraints_compiled,
                    structural.compiled_constraints, structural.skipped_constraints,
                    structural.deletion_monotone, structural.deletion_checks)
            self.contexts[index.head] = proof, closures, prep, excluded, permutation, prefix, tuple(conflicts), residual
            self.coverage_cache[index.head] = {}
            self.stats["head_orders"][str(index.head)] = list(permutation)

    def simplify(self, index, body):
        """Deletion is permitted only with a new legality and safety proof."""
        proof = self.contexts[index.head][0]
        body = tuple(sorted(body))
        for literal in tuple(reversed(body)):
            self.check()
            smaller = tuple(i for i in body if i != literal)
            mask = index.body_mask(smaller)
            if (mask & index.positive_mask and not mask & index.negative_mask
                    and proof.accepts(smaller, check=self.check)):
                body = smaller
        return body, index.body_mask(body)

    def coverage(self, index, mask):
        # Negative witnesses cannot affect the positive coverage signature.
        key = mask & index.positive_mask
        cache = self.coverage_cache[index.head]
        result = cache.get(key)
        if result is not None:
            self.stats["coverage_cache_hits"] += 1
            return result
        result = index.coverage(key)
        if len(cache) >= self.coverage_cache_limit:
            cache.clear()
        cache[key] = result
        return result

    def walk(self, maximum, save, *, progress=None, dominance=None, targets=None, required_coverage=None,
             require_all=False, program_scopes=None, track_targets=False):
        """Enumerate closed legal safe bodies once, with arbitrary literal order.

        Earlier omitted literals are excluded. Adding a literal adds its whole
        mandatory closure; conflicting earlier exclusions reject that branch.
        This is a canonical enumeration of closed sets, including guard cycles.
        Targets must be minimum-cost, nondominated catalogue signatures: the
        safe-prefix/no-op stops intentionally omit more expensive refinements.
        """
        def walk_context(index, context, target_plan=None):
            proof, closures, prep, unavailable, order, prefix, conflicts, structural = context
            if not proof.legal_body_exists:
                return
            body_limit = min(maximum, target_plan.length) if target_plan is not None else maximum
            scope = (program_scopes or {}).get(signature(index.head))
            if scope is not None:
                bound, minimum_cost, positive_scope = scope
                body_limit = min(body_limit, max(0, bound - 1))
            local_goals = targets
            wanted_previous, wanted_mask = None, 0
            def visit(body_bits, start, mask):
                nonlocal wanted_previous, wanted_mask
                self.stats["nodes"] += 1
                self.check()
                if not mask & index.positive_mask:
                    self.stats["pruned_no_positive"] += 1
                    return
                size = body_bits.bit_count()
                if size >= body_limit and mask & index.negative_mask:
                    self.stats["pruned_unsafe_at_limit"] += 1
                    return
                if target_plan is not None:
                    if size + target_plan.remaining_capacity(start, body_bits) < target_plan.length:
                        self.stats["pruned_target_capacity"] += 1
                        return
                    if any(not mask & head for head in target_plan.required_heads):
                        self.stats["pruned_target"] += 1
                        return
                coverage = None
                if scope is not None and size + 1 + minimum_cost > bound:
                    coverage = self.coverage(index, mask)
                    if coverage != positive_scope:
                        self.stats["pruned_target_program_bound"] += 1
                        return
                if required_coverage is not None:
                    wanted = required_coverage()
                    if require_all:
                        covered = 0
                        for i, witnesses in index.positive_heads:
                            if wanted >> i & 1:
                                if not witnesses & mask:
                                    self.stats["pruned_target"] += 1
                                    return
                                covered |= 1 << i
                        if covered != wanted:
                            self.stats["pruned_target"] += 1
                            return
                    else:
                        if wanted != wanted_previous:
                            wanted_mask = 0
                            for i, witnesses in index.positive_heads:
                                if wanted >> i & 1:
                                    wanted_mask |= witnesses
                            wanted_previous = wanted
                        if not mask & wanted_mask:
                            self.stats["pruned_target"] += 1
                            return
                useful_dominance = dominance is not None and size + 1 > getattr(dominance, "minimum_cost", 0)
                if (targets is not None and target_plan is None) or useful_dominance:
                    coverage = self.coverage(index, mask)
                if target_plan is None and local_goals is not None and not any(cost >= size + 1 and wanted & coverage == wanted
                                                       for cost, wanted in local_goals):
                    self.stats["pruned_target"] += 1
                    return
                if useful_dominance and dominance(size + 1, coverage):
                    self.stats["pruned_cheaper_cover"] += 1
                    return
                body = tuple(_members(body_bits))
                forbidden = (prefix[start] & ~body_bits) | unavailable
                if self.legal_pruning:
                    if structural is not None:
                        self.stats["ground_prefix_checks"] += 1
                    if ((structural is not None and not structural.allows(body_bits, forbidden)) or
                        ((structural is None or not structural.all_constraints_compiled) and
                         not proof.possible_completion(body, tuple(_members(forbidden)), check=self.check))):
                        self.stats["pruned_structural_completion"] += 1
                        return
                if not mask & index.negative_mask and proof.accepts(body, check=self.check):
                    self.stats["legal_safe_rules"] += 1
                    if target_plan is None or (size == target_plan.length and
                                               self.coverage(index, mask) == target_plan.coverage):
                        save(index, body, mask)
                    self.stats["pruned_safe_extensions"] += 1
                    return
                if size >= body_limit:
                    return
                for position in range(start, len(order)):
                    literal = order[position]
                    if body_bits >> literal & 1 or unavailable >> literal & 1:
                        continue
                    closure = closures[literal]
                    if closure & ((prefix[position] & ~body_bits) | unavailable):
                        self.stats["pruned_guard_conflict"] += 1
                        continue
                    added = closure & ~body_bits
                    if size + added.bit_count() > body_limit:
                        continue
                    selected = body_bits | added
                    if conflicts[literal] & selected:
                        self.stats["pruned_incompatible"] += 1
                        continue
                    new_mask = mask
                    for i in _members(added):
                        new_mask &= index.literal_masks[i]
                    # Only a proved deletion-monotone literal may be skipped
                    # as a semantic no-op: its deletion preserves the legality
                    # of every future extension, including arbitrary guards.
                    # Other equal masks retain their potentially useful role.
                    if structural is not None and structural.deletion_checks:
                        redundant = False
                        current_forbidden = (prefix[position + 1] & ~selected) | unavailable
                        if prep:
                            implied_selected = 0
                            for i in _members(selected):
                                implied_selected |= prep.implies[i] & selected & ~(1 << i)
                            for i in _members(implied_selected):
                                if structural.can_delete(i, selected, current_forbidden):
                                    redundant = True
                                    break
                        for i in _members(added):
                            if redundant:
                                break
                            without = mask
                            for other in _members(added & ~(1 << i)):
                                without &= index.literal_masks[other]
                            if without == new_mask and structural.can_delete(i, selected, current_forbidden):
                                redundant = True
                                break
                        if redundant:
                            self.stats["pruned_semantic_noop"] += 1
                            continue
                    visit(selected, position + 1, new_mask)
                if progress and self.stats["nodes"] % 1000 == 0:
                    progress(dict(self.stats))
            visit(0, 0, index.all_mask)
        remaining = defaultdict(int)
        if track_targets:
            for index in self.indices:
                remaining[signature(index.head)] += 1
        for index in self.indices:
            context = self.contexts[index.head]
            if targets is None:
                key = signature(index.head)
                if track_targets:
                    item = self.stats['target_search'][f'{key[0]}/{key[1]}']
                    item['started'] = True
                walk_context(index, context)
                if track_targets:
                    remaining[key] -= 1
                    item['search_complete'] = remaining[key] == 0
                continue
            proof, closures, prep, unavailable, order, prefix, conflicts, structural = context
            if not proof.legal_body_exists:
                continue
            for cost, wanted in sorted(targets):
                self.check()
                if cost - 1 > maximum:
                    continue
                plan = _TargetPlan.build(index, proof, closures, unavailable, order, cost, wanted, self.check)
                if plan is None:
                    self.stats["pruned_target"] += 1
                    continue
                self.stats["target_plans"] += 1
                self.stats["target_static_rejected_literals"] += len(self.task.literals) - plan.eligible.bit_count()
                all_literals = (1 << len(self.task.literals)) - 1
                target_context = (proof, closures, prep, unavailable | (all_literals & ~plan.eligible),
                                  plan.order, plan.prefix, conflicts, structural)
                walk_context(index, target_context, plan)


def _single_rule_cegis(task, indices, check, report, *, required_coverage=None):
    """Bounded full-cover heuristic using exact bitmask counterexamples.

    Positive cores forbid conjunction supersets; each negative witness forbids
    bodies contained in its effective truth vector. These cuts are scoped to
    this heuristic and never change the grammar used by complete compilation.
    Progressive caps find a short incumbent before the multi-rule fallback.
    """
    full = ((1 << len(task.positives)) - 1 if required_coverage is None else required_coverage)
    report.update(single_rule_cegis_models=0, single_rule_positive_cores=0,
                  single_rule_negative_witnesses=0)
    for index in indices:
        if sum(1 << i for i, _ in index.positive_heads) != full:
            continue
        positives = [mask for _, mask in index.positive_heads]
        cores = [(i,) for i, mask in enumerate(index.literal_masks)
                 if any(not mask & head for head in positives)]
        negatives = []
        for maximum in range(task.max_body + 1):
            check()
            solver = _HeadSolver(task, index.head, [])
            def add_core(core):
                with solver.control.backend() as backend:
                    backend.add_rule([], [solver.body_atoms[i] for i in core])
            def add_negative(truth):
                with solver.control.backend() as backend:
                    backend.add_rule([], [-atom for i, atom in enumerate(solver.body_atoms)
                                          if atom is not None and not truth >> i & 1])
            with solver.control.backend() as backend:
                backend.add_weight_rule([], maximum + 1, [(a, 1) for a in solver.body_atoms if a is not None])
            for core in cores:
                # A missing atom is already impossible to select.
                if all(solver.body_atoms[i] is not None for i in core):
                    add_core(core)
            for truth in negatives:
                add_negative(truth)
            while True:
                check()
                selected = solver.find(check=check)
                if selected is None:
                    break
                report["single_rule_cegis_models"] += 1
                body = tuple(sorted(selected))
                mask = index.body_mask(body)
                missing = next((head for head in positives if not head & mask), None)
                if missing is not None:
                    core = body
                    for i in body:
                        smaller = tuple(j for j in core if j != i)
                        if not index.body_mask(smaller) & missing:
                            core = smaller
                    cores.append(core)
                    add_core(core)
                    report["single_rule_positive_cores"] += 1
                elif mask & index.negative_mask:
                    witness = (mask & index.negative_mask) & -(mask & index.negative_mask)
                    truth = sum(1 << i for i, literal_mask in enumerate(index.literal_masks) if literal_mask & witness)
                    negatives.append(truth)
                    add_negative(truth)
                    report["single_rule_negative_witnesses"] += 1
                else:
                    return index, body, mask
    return None


def discover_bound(task, indices, seconds, verify):
    """First verified multi-rule incumbent, then exact selection within its pool."""
    started = time.monotonic()
    report = {"status": "disabled" if seconds == 0 else "not_found", "cost": None,
              "elapsed_seconds": 0, "examined": 0, "safe_components": 0,
              "method": "bitmask_cegis_full_cover_then_components"}
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Bound seconds must be finite and nonnegative")
    if seconds == 0:
        return None, report
    deadline = started + seconds
    def check():
        if time.monotonic() >= deadline:
            raise ResourceLimit("Bound search budget reached")
    pool = {}
    full = (1 << len(task.positives)) - 1
    uncovered = full
    cover = None
    class Found(Exception):
        pass
    traversal = None
    try:
        traversal = _Traversal(task, indices, check, lambda: deadline - time.monotonic())
        def save(index, body, mask):
            nonlocal uncovered, cover
            body, mask = traversal.simplify(index, body)
            c = candidate(task, index, body, index.coverage(mask))
            previous = pool.get(c["coverage"])
            if previous is None or (c["cost"], c["rule"]) < (previous["cost"], previous["rule"]):
                pool[c["coverage"]] = c
                uncovered &= ~c["coverage"]
                report["safe_components"] = len(pool)
                if not uncovered:
                    cover = greedy_cover(list(pool.values()), len(task.positives))
                    raise Found
        try:
            if not full:
                cover = []
            else:
                # A single rule covering the remaining labels is often found
                # with much stronger monotone pruning. It is only a bounded
                # heuristic: exhausting it or reaching its small sub-budget
                # always falls back to overlapping multi-rule components.
                class SinglePhaseFinished(Exception):
                    pass
                single_started = time.monotonic()
                single_deadline = min(deadline, single_started + min(5., seconds * .25))
                def single_check():
                    check()
                    if time.monotonic() >= single_deadline:
                        raise SinglePhaseFinished
                try:
                    result = _single_rule_cegis(task, indices, single_check, report)
                    if result is not None:
                        save(*result)
                except SinglePhaseFinished:
                    pass
                finally:
                    report["single_rule_seconds"] = time.monotonic() - single_started
                for size in range(task.max_body + 1):
                    traversal.walk(size, save, required_coverage=lambda: uncovered)
        except Found:
            pass
        if cover is not None:
            check()
            verification = verify([c["rule"] for c in cover])
            if not verification["matched"]:
                raise AssertionError("Proposed bound failed independent verification")
            report.update(status="verified", cost=sum(c["cost"] for c in cover),
                          rules=[c["rule"] for c in cover], verification=verification)
            remaining = deadline - time.monotonic()
            if remaining > 0 and pool:
                exact, pool_report = pool_cover(list(pool.values()), len(task.positives), seconds=remaining)
                report["pool_selection"] = pool_report
                if exact is not None and sum(c["cost"] for c in exact) < report["cost"]:
                    check()
                    verification = verify([c["rule"] for c in exact])
                    if not verification["matched"]:
                        raise AssertionError("Optimized pool bound failed independent verification")
                    cover = exact
                    report.update(cost=sum(c["cost"] for c in cover), rules=[c["rule"] for c in cover],
                                  verification=verification)
    except ResourceLimit:
        if report["status"] != "verified":
            report["status"] = "timeout"
            cover = None
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if traversal is not None:
            report["search"] = traversal.stats
            report["examined"] = traversal.stats["nodes"] + report.get("single_rule_cegis_models", 0)
    return cover, report


def _alpha_key(task, head, body, check=None):
    """Canonical alpha key, computed lazily under the caller's deadline.

    Only permutations of literals with identical predicate/arity need trying.
    Variables are renamed by first occurrence, beginning with the fixed head.
    This does not assume that a variable permutation is an automorphism of the
    entire bias: only already proved legal variants are grouped.
    """
    groups = defaultdict(list)
    for i in body:
        literal = task.literals[i]
        groups[literal.name, len(literal.arguments)].append(literal)
    best = None
    ordered_groups = [groups[key] for key in sorted(groups)]
    def orderings(depth=0, prefix=()):
        if check:
            check()
        if depth == len(ordered_groups):
            yield prefix
            return
        for permutation in permutations(ordered_groups[depth]):
            yield from orderings(depth + 1, prefix + (permutation,))
    for ordering in orderings():
        if check:
            check()
        naming = {}
        def normal(term):
            if term.type != clingo.SymbolType.Function:
                return str(term)
            if term.name == "var__":
                slot = term.arguments[0].number
                if slot not in naming:
                    naming[slot] = len(naming)
                return f"V{naming[slot]}"
            return str(term) if not term.arguments else ("" if term.positive else "-") + term.name + "(" + ",".join(normal(a) for a in term.arguments) + ")"
        key = (normal(head), *(normal(literal) for group in ordering for literal in group))
        if best is None or key < best:
            best = key
    return best


def _finish_variants(task, catalogue, records, check=None):
    alpha_duplicates = 0
    for c in catalogue:
        if check:
            check()
        rows = list({row["rule"]: row for row in records[c["coverage"]]}.values())
        bucket = records[c["coverage"]]
        if isinstance(bucket, VariantBucket):
            c["alpha_collection"] = bucket.export()
            alpha_duplicates += bucket.encountered - len(bucket)
        # Determinism does not depend on traversal order, solver phases, or
        # the order of equal-cost variants returned by Clingo.
        c["variants"] = sorted({row["rule"] for row in rows})
        representative = min(rows, key=lambda row: row["rule"])
        for key in ("head", "body", "rule"):
            c[key] = representative[key]
        if len(rows) == 1:
            c["alpha_variants"] = [[representative["rule"]]]
            c["alpha_variant_count"] = 1
            continue
        alpha = {}
        for row in rows:
            key = _alpha_key(task, clingo.parse_term(row["head"]), row["body"], check)
            alpha.setdefault(key, []).append(row["rule"])
        c["alpha_variants"] = [sorted(set(group)) for _, group in sorted(alpha.items())]
        c["alpha_variant_count"] = len(alpha)
        alpha_duplicates += len(c["variants"]) - len(alpha)
    return alpha_duplicates


def compile_catalogue(task, indices, *, bound=None, seconds=300, progress=None,
                      order="selectivity", preprocessing=True, legal_pruning=True,
                      variants="exhaustive", alpha_collect=True, target_bounds=None):
    """Compile a nondominated catalogue, preserving minimum-cost ties modulo alpha.

    alpha_collect=False keeps raw spellings as the reference ablation.
    target_bounds are caller-verified program costs by (predicate, arity).
    Pruning preserves all global optima, including ties, under those costs;
    some partial signatures unusable at a target optimum may be omitted.
    """
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Compilation seconds must be finite and nonnegative")
    if variants not in {"exhaustive", "targeted", "representative"}:
        raise ValueError("variants must be exhaustive, targeted, or representative")
    from .target_bounds import validate_target_costs, positive_scopes, key_text
    target_costs = validate_target_costs(task, target_bounds)
    started = time.monotonic()
    deadline = started + seconds
    def check():
        if time.monotonic() >= deadline:
            raise ResourceLimit("Compilation time budget reached", dict(traversal.stats) if traversal else {})
    traversal = None
    traversal = _Traversal(task, indices, check, lambda: deadline - time.monotonic(), order=order,
                           preprocess=preprocessing, legal_pruning=legal_pruning)
    maximum = max(0, min(task.max_body, bound - 1)) if bound is not None else task.max_body
    traversal.stats["effective_body_limit"] = maximum
    scopes = positive_scopes(task)
    program_scopes = {}
    for key, cost in target_costs.items():
        minimum = 1 if any(proof.accepts((), check=check) for head, proof in traversal.analysis.heads.items()
                           if signature(head) == key) else 2
        program_scopes[key] = cost, minimum, scopes[key]
    traversal.stats.update(program_bounds={key_text(k):v for k,v in target_costs.items()},
        target_effective_body_limits={key_text(k):min(maximum,max(0,v-1)) for k,v in target_costs.items()},
        target_minimum_useful_rule_cost={key_text(k):v[1] for k,v in program_scopes.items()},
        pruned_target_program_bound=0,
        catalogue_scope="sufficient_under_verified_target_bounds" if target_costs else "nondominated_cost_bounded_catalogue")
    if target_bounds is not None:
        indexed_signatures = {signature(i.head) for i in indices}
        traversal.stats['target_search'] = {key_text(k): {
            'started': False, 'search_complete': k not in indexed_signatures,
            'positive_count': mask.bit_count(), 'verified_target_bound': target_costs.get(k),
            'effective_body_limit': min(maximum, max(0, target_costs[k]-1)) if k in target_costs else maximum}
            for k, mask in scopes.items()}
    classes = {}
    def make_records():
        return VariantRecords(task, check,
            lambda h, b: traversal.analysis.heads[h].accepts(b, check=check), alpha_collect)
    records = make_records()
    dominance_cache, dominance_version = {}, 0
    def save(index, body, mask):
        nonlocal dominance_version
        reduced, reduced_mask = traversal.simplify(index, body)
        if len(reduced) < len(body):
            # A shorter legal safe body covers a superset: the original and
            # all its extensions are strictly dominated. Equal raw masks are
            # never sufficient without this exact legality/safety proof.
            traversal.stats["pruned_semantic_noop"] += len(body) - len(reduced)
            body, mask = reduced, reduced_mask
        c = candidate(task, index, body, index.coverage(mask))
        previous = classes.get(c["coverage"])
        if previous is None or c["cost"] < previous["cost"]:
            dominance_version += 1
            dominated.minimum_cost = min(dominated.minimum_cost, c["cost"])
            classes[c["coverage"]] = c
            records[c["coverage"]] = [c]
        elif c["cost"] == previous["cost"]:
            if variants == "exhaustive":
                records[c["coverage"]].append(c)
            elif c["rule"] < previous["rule"]:
                classes[c["coverage"]] = c
                records[c["coverage"]] = [c]
    def dominated(cost, coverage):
        key = cost, coverage
        cached = dominance_cache.get(key)
        if cached is not None and (cached[0] or cached[1] == dominance_version):
            traversal.stats["dominance_cache_hits"] = traversal.stats.get("dominance_cache_hits", 0) + 1
            return cached[0]
        answer = any(c["cost"] < cost and coverage | c["coverage"] == c["coverage"] for c in classes.values())
        if len(dominance_cache) >= 16384:
            dominance_cache.clear()
        dominance_cache[key] = answer, dominance_version
        return answer
    dominated.minimum_cost = task.max_body + 2
    traversal.walk(maximum, save, progress=progress, dominance=dominated, program_scopes=program_scopes,
                   track_targets=target_bounds is not None)
    retained = []
    for c in sorted(classes.values(), key=lambda c: (c["cost"], c["rule"])):
        check()
        if any(d["cost"] < c["cost"] and d["coverage"] | c["coverage"] == d["coverage"] for d in retained):
            continue
        retained.append(c)
    if variants == "targeted":
        targets = {(c["cost"], c["coverage"]) for c in retained}
        records = make_records()
        for c in retained:
            records[c["coverage"]] = []
        def target_save(index, body, mask):
            c = candidate(task, index, body, index.coverage(mask))
            if (c["cost"], c["coverage"]) in targets:
                records[c["coverage"]].append(c)
        if targets:
            traversal.walk(max(cost for cost, _ in targets) - 1, target_save, progress=progress, targets=targets)
    duplicates = _finish_variants(task, retained, records, check)
    retained.sort(key=lambda c: (c["cost"], c["rule"]))
    for i, c in enumerate(retained):
        c["id"] = f"c{i}"
    traversal.stats.update(complete=True, signatures_before_dominance=len(classes), signatures=len(retained),
                           elapsed_seconds=time.monotonic() - started, variants=variants,
                           alpha_duplicate_variants=duplicates,
                           alpha_collection=records.export(),
                           variant_scope=("one_per_signature" if variants == "representative" else
                                          "minimum_cost_alpha_classes" if alpha_collect else "all_minimum_cost_syntax"))
    return retained, traversal.stats


def mine_variants(task, indices, catalogue, *, seconds=300, progress=None, alpha_collect=True):
    """Reconstruct all variants only for supplied proven cost/coverage pairs.

    The caller must supply minimum-cost, nondominated signatures from a
    complete compilation/selection. Safe-prefix/no-op stops intentionally omit
    more expensive refinements. This function proves exhaustion of this target
    search, not global optimality.
    """
    started = time.monotonic()
    deadline = started + seconds
    traversal = None
    def check():
        if time.monotonic() >= deadline:
            raise ResourceLimit("Targeted variant search time budget reached", dict(traversal.stats) if traversal else {})
    traversal = _Traversal(task, indices, check, lambda: deadline - time.monotonic())
    targets = {(c["cost"], c["coverage"]) for c in catalogue}
    records = VariantRecords(task, check,
        lambda h, b: traversal.analysis.heads[h].accepts(b, check=check), alpha_collect)
    for c in catalogue:
        records[c["coverage"]] = []
    def save(index, body, mask):
        c = candidate(task, index, body, index.coverage(mask))
        if (c["cost"], c["coverage"]) in targets:
            records[c["coverage"]].append(c)
    if targets:
        traversal.walk(max(cost for cost, _ in targets) - 1, save, progress=progress, targets=targets)
    result = [dict(c) for c in catalogue]
    if any(not records[c["coverage"]] for c in result):
        raise ValueError("A requested variant cost/coverage pair has no legal minimum candidate")
    duplicates = _finish_variants(task, result, records, check)
    traversal.stats.update(complete=True, elapsed_seconds=time.monotonic() - started,
                           alpha_duplicate_variants=duplicates, alpha_collection=records.export(),
                           variant_scope="provided_cost_coverage_pairs")
    return result, traversal.stats
