"""Exact catalogue compilation with lazy joins and positive coverage bitmasks.

Atomic positive masks are upper bounds on conjunction coverage: distinct
literals may have incompatible witnesses. They are used only for sound pruning.
Before retaining a rule, correlated coverage and negative safety are established
by CoverageOracle on the original BK. A bounded negative sample affects search
order alone; safety still uses all supplied negatives. No clinical template
restrictions are introduced.
"""

from math import isfinite
from fractions import Fraction
import time

import clingo

from .oracle import ResourceLimit, match
from .alpha_collection import VariantRecords
from .positive_join import has_positive as positive_join_exists
from .relational_order import order_literals
from .search import _analysis, _finish_variants, _members
from .task import signature, variables


class _DynamicBounds:
    """Cheap covers are proposals only; independent BK evaluation certifies them.

    Revisions retain cost equality. The resulting catalogue preserves optimal
    complete programs and their variants, but not every suboptimal signature.
    No cover solver is run in the candidate loop.
    """

    def __init__(self, search, scopes, interval, probes, exact, callback):
        self.search, self.scopes = search, scopes
        self.interval, self.callback = interval, callback
        self.pools = {key: {} for key, mask in scopes.items() if mask}
        self.covered = {key: 0 for key in self.pools}
        self.pending = {key: 0 for key in self.pools}
        self.certificates = {}
        self.chosen = {}
        self.incumbent = None
        self.scope_options = {}
        for key, mask in scopes.items():
            if not mask:
                continue
            contexts = [context for head, context in search.contexts.items() if signature(head) == key]
            minimum = 1 if any(search.accepts(context[0], ()) for context in contexts) else 2
            positions = tuple(_members(mask))
            count = min(probes, len(positions)) if exact else 0
            pilots = (tuple(positions[i * (len(positions) - 1) // (count - 1)] for i in range(count))
                      if count > 1 else positions[:1] if count else ())
            self.scope_options[key] = minimum, mask, pilots
        search.exact_program_bound = exact
        search.stats.update(dynamic_program_bound=True, dynamic_bound_interval=interval,
            dynamic_bound_cover_attempts=0, dynamic_bound_verifications=0,
            dynamic_bound_updates=0, dynamic_global_bound_updates=0,
            dynamic_retired_signatures=0,
            dynamic_target_bounds=self.certificates, dynamic_incumbent=None,
            dynamic_bound_history=[],
            catalogue_scope="sufficient_for_optimal_complete_programs_under_verified_decreasing_bounds")

    def ceiling(self, key):
        limits = [self.search.program_bound]
        if key in self.search.target_program_scopes:
            limits.append(self.search.target_program_scopes[key][0])
        return min((value for value in limits if value is not None), default=None)

    def greedy(self, key):
        self.search.stats["dynamic_bound_cover_attempts"] += 1
        remaining, selected = self.scopes[key], []
        rows = self.pools[key].values()
        while remaining:
            self.search.check()
            useful = [row for row in rows if row["coverage"] & remaining]
            if not useful:
                return None
            chosen = min(useful, key=lambda row: (
                Fraction(row["cost"], (row["coverage"] & remaining).bit_count()), row["cost"], row["rule"]))
            selected.append(chosen)
            remaining &= ~chosen["coverage"]
        for row in list(reversed(selected)):
            self.search.check()
            covered = 0
            for other in selected:
                if other is not row:
                    covered |= other["coverage"]
            if covered == self.scopes[key]:
                selected.remove(row)
        return selected

    def consider(self, head, row):
        key = signature(head)
        previous_union = self.covered[key]
        self.pools[key][row["coverage"]] = row
        self.covered[key] |= row["coverage"]
        self.pending[key] += 1
        if self.covered[key] != self.scopes[key]:
            return
        proposed = None
        if row["coverage"] == self.scopes[key]:
            proposed = [row]
        old = self.chosen.get(key)
        if old is not None:
            # A newly discovered rule can immediately replace several incumbent
            # components. The mask test is exact and needs no cover solve.
            replacement = [item for item in old if item["coverage"] & ~row["coverage"]] + [row]
            if proposed is None or sum(item["cost"] for item in replacement) < sum(item["cost"] for item in proposed):
                proposed = replacement
        if previous_union != self.scopes[key] or self.pending[key] >= self.interval:
            greedy = self.greedy(key)
            self.pending[key] = 0
            if proposed is None or sum(item["cost"] for item in greedy) < sum(item["cost"] for item in proposed):
                proposed = greedy
        if proposed is not None:
            self.publish_target(key, proposed)

    def publish_target(self, key, rows):
        from .target_bounds import key_text, target_view
        from .verification import verify_rules
        cost = sum(row["cost"] for row in rows)
        ceiling = self.ceiling(key)
        if ceiling is not None and (cost > ceiling or
                (cost == ceiling and key_text(key) in self.certificates)):
            return
        rules = [row["rule"] for row in rows]
        self.search.check()
        self.search.stats["dynamic_bound_verifications"] += 1
        # A singleton target can verify the complete task in the same call;
        # this avoids grounding a large original BK twice for one improvement.
        single = len(self.pools) == 1
        verification = verify_rules(self.search.task if single else target_view(self.search.task, key), rules)
        if not verification["matched"]:
            raise ValueError("Dynamic incumbent failed original-BK verification")
        certificate = {"status": "verified", "signature": list(key), "cost": cost,
            "positive_count": self.scopes[key].bit_count(), "positive_mask": hex(self.scopes[key]),
            "rules": rules, "cover": [dict(row, coverage=hex(row["coverage"])) for row in rows],
            "verification": {name: value for name, value in verification.items() if name != "predictions"},
            "predicted_count": len(verification["predictions"]),
            "scope": "Feasible upper bound for this complete target signature; not an optimum"}
        self.certificates[key_text(key)] = certificate
        self.chosen[key] = list(rows)
        self.search.target_program_scopes[key] = (cost, *self.scope_options[key])
        stats = self.search.stats
        stats.setdefault("program_bounds", {})[key_text(key)] = cost
        stats.setdefault("target_minimum_useful_rule_cost", {})[key_text(key)] = self.scope_options[key][0]
        stats.setdefault("target_bound_positive_pilot_positions", {})[key_text(key)] = list(self.scope_options[key][2])
        stats["dynamic_bound_updates"] += 1
        stats["dynamic_bound_history"].append({"signature": key_text(key), "cost": cost,
            "nodes": stats["nodes"], "elapsed_seconds": time.monotonic() - self.search.started})
        target = stats["target_search"][key_text(key)]
        target.update(verified_target_bound=cost,
                      effective_body_limit=min(target["effective_body_limit"], max(0, cost - 1)))
        # Publish completed verification before checking the deadline: an
        # interruption must retain this feasible program and its certificate.
        if single:
            self.publish_global(rows, verification)
        elif len(self.certificates) == len(self.pools):
            combined = [row for chosen in self.chosen.values() for row in chosen]
            self.publish_global(combined)
        self.search.check()

    def publish_global(self, rows, verification=None):
        from .verification import verify_rules
        cost = sum(row["cost"] for row in rows)
        if self.search.program_bound is not None and cost >= self.search.program_bound:
            return
        rules = [row["rule"] for row in rows]
        if verification is None:
            self.search.check()
            self.search.stats["dynamic_bound_verifications"] += 1
            verification = verify_rules(self.search.task, rules)
        if not verification["matched"]:
            raise ValueError("Dynamic complete incumbent failed original-BK verification")
        self.incumbent = {"status": "verified", "cost": cost, "rules": rules,
            "cover": [dict(row, coverage=hex(row["coverage"])) for row in rows],
            "verification": {name: value for name, value in verification.items() if name != "predictions"},
            "predicted_count": len(verification["predictions"]),
            "scope": "Feasible upper bound for all supplied hard labels; not an optimum"}
        self.search.program_bound = cost
        self.search.minimum_rule_cost = min(options[0] for options in self.scope_options.values())
        self.search.stats.update(program_bound=True, program_bound_value=cost,
            exact_program_bound=self.search.exact_program_bound, dynamic_incumbent=self.incumbent)
        self.search.stats["dynamic_global_bound_updates"] += 1
        if self.callback is not None:
            self.callback(self.incumbent)

    def finish(self):
        for key in self.pools:
            if self.pending[key] and self.covered[key] == self.scopes[key]:
                self.publish_target(key, self.greedy(key))
                self.pending[key] = 0


class _Search:
    def __init__(self, task, oracle, seconds, order, progress, alpha_collect=True,
                 ground_prefix_filter=True, prefix_completion=True):
        if seconds is not None and (not isfinite(seconds) or seconds < 0):
            raise ValueError("Compilation seconds must be finite and nonnegative, or None")
        if order not in {"source", "selectivity"}:
            raise ValueError("order must be source or selectivity")
        if type(ground_prefix_filter) is not bool:
            raise ValueError("ground_prefix_filter must be a boolean")
        if type(prefix_completion) is not bool:
            raise ValueError("prefix_completion must be a boolean")
        self.ground_prefix_filter = ground_prefix_filter
        self.prefix_completion = prefix_completion
        self.task, self.oracle, self.progress = task, oracle, progress
        self.positive_cores = getattr(oracle, 'positive_cores', None)
        self.positive_labels = tuple(sorted(task.positives))
        self.global_positive_mask = (1 << len(self.positive_labels)) - 1
        self.program_bound = None
        self.exact_program_bound = False
        self.bound_positive_pilots = ()
        self.minimum_rule_cost = None
        self.target_program_scopes = {}
        self.dynamic_bounds = None
        self.started = time.monotonic()
        self.deadline = self.started + seconds if seconds is not None else float("inf")
        self.original_check = oracle.check
        self.stats = {"backend": "relational_positive_bitmasks", "complete": False,
                      "positive_core_pruning": self.positive_cores is not None, "pruned_positive_core": 0,
                      "positive_join_backend": getattr(oracle, "positive_join_backend", "fixed_labels"),
                      "binding_backend": getattr(oracle, "binding_backend", "reference"),
                      "negative_join_backend": getattr(oracle, "negative_join_backend", "bindings"),
                      "negative_label_limit": getattr(oracle, "negative_label_limit", 0),
                      "negative_join_effective_backend": ("labels_with_budget_fallback"
                          if getattr(oracle, "binding_backend", "reference") == "indexed"
                          and getattr(oracle, "negative_join_backend", "bindings") == "labels"
                          and getattr(oracle, "negative_label_limit", 0) else "bindings"),
                      "positive_pair_pruning": getattr(oracle, "positive_pair_pruning", False),
                      "positive_residual_cache": getattr(oracle, "positive_residual_cache", False),
                      "positive_lookahead": getattr(oracle, "positive_lookahead", False),
                      "positive_residual_normalization": getattr(oracle, "positive_residual_normalization", False),
                      "nodes": 0, "legal_safe_rules": 0, "pruned_no_positive": 0,
                      "program_bound": False, "program_bound_value": None,
                      "minimum_useful_rule_cost": None, "pruned_program_bound": 0,
                      "exact_program_bound": False, "pruned_exact_program_bound": 0,
                      "full_positive_queries": 0, "full_positive_label_checks": 0,
                      "bound_positive_pilot_positions": [], "bound_positive_probe_queries": 0,
                      "bound_positive_probe_label_checks": 0, "pruned_bound_positive_probe": 0,
                      "catalogue_scope": "minimum_cost_nondominated_signatures_within_rule_cost_limit",
                      "pruned_guard_conflict": 0, "pruned_structural_completion": 0,
                      "ground_prefix_filter": ground_prefix_filter, "ground_prefix_checks": 0,
                      "prefix_completion": prefix_completion, "skipped_prefix_completion": 0,
                      "ground_prefix_fallback_heads": 0, "ground_body_rejections": 0,
                      "pruned_safe_extensions": 0, "pruned_cheaper_cover": 0,
                      "pruned_target": 0, "pruned_semantic_noop": 0,
                      "positive_atomic_queries": 0, "positive_atomic_cache_hits": 0,
                      "exact_positive_queries": 0, "exact_negative_queries": 0,
                      "positive_existence_queries": 0,
                      "order": order, "head_orders": {}, "negative_ordering": {},
                      "negative_ordering_seconds": 0.0, "phase": "bias_analysis"}
        self.contexts = {}
        self.classes = {}
        self.alpha_collect = alpha_collect
        self.records = self.make_records()
        self.analysis = _analysis(task, self.check, seconds)
        self.oracle.profile.diagnostics['bias_queries'] = self.analysis
        self.stats["bias_analysis_seconds"] = time.monotonic() - self.started
        self.stats["phase"] = "positive_masks"
        mask_started = time.monotonic()
        oracle.check = self.check
        try:
            for head in task.heads:
                self.check()
                proof = self.analysis.heads[head]
                if ground_prefix_filter and (proof.prefix_filter is None or
                                            not proof.prefix_filter.all_constraints_compiled):
                    self.stats["ground_prefix_fallback_heads"] += 1
                if not proof.legal_body_exists:
                    continue
                positives = []
                for label, position in oracle.positives.items():
                    fixed = match(head, label)
                    if fixed is not None:
                        positives.append((position, fixed))
                all_positive = sum(1 << position for position, _ in positives)
                if not all_positive:
                    continue
                closures = tuple((1 << i) | sum(1 << guard for guard in proof.guards[i])
                                 for i in range(len(task.literals)))
                masks = []
                for i, literal in enumerate(task.literals):
                    self.check()
                    if hasattr(oracle, "atomic_coverage"):
                        # Project each matching relation once instead of
                        # repeating an existential join for every positive.
                        # Correlations between different literals are still
                        # checked by the full query before retaining a rule.
                        self.stats["positive_atomic_queries"] += 1
                        masks.append(oracle.atomic_coverage(head, i))
                        continue
                    slots = variables(literal)
                    answers = {}
                    mask = 0
                    for number, (position, fixed) in enumerate(positives):
                        if number % 256 == 0:
                            self.check()
                        key = tuple((slot, value) for slot, value in sorted(fixed.items()) if slot in slots)
                        if key not in answers:
                            self.stats["positive_atomic_queries"] += 1
                            answers[key] = (oracle.has_binding((i,), dict(key)) if hasattr(oracle, "has_binding")
                                            else next(oracle.bindings((i,), dict(key)), None) is not None)
                        else:
                            self.stats["positive_atomic_cache_hits"] += 1
                        if answers[key]:
                            mask |= 1 << position
                    masks.append(mask)
                unavailable = sum(1 << i for i in proof.unusable_literals)
                unavailable |= sum(1 << i for i, mask in enumerate(masks) if not mask)
                if order == "source":
                    permutation = tuple(range(len(masks)))
                else:
                    # This bounded heuristic changes order only. A partial
                    # negative sample cannot prove rule safety or prune a
                    # branch; full correlated queries remain authoritative.
                    allowance = min(3.0, max(0, self.deadline - time.monotonic())
                                    * .1 / max(1, len(task.heads)))
                    permutation, ordering = order_literals(
                        task, oracle, head, proof, masks, self.check, seconds=allowance)
                    self.stats["negative_ordering"][str(head)] = ordering
                    self.stats["negative_ordering_seconds"] += ordering["elapsed_seconds"]
                prefix = [0]
                for i in permutation:
                    prefix.append(prefix[-1] | (1 << i))
                self.contexts[head] = proof, closures, tuple(masks), unavailable, permutation, prefix, all_positive
                self.stats["head_orders"][str(head)] = list(permutation)
            self.stats["positive_masks_seconds"] = (time.monotonic() - mask_started
                                                    - self.stats["negative_ordering_seconds"])
            self.stats["phase"] = "search"
        except BaseException:
            oracle.check = self.original_check
            raise

    def check(self):
        if time.monotonic() >= self.deadline:
            stats = dict(self.stats, elapsed_seconds=time.monotonic() - self.started)
            if hasattr(self.oracle, "profile"):
                stats["profile"] = self.oracle.profile.export()
            raise ResourceLimit("Relational compilation time budget reached", stats)
        self.original_check()

    def restore(self):
        self.oracle.check = self.original_check

    def make_records(self):
        return VariantRecords(self.task, self.check,
            lambda h, b: self.accepts(self.analysis.heads[h], b), self.alpha_collect)

    def accepts(self, proof, body):
        with self.oracle.profile.measure("bias.legality"):
            return proof.accepts(body, check=self.check)

    def possible_completion(self, proof, body, forbidden):
        with self.oracle.profile.measure("bias.completion"):
            return proof.possible_completion(body, forbidden, check=self.check)

    def row(self, head, body, coverage):
        return {"head": str(head), "body": list(body), "cost": len(body) + 1,
                "coverage": coverage, "rule": self.task.rule(head, body)}

    def has_positive(self, head, body, possible):
        """Check one correlated witness without enumerating all rule outputs."""
        if hasattr(self.oracle, "space") and hasattr(self.oracle, "_head_vars"):
            self.stats["positive_existence_queries"] += 1
            if self.positive_cores is None:
                return positive_join_exists(self.oracle, head, body, possible)
            from .positive_cores import MIN_QUERY_NODES
            before = self.oracle.stats.get('positive_join_nodes', 0)
            started = time.monotonic()
            answer = positive_join_exists(self.oracle, head, body, possible)
            if not answer and self.oracle.stats.get('positive_join_nodes', 0) - before >= MIN_QUERY_NODES:
                self.positive_cores.learn(head, body, time.monotonic() - started)
            return answer
        for position in _members(possible):
            self.check()
            self.stats["positive_existence_queries"] += 1
            if self.oracle.exists(head, body, self.positive_labels[position]):
                return True
        return False

    def positive_coverage(self, head, body, possible):
        """Exact coverage over supplied positives, not a join of all outputs."""
        self.stats["exact_positive_queries"] += 1
        result = 0
        for position in _members(possible):
            self.check()
            self.stats["positive_existence_queries"] += 1
            if self.oracle.exists(head, body, self.positive_labels[position]):
                result |= 1 << position
        return result

    def program_scope(self, head):
        scoped = self.target_program_scopes.get(signature(head))
        if scoped is not None:
            limit, minimum, mask, pilots = scoped
            if self.program_bound is not None:
                limit = min(limit, self.program_bound)
            return limit, minimum, mask, pilots
        return (self.program_bound, self.minimum_rule_cost, self.global_positive_mask,
                self.bound_positive_pilots)

    def full_positive(self, head, body):
        """Prove coverage of every positive, stopping on the first exact failure.

        Used only when the verified program bound forces this rule to stand
        alone. A missing positive also remains missing in every extension.
        An interrupted query is neither a refutation nor a complete mask.
        """
        self.stats["full_positive_queries"] += 1
        for position in _members(self.program_scope(head)[2]):
            self.check()
            self.stats["full_positive_label_checks"] += 1
            if not self.oracle.exists(head, body, self.positive_labels[position]):
                self.check()
                return False
        self.check()
        return True

    def probe_positive(self, head, body):
        """A failed exact pilot query refutes universal coverage; success proves none."""
        self.stats["bound_positive_probe_queries"] += 1
        for position in self.program_scope(head)[3]:
            self.check()
            self.stats["bound_positive_probe_label_checks"] += 1
            if not self.oracle.exists(head, body, self.positive_labels[position]):
                self.check()
                return False
        self.check()
        return True

    def simplify(self, head, body, coverage):
        proof = self.contexts[head][0]
        current = tuple(body)
        for literal in tuple(reversed(body)):
            self.check()
            smaller = tuple(i for i in current if i != literal)
            if not self.accepts(proof, smaller):
                continue
            self.stats["exact_negative_queries"] += 1
            if self.oracle.negative(head, smaller) is None:
                current = smaller
        if current != body:
            self.stats["pruned_semantic_noop"] += len(body) - len(current)
            coverage = self.positive_coverage(head, current, self.contexts[head][-1])
        return current, coverage

    def save(self, head, body, coverage, variants):
        body, coverage = self.simplify(head, body, coverage)
        row = self.row(head, body, coverage)
        previous = self.classes.get(coverage)
        if previous is None or row["cost"] < previous["cost"]:
            self.classes[coverage], self.records[coverage] = row, [row]
            if self.dynamic_bounds is not None:
                self.dynamic_bounds.consider(head, row)
        elif row["cost"] == previous["cost"]:
            if variants == "exhaustive":
                self.records[coverage].append(row)
            elif row["rule"] < previous["rule"]:
                self.classes[coverage], self.records[coverage] = row, [row]

    def walk(self, maximum, variants="exhaustive", goals=None, *, final_pass=False):
        finished_heads = set()
        for head, context in self.contexts.items():
            if final_pass and "target_search" in self.stats:
                self.stats["target_search"][f"{head.name}/{len(head.arguments)}"]["started"] = True
            proof, closures, masks, unavailable, order, prefix, all_positive = context

            def visit(body_bits, start, possible):
                self.stats["nodes"] += 1
                self.check()
                if not possible:
                    self.stats["pruned_no_positive"] += 1
                    return
                if self.positive_cores is not None and self.positive_cores.match(head, body_bits) is not None:
                    self.stats['pruned_positive_core'] += 1
                    return
                size = body_bits.bit_count()
                # Bounds may tighten when a previous branch contributes a new
                # independently verified cover. Never freeze them per head.
                limit, minimum_cost, scope_mask, _ = self.program_scope(head)
                head_maximum = min(maximum, limit - 1) if limit is not None else maximum
                if size > head_maximum:
                    self.stats["pruned_program_bound"] += 1
                    return
                if (limit is not None and possible != scope_mask
                        and size + 1 + minimum_cost > limit):
                    # An extension can only lose positive coverage. A partial
                    # rule in a complete program needs another useful rule;
                    # the two costs already exceed the caller's upper bound.
                    # Equality is retained, and full-global-coverage prefixes
                    # are never removed by this test.
                    self.stats["pruned_program_bound"] += 1
                    return
                if goals is not None:
                    if not any(cost >= size + 1 and wanted & possible == wanted for cost, wanted in goals):
                        self.stats["pruned_target"] += 1
                        return
                elif any(row["cost"] < size + 1 and possible | row["coverage"] == row["coverage"]
                         for row in self.classes.values()):
                    # Possible is only an upper bound, so this condition is
                    # conservative even for head-unsafe or uncorrelated prefixes.
                    self.stats["pruned_cheaper_cover"] += 1
                    return
                body = tuple(_members(body_bits))
                forbidden = (prefix[start] & ~body_bits) | unavailable
                structural = proof.prefix_filter if self.ground_prefix_filter else None
                if structural is not None:
                    self.stats["ground_prefix_checks"] += 1
                # The ground filter only rejects impossible prefixes. If all
                # constraints were lowered, an exact completion solve is not
                # needed for traversal: false positives merely add exploration.
                # Every retained candidate still has exact ASP legality checked.
                if structural is not None and not structural.allows(body_bits, forbidden):
                    self.stats["pruned_structural_completion"] += 1
                    return
                if structural is None or not structural.all_constraints_compiled:
                    if self.prefix_completion:
                        if not self.possible_completion(proof, body, tuple(_members(forbidden))):
                            self.stats["pruned_structural_completion"] += 1
                            return
                    else:
                        # Unknown extendibility permits exploration only. Final
                        # legality below remains exact, including for biases
                        # that the necessary ground filter cannot fully lower.
                        self.stats["skipped_prefix_completion"] += 1
                full_excluded = ((1 << len(self.task.literals)) - 1) & ~body_bits
                possible_body = structural is None or structural.allows(body_bits, full_excluded)
                if not possible_body:
                    self.stats["ground_body_rejections"] += 1
                if possible_body and self.accepts(proof, body):
                    needs_all = (self.exact_program_bound and limit is not None
                                 and size + 1 + minimum_cost > limit)
                    if not self.has_positive(head, body, possible):
                        self.stats["pruned_no_positive"] += 1
                        return
                    if needs_all and not self.probe_positive(head, body):
                        self.stats["pruned_exact_program_bound"] += 1
                        self.stats["pruned_bound_positive_probe"] += 1
                        return
                    self.stats["exact_negative_queries"] += 1
                    if self.oracle.negative(head, body) is None:
                        # Unsafe prefixes remain extensible. Only a safe
                        # component needs a full positive-coverage proof before
                        # collection; doing this first for unsafe rules repeats
                        # thousands of unnecessary fixed-label queries.
                        if needs_all and not self.full_positive(head, body):
                            self.stats["pruned_exact_program_bound"] += 1
                            return
                        coverage = (scope_mask if needs_all else
                                    self.positive_coverage(head, body, possible))
                        self.stats["legal_safe_rules"] += 1
                        if goals is None:
                            self.save(head, body, coverage, variants)
                        elif (size + 1, coverage) in goals:
                            self.records[coverage].append(self.row(head, body, coverage))
                        self.stats["pruned_safe_extensions"] += 1
                        return
                if size >= head_maximum:
                    return
                for position in range(start, len(order)):
                    if self.dynamic_bounds is not None:
                        current_limit = self.program_scope(head)[0]
                        head_maximum = min(maximum, current_limit - 1) if current_limit is not None else maximum
                        if size >= head_maximum:
                            return
                    literal = order[position]
                    if body_bits >> literal & 1 or unavailable >> literal & 1:
                        continue
                    closure = closures[literal]
                    if closure & ((prefix[position] & ~body_bits) | unavailable):
                        self.stats["pruned_guard_conflict"] += 1
                        continue
                    added = closure & ~body_bits
                    if size + added.bit_count() > head_maximum:
                        continue
                    updated = possible
                    for i in _members(added):
                        updated &= masks[i]
                    visit(body_bits | added, position + 1, updated)
                if self.progress and self.stats["nodes"] % 1000 == 0:
                    self.progress(dict(self.stats, elapsed_seconds=time.monotonic() - self.started))

            visit(0, 0, all_positive)
            finished_heads.add(head)
            if final_pass and "target_search" in self.stats:
                key = signature(head)
                target = self.stats["target_search"][f"{key[0]}/{key[1]}"]
                target["search_complete"] = all(h in finished_heads for h in self.contexts
                                                 if signature(h) == key)

    def finish(self, catalogue, variants):
        self.check()
        self.stats["phase"] = "variant_bookkeeping"
        duplicates = _finish_variants(self.task, catalogue, self.records, self.check)
        catalogue.sort(key=lambda row: (row["cost"], row["rule"]))
        for i, row in enumerate(catalogue):
            row["id"] = f"c{i}"
        self.check()
        self.stats.update(complete=True, signatures_before_dominance=len(self.classes), signatures=len(catalogue),
                          elapsed_seconds=time.monotonic() - self.started, variants=variants,
                          alpha_duplicate_variants=duplicates,
                          alpha_collection=self.records.export(), indexes=dict(self.oracle.space.stats),
                          variant_scope=("one_per_signature" if variants == "representative" else
                                         "minimum_cost_alpha_classes" if self.alpha_collect else "all_minimum_cost_syntax"),
                          oracle_queries=dict(self.oracle.stats))
        if hasattr(self.oracle, "profile"):
            self.stats["profile"] = self.oracle.profile.export()
        return catalogue, self.stats


def compile_catalogue(task, oracle, *, bound=None, seconds=300, progress=None,
                      order="selectivity", variants="exhaustive", program_bound=False, alpha_collect=True,
                      exact_program_bound=True, bound_positive_probes=24, ground_prefix_filter=True,
                      prefix_completion=True, target_bounds=None, dynamic_program_bound=False,
                      dynamic_bound_interval=256, on_incumbent=None):
    """Compile a minimum-cost nondominated coverage catalogue.

    The default preserves the usual catalogue restricted by a per-rule bound.
    With program_bound=True, bound must be a positive integer and the result
    need only retain rules usable in complete programs costing at most bound.
    To preserve the unrestricted task optimum, the caller must supply an upper
    bound established from an independently verified feasible complete program.
    This function does not certify an integer bound supplied on its own.

    prefix_completion=False skips only optional exact prefix feasibility
    queries. The necessary ground filter and final exact candidate legality
    still apply. Extra dead branches can increase time or memory.

    target_bounds maps (predicate, arity) signatures to caller-verified feasible
    target costs. All overlapping templates of a signature stay together and
    positive masks retain their original global positions. Missing target
    bounds fall back to the existing global bound or the original grammar.

    dynamic_program_bound=True proposes cheap covers of discovered exact rules
    and independently verifies improving incumbents against the original BK.
    It preserves optimal complete programs, including equal-cost variants, but
    changes the unrestricted catalogue contract and must not be used for a
    partial-positive-coverage objective. Dynamic certificates are returned in
    stats (also ResourceLimit.stats). on_incumbent receives each verified full
    improvement immediately and may interrupt compilation to switch strategy.
    A None time budget has no deadline.

    The extra rule-cost lower bound is derived from empty-body legality in the
    actual bias, never from the discovered pool. Minimum-cost programs <= bound
    retain their direct traversal paths; the proof does not rely on discovering
    them by simplifying larger bodies whose coverage could subsequently grow.
    """
    if type(program_bound) is not bool:
        raise ValueError("program_bound must be a boolean")
    if type(exact_program_bound) is not bool:
        raise ValueError("exact_program_bound must be a boolean")
    if type(dynamic_program_bound) is not bool:
        raise ValueError("dynamic_program_bound must be a boolean")
    if type(dynamic_bound_interval) is not int or dynamic_bound_interval < 1:
        raise ValueError("dynamic_bound_interval must be a positive integer")
    if on_incumbent is not None and (not callable(on_incumbent) or not dynamic_program_bound):
        raise ValueError("on_incumbent requires dynamic_program_bound and must be callable")
    if type(bound_positive_probes) is not int or bound_positive_probes < 0:
        raise ValueError("bound_positive_probes must be a nonnegative integer")
    if program_bound and (type(bound) is not int or bound <= 0):
        raise ValueError("program_bound requires a positive integer complete-program bound")
    if variants not in {"exhaustive", "targeted", "representative"}:
        raise ValueError("variants must be exhaustive, targeted, or representative")
    if bound is not None and (type(bound) is not int or bound < 0):
        raise ValueError("bound must be a nonnegative integer, or None")
    from .target_bounds import positive_scopes, key_text
    scopes = positive_scopes(task)
    if target_bounds is not None:
        if not isinstance(target_bounds, dict) or any(
            key not in scopes or type(value) is not int or value < 0 or (scopes[key] and value == 0)
            for key, value in target_bounds.items()):
            raise ValueError("target_bounds must map known signatures to verified nonnegative costs")
    search = _Search(task, oracle, seconds, order, progress, alpha_collect,
                     ground_prefix_filter, prefix_completion)
    maximum = max(0, min(task.max_body, bound - 1)) if bound is not None else task.max_body
    search.stats["effective_body_limit"] = maximum
    try:
        if program_bound:
            # Contexts include every legally possible head matching at least
            # one supplied positive, including ground heads with empty bodies.
            minimum = 1 if any(search.accepts(context[0], ())
                               for context in search.contexts.values()) else 2
            search.program_bound = bound
            search.exact_program_bound = exact_program_bound
            count = min(bound_positive_probes, len(search.positive_labels)) if exact_program_bound else 0
            search.bound_positive_pilots = (tuple(i * (len(search.positive_labels) - 1) // (count - 1)
                                                  for i in range(count)) if count > 1 else (0,) if count else ())
            search.stats["bound_positive_pilot_positions"] = list(search.bound_positive_pilots)
            search.minimum_rule_cost = minimum
            search.stats.update(program_bound=True, program_bound_value=bound,
                exact_program_bound=exact_program_bound,
                minimum_useful_rule_cost=minimum,
                minimum_rule_cost_basis="actual_bias_empty_body_legality_on_positive_compatible_heads",
                catalogue_scope="sufficient_for_complete_programs_cost_at_most_bound")
        if target_bounds:
            search.exact_program_bound = exact_program_bound
            for key, target_bound in target_bounds.items():
                if not scopes[key]:
                    continue
                relevant = [context for head, context in search.contexts.items() if signature(head) == key]
                minimum = 1 if any(search.accepts(context[0], ()) for context in relevant) else 2
                positions = tuple(_members(scopes[key]))
                count = min(bound_positive_probes, len(positions)) if exact_program_bound else 0
                pilots = (tuple(positions[i * (len(positions) - 1) // (count - 1)] for i in range(count))
                          if count > 1 else positions[:1] if count else ())
                search.target_program_scopes[key] = target_bound, minimum, scopes[key], pilots
            search.stats.update(program_bounds={key_text(key): value for key, value in target_bounds.items()},
                catalogue_scope="sufficient_under_verified_per_target_and_optional_global_bounds",
                target_bound_positive_pilot_positions={key_text(key): list(value[3])
                                                      for key, value in search.target_program_scopes.items()},
                target_minimum_useful_rule_cost={key_text(key): value[1]
                                                for key, value in search.target_program_scopes.items()})
        search.stats["target_search"] = {key_text(key): {
            "started": False, "search_complete": not any(signature(h) == key for h in search.contexts),
            "positive_count": mask.bit_count(),
            "verified_target_bound": (target_bounds or {}).get(key),
            "effective_body_limit": min(maximum, target_bounds[key] - 1)
                if target_bounds and key in target_bounds and mask else maximum}
            for key, mask in scopes.items()}
        if dynamic_program_bound:
            search.dynamic_bounds = _DynamicBounds(search, scopes, dynamic_bound_interval,
                bound_positive_probes, exact_program_bound, on_incumbent)
        # Cheap legal rules often dominate long prefixes. Seed them without
        # altering the later exhaustive traversal or using any target schema.
        if maximum > 1:
            search.walk(1, variants)
        search.walk(maximum, variants, final_pass=True)
        if search.dynamic_bounds is not None:
            search.dynamic_bounds.finish()
        retained = []
        for row in sorted(search.classes.values(), key=lambda row: (row["cost"], row["rule"])):
            search.check()
            if search.dynamic_bounds is not None:
                limit, minimum, mask, _ = search.program_scope(clingo.parse_term(row["head"]))
                if limit is not None and (row["cost"] > limit or
                        (row["coverage"] != mask and row["cost"] + minimum > limit)):
                    # Earlier passes can retain components excluded by a later
                    # verified bound. Retiring them also keeps targeted variant
                    # reconstruction aligned with its final traversal scope.
                    search.stats["dynamic_retired_signatures"] += 1
                    continue
            if any(previous["cost"] < row["cost"] and previous["coverage"] | row["coverage"] == previous["coverage"]
                   for previous in retained):
                continue
            retained.append(row)
        if variants == "targeted" and retained:
            goals = {(row["cost"], row["coverage"]) for row in retained}
            search.records = search.make_records()
            for row in retained:
                search.records[row["coverage"]] = []
            search.walk(max(cost for cost, _ in goals) - 1, goals=goals)
        return search.finish(retained, variants)
    except ResourceLimit as error:
        # ResourceLimit can originate in an oracle or caller rather than our
        # own deadline check; always attach the latest verified incumbents.
        error.stats.update(search.stats, oracle_queries=dict(oracle.stats), indexes=dict(oracle.space.stats),
                           profile=oracle.profile.export(), alpha_collection=search.records.export())
        raise
    finally:
        search.restore()


def mine_variants(task, oracle, catalogue, *, seconds=300, progress=None, order="selectivity", alpha_collect=True,
                  ground_prefix_filter=True, prefix_completion=True):
    """Reconstruct variants for caller-proved cost/coverage pairs only."""
    search = _Search(task, oracle, seconds, order, progress, alpha_collect,
                     ground_prefix_filter, prefix_completion)
    try:
        goals = {(row["cost"], row["coverage"]) for row in catalogue}
        search.records = search.make_records()
        for row in catalogue:
            search.records[row["coverage"]] = []
        if goals:
            search.walk(max(cost for cost, _ in goals) - 1, goals=goals)
        if any(not search.records[row["coverage"]] for row in catalogue):
            raise ValueError("A requested cost/coverage pair has no legal minimum candidate")
        # Preserve catalogue ids: selection/enum already refer to those ids.
        result = [dict(row) for row in catalogue]
        duplicates = _finish_variants(task, result, search.records, search.check)
        search.check()
        search.stats.update(complete=True, elapsed_seconds=time.monotonic() - search.started,
                            alpha_duplicate_variants=duplicates, alpha_collection=search.records.export(),
                            variant_scope="provided_cost_coverage_pairs")
        if hasattr(oracle, "profile"):
            search.stats["profile"] = oracle.profile.export()
        return result, search.stats
    except ResourceLimit as error:
        error.stats.update(oracle_queries=dict(oracle.stats), indexes=dict(oracle.space.stats),
                           profile=oracle.profile.export(), alpha_collection=search.records.export())
        raise
    finally:
        search.restore()


compile_relational_catalogue = compile_catalogue
