"""Exact, head-specific implication proofs and reusable legality queries.

``guards[L]`` contains the other literals selected by every legal body with L.
Proofs use actual ASP stable models under the same head safety, cardinality and
allowed-literal restrictions as Task.accepts; no predicate-specific patterns.
Witness intersections reduce proof queries from O(n²) to O(n * max_body).

For a legal positive body B, t'_L = t_L and all(t_C for C in guards[L]) leaves
its conjunction unchanged. Unusable literals have effective truth false. The
bits are relative to legal bodies, not raw atom truth for arbitrary subsets.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import isfinite
import time
from typing import Callable

import clingo

from .oracle import ResourceLimit
from .task import Task, TaskError, variables


class _GroundObserver:
    """Record the real ground rules, including Clingo's aggregate auxiliaries."""
    def __init__(self):
        self.rules, self.weights, self.external_atoms = [], [], set()
        self.enabled = True

    def rule(self, choice, head, body):
        if self.enabled:
            self.rules.append((choice, tuple(head), tuple(body)))

    def weight_rule(self, choice, head, lower, body):
        if self.enabled:
            self.weights.append((choice, tuple(head), lower, tuple(body)))

    def external(self, atom, _value):
        if self.enabled:
            self.external_atoms.add(atom)


@dataclass(frozen=True)
class PrefixFilter:
    """Necessary ground constraints evaluated with partial body assignments.

    This is deliberately not an existential-completion oracle. Rejecting a
    prefix is a proof, but accepting one only permits continued exploration.
    The final candidate still passes the retained exact ASP legality solver.
    Unsupported definitions, cycles, disjunction and external choices remain
    with that solver. No predicate names or source-level bias idioms are used.
    """
    nogoods: tuple
    bounds: tuple
    incompatible: tuple[int, ...]
    all_constraints_compiled: bool
    compiled_constraints: int
    skipped_constraints: int
    deletion_monotone: int = 0
    deletion_checks: tuple = ()

    def allows(self, selected, excluded):
        for positive, negative in self.nogoods:
            if selected & positive == positive and excluded & negative == negative:
                return False
        for positive, negative, lower, upper in self.bounds:
            if upper is not None and ((selected & positive).bit_count()
                                      + (excluded & negative).bit_count() > upper):
                return False
            if lower is not None and ((positive & ~excluded).bit_count()
                                      + (negative & ~selected).bit_count() < lower):
                return False
        return True

    def can_delete(self, literal, selected, excluded):
        """Prove deletion harmless for every extension of this fixed prefix."""
        bit = 1 << literal
        if self.deletion_monotone & bit:
            return True
        if not self.deletion_checks or self.deletion_checks[literal] is None:
            return False
        nogoods, bounds = self.deletion_checks[literal]
        selected, excluded = selected & ~bit, excluded | bit
        for positive, negative in nogoods:
            # A retained selected negative term or an excluded positive term
            # keeps this forbidden conjunction false in every extension.
            if not negative & selected and not positive & excluded:
                return False
        for positive, negative, lower, upper in bounds:
            if lower is not None and ((positive & selected).bit_count()
                                      + (negative & excluded).bit_count() < lower):
                return False
            if upper is not None and ((positive & ~excluded).bit_count()
                                      + (negative & ~selected).bit_count() > upper):
                return False
        return True


def _prefix_filter(observer, body_atoms, guards=None, fixed_atoms=None):
    """Inline uniquely defined acyclic ground auxiliaries into simple bounds.

    Inlining a single normal definition is exact by stable-model support;
    choices, disjunctive heads, undefined external atoms and cycles are never
    treated as definitions. Only unit-weight flat cardinalities are lowered.
    Other constraints are left to the generic SAT completion fallback.
    """
    inputs = {atom: i + 1 for i, atom in enumerate(body_atoms) if atom is not None}
    fixed_atoms = fixed_atoms or {}
    definitions, unsupported, constraints = {}, set(observer.external_atoms), []
    for choice, head, body in observer.rules:
        if not head:
            constraints.append(("and", body))
        elif choice or len(head) != 1:
            unsupported.update(head)
        else:
            definitions.setdefault(head[0], []).append(("and", body))
    for choice, head, lower, body in observer.weights:
        if not head:
            constraints.append(("weight", lower, body))
        elif choice or len(head) != 1:
            unsupported.update(head)
        else:
            definitions.setdefault(head[0], []).append(("weight", lower, body))
    cache, visiting = {}, set()
    def invert(value):
        if type(value) is bool:
            return not value
        if type(value) is int:
            return -value
        if value is not None and value[0] == "bound":
            _, terms, lower, upper = value
            # The only bounds introduced by ground weight rules have one end.
            return ("bound", terms, None, lower - 1) if lower is not None else ("bound", terms, upper + 1, None)
        return None
    def literal(atom):
        sign, atom = atom > 0, abs(atom)
        if atom in inputs:
            result = inputs[atom]
        elif atom in fixed_atoms:
            result = fixed_atoms[atom]
        elif atom in cache:
            result = cache[atom]
        elif atom in unsupported or atom in visiting or not definitions.get(atom):
            result = None
        else:
            visiting.add(atom)
            alternatives = [expression(rule) for rule in definitions[atom]]
            if any(value is True for value in alternatives):
                result = True
            else:
                alternatives = [value for value in alternatives if value is not False]
                if not alternatives:
                    result = False
                elif len(alternatives) == 1:
                    result = alternatives[0]
                elif all(type(value) is int for value in alternatives):
                    result = ("bound", tuple(set(alternatives)), 1, None)
                else:
                    result = None
            visiting.remove(atom)
            cache[atom] = result
        return result if sign else invert(result)
    def expression(rule):
        if rule[0] == "and":
            values = []
            for atom in rule[1]:
                value = literal(atom)
                if value is False:
                    return False
                if value is True:
                    continue
                if value is None:
                    return None
                if type(value) is tuple and value[0] == "and":
                    values.extend(value[1])
                else:
                    values.append(value)
            if not values:
                return True
            return values[0] if len(values) == 1 else ("and", tuple(values))
        _, lower, terms = rule
        values = []
        for atom, weight in terms:
            value = literal(atom)
            if value is True:
                lower -= weight
            elif value is False:
                continue
            elif type(value) is int and weight == 1:
                values.append(value)
            else:
                return None
        if lower <= 0:
            return True
        if lower > len(values):
            return False
        # Duplicate terms cannot be lowered to a bit count.
        if len(values) != len(set(values)):
            return None
        return ("bound", tuple(values), lower, None)
    nogoods, bounds, skipped, compiled = set(), set(), 0, 0
    def bits(terms):
        return (sum(1 << (i - 1) for i in set(terms) if i > 0),
                sum(1 << (-i - 1) for i in set(terms) if i < 0))
    for rule in constraints:
        value = expression(rule)
        if value is False:
            compiled += 1
            continue
        if value is True:
            nogoods.add((0, 0))
        elif type(value) is int:
            nogoods.add(bits((value,)))
        elif value is not None and value[0] == "and" and all(type(t) is int for t in value[1]):
            nogoods.add(bits(value[1]))
        elif value is not None and value[0] == "bound":
            _, terms, lower, upper = value
            positive, negative = bits(terms)
            bounds.add((positive, negative, upper + 1 if upper is not None else None,
                        lower - 1 if lower is not None else None))
        else:
            skipped += 1
            continue
        compiled += 1
    # A stronger, separately checked certificate permits deleting literals
    # that occur only monotonically in legality constraints. Every body input
    # must be freely supported and the complete auxiliary program acyclic;
    # otherwise stable-model existence itself could change on deletion.
    deletion_monotone, deletion_checks = 0, ()
    context_inputs = set(inputs) | set(fixed_atoms)
    free, forced, acyclic = set(), set(), True
    dependencies = {}
    if observer.external_atoms:
        acyclic = False
    for choice, head, body in observer.rules:
        if choice:
            if any(atom not in context_inputs for atom in head):
                acyclic = False
            if expression(("and", body)) is True:
                free.update(head)
        elif len(head) > 1:
            acyclic = False
        elif head:
            if head[0] in context_inputs:
                if body:
                    acyclic = False
                else:
                    forced.add(head[0])
                    free.add(head[0])
            else:
                dependencies.setdefault(head[0], set()).update(abs(x) for x in body if abs(x) not in context_inputs)
    for choice, head, _, body in observer.weights:
        if head:
            if choice or len(head) != 1 or head[0] in context_inputs:
                acyclic = False
            else:
                dependencies.setdefault(head[0], set()).update(abs(x) for x, _ in body if abs(x) not in context_inputs)
    done, pending = set(), set()
    def no_cycle(atom):
        if atom in pending:
            return False
        if atom in done:
            return True
        pending.add(atom)
        if not all(no_cycle(other) for other in dependencies.get(atom, ())):
            return False
        pending.remove(atom)
        done.add(atom)
        return True
    required_free = set(inputs) | {atom for atom, value in fixed_atoms.items() if value}
    if not skipped and acyclic and required_free <= free and all(no_cycle(a) for a in dependencies):
        protected = sum(1 << (inputs[a] - 1) for a in forced if a in inputs)
        forced_bits = protected
        for _, negative in nogoods:
            protected |= negative
        for positive, negative, lower, upper in bounds:
            if lower is not None:
                protected |= positive
            if upper is not None:
                protected |= negative
        deletion_monotone = ((1 << len(body_atoms)) - 1) & ~protected
        checks = []
        for i in range(len(body_atoms)):
            bit = 1 << i
            if forced_bits & bit:
                checks.append(None)
                continue
            affected_nogoods = tuple((p, n) for p, n in nogoods if n & bit and not p & bit)
            affected_bounds = tuple((p, n, lo if p & bit else None, hi if n & bit else None)
                                    for p, n, lo, hi in bounds if (p ^ n) & bit and
                                    ((p & bit and lo is not None) or (n & bit and hi is not None)))
            checks.append((affected_nogoods, affected_bounds))
        deletion_checks = tuple(checks)
    incompatible = [0] * len(body_atoms)
    for positive, negative in nogoods:
        if not negative and positive.bit_count() == 2:
            for i in range(len(body_atoms)):
                if positive >> i & 1:
                    incompatible[i] |= positive & ~(1 << i)
    for positive, negative, lower, upper in bounds:
        if not negative and upper == 1:
            for i in range(len(body_atoms)):
                if positive >> i & 1:
                    incompatible[i] |= positive & ~(1 << i)
    # Proved guard closures already enforce simple implications. Keeping them
    # in each prefix check would repeat O(n * max_body) Python operations.
    if guards is not None:
        nogoods = {(p, n) for p, n in nogoods if not
                   (p.bit_count() == 1 and n and n & sum(1 << i for i in guards[p.bit_length() - 1]))}
    return PrefixFilter(tuple(sorted(nogoods)), tuple(sorted(bounds, key=repr)), tuple(incompatible),
                        skipped == 0, compiled, skipped, deletion_monotone, deletion_checks)


class _HeadSolver:
    """One grounded copy of the exact legality problem for one head.

    Instances are local, sequential solver state; do not pickle or share across
    concurrent threads. Anonymous external gates avoid collisions with names in
    user meta-programs and are permanently released after each proof query.
    """
    def __init__(self, task, head, diagnostics, *, core_pruning=False):
        self.task = task
        self.head = head
        self.control = clingo.Control(["1"], logger=lambda _, text: diagnostics.append(text.strip()))
        self.ground_observer = _GroundObserver()
        self.control.register_observer(self.ground_observer)
        self.stats = {"solves": 0, "models": 0, "unsat_queries": 0, "solve_seconds": 0.0,
                      "legality_queries": 0, "completion_queries": 0, "cache_hits": 0}
        self.cache = OrderedDict()
        self.cores = None
        if core_pruning:
            from .bias_cores import BiasCores
            self.cores = BiasCores(self.stats, max_body=task.max_body)
        constraints = (f":- not head({head}).\n:- head(H), H != {head}.\n"
                       ":- body(L), not allowed_literal(L).\n")
        try:
            self.control.add("base", [], task.bias_source + "\n" + constraints)
            self.control.ground([("base", [])])
            self.body_atoms = []
            for literal in task.literals:
                atom = self.control.symbolic_atoms[clingo.Function("body", [literal])]
                self.body_atoms.append(None if atom is None else atom.literal)
            # Incremental solves may later clean up forced-false head atoms.
            # Retain the complete mapping from the original grounding.
            self.head_atoms = {atom.literal: atom.symbol.arguments[0] == head
                               for atom in self.control.symbolic_atoms.by_signature("head", 1)}
            with self.control.backend() as backend:
                atoms = [atom for atom in self.body_atoms if atom is not None]
                backend.add_weight_rule([], task.max_body + 1, [(atom, 1) for atom in atoms])
                for slot in variables(head):
                    occurrences = [atom for atom, literal in zip(self.body_atoms, task.literals)
                                   if atom is not None and slot in variables(literal)]
                    backend.add_rule([], [-atom for atom in occurrences])
            self.ground_observer.enabled = False
        except RuntimeError as error:
            raise TaskError(f"Cannot ground actual bias for dependency analysis: {error}") from error

    def find(self, included=(), excluded=(), *, omit_some=(), check=None):
        """Return a selected-body witness or None after proven UNSAT."""
        if check:
            check()
        included, excluded = frozenset(included), frozenset(excluded)
        if any(type(i) is not int or i < 0 or i >= len(self.body_atoms) for i in included | excluded):
            return None
        if included & excluded or len(included) > self.task.max_body:
            return None
        if any(self.body_atoms[i] is None for i in included):
            return None
        assumptions = [self.body_atoms[i] for i in sorted(included)]
        present = tuple(assumptions)
        assumptions.extend(-self.body_atoms[i] for i in sorted(excluded) if self.body_atoms[i] is not None)
        bank = self.cores if not omit_some else None
        signed = frozenset(assumptions) if bank is not None else None
        if bank is not None and bank.match(present, signed):
            return None
        gate = None
        if omit_some:
            with self.control.backend() as backend:
                gate = backend.add_atom()
                backend.add_external(gate, clingo.TruthValue.Free)
                backend.add_rule([], [gate, *(self.body_atoms[i] for i in sorted(omit_some))])
            assumptions.append(gate)
        found, cores = [], []
        def on_model(model):
            found.append(frozenset(i for i, atom in enumerate(self.body_atoms)
                                   if atom is not None and model.is_true(atom)))
        started = time.monotonic()
        self.stats["solves"] += 1
        try:
            if check:
                with self.control.solve(assumptions=assumptions, on_model=on_model, async_=True) as handle:
                    while not handle.wait(0.01):
                        check()
                    result = handle.get()
                    if bank is not None and result.unsatisfiable and result.exhausted and not found:
                        cores.append(tuple(handle.core()))
            else:
                options = {'on_core': lambda core: cores.append(tuple(core))} if bank is not None else {}
                result = self.control.solve(assumptions=assumptions, on_model=on_model, **options)
        except ResourceLimit:
            raise
        except RuntimeError as error:
            raise TaskError(f"Actual bias query failed: {error}") from error
        finally:
            self.stats["solve_seconds"] += time.monotonic() - started
            if gate is not None:
                self.control.release_external(gate)
        if check:
            check()
        if result.unknown:
            raise TaskError("Actual bias query returned UNKNOWN")
        if result.unsatisfiable and result.exhausted and not found:
            self.stats["unsat_queries"] += 1
            if bank is not None:
                for core in cores:
                    bank.store(core, signed)
            return None
        if not result.satisfiable or len(found) != 1:
            raise TaskError(f"Unexpected bias query result: {result}; models={len(found)}")
        selected, = found
        if not included <= selected or selected & excluded:
            raise AssertionError("Bias witness violates assumptions")
        if omit_some and set(omit_some) <= selected:
            raise AssertionError("Bias witness does not refute a possible guard")
        self.stats["models"] += 1
        return selected

    def completion(self, included, excluded, check=None):
        if check:
            check()
        key = frozenset(included), frozenset(excluded)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        answer = self.find(*key, check=check) is not None
        self.cache[key] = answer
        self.cache.move_to_end(key)
        if len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return answer


@dataclass(frozen=True)
class HeadAnalysis:
    head: clingo.Symbol
    guards: tuple[frozenset[int], ...]
    unusable_literals: frozenset[int]
    legal_body_exists: bool
    # Private after expanding each literal by its obligatory guards. Repeated
    # occurrences within one relation retain their equalities during projection.
    private_variables: dict[int, int]
    _checker: _HeadSolver = field(repr=False, compare=False)
    prefix_filter: PrefixFilter | None = field(default=None, repr=False, compare=False)

    @property
    def checker_stats(self):
        return self._checker.stats

    def accepts(self, body, *, check=None):
        """Exact complete-body legality using retained grounding and assumptions."""
        self._checker.stats["legality_queries"] += 1
        body = tuple(body)
        if (len(body) > self._checker.task.max_body or len(body) != len(set(body))
                or any(type(i) is not int or i < 0 or i >= len(self.guards) for i in body)):
            return False
        selected = frozenset(body)
        return self._checker.completion(selected, frozenset(range(len(self.guards))) - selected, check)

    def possible_completion(self, included=(), excluded=(), *, check=None):
        """Whether any legal body contains included and omits excluded literals."""
        self._checker.stats["completion_queries"] += 1
        return self._checker.completion(included, excluded, check)


@dataclass(frozen=True)
class BiasAnalysis:
    heads: dict[clingo.Symbol, HeadAnalysis]
    stats: dict

    def export(self):
        return {str(head): dict(stats=dict(proof.checker_stats),
                    body_atoms=list(proof._checker.body_atoms),
                    core_pruning=proof._checker.cores is not None,
                    cores=proof._checker.cores.export() if proof._checker.cores is not None else None)
                for head, proof in self.heads.items()}

    def accepts(self, head, body, *, check=None):
        return head in self.heads and self.heads[head].accepts(body, check=check)

    def possible_completion(self, head, included=(), excluded=(), *, check=None):
        return head in self.heads and self.heads[head].possible_completion(included, excluded, check=check)


def _private_variables(task, head, guards, unusable):
    occurrences = {}
    head_variables = variables(head)
    for index in range(len(task.literals)):
        if index in unusable:
            continue
        effective = {index} | set(guards[index])
        slots = set().union(*(variables(task.literals[i]) for i in effective)) - head_variables
        for slot in slots:
            occurrences.setdefault(slot, []).append(index)
    return {slot: indices[0] for slot, indices in sorted(occurrences.items()) if len(indices) == 1}


def analyze(task: Task, *, seconds: float = 30,
            check: Callable[[], None] | None = None, core_pruning=None) -> BiasAnalysis:
    """Return complete guard proofs and reusable exact legality checkers.

    Guards exclude their antecedent and are already transitively closed.
    Unusable literals have empty guards, not vacuous implications to everything;
    consult unusable_literals before treating them as usable. ResourceLimit or
    UNKNOWN never returns partial proofs as a complete analysis.

    Analysis reads only the actual bias/templates, never BK or labels. Its
    deadline governs construction/proofs; retained checkers use the optional
    current ``check`` passed to their methods, not this initial deadline.
    core_pruning=None follows the task's compiler runtime choice (off unless
    configured); an explicit boolean overrides it for this fresh analysis.
    """
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Bias analysis requires nonnegative finite seconds")
    if core_pruning is None:
        core_pruning = getattr(task, '_bias_core_pruning', False)
    if type(core_pruning) is not bool:
        raise ValueError("core_pruning must be a boolean")
    started = time.monotonic()
    deadline = started + seconds
    stats = {"method": "stable_model_intersection_and_unsat_proof",
             "head_count": len(task.heads), "literal_count": len(task.literals),
             "solves": 0, "models": 0, "unsat_queries": 0,
             "ground_seconds": 0.0, "solve_seconds": 0.0,
             "proven_guard_edges": 0, "unusable_literal_instances": 0,
             "complete": False, "diagnostics": []}
    def checked():
        stats["elapsed_seconds"] = time.monotonic() - started
        if check:
            check()
        if time.monotonic() >= deadline:
            raise ResourceLimit("Bias dependency analysis time budget reached", dict(stats))
    checked()
    results = {}
    for head in task.heads:
        checked()
        stats["current_head"] = str(head)
        ground_started = time.monotonic()
        solver = _HeadSolver(task, head, stats["diagnostics"], core_pruning=core_pruning)
        stats["ground_seconds"] += time.monotonic() - ground_started
        possible_guards: list[set[int] | None] = [None] * len(task.literals)
        def witness(index=None, omit_some=()):
            selected = solver.find(() if index is None else (index,), omit_some=omit_some, check=checked)
            if selected is not None:
                # Each model simultaneously refines candidates for every
                # antecedent selected in that model, including later ones.
                for item in selected:
                    candidates = set(selected - {item})
                    if possible_guards[item] is None:
                        possible_guards[item] = candidates
                    else:
                        possible_guards[item].intersection_update(candidates)
            return selected
        first = witness()
        unusable = set()
        proven = [frozenset() for _ in task.literals]
        if first is None:
            unusable.update(range(len(task.literals)))
        else:
            for index in range(len(task.literals)):
                checked()
                if possible_guards[index] is None and witness(index) is None:
                    unusable.add(index)
                    continue
                candidates = possible_guards[index]
                while candidates:
                    before = frozenset(candidates)
                    if witness(index, before) is None:
                        break
                    if not candidates < before:
                        raise AssertionError("Refuting a guard must strictly shrink the candidate set")
                proven[index] = frozenset(candidates)
        for index, guards in enumerate(proven):
            if index not in unusable and any(not (proven[g] - {index}) <= guards for g in guards):
                raise AssertionError("Proven guard relation is not transitively closed")
        results[head] = HeadAnalysis(head, tuple(proven), frozenset(unusable), first is not None,
                                     _private_variables(task, head, proven, unusable), solver,
                                     _prefix_filter(solver.ground_observer, solver.body_atoms, proven,
                                                    solver.head_atoms))
        for key in ("solves", "models", "unsat_queries", "solve_seconds"):
            stats[key] += solver.stats[key]
        stats["proven_guard_edges"] += sum(map(len, proven))
        stats["unusable_literal_instances"] += len(unusable)
    checked()
    stats["complete"] = True
    return BiasAnalysis(results, stats)
