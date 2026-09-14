"""Validated, domain-independent inputs for positive-rule compilation.

Learned rules are positive and independent of other learned targets. The
background is normal and acyclic; default negation therefore stays stratified.
Negation and aggregates in the *meta-language bias* are structural constraints,
not learned body literals, and are deliberately permitted.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping

import clingo
import clingo.ast as ast

from .labels import (CompactNegativeSet, NegativeBuilder, NeedsNormalization,
                     canonical_text, parse_atom)


class TaskError(ValueError):
    """An input violates the compiler's supported semantic contract."""


def variables(term: clingo.Symbol) -> set[int]:
    if term.type != clingo.SymbolType.Function:
        return set()
    if term.name == "var__":
        if (not term.positive or len(term.arguments) != 1
                or term.arguments[0].type != clingo.SymbolType.Number
                or term.arguments[0].number < 1):
            raise TaskError("A meta variable must have the form var__(positive_integer)")
        return {term.arguments[0].number}
    return set().union(*(variables(arg) for arg in term.arguments)) if term.arguments else set()


def instantiate(term: clingo.Symbol, binding: Mapping[int, clingo.Symbol]) -> clingo.Symbol:
    if term.type != clingo.SymbolType.Function:
        return term
    if term.name == "var__":
        key, = variables(term)
        return binding[key]
    return clingo.Function(term.name, [instantiate(arg, binding) for arg in term.arguments], term.positive)


def asp_text(term: clingo.Symbol) -> str:
    if term.type != clingo.SymbolType.Function:
        return str(term)
    if term.name == "var__":
        key, = variables(term)
        return f"V{key}"
    if not term.arguments:
        return str(term)
    return ("" if term.positive else "-") + term.name + "(" + ",".join(map(asp_text, term.arguments)) + ")"


def signature(symbol: clingo.Symbol) -> tuple[str, int]:
    if symbol.type != clingo.SymbolType.Function or not symbol.name or not symbol.positive:
        raise TaskError(f"Expected a positive predicate atom, got {symbol}")
    return symbol.name, len(symbol.arguments)


def _walk(node):
    yield node
    for key in node.child_keys:
        value = getattr(node, key)
        if isinstance(value, ast.AST):
            yield from _walk(value)
        elif value is not None:
            for child in value:
                if isinstance(child, ast.AST):
                    yield from _walk(child)


def _atom_signature(node) -> tuple[str, int]:
    """Read only the outer predicate, never constructor terms in arguments."""
    if node.ast_type != ast.ASTType.SymbolicAtom:
        raise TaskError(f"Expected a symbolic predicate atom: {node}")
    term = node.symbol
    if term.ast_type == ast.ASTType.Function and not term.external and term.name:
        return term.name, len(term.arguments)
    if term.ast_type == ast.ASTType.SymbolicTerm:
        return signature(term.symbol)
    raise TaskError(f"Unsupported or negative predicate atom: {node}")


def _head_atoms(node):
    """Head conditions read metadata; they do not define metadata predicates."""
    if node.ast_type == ast.ASTType.ConditionalLiteral:
        yield from _head_atoms(node.literal)
    elif node.ast_type == ast.ASTType.SymbolicAtom:
        yield node
    else:
        for key in node.child_keys:
            value = getattr(node, key)
            if isinstance(value, ast.AST):
                yield from _head_atoms(value)
            elif value is not None:
                for child in value:
                    if isinstance(child, ast.AST):
                        yield from _head_atoms(child)


def _parse(path: Path) -> tuple[list, list[str]]:
    nodes, diagnostics = [], []
    try:
        # Clingo resolves nested #include relative to the including file.
        ast.parse_files([str(path.resolve())], nodes.append,
                        logger=lambda _code, message: diagnostics.append(message.strip()))
    except (RuntimeError, OSError) as error:
        detail = "; ".join(diagnostics) or str(error)
        raise TaskError(f"Cannot parse {path}: {detail}") from error
    for node in nodes:
        for sub in _walk(node):
            if sub.ast_type == ast.ASTType.Function and sub.external:
                raise TaskError(f"External functions are not supported: {sub}")
    # A pool such as individual(a;b) denotes two facts, not a disjunctive
    # choice. Native AST unpooling preserves Clingo's Cartesian semantics for
    # pools in heads and bodies; actual disjunction remains a Disjunction AST
    # and is still rejected by the object-language contract below.
    expanded = []
    for node in nodes:
        expanded.extend(node.unpool() if node.ast_type == ast.ASTType.Rule else [node])
    return expanded, diagnostics


def _common_statement(node, label: str) -> bool:
    """Return true for retained declarations; skip harmless display metadata."""
    if node.ast_type == ast.ASTType.Program:
        if node.name != "base" or node.parameters:
            raise TaskError(f"{label}: only the parameter-free base program is supported")
        return False
    if node.ast_type in (ast.ASTType.Comment, ast.ASTType.ShowSignature, ast.ASTType.ShowTerm):
        return False
    if node.ast_type == ast.ASTType.Definition:
        return True
    if node.ast_type != ast.ASTType.Rule:
        raise TaskError(f"{label}: unsupported directive {node}")
    return True


def _model(source: str, label: str, diagnostics: list[str]) -> tuple[frozenset, clingo.Control]:
    ctl = clingo.Control(["2"], logger=lambda _code, message: diagnostics.append(message.strip()))
    try:
        # We consume all true atoms, never the displayed model. Explicitly
        # disable implicit output: constructing a display for a million fixed
        # BK atoms can dominate solve time despite a trivial unique model.
        ctl.add("base", [], source + "\n#show.\n")
        ctl.ground([("base", [])])
        models = []
        result = ctl.solve(on_model=lambda model: models.append(frozenset(model.symbols(atoms=True))))
    except RuntimeError as error:
        detail = "; ".join(diagnostics[-8:]) or str(error)
        raise TaskError(f"{label}: grounding/solving failed: {detail}") from error
    if not result.satisfiable or result.unknown:
        raise TaskError(f"{label}: expected a stable model, got {result}")
    if len(models) != 1:
        raise TaskError(f"{label}: expected one fixed background interpretation")
    return models[0], ctl


def _background(path: Path, targets: set[tuple[str, int]], *, evaluate=True):
    nodes, diagnostics = _parse(path)
    statements, dependencies = [], {}
    for node in nodes:
        if not _common_statement(node, "BK"):
            continue
        statements.append(str(node))
        if node.ast_type != ast.ASTType.Rule:
            continue
        head = node.head
        if head.ast_type != ast.ASTType.Literal or head.sign != ast.Sign.NoSign:
            raise TaskError(f"BK: only positive normal rules are supported: {node}")
        if head.atom.ast_type == ast.ASTType.BooleanConstant:
            head_signature = None  # Positive integrity constraints may reject the unique model.
        else:
            head_signature = _atom_signature(head.atom)
        body_signatures = []
        for literal in node.body:
            if literal.ast_type != ast.ASTType.Literal or literal.sign not in (ast.Sign.NoSign, ast.Sign.Negation):
                raise TaskError(f"BK: double negation, conditional literals and aggregates are unsupported: {node}")
            if literal.atom.ast_type == ast.ASTType.SymbolicAtom:
                body_signatures.append(_atom_signature(literal.atom))
            elif literal.atom.ast_type not in (ast.ASTType.Comparison, ast.ASTType.BooleanConstant):
                raise TaskError(f"BK: unsupported body expression: {literal}")
        occurrences = set(body_signatures)
        if head_signature is not None:
            occurrences.add(head_signature)
            dependencies.setdefault(head_signature, set()).update(body_signatures)
        forbidden = occurrences & targets
        if forbidden:
            raise TaskError(f"BK contains target predicate signature(s): {sorted(forbidden)}")
    visiting, done = set(), set()
    def visit(predicate):
        if predicate in visiting:
            raise TaskError(f"BK recursion is unsupported (cycle through {predicate[0]}/{predicate[1]})")
        if predicate in done:
            return
        visiting.add(predicate)
        for other in dependencies.get(predicate, ()):
            visit(other)
        visiting.remove(predicate)
        done.add(predicate)
    for predicate in dependencies:
        visit(predicate)
    source = "\n".join(statements) + "\n"
    atoms = _model(source, "BK", diagnostics)[0] if evaluate else frozenset()
    return source, atoms, diagnostics


def _bias(path: Path):
    nodes, diagnostics = _parse(path)
    statements = []
    metadata = {("candidate_head", 1), ("allowed_literal", 1), ("max_body", 1)}
    for node in nodes:
        if not _common_statement(node, "Bias"):
            continue
        statements.append(str(node))
        if node.ast_type != ast.ASTType.Rule:
            continue
        metadata_heads = [sub for sub in _head_atoms(node.head) if _atom_signature(sub) in metadata]
        if metadata_heads:
            if (node.body or node.head.ast_type != ast.ASTType.Literal
                    or node.head.sign != ast.Sign.NoSign
                    or node.head.atom.ast_type != ast.ASTType.SymbolicAtom
                    or any(sub.ast_type == ast.ASTType.Variable for sub in _walk(node))):
                raise TaskError("Bias metadata candidate_head/1, allowed_literal/1 and max_body/1 must be ground facts")
    source = "\n".join(statements) + "\n"
    ctl = clingo.Control(["1"], logger=lambda _code, message: diagnostics.append(message.strip()))
    try:
        ctl.add("base", [], source)
        ctl.ground([("base", [])])
        result = ctl.solve()
    except RuntimeError as error:
        raise TaskError(f"Bias grounding/solving failed: {'; '.join(diagnostics[-8:]) or error}") from error
    if not result.satisfiable or result.unknown:
        raise TaskError(f"Bias has no legal model: {result}")
    def facts(name):
        symbols = tuple(ctl.symbolic_atoms.by_signature(name, 1))
        if any(not atom.is_fact for atom in symbols):
            raise TaskError(f"Bias metadata {name}/1 must be fixed facts")
        return tuple(sorted(atom.symbol.arguments[0] for atom in symbols))
    heads, literals, bounds = facts("candidate_head"), facts("allowed_literal"), facts("max_body")
    if not heads:
        raise TaskError("Bias must declare at least one candidate_head/1 fact")
    if len(bounds) != 1 or bounds[0].type != clingo.SymbolType.Number or bounds[0].number < 0:
        raise TaskError("Bias must declare exactly one nonnegative max_body(N) fact")
    for term in (*heads, *literals):
        signature(term)
        if term.name == "var__":
            raise TaskError("var__/1 is reserved for variable placeholders inside predicate arguments")
        variables(term)
    targets = {signature(head) for head in heads}
    if any(signature(literal) in targets for literal in literals):
        raise TaskError("Target predicates cannot appear in candidate bodies, including acyclic cross-target dependencies")
    return source, heads, literals, bounds[0].number, diagnostics


def _examples_legacy(path: Path, targets: set[tuple[str, int]]):
    """Normalize the complete ASP ground-fact syntax, including constants/pools."""
    nodes, diagnostics = _parse(path)
    statements = []
    for node in nodes:
        if not _common_statement(node, "Examples"):
            continue
        statements.append(str(node))
        if node.ast_type != ast.ASTType.Rule:
            continue
        if (node.body or node.head.ast_type != ast.ASTType.Literal
                or node.head.sign != ast.Sign.NoSign
                or node.head.atom.ast_type != ast.ASTType.SymbolicAtom
                or _atom_signature(node.head.atom) not in {("positive", 1), ("negative", 1)}
                or any(sub.ast_type == ast.ASTType.Variable for sub in _walk(node))):
            raise TaskError("Examples must be ground positive(atom). or negative(atom). facts")
    source = "\n".join(statements) + "\n"
    atoms, _ = _model(source, "Examples", diagnostics)
    positives, negatives = set(), set()
    for atom in atoms:
        value, = atom.arguments
        if signature(value) not in targets:
            raise TaskError(f"Example predicate is absent from candidate heads: {value}")
        (positives if atom.name == "positive" else negatives).add(value)
    if positives & negatives:
        raise TaskError(f"Conflicting positive/negative labels: {sorted(positives & negatives)}")
    return source, frozenset(positives), frozenset(negatives), diagnostics


class _ExamplesNeedFallback(Exception):
    """The input needs the next, more general exact parsing path."""


# A deliberately narrow grammar. Clingo still parses the complete inner term;
# anything outside this grammar goes through the ASP parser, never a rejection
# based on formatting alone. Operators, directives, pools and intervals require
# the legacy normalization path. Quoted strings may contain any of those tokens.
_EXPLICIT_LABEL = re.compile(
    r'(positive|negative)\(((?:[a-zA-Z0-9_(), \t]|"(?:[^"\\]|\\.)*")+)\)\.'
)


class _ExampleCollector:
    """Keep only target symbols, without the positive/negative wrapper atoms."""

    def __init__(self, targets):
        self.targets = targets
        self.positives = set()
        self.negatives = NegativeBuilder()
        self.argument_cache = OrderedDict()
        self.error = None

    def add(self, text):
        match = _EXPLICIT_LABEL.fullmatch(text)
        if match is None:
            raise _ExamplesNeedFallback()
        label, term = match.groups()
        if label == "negative":
            try:
                value = parse_atom(term, self.argument_cache)
            except NeedsNormalization:
                raise _ExamplesNeedFallback() from None
            if (value.name, len(value.arguments)) not in self.targets:
                self.error = self.error or TaskError(f"Example predicate is absent from candidate heads: {value}")
            self.negatives.add(value)
            return
        try:
            value = clingo.parse_term(term)
        except RuntimeError:
            # Variables, malformed terms, and other syntax must receive the
            # same validation/diagnostics as the general ASP parser.
            raise _ExamplesNeedFallback() from None
        try:
            if signature(value) not in self.targets:
                raise TaskError(f"Example predicate is absent from candidate heads: {value}")
        except TaskError as error:
            # A later #const can change a zero-arity term's interpretation.
            # Finish the fast scan before declaring a semantic input error;
            # any declaration triggers the normalizer instead.
            self.error = self.error or error
        self.positives.add(value)

    def finish(self, diagnostics):
        if self.error is not None:
            raise self.error
        negatives = self.negatives.freeze()
        self.argument_cache.clear()
        for value in self.positives:
            if value in negatives:
                raise TaskError(f"Conflicting positive/negative labels: [{value}]")
        # Clear each mutable set after freezing, reducing the peak while the
        # other partition is frozen. Symbol values are shared, not duplicated.
        positives = frozenset(self.positives)
        self.positives.clear()
        return None, positives, negatives, diagnostics


def _examples_lines(path: Path, targets):
    """Fast path for explicit facts written one per line, in constant AST space."""
    collector = _ExampleCollector(targets)
    try:
        with path.open() as stream:
            for line in stream:
                text = line.strip()
                if not text or (text.startswith("%") and not text.startswith("%*")):
                    continue
                collector.add(text)
    except OSError as error:
        raise TaskError(f"Cannot parse {path}: {error}") from error
    return collector.finish([])


def _examples_ast_stream(path: Path, targets):
    """Handle includes/comments/layout with one live AST statement at a time."""
    collector = _ExampleCollector(targets)
    diagnostics = []

    def consume(node):
        # Display metadata is ignored by the contract, but external functions
        # inside a #show term remain forbidden just as in _parse().
        if node.ast_type == ast.ASTType.ShowTerm:
            for sub in _walk(node):
                if sub.ast_type == ast.ASTType.Function and sub.external:
                    raise TaskError(f"External functions are not supported: {sub}")
        if not _common_statement(node, "Examples"):
            return
        if node.ast_type != ast.ASTType.Rule:
            raise _ExamplesNeedFallback()
        collector.add(str(node))

    try:
        ast.parse_files([str(path.resolve())], consume,
                        logger=lambda _code, message: diagnostics.append(message.strip()))
    except (RuntimeError, OSError) as error:
        detail = "; ".join(diagnostics) or str(error)
        raise TaskError(f"Cannot parse {path}: {detail}") from error
    return collector.finish(diagnostics)


def _examples(path: Path, targets: set[tuple[str, int]]):
    """Read explicit labels without materializing ASTs, text copies or a model.

    Constants, arithmetic, pools and other supported ground-fact constructions
    retain the existing full Clingo normalization. No syntax is silently lost
    when an input is outside a fast path.
    """
    try:
        return _examples_lines(path, targets)
    except _ExamplesNeedFallback:
        pass
    try:
        return _examples_ast_stream(path, targets)
    except _ExamplesNeedFallback:
        pass
    return _examples_legacy(path, targets)


def read_ground_model(path: Path):
    """Read our canonical ground-model snapshot, without a new ASP solve.

    The parent has already validated and evaluated the original BK. This
    private transfer format contains one canonical positive ground atom per
    line and deliberately accepts no rules, directives or other ASP syntax.
    """
    atoms = set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith(".\n"):
                raise TaskError("Malformed canonical ground-model snapshot")
            text = line[:-2]
            try:
                atom = clingo.parse_term(text)
            except RuntimeError as error:
                raise TaskError("Malformed canonical ground-model atom") from error
            signature(atom)
            if str(atom) != text:
                raise TaskError("Ground-model snapshot is not canonical")
            atoms.add(atom)
    return frozenset(atoms)


@dataclass(eq=False)
class Task:
    background: str
    atoms: frozenset[clingo.Symbol]
    bias_source: str
    examples_source: str | None
    heads: tuple[clingo.Symbol, ...]
    literals: tuple[clingo.Symbol, ...]
    max_body: int
    positives: frozenset[clingo.Symbol]
    negatives: frozenset[clingo.Symbol] | CompactNegativeSet
    diagnostics: dict[str, list[str]] = field(default_factory=dict)
    _legality_cache: dict = field(default_factory=dict, repr=False)
    examples_path: Path | None = None

    @classmethod
    def load(cls, bk: Path, examples: Path, bias: Path) -> "Task":
        bias_source, heads, literals, max_body, bias_messages = _bias(Path(bias))
        targets = {signature(head) for head in heads}
        background, atoms, bk_messages = _background(Path(bk), targets)
        examples_source, positives, negatives, example_messages = _examples(Path(examples), targets)
        return cls(background, atoms, bias_source, examples_source, heads, literals, max_body,
                   positives, negatives, {"background": bk_messages, "bias": bias_messages, "examples": example_messages},
                   examples_path=Path(examples).resolve())

    def write_examples(self, path: Path):
        """Write a self-contained snapshot without a second in-memory source.

        Stream compact negatives in Clingo order without interning outer
        symbols. Ordinary sets sort their handles one polarity at a time;
        normalized inputs retain their existing canonical ASP source.
        """
        if self.examples_source is not None:
            Path(path).write_text(self.examples_source)
            return
        with Path(path).open("w", buffering=1024 * 1024) as stream:
            for name, examples in (("positive", self.positives), ("negative", self.negatives)):
                for value in canonical_text(examples):
                    stream.write(f"{name}({value}).\n")

    def accepts(self, head: clingo.Symbol, body: tuple[int, ...]) -> bool:
        """Check a complete rule against the exact ASP bias and ASP safety."""
        if head not in self.heads or len(set(body)) != len(body) or len(body) > self.max_body:
            return False
        if any(not isinstance(item, int) or item < 0 or item >= len(self.literals) for item in body):
            return False
        key = head, tuple(sorted(body))
        if key in self._legality_cache:
            return self._legality_cache[key]
        used = set().union(*(variables(self.literals[i]) for i in body)) if body else set()
        if not variables(head) <= used:
            self._legality_cache[key] = False
            return False
        selected = set(body)
        constraints = [f":- not head({head}).", f":- head(H), H != {head}.",
                       ":- body(L), not allowed_literal(L)."]
        constraints.extend(f":- {'not ' if i in selected else ''}body({literal})."
                           for i, literal in enumerate(self.literals))
        diagnostics = []
        ctl = clingo.Control(["1"], logger=lambda _code, message: diagnostics.append(message.strip()))
        try:
            ctl.add("base", [], self.bias_source + "\n".join(constraints))
            ctl.ground([("base", [])])
            result = ctl.solve()
        except RuntimeError as error:
            raise TaskError(f"Bias legality check failed: {'; '.join(diagnostics) or error}") from error
        if result.unknown:
            raise TaskError("Bias legality check returned UNKNOWN")
        self._legality_cache[key] = result.satisfiable
        return result.satisfiable

    def rule(self, head: clingo.Symbol, body: tuple[int, ...]) -> str:
        text = asp_text(head)
        return text + (" :- " + ", ".join(asp_text(self.literals[i]) for i in sorted(body)) if body else "") + "."

    def stats(self) -> dict:
        return {"background_atoms": len(self.atoms), "head_templates": len(self.heads),
                "target_signatures": [f"{name}/{arity}" for name, arity in sorted({signature(h) for h in self.heads})],
                "literal_templates": len(self.literals), "max_body_literals": self.max_body,
                "positive_examples": len(self.positives), "negative_examples": len(self.negatives),
                "unlabelled_atoms": "unconstrained", "diagnostics": self.diagnostics,
                "negative_storage": (self.negatives.stats() if isinstance(self.negatives, CompactNegativeSet)
                                     else {"negative_storage": "symbol_frozenset"})}
