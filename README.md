# Nidavellir

**Nidavellir: A Semantic Compiler for Finite Inductive Logic Programming**

Nidavellir compiles a finite, explicitly biased inductive logic programming
(ILP) task into a minimum-cost set of positive rules. It evaluates candidate
rules against a fixed Answer Set Programming (ASP) background, compiles their
exact positive coverage, rejects rules covering supplied negative examples,
and solves the resulting weighted set-cover problem.

The compiler is domain independent. Its three inputs are a background
knowledge file (BK), hard positive and negative examples, and an ASP bias that
defines the finite rule language.

## Requirements

- Python 3.10 or newer.
- The Python package `clingo>=5.7,<6` (installed automatically by `pip`).
- A POSIX environment such as Linux or macOS is currently supported and tested.
- ILASP 4 is optional. It is used only with `--optimizer ilasp` or
  `--verify-ilasp`; the default internal optimizer does not require it.

The standalone `clingo` command is not required. Nidavellir imports Clingo as a
Python module. See [Installation](docs/installation.md) for the complete
dependency and environment matrix.

## Installation

From the repository root:

```bash
conda create --name nidavellir python=3.10 -y
conda activate nidavellir
python -m pip install --upgrade pip
python -m pip install .
```

For editable development:

```bash
conda activate nidavellir
python -m pip install -e .
```

Check both installed entry points:

```bash
nidavellir --version
python -m nidavellir --version
```

## Five-minute example

The output directory must not already exist:

```bash
nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-example
```

The equivalent module invocation is:

```bash
python -m nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-module-example
```

This task needs two overlapping rules to cover all positives without covering a
negative. On a complete run, `learned.lp` contains rules equivalent to:

```prolog
t(V1) :- p(V1).
t(V1) :- q(V1).
```

The rule cost is one for the head plus one per selected body literal, so this
two-rule program has cost four.

## Input contract

Every run requires three ASP files:

1. `background.lp` is the BK. It must define one finite, fixed interpretation
   using facts and supported acyclic normal rules. Target predicates must not
   occur in the BK.
2. `examples.lp` contains ground hard labels of the form `positive(atom).` and
   `negative(atom).`. Unlabelled target atoms are unconstrained; Nidavellir does
   not apply a closed-world assumption to them.
3. `bias.lp` declares ground `candidate_head/1`, `allowed_literal/1`, and exactly
   one nonnegative `max_body/1` fact. It also defines legal selections through
   `head/1` and `body/1` in the ASP meta-language.

`var__(N)`, where `N` is a positive integer, is a compiler placeholder for a
rule variable. It is not a BK constant or an ASP variable. For example:

```prolog
candidate_head(reachable(var__(1))).
allowed_literal(edge(var__(1),var__(2))).
allowed_literal(marked(var__(2))).
max_body(2).

1 { head(H) : candidate_head(H) } 1.
0 { body(L) : allowed_literal(L) } N :- max_body(N).
```

Learned clauses contain only positive body literals selected from
`allowed_literal/1`. The bias itself may use negation and aggregates to express
structural constraints. The full normative contract, including supported ASP
constructs and rejection conditions, is in [Input format](docs/input-format.md).

## Outputs

The requested output path is created as a new run directory and is never
overwritten. Individual JSON artifacts are written atomically. Main artifacts
include:

- `status.json` and `report.md`: final status and human-readable summary;
- `inputs/`: normalized, hashed snapshots of all three inputs;
- `manifest.json`: versions, arguments, hashes, task sizes, and engine hashes;
- `planning.json`, `oracle.json`, and `bound.json`: backend and proof metadata;
- `catalogue.json`: retained negative-safe cost/coverage candidates;
- `compiled_set_cover.lp` and `compiled_ilasp.las`: compiled selection tasks;
- `solver.json`: selection result and its proof/completion state;
- `learned.lp`, `predictions.lp`, and `verification.json`: emitted hypothesis
  and independent evaluation in the original BK;
- `trace.jsonl`: machine-readable progress events.

`first_feasible.lp`, `enumeration.json`, `variant_catalogue.json`, and
`variants.json` are conditional artifacts. Multi-signature tasks in the default
`separate` mode also contain a `targets/` tree with one isolated run per target
signature.

See [Compiler pipeline](docs/compiler-pipeline.md) for the lifecycle and
[Examples](docs/examples.md) for artifact inspection commands.

## Guarantees and limits

When `status.json` reports `"status": "complete"`, Nidavellir has completed the
finite catalogue and selection needed to prove the reported minimum cost under
the supplied BK, labels, and bias. Every emitted hypothesis is independently
checked against all supplied hard labels in the original BK.

Important boundaries:

- A `feasible` result is verified as covering the supplied positives and no
  supplied negatives, but is not a proof of optimality.
- A timeout or resource limit means `incomplete`, never unsatisfiable.
- Catalogue completion does not automatically enumerate every tied optimum or
  every alpha-renamed spelling. Request `--enumerate-optima --expand-variants`
  and inspect their separate completion fields.
- Optional ILASP verification checks minimum-cost selection from Nidavellir's
  compiled finite catalogue. It is not an independent reconstruction of the
  original unrestricted ILASP hypothesis space.
- Guarantees are relative to a finite positive-rule language, a unique acyclic
  BK interpretation, explicit hard labels, and the exact supplied bias.
- Runtime and memory can grow rapidly with the grounded BK, hidden variables,
  join density, number of allowed literals, `max_body`, and number of tied
  programs.

Read [Semantics and guarantees](docs/semantics-and-guarantees.md) before making
completeness or optimality claims.

## Tests

Install the package first, then run the complete standard-library suite:

```bash
python -m unittest discover -s tests -v
```

Tests that exercise real ILASP integration run when an `ILASP` binary is on
`PATH` or next to the active Python executable. Without it, only those optional
integration tests are skipped; the internal compiler and optimizer remain fully
testable.

## Documentation

- [Architecture](docs/architecture.md)
- [Input format](docs/input-format.md)
- [Compiler pipeline](docs/compiler-pipeline.md)
- [Semantics and guarantees](docs/semantics-and-guarantees.md)
- [Installation](docs/installation.md)
- [Examples](docs/examples.md)
- [Development](docs/development.md)

## Contributing

Create a focused branch, install the project in editable mode, add tests for
semantic changes, run the full suite, and update the relevant documentation and
changelog. Preserve the explicit distinction between feasible, complete,
optimal, and timed-out outcomes. See [Development](docs/development.md) for the
required invariants and validation checklist.

## License

Nidavellir is released under the [MIT License](LICENSE).
