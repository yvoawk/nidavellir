# Changelog

All notable changes to Nidavellir are documented in this file. The project
follows semantic versioning from its first standalone release.

## [0.3.28] - 2026-10-16

### Added

- Add `--proof-strategy auto|catalogue|cegis`. Eligible strict relational runs
  now seek a program cheaper than an independently verified incumbent and
  tighten its cost after each verified improvement. Only exhausted UNSAT proves
  optimality; direct proof publishes `program_proof.json` without claiming a
  complete catalogue.
- Revise verified global and per-target bounds during relational compilation.
  Preserve equality and optimal variants, retain improved incumbents after a
  resource limit, and allow immediate transfer to direct program proof. The CLI
  enables this by default; `--no-dynamic-program-bounds` disables it. The Python
  catalogue API remains opt-in with `dynamic_program_bound=True`.
- Add `--no-time-limit` for explicit relational execution without optimum or
  variant enumeration. Oracle, compilation, proof, and selection deadlines
  become unbounded; initial bound discovery retains its finite heuristic budget.

### Fixed

- Certify empty target programs from the validated fixed BK without repeatedly
  grounding that same BK for declarations with no positive examples. Explicit
  `--target-signature predicate/arity` continues to isolate one target while
  preserving its complete rule-level bias.
- Recover a verified global incumbent from complete per-target certificates
  when the discovery worker times out before publishing its global checkpoint.
- Compile supported grounded bias programs into an exact legality evaluator
  and reuse witnessed legal completions. Unsupported ground constructs retain
  the ASP solver fallback; no predicate-specific bias patterns are introduced.
- Re-read decreasing bounds during recursive compilation and retire previously
  retained components excluded by the final verified bound before targeted
  variant reconstruction.

### Preserved

- Retain the catalogue route for bitmask compilation, partial coverage, optimum
  enumeration, variant expansion, and ILASP selection or verification.
- Keep independently verified feasibility distinct from proved optimality.
  Timeouts retain available incumbents without turning UNKNOWN into UNSAT.
- Add differential tests for direct program proofs, dynamic bounds, grounded
  bias legality, and target isolation. Runtime improvements require separate
  benchmark evidence; no universal speedup is claimed for this release.

### Validation

- The complete 572-test suite passes, including the optional ILASP integration
  checks in the validation environment. The built wheel was installed separately
  and all 43 packaged Python modules match the validated source.
- The archived AKI workload retains its automatic bitmask route. Across 24
  sequential runs (two targets, representative and complete-variant modes,
  two versions, three repetitions), median wall-time changes versus 0.3.27
  range from -0.50% to +0.13%. Proven costs, representative predictions and the
  complete optimal program sets are unchanged. These are local descriptive
  measurements, not a general performance guarantee.
- A supplementary forced-relational termination comparison remains unresolved:
  the old-version process received SIGKILL and the new version reached its
  compilation limit without an optimum. These are unfinished searches, not
  comparable completed runtimes; the expanded 36-run protocol was not completed.

## [0.3.27] - 2026-10-12

### Added

- Add opt-in `--allow-partial-coverage` and the corresponding Python selection
  API option. Maximize covered positives, then minimize rule cost, while keeping
  every supplied negative strict. Report coverage and cost proofs separately in
  `coverage.json`, including the exact uncoverable positive labels.
- Support this objective with both compiler backends, both optimizers,
  separate/grouped targets, and optimum/variant enumeration. Keep the default
  full-coverage objective and reject incompatible resumed full-cover bounds.
- Include a domain-independent partial-coverage example and differential tests
  against exhaustive rule/program enumeration and independent Clingo evaluation.

### Performance

- Encode individual JSON records with the standard encoder while retaining
  lazy arrays, bounded buffering, deadlines, and atomic output replacement.
- Compile supported conjunctions of ground cardinality conditions into bitmask
  prefix checks. Preserve exact ASP legality checks for completed candidates and
  the generic fallback for unsupported constraints.
- Restrict hidden-variable domains using proved functional carriers and bias
  coverage obligations. Prefer head-dependent carriers when sufficient, preserve
  shared-variable correlations, and enumerate small assignment products through
  grouped relation lookups. Retain projected solving for larger products.

### Validation

- The controlled 20-fold campaign measured a mean wall-time reduction from
  158.62 s to 28.43 s (5.58x) against the frozen reference, with identical retained
  catalogues, learned programs, and train/test predictions on every fold.
  These are single-run descriptive measurements on one shared host, not a
  general speedup guarantee.
- The complete 516-test regression suite passes, including optional ILASP
  integration checks in the validation environment.

### Packaging

- Exclude local research notes and experiment artifacts from source
  distributions, matching the repository's Git exclusions.

## [0.3.26] - 2026-10-11

### Fixed

- Use Clingo core-guided `usc,oll` optimization for internal weighted set cover.
  Keep `--internal-opt-strategy bb,lin` for reproducing the previous strategy and
  record the selected strategy in solver statistics. The catalogue and objective
  are unchanged.
- Preserve the verified greedy cover on internal or ILASP selection timeout when
  no better solver incumbent exists. The CLI also retains a cheaper initial
  bound program. Feasibility never becomes an optimality claim after timeout.
- Retain the objective during fixed-cost optimum enumeration and use core-guided
  `optN`. Preliminary optimization models are excluded from enumeration counts
  and limits; complete enumeration still requires solver exhaustion.

## [0.3.25] - 2026-09-14

### Added

- First standalone Nidavellir package using a modern `src/` layout.
- Installable `nidavellir` console command and `python -m nidavellir` entry point.
- Domain-independent overlapping-rule and temporal sensor examples.
- Architecture, input-contract, pipeline, semantics, installation, example,
  and development documentation.
- Full compiler regression suite adapted to the standalone package namespace.

### Preserved

- Exact bitmask and relational compilation backends.
- Minimum-cost multi-rule weighted set-cover selection.
- Independent rule verification, optional ILASP selection verification,
  progressive bound search, target isolation, checkpoints, and explicit
  timeout/completeness states from compiler version 0.3.25.
