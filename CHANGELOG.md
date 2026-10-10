# Changelog

All notable changes to Nidavellir are documented in this file. The project
follows semantic versioning from its first standalone release.

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
