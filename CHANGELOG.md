# Changelog

All notable changes to Nidavellir are documented in this file. The project
follows semantic versioning from its first standalone release.

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
