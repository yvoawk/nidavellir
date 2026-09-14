# Development

## Environment

Create an isolated editable installation:

```bash
conda create --name nidavellir python=3.10 -y
conda activate nidavellir
python -m pip install --upgrade pip
python -m pip install -e .
```

Required development-time runtime dependencies are the same as production:
Python 3.10+ and `clingo>=5.7,<6`. The tests use `unittest` from the standard
library. ILASP 4 is optional but required to exercise the real ILASP integration
tests. `build` and `twine` are optional release tools, not compiler dependencies.

## Test suite

Run all tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run a focused module while iterating:

```bash
python -m unittest tests.test_task -v
python -m unittest tests.test_cli -v
```

The packaging tests expect Nidavellir to be installed in the active environment.
They execute the console script and `python -m nidavellir` from an unrelated
temporary working directory.

## Semantic invariants

Changes to the compiler must preserve all of the following:

- No candidate reaches selection without exact bias legality, correlated
  positive coverage, and all-negative safety.
- A timeout, interruption, UNKNOWN result, or resource limit is never reported
  as UNSAT or a completed catalogue.
- A verified feasible hypothesis is not described as optimal without exhausted
  compilation and selection evidence.
- Program-bound pruning keeps the equality case and uses only verified bounds.
- Atomic literal masks are upper bounds for conjunctions, not proofs of a joint
  witness.
- Unlabelled target atoms remain unconstrained.
- Multi-rule costs are additive, and overlapping rules are not merged.
- Exact duplicates, alpha-equivalent spellings, semantic candidate signatures,
  and complete optimal program enumeration remain separate concepts.
- Optimizations derive only from the actual BK, bias, labels, or verified
  certificates; they must not encode application-specific predicate knowledge.

Add differential tests against a simpler exact backend whenever optimizing a
join, projection, cache, dominance rule, or prefix filter.

## Documentation checks

Before release:

1. Compare `nidavellir --help` with every documented option.
2. Run both repository examples exactly as shown in the README.
3. Inspect `status.json`, `learned.lp`, and `verification.json` for each run.
4. Run the installed command from a directory outside the checkout.
5. Run the complete test suite and record skipped optional integrations.
6. Search source, tests, examples, and documentation for stale package names or
   domain-specific assumptions.

## Building distributions

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Install the wheel, not the checkout, in a second clean environment and repeat
the console/module smoke tests. `MANIFEST.in` includes documentation, examples,
and tests in the source distribution; the wheel contains the runtime package.

## Change scope

Keep feature work focused. Avoid opportunistic refactors during semantic
changes. Update `CHANGELOG.md` for user-visible behavior and version
`src/nidavellir/__init__.py` together with `pyproject.toml`.
