# Installation and dependencies

## Runtime dependencies

Nidavellir has a deliberately small runtime dependency set:

| Dependency | Supported version | Purpose | Required |
| --- | --- | --- | --- |
| Python | 3.10 or newer | Package runtime, CLI, multiprocessing, and tests | Yes |
| `clingo` Python package | `>=5.7,<6` | ASP parsing, grounding, model computation, exact checks, and internal optimization | Yes |
| ILASP executable | Version 4 command-line interface | Optional selection backend or cross-check of compiled selection | No |

`pip install .` installs the Python Clingo binding automatically. A separate
system `clingo` executable is not used by the compiler. No scientific Python
stack, database, or domain-specific package is required.

The compiler is currently supported and tested on POSIX systems. Process-group
termination is used to enforce hard subprocess timeouts. Linux and macOS are
the intended platforms for this release; native Windows behavior has not been
validated.

## Recommended Conda environment

From a checkout of the repository:

```bash
conda create --name nidavellir python=3.10 -y
conda activate nidavellir
python -m pip install --upgrade pip
python -m pip install .
```

Verify the installed distribution and both entry points:

```bash
python -m pip show nidavellir clingo
nidavellir --version
python -m nidavellir --version
```

The console command and module command expose the same parser and behavior.

## Editable development installation

```bash
conda activate nidavellir
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Standard-library `venv` alternative

Conda is the recommended setup for this project. If Conda is unavailable, an
isolated standard-library environment is also supported:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The test suite uses only the Python standard library plus the runtime
dependencies. `pytest` is not required, although it can discover the same
`unittest` tests if installed separately.

## Optional ILASP integration

Nidavellir searches for `ILASP` on `PATH`. It also checks next to the active
Python executable. A specific binary can be supplied explicitly:

```bash
nidavellir ... --verify-ilasp --ilasp-binary /path/to/ILASP
```

or used as the primary selector:

```bash
nidavellir ... --optimizer ilasp --ilasp-binary /path/to/ILASP
```

`--verify-ilasp` applies only when the primary optimizer is `internal`. ILASP
receives the finite weighted selection problem emitted in
`compiled_ilasp.las`; it does not regenerate candidate rules from the original
bias.

## Build and publication tools

These tools are not runtime dependencies. Install them only when preparing a
distribution:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Before publishing, inspect both the wheel and source archive, run the installed
wheel from a fresh environment, and execute the validation checklist in
[Development](development.md).
