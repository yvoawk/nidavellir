# Architecture

Nidavellir separates parsing, semantic compilation, finite optimization, and
independent verification. The separation is intentional: a fast search path
may propose or prune candidates, but only exact checks can publish a rule or a
proof state.

## Package layout

| Area | Main modules | Responsibility |
| --- | --- | --- |
| Public execution | `__main__.py`, `target_execution.py`, `terminal_progress.py` | CLI validation, run directories, isolated target processes, progress, and status artifacts |
| Input model | `task.py`, `labels.py` | Parse and validate BK, examples, bias, variable slots, and compact negative labels |
| Bias reasoning | `bias_analysis.py`, `bias_cores.py` | Legal-body analysis, guards, prefix conditions, and optional reusable UNSAT cores |
| Backend planning | `planning.py`, `reduction.py`, `witness_space.py` | Choose an exact representation and derive task-relative reductions |
| Bitmask compiler | `oracle.py`, `optimized_oracle.py`, `conditional_anchors.py`, `projected_oracle.py`, `search.py` | Materialize or compress witness truth vectors using proved reductions and compile coverage classes |
| Relational compiler | `relational_coverage.py`, `relational_compile.py`, `indexed_join.py`, `positive_join.py`, `negative_join.py` | Query exact correlated witnesses lazily and compile coverage classes |
| Feasible bounds | `bound_worker.py`, `bitmask_bound.py`, `relational_bound.py`, `short_bound.py`, `target_bounds.py` | Search for verified upper bounds without conflating failure or timeout with UNSAT |
| Direct program proof | `program_proof.py` | Refine whole-program proposals below a verified cost bound; prove optimality only after exhausted UNSAT |
| Optimization | `solvers.py` | Internal Clingo weighted set-cover, optional ILASP selection, optimum enumeration, and variant expansion |
| Verification | `verification.py` | Reground original BK plus emitted rules and recheck every hard label |
| Diagnostics | `profiling.py`, `join_profile.py`, `oracle_cache.py`, `positive_cores.py`, `positive_support.py` | Reproducible metrics, complete caches, and sound optional pruning evidence |
| Artifact serialization | `json_stream.py` | Stream lazy JSON arrays with bounded buffering, deadlines, and atomic replacement |

The CLI is the stable user-facing interface in this release. Internal Python
modules are documented for maintainers but should be treated as provisional APIs.

## Core data model

`Task` holds the normalized BK source, its fixed ground model, the original
bias, candidate head and literal templates, `max_body`, and explicit labels.
Templates remain Clingo symbols containing `var__(N)` placeholders. Candidate
bodies use integer indices into the task's literal tuple, avoiding repeated
term parsing during search.

A compiled candidate record contains:

- a stable candidate identifier;
- its head template and selected body indices;
- the rendered ASP rule;
- cost `1 + body length`;
- a bitmask over positive-example positions.

Only negative-safe candidates can enter the selection catalogue.

## Process isolation

Feasible-bound discovery runs in a subprocess with a hard wall-clock budget.
For multi-signature tasks, the default `--target-mode separate` also runs each
predicate/arity signature in a fresh sequential process. This gives every
target its own phase budgets and preserves completed target artifacts if a later
target fails.

The parent records normalized input snapshots, hashes, arguments, engine file
hashes, Python and Clingo versions, checkpoints, and final composition state.
The working directory is never used to locate package modules; installed module
resolution is used by child processes.

## Exactness boundary

Performance mechanisms may change traversal order or provide necessary
conditions, upper bounds, and cache hits. They do not replace these publication
checks:

1. Candidate structure is legal under the exact original bias.
2. Correlated positive coverage is computed against the fixed BK.
3. Negative safety is checked against every supplied negative label.
4. By default, the selected rule set covers every supplied positive. In optional
   partial-coverage mode, a complete result proves maximum positive coverage
   first and minimum cost at that coverage second.
5. Emitted clauses are independently evaluated with the original BK.

Resource-limit exceptions carry partial statistics and any dynamically verified
incumbent, but do not publish an incomplete catalogue as complete. Direct
program proof can establish optimality without a complete catalogue; its
separate certificate records the exhausted strict-improvement search.

## Choice of proof strategy

`--proof-strategy auto` selects direct program proof for strict relational
representative runs with the internal optimizer and without ILASP verification.
`--proof-strategy catalogue` forces the previous compilation/selection route;
`--proof-strategy cegis` requires eligibility for direct proof.

The direct route starts from a verified program of cost `U`. Its finite ASP
abstraction copies the actual grounded bias for each potential rule slot and
admits every legal program costing less than `U`. Exact positive-failure cores
and real negative witnesses refine proposals. Every improving program is
independently evaluated in the original BK before it becomes the new incumbent.
Only exhausted UNSAT of the remaining relaxation proves that none is cheaper.

The catalogue architecture remains useful when its output is needed: it exposes
coverage classes for ILASP selection, supports complete optimum and variant
enumeration, and computes the maximum achievable coverage for the partial
objective. Direct strict-improvement proof produces one optimum and does not
provide these catalogue or enumeration guarantees. These modes therefore keep
the catalogue route, as does the bitmask backend.

If initial bound discovery finds no feasible program, relational compilation
can discover one. Dynamic bound tracking considers cheap covers when coverage
first becomes complete, when a new rule cheaply replaces incumbent components,
and periodically after retained-candidate changes. Only verified improvements
tighten bounds. Eligible direct-proof runs transfer immediately to program
proof using the remaining compilation budget. Catalogue runs retain cost
equality and the variants needed for optimal programs.

## Backend families

### Bitmask compiler

The bitmask path represents ground witness classes by literal truth vectors.
The optimized backend uses proven guards, private-variable reductions, and
functional dependencies derived from the real task. Projected and Cartesian
oracles remain available as reference implementations. Cartesian enumeration
can be prohibitively large and is protected by `--max-witnesses`.

### Relational compiler

The relational path avoids a full Cartesian witness catalogue. It uses atomic
positive masks only as upper bounds, then performs exact correlated joins for
coverage and safety before accepting a rule. Indexed and reference join modes
support differential tests and controlled ablations.

Supported grounded bias programs are compiled into an exact Boolean legality
evaluator with stratified auxiliary definitions. Legal completion witnesses
can also answer compatible prefix queries. Unsupported constructs continue to
use the retained Clingo solver. Necessary prefix filters remain distinct from
these exact legality checks: accepting a necessary condition alone is never a
legality certificate.

### Automatic selection

`--compiler-backend auto` analyzes how many hidden variables remain after
proved reductions. At most two selects bitmask; a larger number selects the
relational backend. This threshold is a performance heuristic only. Both
backends represent the same finite legal rule language.

### Selection backends

On the catalogue route, the default internal selector sends weighted set-cover
to Clingo and records
whether optimization exhausted the search. The optional ILASP selector receives
an equivalent finite coverage problem. `--verify-ilasp` compares ILASP's result
with a completed internal selection, including tied-cover tolerance through
cost and coverage rather than textual rule identity.

## Deadline representation

`--no-time-limit` requires explicit relational execution and excludes optimum
enumeration and variant expansion. It passes `None` for oracle, compilation,
proof, and selection budgets. There is no large numeric deadline sentinel.
The initial bound subprocess still uses the finite `--bound-seconds` heuristic
budget; disabling deadlines does not remove unrelated resource constraints or
turn an interrupted run into an optimum proof.
