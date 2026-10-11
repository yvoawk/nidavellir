# Compiler pipeline

One CLI invocation proceeds through explicit, persisted phases. The current
phase and completion state are always recorded, so a timeout can be interpreted
without guessing what was proved.

## 1. Parse and validate inputs

The bias is parsed first to obtain target signatures and the finite grammar.
The BK is then checked for supported syntax, acyclic dependencies, absence of
targets, satisfiability, and a unique stable model. Finally, ground hard labels
are loaded and checked against declared target signatures.

Normalized snapshots are written under `inputs/`; hashes of original files and
snapshots are stored in `manifest.json`.

## 2. Isolate target signatures

By default, a task containing multiple predicate/arity signatures is decomposed
into fresh sequential subprocesses. Head templates sharing one signature stay
together. Because target predicates cannot occur in the BK or candidate bodies,
these subproblems are independent and their minimum costs are additive.

`--target-mode grouped` keeps them in one compiler call. The legal bias remains
unchanged in both modes; only head enumeration and labels are restricted in a
single-target child.

## 3. Analyze the bias and choose a compiler

Bias analysis derives legal-body existence, required guards, capacity limits,
and safe necessary prefix filters. Task-relative witness reductions can remove
variables only when justified by the BK and bias.

Automatic planning selects the bitmask backend when the proved reductions leave
at most two hidden variables, otherwise the relational backend. The analysis and
choice are stored in `planning.json`.

The ground prefix filter also handles conjunctions of unit-weight cardinality
conditions, including a used variable requiring one of several binders. It
rejects a prefix only if an integrity constraint must be violated. An accepted
prefix need not have a legal completion: complete candidates still pass the
exact ASP checker. Unsupported ground definitions retain the solver fallback.

## 4. Search for a feasible upper bound

With `--allow-partial-coverage`, this phase and all full-cover cost bounds are
disabled. Compilation uses the original grammar limit; partial coverage is
quantified only after the catalogue is complete.

A separate worker tries to find a multi-rule hypothesis that covers all
positives and no negatives. The short phase considers small legal components;
progressive CEGIS then increases the effective body cap and replays exact
positive impossibility cores and negative witnesses.

A returned bound is accepted only after independent verification. It is an
upper bound `U`, not automatically an optimum. If the worker times out without a
verified incumbent, compilation continues with the original bias limit. Failure
to find a bound never means the task is unsatisfiable.

Verified bounds can soundly prune candidates that cannot belong to a program of
cost at most `U`. Equality is retained. Per-target certificates are recorded in
`bound.json`; relational checkpoints can be resumed with `--resume-bound` when
their fingerprints match.

## 5. Build semantic indices or lazy relations

The bitmask family groups ground substitutions by truth-vector behavior. The
relational family builds indexed relations and computes atomic positive masks
without materializing the whole Cartesian witness space.

The optimized bitmask oracle can restrict a hidden variable to the outputs of
head-determined functional carriers. Two proofs are required: functionality in
the fixed BK, and bias UNSAT when a legal body uses that variable but selects
none of those carriers. The union of carrier outputs preserves every legal
body's witnesses; variables unused by a body retain an arbitrary completion.
For at most 65,536 remaining assignments per labelled head, grouped relation
lookups enumerate truth profiles directly. Larger products retain the projected
solver. `oracle.json` records the proved carriers and execution counts.

Oracle diagnostics remain streamed: individual records use the standard JSON
encoder, with bounded buffering and atomic file replacement.

Atomic masks are only upper bounds for conjunctions: two literals can each
match a positive label through incompatible hidden bindings. Exact joint
queries are therefore required before any candidate is published.

## 6. Generate and validate candidate rules

For every compatible head, the compiler explores bodies within `max_body` while
respecting proved guards and bias constraints. Search can prune a prefix using:

- exact bias impossibility;
- absent positive support;
- correlated empty joins;
- safe dominance or cost bounds;
- optional independently verifiable positive-failure or bias UNSAT cores.

Before retention, a completed candidate passes exact bias legality, correlated
positive coverage, and all-negative safety. Candidate cost is one plus the
number of selected body literals. Equivalent coverage/cost signatures are
compressed without discarding structural ties needed for requested variant
reconstruction.

The result is `catalogue.json`, plus an internal Clingo set-cover program and an
ILASP selection task.

## 7. Select a minimum-cost program

With `--allow-partial-coverage`, first take the union of all safe candidate
positive masks, then minimize cost over covers of that union. The union is
the maximum achievable positive coverage. Emit `coverage.json` with original
uncoverable labels and separate coverage/cost proofs. Both optimizer backends
and enumeration use the same reduction, with the dense-to-original index map
recorded in the selection artifacts.

The selector chooses a set of candidate rules whose union covers all positive
positions. Costs are additive even when rules overlap. The default internal
backend uses Clingo optimization. ILASP can be the primary finite selector, or
it can verify a completed internal optimum.

A solver incumbent obtained before timeout is `feasible`; only an exhausted
optimization search can publish `optimum_cost`. If the complete catalogue has no
cover and the solver exhausts the task, the result is `unsatisfiable`.

## 8. Optionally enumerate optima and variants

`--enumerate-optima` enumerates all candidate subsets at an already proved
optimum cost. `--expand-variants` then reconstructs legal syntactic rules for the
selected semantic cost/coverage signatures and removes exact duplicate programs.

Enumeration and reconstruction have their own budgets and completion fields. A
timeout here does not erase a previously proved optimum, but it prevents a claim
that all optimal programs or all requested variants were returned.

## 9. Verify and publish

Selected rules are regrounded with the original BK in a fresh Clingo control.
The resulting unique target model must include every positive and exclude every
negative. The compiler writes `learned.lp`, `predictions.lp`, and
`verification.json` only for a verified hypothesis.

Machine-readable state is printed as JSON on standard output and mirrored in
`status.json`. Compact progress goes to standard error. Exit codes are:

- `0`: complete optimum or proved unsatisfiable;
- `2`: invalid input or execution error;
- `3`: a valid but nonterminal outcome such as feasible, incomplete,
  verification-incomplete, enumeration-incomplete, or interrupted.
