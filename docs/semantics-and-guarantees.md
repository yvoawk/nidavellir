# Semantics and guarantees

## Learning objective

For a validated task, let the bias define a finite set of legal positive rules.
A rule is safe for selection when evaluating it with the fixed BK covers none of
the supplied negatives. Nidavellir chooses a set of safe rules whose union
covers every supplied positive and minimizes:

```text
sum over selected rules of (1 + number of body literals)
```

Overlapping coverage is allowed and often necessary. The optimizer solves a
multi-rule weighted set-cover problem; it does not assume one rule per target or
one rule per example.

### Opt-in partial coverage

`--allow-partial-coverage` changes the objective to the lexicographic pair
`(number of missed supplied positives, total rule cost)`. Negatives stay strict.
This is not a weighted error/cost compromise. The default objective above is
unchanged. Input consistency and every other task contract still apply.

Let `U` be the union of the exact positive masks of every negative-safe rule.
Because learned rules are positive and independent, their unions remain safe.
Thus `U` is attainable and no admissible program covers a positive outside it.
The selector minimizes cost subject to covering all of `U`. This reduction is
exact for the lexicographic objective, including ties and empty programs.
The generated selection encodings use dense indices for `U`, with an explicit
mapping to the original positive positions. Input labels are never rewritten.

Safe compiler reductions still apply: replacing a rule by a strictly cheaper
negative-safe rule covering a superset cannot reduce coverage and improves cost
whenever needed. Equivalent masks preserve their minimum costs and tied
variants. Full-cover heuristic bounds are disabled; compilation therefore
retains every signature needed to certify maximum coverage and its minimum
cost. Target-independent maxima and subsequent minimum costs add across
signatures, so separate execution is equivalent to grouped execution.

`coverage.json` quantifies training coverage and lists uncoverable labels.
`maximum_coverage_proven` requires completed compilation at the CLI level.
`cost_optimal` additionally requires completed selection. A compiler timeout
is incomplete, even if some useful rules were discovered. A selection timeout
can retain a verified maximum-coverage cover without an optimal cost. The
Python selector's coverage certificate concerns only its supplied catalogue.

## What `complete` guarantees

A top-level `complete` result guarantees, relative to the accepted input
contract:

1. The required finite candidate compilation finished under all enabled sound
   reductions.
2. The optimizer completed and proved the reported minimum cost over that
   catalogue.
3. The catalogue reductions preserve minimum-cost programs in the finite legal
   grammar.
4. The emitted hypothesis was independently evaluated in the original fixed BK.
5. No supplied negative is predicted. In default strict mode every supplied
   positive is predicted; in partial mode the reported maximum number of
   positives is predicted and cost is minimal among programs attaining it.

This is a task-relative proof. It does not establish that the BK is a faithful
model of an external world, that missing labels are negative, or that a better
rule exists outside the supplied bias.

## Feasible, incomplete, and unsatisfiable

These states are deliberately distinct:

- `feasible`: a hypothesis has passed independent semantic verification, but
  compilation or selection did not prove it optimal.
- `incomplete`: a required phase exhausted a time, witness, class, or other
  resource budget without a verified hypothesis.
- `unsatisfiable`: complete candidate compilation and exhausted selection prove
  that no program in the supplied finite language covers the hard labels.
  Under partial coverage alone, valid tasks always admit at least the empty
  program, so uncoverable positives are quantified rather than reported UNSAT.
- `verification_incomplete`: internal selection finished, but requested ILASP
  verification was unavailable or incomplete.
- `enumeration_incomplete`: an optimum is proved, but the optional enumeration
  or syntactic expansion request did not finish.

A timeout is never converted into UNSAT. A correct prediction file does not by
itself establish catalogue completeness or global optimality.

## Candidate and program completeness

Conditional functional domains preserve coverage for every legal body. If a
body uses a restricted variable, the bias proof forces a selected carrier; its
functional output is among the retained values. Variables absent from the body
can be assigned any original-domain value without changing that body's truth.
Conversely, retained assignments remain actual assignments in the fixed BK.
Shared variables are evaluated with the same binding across all literals.
Missing carrier outputs do not become facts, and nonfunctional or unproved
carriers cannot justify a restriction. This is a legal-body equivalence, not a
claim that raw profiles or illegal bodies have identical truth vectors.

The default catalogue is optimized for minimum-cost selection. It may omit
strictly dominated rules and collapse alpha-renamed spellings while retaining
the cost/coverage information and structural ties required to preserve optima.
It is not intended to be a raw list of every syntactically expressible rule.

Three claims must therefore remain separate:

- catalogue completion: all optimum-relevant semantic candidate signatures
  required by the compiled objective have been handled;
- optimum enumeration: every candidate subset at the proved cost has been
  returned;
- variant expansion: every requested legal spelling for those semantic
  signatures has been reconstructed and exact duplicate programs removed.

Use `--enumerate-optima --expand-variants`, then inspect
`enumeration.json.complete` and `variants.json.all_optimal_programs_complete`
before claiming a complete set of optimal programs. `--raw-alpha-variants` is a
reference ablation for retaining raw alpha spellings during collection; it is
not normally needed.

## Sound pruning and verified bounds

Nidavellir derives reductions only from the current BK, bias, labels, or a
verified feasible cost bound. Atomic masks and sampled negative statistics may
overestimate support or change traversal order, but they cannot establish final
coverage or safety.

When a verified program bound `U` is available, a partial rule may be pruned if
its current cost plus a proved lower bound for another useful rule is strictly
greater than `U`. Equality is preserved. Exact positive probes can disprove that
an expensive rule covers all positives; successful probes never substitute for
full coverage verification.

Optional positive-core and bias-core pruning is disabled by default. Enabling
it reuses only independently characterized failure evidence. Exact final checks
remain in place.

## Backends and equivalence

The bitmask and relational compilers implement the same finite rule semantics:

- Bitmask compilation compresses witness truth vectors and is effective when
  few hidden variables remain after proved reductions.
- Relational compilation performs lazy exact joins and avoids materializing a
  large Cartesian space.

The `optimized`, `projected`, and `cartesian` oracle choices are representation
and validation alternatives inside bitmask compilation. The Cartesian oracle is
the clearest reference but can exceed practical witness limits.

The internal and ILASP optimizers solve the same emitted finite selection
problem. Optional ILASP agreement is useful differential evidence, but it does
not make ILASP independently validate candidate generation from the original
bias.

## Difference from a classic ILASP workflow

A classic ILASP workflow normally gives ILASP background knowledge, mode bias,
and examples, allowing ILASP to generate and optimize hypotheses directly.
Nidavellir instead:

1. accepts its own explicit finite template contract;
2. evaluates and compresses rule semantics before optimization;
3. emits a finite weighted coverage problem;
4. solves that problem internally by default;
5. optionally asks ILASP to solve or verify only the compiled selection layer.

Consequently, Nidavellir's `bias.lp` is not a drop-in ILASP mode-declaration
file, and `compiled_ilasp.las` is a generated selection artifact rather than the
original learning task.

## Scalability limits

The legal grammar is finite, but worst-case growth is combinatorial. Important
drivers are:

- the number of `allowed_literal/1` templates and `max_body`;
- head templates and target signatures;
- hidden variable domains and Cartesian witness products;
- dense joins and weakly selective predicates;
- the count of hard labels, especially large structured negative sets;
- the number of equal-cost/equal-coverage candidates and tied optimal programs;
- optional full enumeration and alpha-variant expansion.

Resource budgets protect a run from silent overclaiming, not from all expensive
grounding or native solver calls. Some Clingo grounding operations cannot be
interrupted at arbitrary Python instruction boundaries. Use isolated processes,
fresh output directories, explicit budgets, and persisted status artifacts for
large tasks.
