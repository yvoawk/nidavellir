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
5. Every supplied positive is predicted and no supplied negative is predicted.

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
- `verification_incomplete`: internal selection finished, but requested ILASP
  verification was unavailable or incomplete.
- `enumeration_incomplete`: an optimum is proved, but the optional enumeration
  or syntactic expansion request did not finish.

A timeout is never converted into UNSAT. A correct prediction file does not by
itself establish catalogue completeness or global optimality.

## Candidate and program completeness

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
