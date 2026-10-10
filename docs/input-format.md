# Input format

Nidavellir accepts three paths: `--background`, `--examples`, and `--bias`.
All inputs use Clingo/ASP syntax and are normalized into hashed snapshots inside
the output directory. This document is the normative public input contract for
version 0.3.26.

## Background knowledge (BK)

The BK describes a fixed finite interpretation against which every candidate
rule is evaluated.

```prolog
edge(a,b).
edge(b,c).
marked(c).
path(X,Y) :- edge(X,Y).
path(X,Z) :- edge(X,Y), edge(Y,Z).
```

Supported BK constructs are:

- positive ground facts;
- positive normal rules;
- positive or default-negated symbolic body atoms;
- comparisons and Boolean constants in rule bodies;
- `#const`, pools, intervals, and `#include` when Clingo can normalize them
  within the rest of this contract;
- harmless display declarations, which are ignored because Nidavellir reads
  all true atoms rather than only shown atoms.

The following conditions are enforced:

- The grounded BK must be satisfiable and have exactly one stable model.
- Predicate dependencies must be acyclic. Recursion is rejected.
- Rule heads must be ordinary positive atoms. Choice, disjunctive, aggregate,
  and external heads are rejected.
- Body aggregates, conditional literals, double negation, external functions,
  and non-base parameterized programs are rejected.
- A target predicate signature, identified by both name and arity, must not
  occur anywhere in the BK. Learned targets are also forbidden in candidate
  bodies, so learned rules are independent rather than recursively composed.

Default negation in an accepted acyclic BK remains stratified. The single model
is therefore the fixed world used for compilation and final verification.

## Positive and negative examples

Examples are ground hard labels:

```prolog
positive(reachable(a)).
positive(reachable(b)).
negative(reachable(d)).
```

Each wrapped atom must match a name/arity signature declared by at least one
`candidate_head/1` fact. An atom cannot be both positive and negative. Variables,
rules, weak constraints, or arbitrary facts in the examples file are rejected.

Nidavellir supports comments, includes, constants, pools, and other Clingo
ground-fact normalization features. A fast line-oriented parser handles simple
one-fact-per-line files without constructing a large wrapper model; more general
supported syntax falls back to exact Clingo normalization.

Labels are explicit:

- Every positive must be covered by at least one selected rule.
- No negative may be covered by any selected rule.
- Unlabelled target atoms are unconstrained and may be predicted.

This is not a closed-world interpretation of missing labels.

## Bias metadata

The bias defines a finite set of head templates, a finite set of allowed body
literal templates, and a maximum body length:

```prolog
candidate_head(reachable(var__(1))).
allowed_literal(edge(var__(1),var__(2))).
allowed_literal(marked(var__(2))).
max_body(2).
```

All `candidate_head/1`, `allowed_literal/1`, and `max_body/1` declarations must
be unconditional ground facts in the ASP meta-language. Their arguments are
compiler terms, so `var__(1)` is ground from Clingo's perspective even though it
denotes a variable slot in the learned rule.

Requirements:

- At least one `candidate_head/1` fact is required.
- Exactly one `max_body(N)` fact is required, with `N >= 0`.
- Zero or more `allowed_literal/1` facts may be declared.
- Every head and literal must have a positive outer predicate symbol.
- Target signatures cannot occur in `allowed_literal/1`.
- `var__/1` is reserved and cannot itself be a head or body predicate.

## `var__(N)` placeholders

`var__(N)` denotes variable slot `N`, where `N` is a strictly positive integer.
The same number denotes the same logical variable everywhere in one template.
Different numbers denote different slots before legal alpha-renaming.

```prolog
candidate_head(linked(var__(1),var__(1))).
allowed_literal(edge(var__(1),var__(2))).
allowed_literal(edge(var__(2),var__(3))).
allowed_literal(tagged(pair(var__(3),fixed))).
```

Repeated variables, constants, arbitrary predicate arities, nullary atoms, and
nested function terms are supported. In learned ASP output, `var__(1)` is
rendered as `V1`, `var__(2)` as `V2`, and so on.

Every variable occurring in a candidate head must occur in the selected body.
This safety requirement is enforced independently of bias rules. Body-only
slots are existential variables during coverage evaluation.

## Defining legal selections

The bias is also an ASP meta-program over `head/1` and `body/1`. A minimal
general scaffold is:

```prolog
1 { head(H) : candidate_head(H) } 1.
0 { body(L) : allowed_literal(L) } N :- max_body(N).
#show head/1.
#show body/1.
```

Additional constraints can express typing, guards, mutual exclusion, or other
structural restrictions. For example:

```prolog
:- body(marked(var__(2))), not body(edge(var__(1),var__(2))).
```

This requires the edge literal whenever the marked literal is selected. Bias
negation and aggregates constrain rule structure; they do not become negated
or aggregate literals in the learned rule.

For every completed candidate, Nidavellir fixes exactly one declared head and
the chosen set of body templates, then asks Clingo whether the original bias is
satisfiable. Optimized prefix filters are only used when they are proved
necessary conditions; final candidate legality remains an exact bias check.

## Finite-language and cost model

The explicit templates and `max_body/1` define the finite hypothesis language.
A rule with `k` body literals costs `k + 1`. Program cost is the sum of its rule
costs. Selection minimizes that additive cost while covering all positive
examples.

Duplicate semantic coverage does not automatically erase all syntax: a
strictly more expensive rule may be dominated, while equal-cost structural
ties needed for complete variant reconstruction are retained according to the
catalogue mode. See [Semantics and guarantees](semantics-and-guarantees.md).

## Validation errors

Input-contract violations produce exit code 2 and an `invalid_input` status
when the run directory has been created. The reason is recorded in
`status.json` and `report.md`. Invalid option combinations are rejected by the
argument parser before a run directory is created.
