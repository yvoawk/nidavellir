# Examples

The repository contains three self-contained synthetic examples. None needs
external data or a domain adapter.

## Overlapping rules

`examples/overlapping_rules` demonstrates why the learned hypothesis may need
more than one rule. Predicate `p/1` covers two positives and `q/1` covers two;
they overlap on one positive. The distractor `r/1` also covers a negative.

```bash
nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-overlap
```

Expected complete result: two one-literal rules, total cost four, with all three
positives covered and neither negative covered.

Inspect the result:

```bash
cat /tmp/nidavellir-overlap/learned.lp
cat /tmp/nidavellir-overlap/verification.json
python -m json.tool /tmp/nidavellir-overlap/status.json
```

To enumerate tied optimal semantic programs and reconstruct legal syntactic
variants:

```bash
nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-overlap-variants \
  --enumerate-optima \
  --expand-variants
```

## Temporal sensor rise

`examples/temporal_rise` uses device readings at ordered integer times. The BK
derives `increased/2` when a current value is at least 1.5 times a previous
value, and `before/2` for temporal order. The bias allows the compiler to join
two readings through hidden value and time variables.

```bash
python -m nidavellir \
  --background examples/temporal_rise/background.lp \
  --examples examples/temporal_rise/examples.lp \
  --bias examples/temporal_rise/bias.lp \
  --output /tmp/nidavellir-temporal
```

The expected complete hypothesis has one four-literal rule and cost five. This
example exercises correlated hidden-variable joins: matching each literal in
isolation would not be sufficient evidence for the conjunction.

## Partial positive coverage

`examples/partial_coverage` contains three positives, of which only one can be
covered without covering a negative. The default strict mode therefore reports
`unsatisfiable`.

```bash
nidavellir \
  --background examples/partial_coverage/background.lp \
  --examples examples/partial_coverage/examples.lp \
  --bias examples/partial_coverage/bias.lp \
  --output /tmp/nidavellir-partial \
  --allow-partial-coverage
```

The expected complete result covers one positive and no negatives, using one
one-literal rule of cost two. The lexicographic objective is `[2, 2]`: two
uncovered positives, then program cost two. `coverage.json` identifies the
covered and uncovered labels and records the coverage proof. The two equivalent
predicates `q/1` and `r/1` also permit tied optimal rules.

## Trying explicit backends

The automatic planner is recommended. Small examples can also be used for
differential checks:

```bash
nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-relational \
  --compiler-backend relational
```

This strict representative run uses the automatic direct-proof route when a
verified incumbent is available. Inspect `program_proof.json` and `solver.json`
for the proof of no cheaper program. To compare catalogues instead, add
`--proof-strategy catalogue`; also add `--no-dynamic-program-bounds` and
`--bound-seconds 0` when comparing an unrestricted reference catalogue rather
than optimal-program preservation. Optimum and variant enumeration automatically retain the
catalogue route.

Use a fresh output path for every invocation. Nidavellir refuses to overwrite
an existing run directory so that evidence from an earlier run is not silently
mixed with new results.

Each example directory also has a local README explaining its predicates and
expected result.
