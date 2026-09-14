# Examples

The repository contains two self-contained synthetic examples. Neither needs
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

Use a fresh output path for every invocation. Nidavellir refuses to overwrite
an existing run directory so that evidence from an earlier run is not silently
mixed with new results.

Each example directory also has a local README explaining its predicates and
expected result.
