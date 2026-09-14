# Overlapping-rule example

This finite synthetic task learns `t/1` from three unary predicates.

- `p/1` covers `a` and `b`.
- `q/1` covers `b` and `c`.
- `r/1` is a distractor because it also covers the negative example `d`.

No single legal one-literal rule covers all positives. The minimum-cost program
therefore contains the rules `t(V1) :- p(V1).` and `t(V1) :- q(V1).`, with cost
four under the head-plus-body cost model.

From the repository root:

```bash
nidavellir \
  --background examples/overlapping_rules/background.lp \
  --examples examples/overlapping_rules/examples.lp \
  --bias examples/overlapping_rules/bias.lp \
  --output /tmp/nidavellir-overlap
```

The output path must not already exist.
