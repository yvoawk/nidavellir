# Partial positive coverage

This synthetic example has three positive labels (`answer(a)`, `answer(b)`,
and `answer(c)`) and one negative label (`answer(n)`).

- `p/1` holds for both `a` and `n`, so a rule using only `p/1` is unsafe.
- `q/1` and `r/1` each hold only for `b` and provide equivalent safe rules.
- No allowed predicate holds for `c`.

The strict default objective is unsatisfiable. Optional partial coverage
maximizes the number of covered positives before minimizing program cost,
while still forbidding every negative prediction.

Run from the repository root:

```bash
nidavellir \
  --background examples/partial_coverage/background.lp \
  --examples examples/partial_coverage/examples.lp \
  --bias examples/partial_coverage/bias.lp \
  --output /tmp/nidavellir-partial \
  --allow-partial-coverage
```

The expected result is `complete`, with one covered positive, two uncovered
positives, no covered negatives, and a single one-literal rule of cost two.
The reported lexicographic objective `[2, 2]` represents uncovered-positive
count followed by program cost. Either of these rules is optimal:

```prolog
answer(V1) :- q(V1).
answer(V1) :- r(V1).
```

Inspect `learned.lp`, `coverage.json`, `verification.json`, and `status.json`.
To compare with strict behavior, omit `--allow-partial-coverage` and choose a
fresh output directory; the result should be `unsatisfiable`.
