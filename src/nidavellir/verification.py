"""Independent evaluation in the original fixed BK, exporting only targets."""
import clingo

from .task import signature


def verify_rules(task, rules, *, allow_partial_coverage=False):
    if type(allow_partial_coverage) is not bool:
        raise ValueError("allow_partial_coverage must be a boolean")
    diagnostics = []
    ctl = clingo.Control(["2"], logger=lambda code, text: diagnostics.append(text))
    targets = {signature(h) for h in task.heads}
    # The complete BK is still grounded and solved independently. Projection
    # only prevents transferring millions of irrelevant BK symbols to Python.
    projection = "\n#show.\n" + "\n".join(f"#show {name}/{arity}." for name, arity in sorted(targets))
    ctl.add("base", [], task.background + "\n" + "\n".join(rules) + projection)
    ctl.ground([("base", [])])
    models = []
    result = ctl.solve(on_model=lambda m: models.append({a for a in m.symbols(shown=True)
                                                        if signature(a) in targets}))
    if not result.satisfiable or result.unknown or len(models) != 1:
        raise RuntimeError("Learned program did not produce one definite Clingo model")
    actual = models[0]
    missing = task.positives - actual
    forbidden = task.negatives & actual
    return {"matched": not forbidden and (allow_partial_coverage or not missing),
            "hard_labels_matched": not missing and not forbidden,
            "coverage_mode": "partial" if allow_partial_coverage else "strict",
            "positive_count": len(task.positives),
            "covered_positives": len(task.positives) - len(missing),
            "missed_positives": len(missing),
            "positive_recall": (len(task.positives) - len(missing)) / len(task.positives) if task.positives else 1.0,
            "missing_positives": sorted(map(str, missing)),
            "covered_negatives": sorted(map(str, forbidden)),
            "predictions": sorted(map(str, actual)),
            "scope": ("original BK + learned clauses, strict negatives and quantified positive coverage; no optimality claim"
                      if allow_partial_coverage else
                      "original BK + learned clauses, all supplied hard labels; unlabelled atoms unconstrained"),
            "diagnostics": diagnostics}
