"""Independent evaluation in the original fixed BK, exporting only targets."""
import clingo

from .task import signature


def verify_rules(task, rules):
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
    return {"matched": not missing and not forbidden,
            "missing_positives": sorted(map(str, missing)),
            "covered_negatives": sorted(map(str, forbidden)),
            "predictions": sorted(map(str, actual)),
            "scope": "original BK + learned clauses, all supplied hard labels; unlabelled atoms unconstrained",
            "diagnostics": diagnostics}
