"""Domain-independent performance choice between two exact compilers."""
from .bias_analysis import analyze
from .reduction import plan_reduction
from .task import variables
from .witness_space import WitnessSpace


def choose_backend(task, check=lambda: None, *, hidden_limit=2):
    """Choose a representation, without changing any rule or example.

    This is a performance heuristic, not a completeness assumption. Both paths
    retain the same legal finite language. Users may explicitly choose either.
    """
    check()
    analysis = getattr(task, "_bias_analysis", None)
    if analysis is None:
        analysis = analyze(task, check=check)
        task._bias_analysis = analysis
    space = WitnessSpace(task, check)
    plans, details = {}, {}
    maximum = 0
    for head in task.heads:
        info = analysis.heads[head]
        if not info.legal_body_exists:
            continue
        plan = plan_reduction(task, head, info, space, check)
        plans[head] = plan
        hidden = sorted(space.all_vars - variables(head) - plan.eliminated)
        maximum = max(maximum, len(hidden))
        details[str(head)] = {"remaining_hidden_variables": hidden, **plan.stats}
    backend = "bitmask" if maximum <= hidden_limit else "relational"
    # Only immutable plans are reused; the space's deadline callback belongs to
    # this planning phase and must not leak into a later oracle phase.
    task._reduction_plans = plans
    return backend, {"method": "remaining_hidden_variable_count", "hidden_limit": hidden_limit,
                     "maximum_remaining_hidden_variables": maximum, "heads": details,
                     "chosen_backend": backend,
                     "guarantee": "performance heuristic only; both backends use the complete legal bias"}
