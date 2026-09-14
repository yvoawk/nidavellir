"""Private subprocess for a hard wall-clock bound-search budget.

Only strings and integer masks cross the process boundary. Clingo Symbols hold
process-local interned handles and must never be pickled into another process.
"""
import json
from pathlib import Path
import sys
import time

import clingo

from .oracle import MaskIndex
from .search import discover_bound
from .task import Task, _examples, _model, read_ground_model, signature
from .verification import verify_rules
from .labels import CompactNegativeSet


def main():
    started = time.monotonic()
    if len(sys.argv) > 1:
        with Path(sys.argv[1]).open() as stream:
            data = json.load(stream)
    else:
        data = json.load(sys.stdin)
    t = data["task"]
    relational = data.get("strategy") == "relational"
    parse = lambda key: tuple(clingo.parse_term(s) for s in t[key])
    heads, literals = parse("heads"), parse("literals")
    if "background_file" in t:
        background = Path(t["background_file"]).read_text()
        positives = frozenset(parse("positives"))
        negatives = frozenset()
        if relational and t.get("compact_negative_file"):
            negatives = CompactNegativeSet.load(t["compact_negative_file"],
                                                 source_sha256=t["examples_sha256"],
                                                 artifact_sha256=t["compact_negative_sha256"])
            if not set(negatives._tables) <= {signature(h) for h in heads}:
                raise ValueError("Compact negatives contain an undeclared target")
            if any(atom in negatives for atom in positives):
                raise ValueError("Conflicting labels in compact negative transfer")
        elif relational or not t["explicit_examples"]:
            # Rare normalization syntax (#const/pools) retains its exact
            # interpretation. Large explicit snapshots use the streaming
            # negative check below instead of constructing another label set.
            _, positives, negatives, _ = _examples(Path(t["examples_file"]), {signature(h) for h in heads})
        atoms = frozenset()
        if relational:
            if t.get("background_transfer") == "replay":
                atoms, control = _model(background, "Bound worker BK", [])
                del control
            else:
                atoms = read_ground_model(Path(t["background_model_file"]))
        task = Task(background, atoms, t["bias_source"], None, heads, literals,
                    t["max_body"], positives, negatives)
    else:
        # Accept older small payloads for reproducibility of saved commands.
        task = Task(t["background"], frozenset(), t["bias_source"], t.get("examples_source"),
                    heads, literals, t["max_body"], frozenset(parse("positives")), frozenset(parse("negatives")))
    indices = [MaskIndex(clingo.parse_term(i["head"]), tuple(int(m, 16) for m in i["literal_masks"]),
                         int(i["all_mask"], 16), int(i["positive_mask"], 16), int(i["negative_mask"], 16),
                         tuple((n, int(m, 16)) for n, m in i["positive_heads"]), []) for i in data["indices"]]
    last_verification = {}
    def evaluate(view, rules):
        result = verify_rules(view, rules)
        if t.get("explicit_examples") and not relational:
            actual = set(result["predictions"])
            forbidden = set()
            checked = 0
            # This file is the self-contained canonical snapshot produced by
            # Task.write_examples(), not arbitrary ASP input. Comparing its
            # canonical atom text preserves Clingo's symbol identity exactly.
            with Path(t["examples_file"]).open() as stream:
                for line in stream:
                    if line.startswith("negative("):
                        if not line.endswith(").\n"):
                            raise RuntimeError("Malformed canonical negative snapshot")
                        atom = line[9:-3]
                        checked += 1
                        if atom in actual:
                            forbidden.add(atom)
            result["covered_negatives"] = sorted(forbidden)
            result["matched"] = not result["missing_positives"] and not forbidden
            result["negative_labels_checked"] = checked
            result["negative_verification"] = "streamed canonical labels against direct Clingo predictions"
        return result

    def verify(rules):
        nonlocal last_verification
        result = evaluate(task, rules)
        last_verification = dict(result, rules=rules) if result["matched"] else {}
        return result

    def checkpoint(value):
        if data.get("checkpoint_path"):
            path = Path(data["checkpoint_path"])
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(dict(value, _verification=last_verification)) + "\n")
            temporary.replace(path)

    seconds = max(0, data["seconds"] - (time.monotonic() - started))
    if not seconds:
        result = (None, {"status": "timeout", "cost": None, "reason": "Worker input budget exhausted"})
    elif relational:
        from .relational_bound import discover_relational_bound
        state = json.loads(Path(data["resume_path"]).read_text()) if data.get("resume_path") else None
        result = discover_relational_bound(task, seconds, verify, state=state, checkpoint=checkpoint,
                                           fingerprint=data.get("fingerprint"),
                                           per_target_bounds=data.get("per_target_bounds", False))
    elif data.get("per_target_bounds", False):
        from .bitmask_bound import discover_target_bounds
        from .target_bounds import target_view
        result = discover_target_bounds(task, indices, seconds, verify,
            verify_target=lambda key, rules: evaluate(target_view(task, key), rules),
            checkpoint=checkpoint, fingerprint=data.get("fingerprint"))
    else:
        result = discover_bound(task, indices, seconds, verify)
    if result[0] is not None:
        for c in result[0]:
            c["coverage"] = hex(c["coverage"])
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
