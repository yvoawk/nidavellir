"""Collect one legal representative per alpha class without storing raw rows.

No search branch is pruned. Both spellings pass the actual bias before a
duplicate is discarded. The canonical key proves a bijection of variables
including the head and all nested/repeated occurrences. Thus no global
automorphism of an arbitrary ASP bias is assumed or required.
"""
import clingo


class VariantBucket(list):
    def __init__(self, task, check, accepts, rows=()):
        super().__init__()
        self.task, self.check, self.accepts = task, check, accepts
        self.positions = {}
        self.encountered = 0
        for row in rows:
            self.append(row)

    def append(self, row):
        from .search import _alpha_key
        self.check()
        head = clingo.parse_term(row["head"])
        body = tuple(row["body"])
        # This checks template membership, guards, dependencies, cardinality,
        # arbitrary ASP constraints and head safety on each actual spelling.
        if not self.accepts(head, body):
            raise ValueError("Alpha collection requires a rule legal in the actual bias")
        key = _alpha_key(self.task, head, body, self.check)
        position = self.positions.get(key)
        self.check()
        # Publish as one transaction only after the entire key and proof.
        self.encountered += 1
        if position is None:
            self.positions[key] = len(self)
            super().append(row)
        elif row["rule"] < self[position]["rule"]:
            self[position] = row

    def export(self):
        return dict(syntax_encounters=self.encountered, alpha_classes=len(self),
                    duplicates_avoided=self.encountered - len(self), retained_rows=len(self),
                    canonical_representatives=[dict(key=list(key), rule=self[position]["rule"])
                                              for key, position in sorted(self.positions.items())])


class VariantRecords(dict):
    def __init__(self, task, check, accepts, enabled=True):
        super().__init__()
        if type(enabled) is not bool:
            raise ValueError("alpha_collect must be a boolean")
        self.task, self.check, self.accepts, self.enabled = task, check, accepts, enabled

    def __setitem__(self, key, rows):
        value = VariantBucket(self.task, self.check, self.accepts, rows) if self.enabled else rows
        super().__setitem__(key, value)

    def export(self):
        buckets = [v.export() for v in self.values()] if self.enabled else []
        return dict(enabled=self.enabled, scope="current minimum-cost collection; encounters include repeated visits",
                    syntax_encounters=sum(b["syntax_encounters"] for b in buckets),
                    alpha_classes=sum(b["alpha_classes"] for b in buckets),
                    duplicates_avoided=sum(b["duplicates_avoided"] for b in buckets),
                    retained_rows=sum(map(len, self.values())))
