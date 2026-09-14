"""Verified bounds per target signature over one shared learning task.

Target predicates are absent from learned bodies and from the fixed BK.
Hence rules for distinct (predicate, arity) signatures have disjoint labelled
coverage and additive costs. Head templates within a signature stay together.
"""
from dataclasses import replace

import clingo

from .labels import CompactNegativeSet
from .search import greedy_cover
from .task import signature
from .verification import verify_rules


def key_text(key):
    return f"{key[0]}/{key[1]}"


def positive_scopes(task):
    scopes={signature(head):0 for head in task.heads}
    for position,label in enumerate(sorted(task.positives)):
        scopes[signature(label)]|=1<<position
    return scopes


def target_view(task,key):
    """Share the immutable BK and compact label tables; do not copy their atoms."""
    heads=tuple(h for h in task.heads if signature(h)==key)
    if not heads:
        raise ValueError("Unknown target signature")
    if isinstance(task.negatives,CompactNegativeSet):
        table=task.negatives._tables.get(key)
        negatives=CompactNegativeSet([table] if table is not None else [])
    else:
        negatives=frozenset(a for a in task.negatives if signature(a)==key)
    return replace(task,heads=heads,positives=frozenset(a for a in task.positives if signature(a)==key),
                   negatives=negatives,examples_source=None,examples_path=None)


def collect_verified_target_bounds(task,pool,previous,analysis,check,*,verify_target=None):
    """Improve target incumbents from safe components, independently verifying each.

    Pool masks keep the original global positive positions. Only the temporary
    greedy-cover inputs are compacted to one target's positions. A deadline
    before publication leaves the previous certificates intact. The caller
    associates these certificates with its current input/engine fingerprint.
    """
    scopes=positive_scopes(task)
    grouped={key:[] for key in scopes}
    for candidate in pool.values():
        check()
        head=clingo.parse_term(candidate['head'])
        key=signature(head)
        if (key not in scopes or type(candidate['coverage']) is not int
                or candidate['coverage']<0 or candidate['coverage'] & ~scopes[key]):
            raise ValueError("Component coverage crosses its target signature")
        if (type(candidate['cost']) is not int or candidate['cost']!=len(candidate['body'])+1
                or task.rule(head,candidate['body'])!=candidate['rule']):
            raise ValueError("Component cost or syntax is inconsistent")
        grouped[key].append(candidate)
    for key,mask in scopes.items():
        check()
        rows=grouped[key]
        covered=0
        for row in rows:
            covered|=row['coverage']
        if covered!=mask:
            continue
        positions=[i for i in range(len(task.positives)) if mask>>i&1]
        local=[dict(row,coverage=sum(1<<i for i,p in enumerate(positions) if row['coverage']>>p&1))
               for row in rows]
        chosen=greedy_cover(local,len(positions)) if mask else []
        if chosen is None:
            continue
        originals={r['rule']:r for r in rows}
        chosen=[originals[r['rule']] for r in chosen]
        cost=sum(r['cost'] for r in chosen)
        text=key_text(key)
        if text in previous and previous[text]['cost']<=cost:
            continue
        for row in chosen:
            check()
            head=clingo.parse_term(row['head'])
            if not analysis.heads[head].accepts(tuple(row['body']),check=check):
                raise ValueError("Target incumbent is not legal in the original bias")
        rules=[row['rule'] for row in chosen]
        verification=(verify_rules(target_view(task,key),rules) if verify_target is None
                      else verify_target(key,rules))
        check()
        if not verification['matched']:
            raise ValueError("Target incumbent failed original-BK verification")
        certificate={'status':'verified','signature':list(key),'cost':cost,
            'positive_count':len(positions),'positive_mask':hex(mask),'rules':rules,
            'cover':[dict(row,coverage=hex(row['coverage'])) for row in chosen],
            'verification':{k:v for k,v in verification.items() if k!='predictions'},
            'predicted_count':len(verification['predictions']),
            'scope':'Feasible upper bound for this complete target signature; not an optimum'}
        previous[text]=certificate
    return previous


def validate_target_costs(task,bounds):
    """Validate caller-verified per-signature costs without changing their scope."""
    if bounds is None:
        return {}
    if not isinstance(bounds,dict):
        raise ValueError("target_bounds must be a dictionary")
    scopes=positive_scopes(task)
    for key,value in bounds.items():
        if (key not in scopes or type(value) is not int or value<0
                or (scopes[key] and value==0)):
            raise ValueError("Target costs require known signatures and nonnegative integers; positive targets need positive costs")
    return dict(bounds)


def costs_from_certificates(task,certificates):
    """Decode internally verified, fingerprint-bound worker certificates."""
    scopes=positive_scopes(task)
    costs={}
    for name,item in certificates.items():
        key=tuple(item['signature'])
        if (key not in scopes or name!=key_text(key) or item.get('status')!='verified'
                or not item.get('verification',{}).get('matched')
                or type(item['cost']) is not int or item['cost']<0
                or int(item['positive_mask'],16)!=scopes[key]
                or item['positive_count']!=scopes[key].bit_count()
                or item['cost']!=sum(row['cost'] for row in item['cover'])
                or item['rules']!=[row['rule'] for row in item['cover']]):
            raise ValueError("Invalid per-target bound certificate")
        if scopes[key] and item['cost']==0:
            raise ValueError("A nonempty positive target cannot have zero cost")
        covered=0
        for row in item['cover']:
            head=clingo.parse_term(row['head'])
            coverage=int(row['coverage'],16)
            if (head not in task.heads or signature(head)!=key or type(row['cost']) is not int
                    or row['cost']!=len(row['body'])+1 or coverage<0 or coverage & ~scopes[key]
                    or row['rule']!=task.rule(head,row['body'])):
                raise ValueError("Invalid per-target bound component")
            covered|=coverage
        if covered!=scopes[key]:
            raise ValueError("Target bound components do not cover the target")
        costs[key]=item['cost']
    return costs
