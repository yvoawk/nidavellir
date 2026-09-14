"""Recheck saved signed cores with separate source-level Clingo programs."""
import argparse
import json
from pathlib import Path
import time

import clingo

from ..task import variables
from .replay_positive_cores import digest


def verify(bias, literals, max_body, proofs):
    results = []
    for head, proof in proofs.items():
        bank = proof['cores']
        if bank is None:
            continue
        inverse = {}
        for index, atom in enumerate(proof['body_atoms']):
            if atom is not None:
                inverse.setdefault(atom, index)
        constraints = [f':- not head({head}).', f':- head(H), H != {head}.',
            ':- body(L), not allowed_literal(L).',
            f':- #count {{ L : body(L) }} > {max_body}.']
        symbols = list(map(clingo.parse_term, literals))
        for slot in variables(clingo.parse_term(head)):
            occurrences = [f'body({literal})' for literal, symbol in zip(literals, symbols)
                           if slot in variables(symbol)]
            constraints.append(':- ' + ', '.join('not ' + atom for atom in occurrences) + '.')
        for core in bank['cores']:
            assert all(abs(literal) in inverse for literal in core)
            selected = [inverse[atom] for atom in core if atom > 0]
            excluded = [inverse[-atom] for atom in core if atom < 0]
            fixed = [f':- not body({literals[i]}).' for i in selected]
            fixed += [f':- body({literals[i]}).' for i in excluded]
            control = clingo.Control(['1'], logger=lambda *_: None)
            control.add('base', [], bias + '\n' + '\n'.join(constraints + fixed))
            control.ground([('base', [])])
            answer = control.solve()
            results.append(dict(head=head, selected=selected, excluded=excluded,
                                unsatisfiable=answer.unsatisfiable, exhausted=answer.exhausted,
                                unknown=answer.unknown))
    return dict(matched=all(r['unsatisfiable'] and r['exhausted'] and not r['unknown'] for r in results),
                core_count=len(results), results=results,
                scope='Fresh source-level bias plus head, safety, length and signed core constraints; no _HeadSolver')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output')
    manifest = json.loads((args.compiler / 'manifest.json').read_text())
    catalogue = json.loads((args.compiler / 'catalogue.json').read_text())
    bias_path = args.compiler / 'inputs/bias.lp'
    assert digest(bias_path) == manifest['sha256']['bias.lp']
    started = time.monotonic()
    result = verify(bias_path.read_text(), manifest['literal_templates'], manifest['task']['max_body_literals'],
                    catalogue['stats']['profile']['diagnostics']['bias_queries'])
    result.update(seconds=time.monotonic()-started, clingo_version=clingo.__version__,
                  bias_sha256=digest(bias_path), catalogue_sha256=digest(args.compiler / 'catalogue.json'))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))
    return 0 if result['matched'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
