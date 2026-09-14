"""Cold sequential replay of recorded queries, with online learning only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import clingo

from .. import __version__
from ..oracle import ResourceLimit
from ..positive_cores import MIN_QUERY_NODES
from ..positive_join import has_positive
from ..relational_coverage import CoverageOracle
from ..target_execution import select_signature
from ..task import Task, asp_text


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): digest(p) for p in sorted([*root.glob('*.py'), Path(__file__)])}


def verify_cores(task, cores):
    """Independent Clingo query over BK, original head labels, and partial bodies."""
    prefix = 'nidavellir_core_validation'
    names = {atom.name for atom in task.atoms}
    while prefix + '_label' in names or prefix + '_failure' in names:
        prefix += '_x'
    facts = '\n'.join(f'{prefix}_label({label}).' for label in sorted(task.positives))
    rules = []
    for number, core in enumerate(cores):
        head = asp_text(clingo.parse_term(core['head']))
        body = [f'{prefix}_label({head})', *(asp_text(task.literals[i]) for i in core['body'])]
        rules.append(f'{prefix}_failure({number}) :- ' + ', '.join(body) + '.')
    control = clingo.Control(['0'], logger=lambda *_: None)
    control.add('base', [], task.background + '\n' + facts + '\n' + '\n'.join(rules)
                + f'\n#show.\n#show {prefix}_failure/1.')
    control.ground([('base', [])])
    models = []
    result = control.solve(on_model=lambda model: models.append(list(map(str, model.symbols(shown=True)))))
    return dict(matched=result.satisfiable and result.exhausted and models == [[]], models=models,
                scope='Independent Clingo: no original supplied positive satisfies any retained conjunction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('background', 'bias', 'examples', 'profile', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--enabled', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.count < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('Use new output and positive finite limits')
    paths = {k: getattr(args, k) for k in ('background', 'examples', 'bias')}
    input_hashes = {k: digest(p) for k, p in paths.items()}
    profile = json.loads(args.profile.read_text())
    manifest = json.loads(args.profile.with_name('manifest.json').read_text())
    if input_hashes != manifest['input_files_sha256'] or manifest['clingo_version'] != clingo.__version__:
        parser.error('Inputs and Clingo must match the source archive')
    frozen = hashes()
    record = dict(version=__version__, enabled=args.enabled, engine_sha256=frozen,
        input_sha256=input_hashes, inputs={k: str(p.resolve()) for k, p in paths.items()},
        platform=platform.platform(), python=sys.version, clingo_version=clingo.__version__,
        profile_sha256=digest(args.profile), source_profile=str(args.profile.resolve()),
        scope='Archived slow queries in descending archived duration; fresh process/oracle, no preloaded cores',
        results=[], complete=False)
    try:
        started = time.monotonic()
        task = Task.load(paths['background'], paths['examples'], paths['bias'])
        if manifest['arguments'].get('target_signature'):
            task = select_signature(task, manifest['arguments']['target_signature'])
        if list(map(str, task.literals)) != manifest['literal_templates']:
            raise AssertionError('Literal index order changed')
        record['input_seconds'] = time.monotonic() - started
        oracle = CoverageOracle(task, positive_core_pruning=args.enabled)
        queries = [r for r in profile['stats']['profile']['diagnostics']['positive_join']['slowest_calls']
                   if r['outcome'] == 'false'][:args.count]
        loop_started = time.monotonic()
        def check():
            if time.monotonic() - loop_started >= args.seconds:
                raise ResourceLimit('Replay deadline')
        oracle.check = check
        for query in queries:
            head, body = clingo.parse_term(query['head']), tuple(query['body'])
            if task.rule(head, body) != query['rule']:
                raise AssertionError('Archived rule/template mismatch')
            at = time.monotonic()
            bank = oracle.positive_cores
            hit = bank.match(head, sum(1 << i for i in body)) if bank else None
            if hit is not None:
                answer = False
            else:
                before = oracle.stats['positive_join_nodes']
                answer = has_positive(oracle, head, body, int(query['possible_hex'], 16))
                elapsed = time.monotonic() - at
                if bank and not answer and oracle.stats['positive_join_nodes'] - before >= MIN_QUERY_NODES:
                    bank.learn(head, body, elapsed)
            if answer is not False:
                raise AssertionError('Answer differs from the completed archived failure')
            row = dict(body=list(body), answer=answer, core_hit=hit is not None, seconds=time.monotonic() - at)
            record['results'].append(row)
            print(json.dumps(row), flush=True)
        record.update(query_seconds=time.monotonic() - loop_started, oracle_stats=dict(oracle.stats),
                      profile=oracle.profile.export())
        if oracle.positive_cores:
            started = time.monotonic()
            record['verification'] = verify_cores(task, oracle.positive_cores.export()['cores'])
            record['verification_seconds'] = time.monotonic() - started
            if not record['verification']['matched']:
                raise AssertionError('Independent Clingo found an invalid global core')
        record['complete'] = bool(queries)
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    record.update(engine_unchanged=frozen == hashes(), inputs_unchanged=input_hashes == {k:digest(p) for k,p in paths.items()},
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024))
    record['complete'] &= record['engine_unchanged'] and record['inputs_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k in ('complete','error','query_seconds','verification_seconds')}))
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
