"""Compare a fixed DFS prefix, not node counts distorted by sound pruning.

This validation-only stop hook is injected identically into both runs. It does
not alter choices, proofs or pruning. A reference stops BEFORE its Nth visit;
the challenger stops at or beyond that DFS frontier (a core may skip it).
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import resource
import sys
import textwrap
import time
from unittest.mock import patch

import clingo

from .. import relational_compile
from ..bias_analysis import analyze
from ..oracle import ResourceLimit
from ..relational_coverage import CoverageOracle
from ..task import Task
from ..target_execution import select_signature
from .replay_positive_cores import digest, verify_cores
from .verify_bias_cores import verify as verify_bias_cores


def instrument(reference, node_limit):
    original = textwrap.dedent(inspect.getsource(relational_compile._Search.walk))
    source = original
    changes = {
        '    finished_heads = set()':
            '    self._validation_walk = getattr(self, "_validation_walk", 0) + 1\n    finished_heads = set()',
        '    for head, context in self.contexts.items():':
            '    for head_number, (head, context) in enumerate(self.contexts.items()):',
        'def visit(body_bits, start, possible):':
            'def visit(body_bits, start, possible, validation_path=()):',
        '            self.stats["nodes"] += 1':
            '            self.stats["nodes"] += 1\n            record_frontier(self, head_number, validation_path, body_bits)',
        'visit(body_bits | added, position + 1, updated)':
            'visit(body_bits | added, position + 1, updated, validation_path + (position,))',
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise AssertionError('Compiler traversal changed; review the stop hook')
        source = source.replace(old, new)

    def record_frontier(search, head_number, positions, body):
        frontier = [search._validation_walk, head_number, *positions]
        reached = (tuple(frontier) >= tuple(reference['frontier']) if reference
                   else search.stats['nodes'] >= node_limit)
        if reached:
            search.check()  # Real deadline has priority over diagnostic completion.
            stats = dict(search.stats, frontier=frontier, frontier_body_bits=hex(body),
                         validation_frontier_reached=True,
                         elapsed_seconds=time.monotonic() - search.started,
                         partial_classes=list(search.classes.values()))
            raise ResourceLimit('Validation DFS frontier reached; catalogue remains incomplete', stats)

    namespace = dict(vars(relational_compile), record_frontier=record_frontier)
    exec(compile(source, '<positive-core-prefix-stop-hook>', 'exec'), namespace)
    return namespace['walk'], dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                  instrumented_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                  instrumented_source=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('background', 'bias', 'examples', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--target-signature', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--enabled', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bias-core-pruning', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.output.exists() or args.nodes < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('Choose a fresh output and a positive node limit')
    paths = {k: getattr(args, k) for k in ('background', 'examples', 'bias')}
    inputs = {k: digest(p) for k,p in paths.items()}
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob('*.py'), Path(__file__), Path(__file__).with_name('replay_positive_cores.py'),
                    Path(__file__).with_name('verify_bias_cores.py')])
    def hashes():
        return {str(p.relative_to(root)): digest(p) for p in files}
    frozen = hashes()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (not reference['reached'] or inputs != reference['input_sha256']
                      or frozen != reference['engine_sha256'] or args.target_signature != reference['target_signature']
                      or reference['clingo_version'] != clingo.__version__ or reference['python'] != sys.version):
        parser.error('Reference must have reached its frontier on identical sources, inputs and target')
    record = dict(enabled=args.enabled, bias_core_pruning=args.bias_core_pruning,
        engine_sha256=frozen, input_sha256=inputs,
        inputs={k:str(p.resolve()) for k,p in paths.items()}, target_signature=args.target_signature,
        node_limit=args.nodes, reference=str(args.reference) if args.reference else None,
        python=sys.version, clingo_version=clingo.__version__, platform=platform.platform(), reached=False, complete=False,
        scope='Identical instrumented DFS prefix; no claim of a complete catalogue or optimum')
    try:
        started = time.monotonic()
        task = select_signature(Task.load(paths['background'], paths['examples'], paths['bias']), args.target_signature)
        record['input_seconds'] = time.monotonic() - started
        record['literal_templates'] = list(map(str, task.literals))
        oracle = CoverageOracle(task, positive_core_pruning=args.enabled)
        analysis_started = time.monotonic()
        task._bias_analysis = analyze(task, seconds=args.seconds, core_pruning=args.bias_core_pruning)
        record['analysis_seconds'] = time.monotonic() - analysis_started
        remaining = args.seconds - record['analysis_seconds']
        if remaining <= 0:
            raise ResourceLimit('Prefix budget exhausted by bias analysis')
        method, hook = instrument(reference, args.nodes)
        record['stop_hook'] = hook
        if reference:
            assert record['literal_templates'] == reference['literal_templates']
            assert hook == reference['stop_hook']
        with patch.object(relational_compile._Search, 'walk', method):
            try:
                rows, stats = relational_compile.compile_catalogue(task, oracle, seconds=remaining,
                    variants='representative', progress=lambda s: print(json.dumps(
                        {k:s[k] for k in ('nodes','elapsed_seconds','pruned_positive_core')}), flush=True))
                record.update(stats=stats, catalogue=rows)
            except ResourceLimit as error:
                record.update(stats=error.stats, reason=str(error))
        stats = record['stats']
        record['compile_seconds_with_analysis'] = record['analysis_seconds'] + stats['elapsed_seconds']
        record['reached'] = stats.get('validation_frontier_reached', False)
        if record['reached']:
            record['frontier'] = stats['frontier']
            if reference:
                assert stats['head_orders'] == reference['stats']['head_orders']
                assert stats['partial_classes'] == reference['stats']['partial_classes']
                record['partial_classes_identical'] = True
        if oracle.positive_cores:
            at = time.monotonic()
            record['verification'] = verify_cores(task, oracle.positive_cores.export()['cores'])
            record['verification_seconds'] = time.monotonic() - at
            assert record['verification']['matched']
        if args.bias_core_pruning:
            at = time.monotonic()
            record['bias_verification'] = verify_bias_cores(task.bias_source, record['literal_templates'],
                                                           task.max_body, task._bias_analysis.export())
            record['bias_verification_seconds'] = time.monotonic() - at
            assert record['bias_verification']['matched']
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    record.update(engine_unchanged=frozen == hashes(), inputs_unchanged=inputs == {k:digest(p) for k,p in paths.items()},
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024))
    record['valid_comparison'] = record['reached'] and record['engine_unchanged'] and record['inputs_unchanged'] and 'error' not in record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k in ('reached','error','valid_comparison','frontier')}))
    return 0 if record['valid_comparison'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
