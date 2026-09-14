"""Independent semantic and IO review of exact compressed negative labels."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.labels import (
    AtomRecord, CompactNegativeSet, NegativeBuilder, canonical_text, parse_atom,
)
from nidavellir.task import TaskError, _examples, _examples_legacy


def compact(texts):
    builder, cache = NegativeBuilder(), OrderedDict()
    for text in texts:
        builder.add(parse_atom(text, cache))
    return builder.freeze()


def rewrite_artifact(path, transform):
    """Recompute the checksum so validation tests exercise semantic checks."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records.pop()
    transform(records)
    payload = ''.join(json.dumps(record, separators=(',', ':')) + '\n' for record in records)
    footer = {'type': 'complete', 'sha256': hashlib.sha256(payload.encode()).hexdigest()}
    path.write_text(payload + json.dumps(footer) + '\n')


class CompactNegativeReviewTests(unittest.TestCase):
    def test_canonical_order_equals_clingo_for_mixed_signatures_and_arguments(self):
        texts = ['z', 'a', 'q(2)', 'q(10)', 'q(-2)', 'q(#inf)', 'q(#sup)',
                 'q("z")', 'q("a\\\"b")', 'q(a)', 'q(z)', 'q(f(1))',
                 'q((a,1))', 'q(g(2,1))', 'a(z,z)', 'z(a,a)',
                 'q(2,1)', 'q(1,2)', 'q(1,10)', 'q(f(2),"x,(y)")']
        expected = sorted({clingo.parse_term(text) for text in texts})
        value = compact(reversed(texts + texts[:4]))
        self.assertEqual(list(value), expected)
        self.assertEqual(list(value.iter_text()), list(map(str, expected)))
        self.assertEqual(list(canonical_text(value)), list(map(str, expected)))
        self.assertEqual(len(value), len(expected))

    def test_tuple_holes_and_sparse_storage_never_complete_a_product(self):
        # A diagonal creates >4095 argument IDs with one value per bucket.
        texts = [f't({i},{i})' for i in range(4205)]
        value = compact(reversed(texts + texts[-10:]))
        self.assertEqual(len(value), 4205)
        self.assertLess(value.stats()['bitmap_buckets'], value.stats()['buckets'])
        for i in (0, 1, 4095, 4096, 4204):
            self.assertIn(clingo.parse_term(f't({i},{i})'), value)
            self.assertNotIn(clingo.parse_term(f't({i},{(i + 1) % 4205})'), value)
        expected = {clingo.parse_term(text) for text in texts}
        self.assertEqual(value, expected)
        self.assertEqual(expected, value)

    def test_set_operations_in_both_directions_and_non_mutating_queries(self):
        value = compact(['t(a)', 't(b)', 'u(a,b)', 'z'])
        expected = {clingo.parse_term(text) for text in ['t(a)', 't(b)', 'u(a,b)', 'z']}
        other = {clingo.parse_term(text) for text in ['t(b)', 'u(a,b)', 't(c)']}
        before = value.stats()
        for absent in ['t(c)', 't(a,b)', 'u(b,a)', 'u(a)', '-t(a)', '1', '"t(a)"']:
            self.assertNotIn(clingo.parse_term(absent), value)
        self.assertNotIn('t(a)', value)
        self.assertEqual(value.stats(), before)
        for rhs in (set(other), frozenset(other), compact(map(str, other))):
            self.assertEqual(value & rhs, expected & other)
            self.assertEqual(rhs & value, other & expected)
            self.assertEqual(value | rhs, expected | other)
            self.assertEqual(rhs | value, other | expected)
            self.assertEqual(value - rhs, expected - other)
            self.assertEqual(rhs - value, other - expected)
            self.assertEqual(value ^ rhs, expected ^ other)
            self.assertEqual(rhs ^ value, other ^ expected)
        self.assertTrue(value <= expected)
        self.assertTrue(expected <= value)

    def test_parse_and_io_never_intern_outer_negative_symbols(self):
        texts = ['target(f(a,2),"x,(y)\\\"z")', 'target(f(b,10),(a,2))', 'zero']
        expected = list(map(str, sorted(map(clingo.parse_term, texts))))
        original_parse = clingo.parse_term
        parsed = []
        def arguments_only(text):
            parsed.append(text)
            self.assertFalse(text.startswith('target('), text)
            self.assertNotEqual(text, 'zero')
            return original_parse(text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            with patch('nidavellir.labels.clingo.parse_term', side_effect=arguments_only), \
                 patch.object(AtomRecord, 'to_symbol', side_effect=AssertionError('outer Symbol')):
                value = compact(texts)
                self.assertEqual(list(value.iter_text()), expected)
                fingerprint = value.dump(path, source_sha256='source')
                restored = CompactNegativeSet.load(path, source_sha256='source', artifact_sha256=fingerprint)
                self.assertEqual(list(restored.iter_text()), expected)
                self.assertEqual(len(list(restored.iter_records())), 3)
        self.assertTrue(parsed)

    def test_artifact_empty_nullary_dense_and_sparse_roundtrips(self):
        cases = [[], ['z'], ['t(1)', 't(2)', 't(1)'],
                 [f't({i},{i})' for i in range(4205)]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            for texts in cases:
                with self.subTest(count=len(texts)):
                    expected = compact(texts)
                    fingerprint = expected.dump(path, source_sha256='source')
                    actual = CompactNegativeSet.load(path, source_sha256='source', artifact_sha256=fingerprint)
                    self.assertEqual(list(actual.iter_text()), list(expected.iter_text()))
                    self.assertEqual(actual.stats(), expected.stats())

    def test_corruption_partial_wrong_source_and_wrong_external_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            value = compact(['t(a,b)', 't(a,c)'])
            fingerprint = value.dump(path, source_sha256='source')
            original = path.read_text()
            for text in (original.rsplit('\n', 2)[0] + '\n', original.replace('"count":2', '"count":3'),
                         original + '{}\n'):
                with self.subTest(text=text[-70:]):
                    path.write_text(text)
                    with self.assertRaises((ValueError, KeyError)):
                        CompactNegativeSet.load(path, source_sha256='source')
            path.write_text(original)
            with self.assertRaises(ValueError):
                CompactNegativeSet.load(path, source_sha256='wrong-source')
            with self.assertRaises(ValueError):
                CompactNegativeSet.load(path, source_sha256='source', artifact_sha256='0' * 64)
            self.assertEqual(CompactNegativeSet.load(path, source_sha256='source', artifact_sha256=fingerprint), value)

    def test_recomputed_checksum_does_not_bypass_structural_validation(self):
        mutations = [
            lambda rows: rows[0].update(count=999),
            lambda rows: rows[0].update(tables=999),
            lambda rows: rows[1].update(pivot=99),
            lambda rows: rows[1].update(zero=True),
            lambda rows: rows[2].update(key=[999]),
            lambda rows: rows[2].update(bits='0x100000'),
            lambda rows: rows.insert(-1, dict(rows[2])),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            for mutation in mutations:
                compact(['t(a,b)', 't(a,c)']).dump(path, source_sha256='source')
                rewrite_artifact(path, mutation)
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                    CompactNegativeSet.load(path, source_sha256='source')

    def test_atomic_dump_preserves_previous_complete_artifact_when_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            compact(['t(a)']).dump(path, source_sha256='old')
            previous = path.read_bytes()
            calls = 0
            def interrupt():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise TimeoutError('review cancellation')
            with self.assertRaises(TimeoutError):
                compact(['t(b)', 'u(c)']).dump(path, source_sha256='new', check=interrupt)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(CompactNegativeSet.load(path, source_sha256='old').iter_text()), ['t(a)'])

    def test_fallback_reparses_all_buffered_labels_and_late_constant_aliases(self):
        texts = [
            'negative(alias).\n#const alias=t(a).\npositive(t(b)).\n',
            'negative(t(a)).\nnegative(t(1..3)).\npositive(t(9)).\n',
            'negative(t(a)).\nnegative(t(1+2)).\npositive(t(4)).\n',
            'negative(t(a)). % end\nnegative(t(b)).\npositive(t(c)).\n',
            'negative(t(a)).\n#include "extra.lp".\npositive(t(c)).\n',
        ]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / 'labels.lp'
            (folder / 'extra.lp').write_text('negative(t(b)).\n')
            for text in texts:
                path.write_text(text)
                with self.subTest(text=text):
                    expected = _examples_legacy(path, {('t', 1)})
                    actual = _examples(path, {('t', 1)})
                    self.assertEqual(actual[1:3], expected[1:3])
            for text in ['negative(alias).\npositive(t(a)).\n#const alias=t(a).\n',
                         'negative(t(a)).\npositive(t(a)).\n']:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(TaskError):
                    _examples(path, {('t', 1)})


if __name__ == '__main__':
    unittest.main()
