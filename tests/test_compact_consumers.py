"""Canonical IO and direct-API fingerprints preserve exact compact labels."""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clingo

from nidavellir.labels import AtomRecord, NegativeBuilder, parse_atom
from nidavellir.oracle_cache import fingerprint
from nidavellir.relational_bound import _fingerprint
from nidavellir.task import Task


class CompactConsumerTests(unittest.TestCase):
    def test_snapshots_and_fingerprints_equal_sets_without_negative_interning(self):
        terms = ['t(10)', 't(2)', 't(a)', 't(f(a,2))']
        negatives = frozenset(map(clingo.parse_term, terms))
        builder, cache = NegativeBuilder(), OrderedDict()
        for text in reversed(terms):
            builder.add(parse_atom(text, cache))
        compact = builder.freeze()
        reference = Task('', frozenset(), '', None, (clingo.parse_term('t(var__(1))'),),
                         (), 1, frozenset([clingo.parse_term('t(z)')]), negatives)
        target = replace(reference, negatives=compact)
        expected_cache, expected_bound = fingerprint(reference), _fingerprint(reference)
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'set.lp', Path(directory)/'compact.lp'
            reference.write_examples(left)
            with patch.object(AtomRecord, 'to_symbol', side_effect=AssertionError('outer Symbol')):
                target.write_examples(right)
                self.assertEqual(fingerprint(target), expected_cache)
                self.assertEqual(_fingerprint(target), expected_bound)
            self.assertEqual(right.read_bytes(), left.read_bytes())


if __name__ == '__main__':
    unittest.main()
