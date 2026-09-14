"""The low-memory paths must preserve the established ASP label contract."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clingo

from nidavellir.task import Task, TaskError, _examples, _examples_legacy


class StreamingExampleTests(unittest.TestCase):
    def compare(self, text, targets=None, include=None):
        targets = targets or {("t", 1), ("u", 2), ("ok", 0)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.lp"
            path.write_text(text)
            if include:
                (Path(folder) / "more.lp").write_text(include)
            expected = _examples_legacy(path, targets)
            actual = _examples(path, targets)
        self.assertEqual(actual[1:3], expected[1:3])
        return actual

    def test_plain_facts_do_not_parse_global_ast_or_solve(self):
        text = 'positive(t(a)).\nnegative(t(b)).\npositive(u(f(1),"a,;#%@.\\\"b")).\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.lp"
            path.write_text(text)
            with patch("nidavellir.task.ast.parse_files", side_effect=AssertionError("AST parser")), \
                 patch("nidavellir.task._model", side_effect=AssertionError("Clingo model")):
                source, positive, negative, _ = _examples(path, {("t", 1), ("u", 2)})
        self.assertIsNone(source)
        self.assertEqual(len(positive), 2)
        self.assertEqual(negative, frozenset({clingo.parse_term("t(b)")}))

    def test_deduplicates_both_polarities_and_handles_empty_files(self):
        for text in ("", "% Nothing labelled\n", "positive(ok).\npositive(ok).\nnegative(t(a)).\nnegative(t(a)).\n"):
            with self.subTest(text=text):
                result = self.compare(text)
                self.assertIsNone(result[0])

    def test_layout_comments_and_includes_use_streaming_ast_without_model(self):
        texts = [
            "positive(t(a)). negative(t(b)).",
            "positive (\n t(a)\n).\nnegative(t(b)).",
            "%* block\n comment *% positive(t(a)).\nnegative(t(b)). % end\n",
            '#include "more.lp".\npositive(t(a)).',
            "#show ignored/1.\n#show u(X,1) : something(X).\npositive(t(a)).",
        ]
        for text in texts:
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "labels.lp"
                    path.write_text(text)
                    (Path(folder) / "more.lp").write_text("negative(t(b)).")
                    expected = _examples_legacy(path, {("t", 1)})
                    with patch("nidavellir.task._model", side_effect=AssertionError("Clingo model")):
                        actual = _examples(path, {("t", 1)})
                    self.assertIsNone(actual[0])
                    self.assertEqual(actual[1:3], expected[1:3])

    def test_constants_arithmetic_intervals_and_pools_keep_normalization(self):
        texts = [
            "#const n=3. positive(t(n)). negative(t(n+1)).",
            "positive(t(1+2)). negative(t(6/2+1)).",
            "positive(t(1..3)). negative(t(4;5)).",
            "positive(t(a;b)). negative(u(c,1;d,2)).",
            "positive(t(-2)). negative(t(#inf)). negative(t(#sup)).",
            "positive(t(1/0)). negative(t(b)).",
        ]
        for text in texts:
            with self.subTest(text=text):
                result = self.compare(text)
                self.assertIsInstance(result[0], str)

    def test_later_constant_can_reinterpret_an_apparently_invalid_symbol(self):
        result = self.compare("positive(n).\n#const n=ok.\n")
        self.assertEqual(result[1], frozenset({clingo.parse_term("ok")}))

    def test_nested_terms_strings_and_tuple_constructors(self):
        self.compare('positive(t(f(g(a),2))).\nnegative(u("% not a comment",(a,1))).\n')

    def test_rejects_all_non_fact_label_forms_as_before(self):
        texts = [
            "positive(t(a)).\nnegative(t(a)).\n",
            "positive(other(a)).",
            "positive(t(X)).",
            "positive(t(a)) :- other(a).",
            "{ positive(t(a)) }.",
            "positive(t(a)); negative(t(b)).",
            "not positive(t(a)).",
            "-positive(t(a)).",
            "positive(-t(a)).",
            "positive(1).",
            "positive(t(@f())).",
            "#show @f().\npositive(t(a)).",
            "#program another.\npositive(t(a)).",
            "positive(t(a)).\nunrelated(a).",
            "positive(t(a)).\n:- positive(t(a)).",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.lp"
            for text in texts:
                path.write_text(text)
                with self.subTest(text=text):
                    with self.assertRaises(TaskError):
                        _examples_legacy(path, {("t", 1)})
                    with self.assertRaises(TaskError):
                        _examples(path, {("t", 1)})

    def test_invalid_syntax_and_missing_include_are_errors(self):
        for text in ("positive(t(a).", '#include "missing.lp".\npositive(t(a)).'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "labels.lp"
                path.write_text(text)
                with self.assertRaises(TaskError):
                    _examples(path, {("t", 1)})

    def test_snapshot_is_self_contained_after_original_input_disappears(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            (source / "bk.lp").write_text("p(a). p(b).")
            (source / "bias.lp").write_text("candidate_head(t(var__(1))). allowed_literal(p(var__(1))).\n"
                                           "max_body(1). head(H) :- candidate_head(H). {body(L)} :- allowed_literal(L).")
            (source / "labels.lp").write_text('#include "more.lp".\npositive(t(a)).')
            (source / "more.lp").write_text("negative(t(b)).")
            task = Task.load(source / "bk.lp", source / "labels.lp", source / "bias.lp")
            self.assertIsNone(task.examples_source)
            self.assertEqual(task.examples_path, (source / "labels.lp").resolve())
            (source / "labels.lp").unlink()
            (source / "more.lp").unlink()
            snapshot = Path(folder) / "snapshot.lp"
            task.write_examples(snapshot)
            reloaded = _examples_legacy(snapshot, {("t", 1)})
            self.assertEqual(reloaded[1], task.positives)
            self.assertEqual(reloaded[2], task.negatives)

    def test_legacy_snapshot_retains_normalized_source(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "bk.lp").write_text("p(a).")
            (folder / "bias.lp").write_text("candidate_head(t(var__(1))). allowed_literal(p(var__(1))).\n"
                                           "max_body(1). head(H) :- candidate_head(H). {body(L)} :- allowed_literal(L).")
            (folder / "labels.lp").write_text("#const n=2. positive(t(1..n)).")
            task = Task.load(folder / "bk.lp", folder / "labels.lp", folder / "bias.lp")
            snapshot = folder / "snapshot.lp"
            task.write_examples(snapshot)
            self.assertEqual(snapshot.read_text(), task.examples_source)
            reloaded = _examples_legacy(snapshot, {("t", 1)})
            self.assertEqual(reloaded[1], task.positives)


if __name__ == "__main__":
    unittest.main()
