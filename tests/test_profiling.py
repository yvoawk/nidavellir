import unittest

from nidavellir.profiling import SearchProfile, timed


class ProfilingTests(unittest.TestCase):
    def test_nested_timeout_charges_each_interval_once_and_unwinds(self):
        timestamps = iter((0, 2, 5, 8, 10, 12))
        profile = SearchProfile(lambda: next(timestamps))
        with self.assertRaises(TimeoutError):
            with profile.measure("outer"):
                with profile.measure("inner"):
                    raise TimeoutError()
        with profile.measure("after"):
            pass
        spans = profile.export()["timings"]
        self.assertEqual(spans["outer"], {"calls": 1, "seconds": 8, "exclusive_seconds": 5})
        self.assertEqual(spans["inner"], {"calls": 1, "seconds": 3, "exclusive_seconds": 3})
        self.assertEqual(spans["after"]["seconds"], 2)
        self.assertEqual(sum(s["exclusive_seconds"] for s in spans.values()), 10)
        self.assertEqual(profile.stack, [])

    def test_decorator_counts_repeated_queries_and_preserves_result(self):
        timestamps = iter(range(4))
        class Oracle:
            profile = SearchProfile(lambda: next(timestamps))
            @timed("exists")
            def query(self, value):
                return value > 0
        oracle = Oracle()
        self.assertTrue(oracle.query(1))
        self.assertFalse(oracle.query(0))
        self.assertEqual(oracle.profile.export()["timings"]["exists"],
                         {"calls": 2, "seconds": 2, "exclusive_seconds": 2})


if __name__ == "__main__":
    unittest.main()
