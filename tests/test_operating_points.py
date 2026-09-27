"""Degenerate comparisons and attainable thresholds, with no real data."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import train


class OperatingPointsTests(unittest.TestCase):
    def test_empty_and_no_positive(self):
        for y in (np.array([], dtype=int), np.zeros(12, dtype=int)):
            p = np.zeros(len(y))
            self.assertIsNone(train._at_threshold(y, p, .5)["recall"])
            self.assertIsNone(train._point_for(y, p, .8, "recall"))
            self.assertIsNone(train._point_for(y, p, .65, "precision"))

    def test_ties_must_meet_target(self):
        y = np.array([1, 0, 0, 0])
        p = np.array([.9, .9, .8, .8])
        self.assertIsNone(train._point_for(y, p, .65, "precision"))
        got = train._point_for(y, p, .8, "recall")
        self.assertEqual(got["n_flagged"], 2)
        self.assertEqual(got["recall"], 1)
        for metric in ("recall", "precision"):
            for target in (.2, .5, .8, 1):
                point = train._point_for(y, p, target, metric)
                if point is not None:
                    self.assertGreaterEqual(point[metric], target)

    def test_unavailable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            with patch.object(train.paths, "BASELINE_REPORT", report), contextlib.redirect_stdout(io.StringIO()):
                for y in (np.array([], dtype=int), np.zeros(12), np.ones(12), np.array([0, 1])):
                    got = train.compare_operating_points(y, np.zeros(len(y)), np.zeros(len(y)), {}, "model")
                    self.assertIn("unavailable", got)
                    self.assertIn("не определен", report.read_text())
                y = np.array([0, 1] * 10)
                results = {k: {"at_0.5": {"roc_auc": .5, "pr_auc": .5}}
                           for k in ("baseline_subcat", "model")}
                self.assertIn("unavailable", train.compare_operating_points(y, y, y, results, "model"))


if __name__ == "__main__":
    unittest.main()
