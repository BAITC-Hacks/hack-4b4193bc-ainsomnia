"""Fixed health/drift boundaries on artificial aggregates, without real files."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from src import data_health as health, paths
from src.data_drift import compare
from src.dataset_manifest import digest, write_json

REGION = "Костанайская область"


def sample():
    frame = pd.DataFrame({"created_at": pd.to_datetime(["2025-01-01", "2025-01-09", "2025-01-18"]),
                          "region": [REGION] * 3, "category": ["Дороги"] * 3,
                          "topic": ["дороги"] * 3, "appeal_class": ["problem"] * 3,
                          "district": [None] * 3, "executor": ["private marker"] * 3,
                          "status": [None] * 3, "sla_breach": [None] * 3})
    return frame, health.profile(frame, {"dataset_id": "a" * 64}, {}, source="fake",
                                 built_at="2026-09-27T00:00:00+00:00")


class HealthTests(unittest.TestCase):
    def test_gaps_and_ready_file(self):
        frame, result = sample()
        item = result["regions"][REGION]
        self.assertEqual(item["history_days"], 18)
        self.assertEqual(item["observed_days"], 3)
        self.assertEqual(item["longest_gap"], 8)
        self.assertEqual([x["days"] for x in item["gaps_ge7"]], [7, 8])
        self.assertEqual(item["class_shares"], {"problem": 1., "info": 0., "system": 0.})
        self.assertFalse(item["forecast"]["available"])
        self.assertEqual(health.missing_runs(pd.Series([], dtype="datetime64[ns]")), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            write_json(path, result)
            self.assertNotIn("private marker", path.read_text())
            self.assertNotIn("Дороги", path.read_text())
        with self.assertRaises(ValueError):
            health.profile(frame.iloc[:0], {"dataset_id": "a" * 64}, {}, source="fake", built_at="now")

    def test_drift_thresholds_and_zero_base(self):
        _, old = sample()
        self.assertFalse(compare(old, None)["available"])
        self.assertEqual(compare(old, old)["changes"], [])
        new = copy.deepcopy(old)
        a, b = old["regions"][REGION], new["regions"][REGION]
        a["rows"], b["rows"] = 100, 120
        b["class_shares"] = {"problem": .9, "info": .1, "system": 0}
        b["completeness"]["category"] = .9
        b["categories"]["b" * 64] = 20
        b["last_date"] = "2025-01-20"
        result = compare(new, old)
        flags = {x["field"]: x["warning"] for x in result["changes"]}
        self.assertTrue(flags["число строк"])
        self.assertTrue(flags["class_shares:info"])
        self.assertTrue(flags["completeness:category"])
        self.assertTrue(flags["состав категорий"])
        self.assertFalse(flags["last_date"])
        a["rows"] = 0
        self.assertTrue(compare(new, old)["available"])
        self.assertTrue(compare({**new, "regions": {}}, old)["changes"][0]["warning"])
        with self.assertRaises(ValueError):
            compare({**new, "source": "real"}, old)

    def test_profile_mismatch_and_build_failure(self):
        _, profile = sample()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, marker = root / "unified.parquet", root / "SOURCE"
            unified.write_bytes(b"fixture fingerprint")
            marker.write_text("fake\n")
            profile["unified_sha256"] = digest(unified)
            profile["drift"] = compare(profile, None)
            report, state = root / "health.json", root / "last_build.json"
            write_json(report, profile)
            with patch.object(health, "PROFILE", report), patch.object(health, "BUILD_STATE", state), \
                 patch.object(paths, "UNIFIED", unified), patch.object(paths, "SOURCE_MARK", marker), \
                 patch.object(paths, "SOURCE", "fake"), patch.object(paths, "FAKE", True):
                self.assertIsNone(health.load()[1])
                write_json(state, {"state": "failed", "source": "fake"})
                self.assertIsNotNone(health.load()[1])
                write_json(state, {"state": "ok", "source": "fake"})
                profile["source"] = "real"
                write_json(report, profile)
                self.assertIsNotNone(health.load()[1])
                profile["source"] = "fake"
                write_json(report, profile)
                unified.write_bytes(b"changed bytes")
                self.assertIsNotNone(health.load()[1])


if __name__ == "__main__":
    unittest.main()
