"""Manifest unit tests use only fake CSV copies; source guards remain independent."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.dataset_manifest import (ManifestError, accept, generate, read_manifest,
                                  verify, write_json)
from src import paths


class ManifestTests(unittest.TestCase):
    def test_version_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            shutil.copytree(paths.FAKE_RAW, raw)
            candidate, accepted = root / "candidate.json", root / "accepted.json"
            kwargs = {"source": "fake", "include_xlsx": False}
            first = generate(raw, **kwargs)
            self.assertEqual(first["dataset_id"], generate(raw, **kwargs)["dataset_id"])
            self.assertEqual(sum(f["rows"] for f in first["files"]), 12950)
            write_json(candidate, first)
            with self.assertRaises(ManifestError):
                verify(raw, accepted, **kwargs)
            with self.assertRaises(ManifestError):
                accept(raw, candidate, "wrong", accepted, **kwargs)
            accept(raw, candidate, first["dataset_id"], accepted, **kwargs)
            self.assertEqual(verify(raw, accepted, **kwargs)["dataset_id"], first["dataset_id"])
            file = raw / first["files"][0]["relative_path"]
            # A blank line changes the fingerprint, while preserving the CSV schema/rows.
            file.write_bytes(file.read_bytes() + b"\n")
            with self.assertRaisesRegex(ManifestError, "отличается"):
                verify(raw, accepted, **kwargs)
            with self.assertRaises(ManifestError):
                accept(raw, candidate, first["dataset_id"], accepted, **kwargs)
            second = generate(raw, **kwargs)
            self.assertNotEqual(first["dataset_id"], second["dataset_id"])
            write_json(candidate, second)
            accept(raw, candidate, second["dataset_id"], accepted, **kwargs)
            self.assertEqual(len(list((root / "manifests").glob("*.json"))), 2)
            self.assertEqual(verify(raw, accepted, **kwargs)["dataset_id"], second["dataset_id"])
            moved = file.with_suffix(".missing")
            file.rename(moved)
            with self.assertRaises(ManifestError):
                verify(raw, accepted, **kwargs)
            moved.rename(file)
            content = json.loads(accepted.read_text())
            content["files"][0]["rows"] += 1
            write_json(accepted, content)
            with self.assertRaises(ManifestError):
                read_manifest(accepted, **kwargs)
            accepted.write_text("broken")
            with self.assertRaises(ManifestError):
                read_manifest(accepted, **kwargs)
            file.write_text("unexpected_column\nnot a row\n")
            with self.assertRaisesRegex(ManifestError, "заголовок"):
                generate(raw, **kwargs)


if __name__ == "__main__":
    unittest.main()
