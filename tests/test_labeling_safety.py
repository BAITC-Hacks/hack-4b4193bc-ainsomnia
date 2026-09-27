"""Ready-file safety and symmetric category exchange without real data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from src import paths
from src.checks import labeling


class LabelingTests(unittest.TestCase):
    def test_unknown_text_and_symmetric_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, base, review, marker = [root / n for n in
                                          ("unified.parquet", "base.json", "review.csv", "SOURCE")]
            marker.write_text(paths.SOURCE)
            base.write_text(json.dumps({"Дороги": ["дороги", "problem"],
                                        "Вода": ["водоснабжение и канализация", "problem"]}))
            with patch.object(labeling, "DATA", data), patch.object(labeling, "BASE", base), \
                 patch.object(labeling, "REVIEW", review), patch.object(paths, "SOURCE_MARK", marker), \
                 patch("sys.argv", ["labeling"]):
                labeling.known_categories.cache_clear()
                frame = pd.DataFrame({"region": ["Тестовый регион"] * 2,
                                      "category": ["Дороги", "Вода"],
                                      "topic": ["водоснабжение и канализация", "дороги"],
                                      "appeal_class": ["problem"] * 2})
                frame.to_parquet(data)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(labeling.main(), 1)
                marker_text = "произвольный закрытый текст категории"
                frame.loc[:, "category"] = marker_text
                frame.loc[:, "topic"] = "дороги"
                frame.to_parquet(data)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(labeling.main(), 0)
                saved = review.read_text()
                self.assertNotIn(marker_text, saved + output.getvalue())
                self.assertIn("category-sha256:", saved)
                self.assertEqual(int(pd.read_csv(review)["строк"].sum()), 2)
                labeling.known_categories.cache_clear()


if __name__ == "__main__":
    unittest.main()
