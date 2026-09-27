"""Fake-only UI scenarios, invoked by test_fake_export after build and train."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CASES = ("filters", "no_data", "no_predictions", "corrupt", "no_font", "no_chrome")


def exercise(case):
    from src import paths
    assert paths.FAKE, "UI tests require FAKE"
    if case == "no_data":
        paths.UNIFIED.rename(paths.UNIFIED.with_suffix(".unused"))
    elif case == "no_predictions":
        paths.PREDICTIONS.rename(paths.PREDICTIONS.with_suffix(".unused"))
    elif case == "corrupt":
        paths.UNIFIED.write_bytes(b"deliberately invalid parquet")
    elif case == "no_font":
        import src.export as ex
        ex.FONT_CANDIDATES[:] = []
    elif case == "no_chrome":
        import plotly.graph_objects as go
        def unavailable(*args, **kwargs):
            raise RuntimeError("test browser unavailable")
        go.Figure.to_image = unavailable

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=180).run()
    assert not at.exception, case
    text = lambda kind: "\n".join(e.value for e in getattr(at, kind))
    if case == "no_data":
        assert "Нет данных для витрины" in text("error")
    elif case == "no_predictions":
        assert "Нет готового" in text("warning") + text("info")
        assert any("Общие цифры" in s.value for s in at.subheader)
    elif case == "corrupt":
        assert "повреждены" in text("error")
        assert "Traceback" not in text("error") and "invalid parquet" not in text("error")
        assert not at.subheader
    elif case in ("no_font", "no_chrome"):
        next(b for b in at.button if b.label == "Собрать PDF").click().run()
        assert not at.exception
        assert "PDF" in text("warning") + text("error")
    else:
        def cards_match():
            assert not at.exception
            card = next(m.value for m in at.markdown if "Всплесков требуют внимания" in m.value)
            number = int(re.findall(r">(\d+)</div>$", card)[0])
            section = next(m.value for m in at.markdown if m.value.startswith("#### Требует внимания"))
            assert number == int(re.findall(r"\d+", section)[0])
        cards_match()
        at.radio(key="ev_order").set_value("по тому, во сколько раз больше обычного").run()
        cards_match()
        at.multiselect(key="ev_reg").set_value(["Костанайская область"]).run()
        cards_match()
        at.number_input(key="ev_new").set_value(1).run()
        cards_match()
        at.multiselect(key="ev_reg").set_value([]).run()
        cards_match()
        general_period = [v for v in at.date_input if not str(v.key).startswith("ev_")][0]
        day = general_period.value[0]
        general_period.set_value((day,)).run()
        assert not at.exception
        general_period.set_value((day, day)).run()
        assert not at.exception
        general_regions = [v for v in at.multiselect if v.label == "Регион" and v.key != "ev_reg"][0]
        general_regions.set_value([]).run()
        assert not at.exception
        assert "не попало ни одного обращения" in text("warning")
        assert any("Общие цифры" in s.value for s in at.subheader)
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert config["client"]["showErrorDetails"] == "none"


def main():
    assert os.environ.get("NAZAR_SOURCE") == "fake", "Run via tests/test_fake_export.py"
    work = Path(os.environ["NAZAR_WORK_DIR"])
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix="nazar-ui-") as directory:
            for name in ("data", "reports"):
                shutil.copytree(work / name, Path(directory) / name)
            result = subprocess.run([sys.executable, "-m", "tests.test_dashboard", case],
                                    cwd=ROOT, env={**os.environ, "NAZAR_WORK_DIR": directory},
                                    capture_output=True, text=True)
            if result.returncode:
                # Only fake fixture is present, but do not echo UI tables/exception payloads.
                print("FAIL UI scenario:", case)
                return 1
            print("UI scenario OK:", case)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2:
        exercise(sys.argv[1])
    else:
        raise SystemExit(main())
