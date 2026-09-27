"""Единственное место, где задаются пути к выгрузке и к результатам (CLAUDE.md, 5p).

    NAZAR_SOURCE=real  (по умолчанию) — настоящая выгрузка, результаты в data/, reports/, models/
    NAZAR_SOURCE=fake  — поддельная выгрузка из tests/fixtures/fake_export/,
                         результаты в fake_run/data, fake_run/reports, fake_run/models
    NAZAR_RAW_DIR=...  — другой каталог выгрузки (по умолчанию прежний; в режиме fake — для
                         варианта набора из теста)
    NAZAR_WORK_DIR=... — только в режиме fake: другой каталог результатов вместо fake_run/

Поддельный прогон не пишет ни в один отслеживаемый git файл: разные пути столкнуться
не могут, в отличие от отметки, которую можно не прочитать. Файлы репозитория —
CLAUDE.md, tests/*.json — читаются из корня в обоих режимах.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE = os.environ.get("NAZAR_SOURCE", "real").strip().lower() or "real"
if SOURCE not in ("real", "fake"):
    raise SystemExit(f"NAZAR_SOURCE={SOURCE!r}: допустимо real или fake (CLAUDE.md, 5p)")
FAKE = SOURCE == "fake"

REAL_RAW = ROOT / "drive-download-20260907T161509Z-1-001"
FAKE_RAW = ROOT / "tests" / "fixtures" / "fake_export"
_raw = os.environ.get("NAZAR_RAW_DIR")
RAW_DIR = Path(_raw).expanduser() if _raw else (FAKE_RAW if FAKE else REAL_RAW)

_work = os.environ.get("NAZAR_WORK_DIR")
if _work and not FAKE:
    raise SystemExit("NAZAR_WORK_DIR действует только при NAZAR_SOURCE=fake: результаты настоящей "
                     "выгрузки пишутся в data/, reports/, models/ репозитория (CLAUDE.md, 5p)")
WORK = (Path(_work).expanduser() if _work else ROOT / "fake_run") if FAKE else ROOT
DATA_DIR = WORK / "data"
REPORTS_DIR = WORK / "reports"
MODELS_DIR = WORK / "models"

UNIFIED = DATA_DIR / "unified.parquet"
BY_REGION = DATA_DIR / "by_region"
SOURCE_MARK = DATA_DIR / "SOURCE"
PREDICTIONS = REPORTS_DIR / "predictions.csv"
METRICS = REPORTS_DIR / "metrics.json"
BASELINE_REPORT = REPORTS_DIR / "baseline_vs_model.md"
MODEL = MODELS_DIR / "model.pkl"

KARAGANDA_CSV = "Обращения граждан 109 - Карагандинская область.csv"


def data_is_fake() -> bool:
    """Собранные данные — из поддельной выгрузки: по отметке, которую пишет сборка."""
    try:
        return SOURCE_MARK.read_text(encoding="utf-8").strip() == "fake"
    except OSError:
        return FAKE
