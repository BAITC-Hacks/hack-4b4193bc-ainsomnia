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
import sys
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SourceError(SystemExit):
    """Отказ до обработки данных; CLI получает код 2, UI может показать причину."""

    def __init__(self, message):
        self.message = "ОШИБКА ИСТОЧНИКА: " + message
        print(self.message, file=sys.stderr)
        super().__init__(2)

    def __str__(self):
        return self.message


def canonical(path):
    return Path(path).expanduser().resolve()


def overlaps(a, b):
    return a.is_relative_to(b) or b.is_relative_to(a)

SOURCE = os.environ.get("NAZAR_SOURCE", "real").strip().lower() or "real"
if SOURCE not in ("real", "fake"):
    raise SystemExit(f"NAZAR_SOURCE={SOURCE!r}: допустимо real или fake (CLAUDE.md, 5p)")
FAKE = SOURCE == "fake"

REAL_RAW = ROOT / "drive-download-20260907T161509Z-1-001"
FAKE_RAW = ROOT / "tests" / "fixtures" / "fake_export"
_raw = os.environ.get("NAZAR_RAW_DIR")
RAW_DIR = canonical(_raw if _raw else (FAKE_RAW if FAKE else REAL_RAW))


@lru_cache(maxsize=1)
def fake_fingerprints():
    """Единственный список известных raw SHA256 — версионируемый manifest fixture."""
    try:
        manifest = json.loads((FAKE_RAW / "manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("empty files")
        hashes = frozenset(files.values())
        if any(not isinstance(h, str) or len(h) != 64
               or any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("invalid SHA256")
        return hashes
    except (OSError, ValueError, KeyError, TypeError):
        raise SourceError("не удалось прочитать SHA256 manifest fake fixture; "
                          "восстановите tests/fixtures/fake_export/manifest.json.") from None


def validate_raw_source(path, *, recursive=False):
    """Известные расположения и точные копии fake raw; неизвестные SHA256 допустимы."""
    resolved = canonical(path)
    if not FAKE:
        for forbidden in (FAKE_RAW, ROOT / "tests/fixtures/synth", ROOT / "data/synth"):
            if resolved.is_relative_to(canonical(forbidden)):
                raise SourceError("fake fixture / synthetic corpus нельзя использовать в real mode; "
                                  "выберите настоящий источник или NAZAR_SOURCE=fake для теста.")
        if resolved.is_file():
            known = fake_fingerprints()
            with resolved.open("rb") as stream:
                fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
            if fingerprint in known:
                raise SourceError("известная fake fixture по SHA256 fingerprint: "
                                  "этот файл нельзя использовать в real mode.")
    if recursive and not FAKE and resolved.is_dir():
        pending, seen = [resolved], set()
        while pending:
            directory = pending.pop()
            if directory in seen:
                continue
            seen.add(directory)
            for child in directory.iterdir():
                checked = validate_raw_source(child)
                if checked.is_dir():
                    pending.append(checked)
    return resolved

_work = os.environ.get("NAZAR_WORK_DIR")
if _work and not FAKE:
    raise SystemExit("NAZAR_WORK_DIR действует только при NAZAR_SOURCE=fake: результаты настоящей "
                     "выгрузки пишутся в data/, reports/, models/ репозитория (CLAUDE.md, 5p)")
WORK = canonical((_work if _work else ROOT / "fake_run") if FAKE else ROOT)
RUNTIME = os.environ.get('NAZAR_RUNTIME_DIR')
STAGING = os.environ.get('_NAZAR_BUILD_RELEASE')
RELEASE_ID = None
RELEASE_ERROR = None
if STAGING and not RUNTIME:
    raise SourceError('Staging context требует отдельный runtime.')
if RUNTIME:
    from src.release_store import runtime_path, selected, ReleaseError
    if _work:
        raise SourceError('NAZAR_WORK_DIR и release mode нельзя использовать одновременно.')
    try:
        RUNTIME = runtime_path(RUNTIME, SOURCE, ROOT)
        WORK, RELEASE_ID = selected(RUNTIME, SOURCE, STAGING)
    except (ReleaseError, OSError, ValueError):
        RELEASE_ERROR = 'Нет совместимого ACTIVE release; проверьте nazar-health и nazar-refresh.'
        if STAGING:
            raise SourceError('Некорректный staging context.') from None
        WORK = ROOT / 'runtime-unavailable'


def require_writable():
    if RUNTIME and not STAGING:
        raise SourceError('ACTIVE release неизменяем: используйте nazar-refresh; обучение — отдельно в legacy mode.')


def validate_work_dir(path):
    require_writable()
    work = canonical(path)
    if FAKE:
        if work == ROOT:
            raise SourceError("fake work dir не может совпадать с корнем проекта.")
        protected = [canonical(ROOT / name) for name in ("data", "reports", "models")]
        for name in ("data", "reports", "models"):
            output = work / name
            candidates = [canonical(output)]
            if output.is_dir():
                candidates.extend(canonical(p) for p in output.rglob("*") if p.is_symlink())
            if any(overlaps(candidate, real) for candidate in candidates for real in protected):
                raise SourceError("fake outputs пересекаются с рабочими data/, reports/ или models/; "
                                  "выберите отдельный NAZAR_WORK_DIR.")
    return work


validate_raw_source(RAW_DIR)
if not RUNTIME or STAGING:
    validate_work_dir(WORK)
DATA_DIR = WORK / "data"
REPORTS_DIR = WORK / "reports"
MODELS_DIR = WORK / "models"

UNIFIED = DATA_DIR / "unified.parquet"
BY_REGION = DATA_DIR / "by_region"
SOURCE_MARK = DATA_DIR / "SOURCE"
RISK_SOURCE_MARK = REPORTS_DIR / "SOURCE"
PREDICTIONS = REPORTS_DIR / "predictions.csv"
METRICS = REPORTS_DIR / "metrics.json"
BASELINE_REPORT = REPORTS_DIR / "baseline_vs_model.md"
MODEL = MODELS_DIR / "model.pkl"

KARAGANDA_CSV = "Обращения граждан 109 - Карагандинская область.csv"


def write_source_marker(marker, raw):
    """Вызывать после успешного создания артефактов из проверенного источника."""
    validate_raw_source(raw, recursive=True)
    validate_work_dir(WORK)
    Path(marker).write_text(SOURCE + "\n", encoding="utf-8")


def require_source_marker(marker):
    try:
        source = Path(marker).read_text(encoding="utf-8").strip()
    except OSError:
        raise SourceError("нет SOURCE-маркера; повторите nazar-build-data для данных "
                          "или python train.py для результатов риска.") from None
    if source != SOURCE:
        raise SourceError("SOURCE-маркер не совпадает с выбранным режимом; "
                          "смешивать real и fake нельзя. Пересоберите нужный источник.")
    return source


def data_is_fake() -> bool:
    return require_source_marker(SOURCE_MARK) == "fake"
