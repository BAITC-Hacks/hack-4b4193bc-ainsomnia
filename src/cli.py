"""Две команды проекта и понятные сообщения, когда не хватает файла.

После `uv pip install --python .venv/bin/python -e .` в .venv/bin появляются:

    nazar-build-data   сырые выгрузки -> data/unified.parquet с темами, плюс проверка
    nazar-dashboard    витрина руководителя (аргументы streamlit передаются дальше,
                       например --server.port 8600)

Пути в проекте относительные (data/, reports/, CLAUDE.md), поэтому обе команды
сначала переходят в корень проекта и работают из любого каталога. Корень — это
каталог над src/ (пакет ставится в режиме -e, код остаётся в репозитории); так же
его находят адаптеры, поэтому отдельной настройки корня нет.

Модуль лёгкий — без pandas и streamlit на верхнем уровне: его импортируют
остальные модули ради `require`, и импорт не должен ничего тянуть.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from src import paths

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = paths.UNIFIED
BUILD_HINT = ("соберите её одной командой: .venv/bin/nazar-build-data "
              "(раздел 0 CLAUDE.md; нужны сырые выгрузки)")
RAW_HINT = ("сырых выгрузок в репозитории нет и не будет — в них персональные данные. "
            "Получите их у владельца данных и положите в корень репозитория "
            "(раздел 0 CLAUDE.md, шаг 3): 8 CSV и 4 XLSX по семи регионам, Павлодар — "
            "в подкаталоге «Обращения граждан Павлодар»")


def require(path, what, how):
    """Нет файла — сообщение «чего нет, где искали, что сделать» и выход с кодом 2.

    Трейсбек FileNotFoundError говорит, где упало, но не говорит, что делать."""
    p = Path(path)
    if p.exists():
        return p
    sys.stderr.write(f"\nНЕ ХВАТАЕТ: {what}\n  искали здесь: {p.resolve()}\n"
                     f"  что сделать: {how}\n\n")
    raise SystemExit(2)


def require_unified():
    p = require(UNIFIED, "сводной таблицы обращений", BUILD_HINT)
    paths.require_source_marker(paths.SOURCE_MARK)
    return p


def require_raw(path):
    return require(paths.validate_raw_source(path), "сырого файла выгрузки", RAW_HINT)


# ---------------------------------------------------------------- команды
BUILD_STEPS = (
    ("адаптеры схем: 12 файлов -> data/by_region/", "src.adapters.adapters"),
    ("сводная таблица -> data/unified.parquet", "src.adapters.build_unified"),
    ("темы и классы обращения", "src.topic_mapping"),
    # Свойства разметки, а не абсолютные числа: на новой выгрузке числа по темам
    # меняются всегда. tests.test_topic_coverage — для правок src/topic_mapping.py.
    ("проверка разметки: «прочее», сопоставление категорий, новые категории",
     "src.checks.labeling"),
    ("качество, изменение и свежесть данных", "src.data_health"),
)


def build_data():
    """Шаги 4–7 раздела 0 одной командой, строго по порядку.

    Каждый шаг — отдельный процесс: следующий читает то, что записал предыдущий,
    и на первой ошибке команда останавливается, а не собирает витрину из
    полуготовых файлов."""
    os.chdir(ROOT)
    from src.adapters.adapters import BASE_DIR
    require(BASE_DIR, "каталога с сырыми выгрузками", RAW_HINT)
    paths.validate_raw_source(BASE_DIR, recursive=True)
    paths.validate_work_dir(paths.WORK)
    from src.dataset_manifest import ManifestError, generate, verify, write_json
    from datetime import datetime, timezone
    input_manifest = None
    def state(value, stage):
        write_json(paths.DATA_DIR / "last_build.json", {"state": value, "stage": stage,
                   "source": paths.SOURCE, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "dataset_id": input_manifest["dataset_id"] if input_manifest else None})
    try:
        input_manifest = (generate(BASE_DIR, include_xlsx=False, source="fake") if paths.FAKE else verify())
    except ManifestError as exc:
        sys.stderr.write(str(exc) + "\n")
        state("failed", 0)
        return 2
    state("running", 0)
    if paths.FAKE:
        print(f"ПОДДЕЛЬНАЯ ВЫГРУЗКА: {BASE_DIR} -> результаты в {paths.DATA_DIR} (CLAUDE.md, 5p)")
    for i, (title, module) in enumerate(BUILD_STEPS, start=1):
        state("running", i)
        print(f"\n=== Шаг {i} из {len(BUILD_STEPS)}: {title}", flush=True)
        code = subprocess.run([sys.executable, "-m", module]).returncode
        if code:
            state("failed", i)
            sys.stderr.write(
                f"\nШаг {i} «{title}» завершился с ошибкой (код {code}). Дальше не иду: "
                f"следующие шаги читают то, что пишет этот. Сообщение шага — выше.\n")
            return code
    state("ok", len(BUILD_STEPS))
    print("\nГотово: data/unified.parquet собран и проверен. Витрина: "
          ".venv/bin/nazar-dashboard")
    return 0


def dashboard():
    """streamlit run src/dashboard.py из корня проекта.

    Без данных витрина всё равно поднимается и говорит на странице, чего не
    хватает, — это понятнее человеку в браузере, чем упавший процесс. Здесь
    то же предупреждение печатается в терминал."""
    os.chdir(ROOT)
    if not UNIFIED.exists():
        sys.stderr.write(f"\nПРЕДУПРЕЖДЕНИЕ: нет {UNIFIED.resolve()} — витрина откроется "
                         f"с инструкцией вместо данных; {BUILD_HINT}.\n\n")
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(ROOT / "src" / "dashboard.py"), *sys.argv[1:]]
    return stcli.main()
