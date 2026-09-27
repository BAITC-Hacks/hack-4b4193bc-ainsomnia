#!/usr/bin/env python3
"""Сквозной тест на поддельной выгрузке (CLAUDE.md, 5p). Реальные данные не нужны.

    .venv/bin/python tests/test_fake_export.py

Шаги — как у человека без доступа к данным, плюс сверка с эталоном 5p, записанным до
генерации:
  1. набор перегенерируется во временный каталог и совпадает с tests/fixtures/ до байта;
  2. три команды в режиме fake: nazar-build-data, train.py, витрина (через AppTest);
  3. сводная таблица, отсев, повторы, разметка, детектор, пропуск, ПДн — числа 5p;
  4. отслеживаемые git файлы не изменились;
  5. вариант с новой категорией, уходящей в info, роняет проверку разметки.
Код выхода 0 — всё сошлось; иначе печатается, что не сошлось.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BIN = Path(PY).parent
FAIL: list[str] = []


def check(ok, what):
    print(("  ок    " if ok else "  ПРОВАЛ ") + what)
    if not ok:
        FAIL.append(what)


def run(cmd, env_extra=None):
    env = {**os.environ, "NAZAR_SOURCE": "fake", **(env_extra or {})}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def git_status():
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout


def main() -> int:
    before = git_status()
    if __package__:
        from .test_source_isolation import snapshot
    else:
        from test_source_isolation import snapshot
    real_before = snapshot()
    guard = run([PY, str(ROOT / "tests/test_source_isolation.py")])
    check(guard.returncode == 0, "изоляция источников: отрицательные регрессии и SHA256 real")
    if guard.returncode:
        return 1

    print("1. Набор воспроизводится до байта")
    with tempfile.TemporaryDirectory() as tmp:
        r = run([PY, "-m", "src.fake.generate", "--out", tmp])
        check(r.returncode == 0, "генератор и его самопроверка: " + r.stdout.strip().splitlines()[-1]
              if r.returncode == 0 else "генератор упал: " + r.stderr[-500:])
        committed = json.loads((ROOT / "tests/fixtures/fake_export/manifest.json").read_text())
        regen = json.loads((Path(tmp) / "manifest.json").read_text())
        check(committed["files"] == regen["files"], "суммы sha256 совпадают с tests/fixtures/fake_export")

    print("2. Три команды")
    r = run([str(BIN / "nazar-build-data")])
    check(r.returncode == 0, "nazar-build-data — код 0")
    check("ОК: свойства разметки в норме" in r.stdout, "проверка разметки: ОК")
    check("Новые категории — список на просмотр: 0" in r.stdout, "новых категорий 0")
    t = run([PY, "train.py"])
    check(t.returncode == 0, "train.py — код 0")
    check(t.stdout.startswith("ВНИМАНИЕ. ПОДДЕЛЬНЫЕ ДАННЫЕ"), "train.py первой строкой говорит о поддельных данных")

    print("3. Эталон 5p")
    env = {**os.environ, "NAZAR_SOURCE": "fake"}
    probe = subprocess.run([PY, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True)
    if probe.returncode:
        check(False, "замер эталона упал: " + probe.stderr[-800:])
    else:
        got = json.loads(probe.stdout.strip().splitlines()[-1])
        for k, want in EXPECT.items():
            check(got.get(k) == want, f"{k}: {got.get(k)!r}" + ("" if got.get(k) == want else f", ожидалось {want!r}"))

    print("4. Отслеживаемые файлы")
    check(git_status() == before, "git status до и после прогона одинаков — поддельный прогон не пишет в git")
    check(snapshot() == real_before, "SHA256 настоящих артефактов после fake-прогона неизменны")

    print("5. Вариант с новой категорией")
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as work:
        g = run([PY, "-m", "src.fake.generate", "--out", raw, "--variant", "new-category"])
        check(g.returncode == 0, "вариант сгенерирован")
        b = run([str(BIN / "nazar-build-data")], {"NAZAR_RAW_DIR": raw, "NAZAR_WORK_DIR": work})
        check(b.returncode == 1, f"сборка упала с кодом 1 (фактически {b.returncode})")
        check("ВЫПАДАЕТ ИЗ ЖАЛОБ" in b.stdout and "класс info, 40 строк" in b.stdout,
              "сообщение: «ВЫПАДАЕТ ИЗ ЖАЛОБ … класс info, 40 строк»")

    print("\nИТОГ:", "всё сошлось с 5p" if not FAIL else f"не сошлось {len(FAIL)}")
    return 1 if FAIL else 0


EXPECT = {
    "строк": 12910,
    "по регионам и классам": {
        "Акмолинская область": [390, 5, 2], "Алматинская область": [318, 170, 0],
        "Восточно-Казахстанская область": [197, 300, 0], "Карагандинская область": [6000, 3000, 978],
        "Костанайская область": [495, 3, 2], "Павлодарская область": [420, 180, 0],
        "Туркестанская область": [446, 3, 1]},
    "в группах повторов / лишних копий": [66, 33],
    "train.py: выборка, train, test, отброшено по кавычкам и appeal_type": [5000, 3000, 2000, 20, 2],
    "metrics.json source": "fake",
    "сохранённый отчёт: плашка первой строкой": True,
    "всплески": [["Костанайская область", "благоустройство и озеленение", "2025-08-27", 40, 1.0, 1.0, 5.0, 40.0, 39.0, 1]],
    "пропуски выгрузки": [["Туркестанская область", "2025-03-01", "2025-05-31", 92]],
    "ПДн в сырье": {"телефон": 4, "12 цифр": 4, "адрес": 7, "отчество": 52, "названий-людей": 2, "строк с ними": 55},
    "ПДн в сводной": {"телефон": 0, "12 цифр": 0, "адрес": 0, "отчество": 0, "названий-людей": 2, "строк с ними": 55},
    "витрина: исключений": 0,
    "витрина: блоки": ["Что сейчас важно", "Всплески жалоб", "Риск просрочки — только Карагандинская область",
                       "Из чего состоит поток обращений по регионам", "О чём жалуются", "Жалобы по месяцам",
                       "Общие цифры", "Выгрузка отчётов"],
    "витрина: плашка поддельных данных": True,
    "витрина: требуют внимания": "1",
    "витрина: блок риска маскирует ИП": True,
}

# Замер — отдельным процессом, чтобы src.paths прочитал NAZAR_SOURCE=fake при импорте.
PROBE = r'''
import json, re, pandas as pd
from src import paths
from src.dashboard import load_events
from src import spikes
from src.checks.pii_scan import scan_dir, scan_frame, total
out = {}
u = pd.read_parquet(paths.UNIFIED)
out["строк"] = len(u)
t = pd.crosstab(u.region, u.appeal_class).reindex(columns=["problem", "info", "system"], fill_value=0)
out["по регионам и классам"] = {r: [int(x) for x in row] for r, row in t.iterrows()}
out["в группах повторов / лишних копий"] = [int(u.duplicated(keep=False).sum()), int(u.duplicated().sum())]
m = json.loads(paths.METRICS.read_text())
d = m["data"]
out["train.py: выборка, train, test, отброшено по кавычкам и appeal_type"] = [
    d["sample"], d["train_n"], d["test_n"], d["dropped_broken_quotes"], d["dropped_bad_appeal_type"]]
out["metrics.json source"] = m.get("source")
first_line = next(line for line in paths.BASELINE_REPORT.read_text().splitlines() if line.strip())
out["сохранённый отчёт: плашка первой строкой"] = (
    "ПОДДЕЛЬНЫЕ ДАННЫЕ" in first_line and "бессмысленны" in first_line)
_, _, ev = load_events()
out["всплески"] = [[r["регион"], r["тема"], str(pd.Timestamp(r["дата"]).date()), int(r["обращений"]),
                    float(r["медиана окна"]), float(r["MAD"]), float(r["порог"]), round(float(r["кратность"]), 1),
                    float(r["прирост"]), int(r["дней подряд"])] for _, r in ev.iterrows()]
daily, bounds = spikes.daily_counts(u[u.appeal_class == "problem"])
gaps = []
for reg, mask in spikes.gap_days(u, bounds).items():
    if mask.any():
        s = mask[mask]
        gaps.append([reg, str(s.index.min().date()), str(s.index.max().date()), int(mask.sum())])
out["пропуски выгрузки"] = gaps
out["ПДн в сырье"] = total(scan_dir(paths.RAW_DIR))
out["ПДн в сводной"] = scan_frame(u.drop(columns=["source_file"]), "canon")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(paths.ROOT / "src" / "dashboard.py"), default_timeout=600).run()
out["витрина: исключений"] = len(at.exception)
out["витрина: блоки"] = [s.value for s in at.subheader]
out["витрина: плашка поддельных данных"] = any("ПОДДЕЛЬНЫЕ ДАННЫЕ" in e.value for e in at.error)
card = [m.value for m in at.markdown if "Всплесков требуют внимания" in m.value]
out["витрина: требуют внимания"] = re.findall(r">(\d+)</div>$", card[0])[0] if card else None
risk = [x.value for x in at.dataframe if "служба" in x.value.columns]
out["витрина: блок риска маскирует ИП"] = bool(risk) and any(s.startswith("ИП-") for s in risk[0]["служба"]) \
    and not any("Тестова" in s for s in risk[0]["служба"])
print(json.dumps(out, ensure_ascii=False, default=str))
'''

if __name__ == "__main__":
    # По умолчанию весь сквозной прогон проверяет внешний каталог результатов.
    # Явный NAZAR_WORK_DIR позволяет отдельно проверить внутренний путь.
    if os.environ.get("NAZAR_WORK_DIR"):
        sys.exit(main())
    with tempfile.TemporaryDirectory(prefix="nazar-fake-work-") as work:
        os.environ["NAZAR_WORK_DIR"] = work
        sys.exit(main())
