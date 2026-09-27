"""Проверка ПДн по всем ячейкам csv-выгрузки — положительный тест (CLAUDE.md, 5p).

    .venv/bin/python -m src.checks.pii_scan                 # каталог выгрузки из src/paths.py
    .venv/bin/python -m src.checks.pii_scan --canon         # сводная таблица после адаптеров

Все проверки ПДн до 5p прогонялись только там, где ПДн быть не должно, и «0 срабатываний»
не отличал работающую проверку от слепой. На поддельной выгрузке ПДн стоят нарочно, в
известных местах, и эта проверка обязана найти каждое. Печатает только счётчики.

Шаблоны — те же, что у проверки выгрузки Excel и PDF (src/export.py), плюс окончание
отчества и `person_hits` для названий-людей. `person_hits` применяется только к колонкам
исполнителя: на свободном тексте его правило «начинается с личного имени» шумит.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src import paths
from src.checks.person_names import person_hits
from src.export import ADDR, LONG_DIGITS, PHONE

PATRONYMIC = re.compile(r"(?<![\wА-Яа-яЁё])[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна|ична)(?![\wА-Яа-яЁё])")
# Колонки исполнителя по семействам (раздел 5c; у Караганды сырая `category` — тоже организация).
EXECUTOR_COLUMNS = {"executor_gov_org", "organizationname", "executor"}
EXECUTOR_BY_FILE = {"Карагандинская": {"category"}, "Восточно-Казахстанская": {"contractor"}}
KINDS = ("телефон", "12 цифр", "адрес", "отчество")


def _executor_cols(name: str, columns) -> set[str]:
    extra = set().union(*(v for k, v in EXECUTOR_BY_FILE.items() if k in name))
    return {c for c in columns if c in EXECUTOR_COLUMNS | extra}


def scan_frame(df: pd.DataFrame, name: str = "") -> dict:
    out = {k: 0 for k in KINDS}
    people: set[str] = set()
    rows = pd.Series(False, index=df.index)
    ex_cols = _executor_cols(name, df.columns)
    for c in df.columns:
        s = df[c].dropna().astype(str)
        for v in s:
            out["телефон"] += len(PHONE.findall(v))
            out["12 цифр"] += len(LONG_DIGITS.findall(v))
            out["адрес"] += 1 if ADDR.search(v) else 0
            out["отчество"] += len(PATRONYMIC.findall(v))
        if c in ex_cols:
            hit = {v for v in s.unique() if person_hits(v)}
            people |= hit
            rows |= df[c].isin(hit)
    out["названий-людей"] = len(people)
    # строки, а не ячейки: у Караганды одно имя стоит и в `category`, и в `executor_gov_org`
    out["строк с ними"] = int(rows.sum())
    return out


def scan_dir(root: Path) -> dict[str, dict]:
    res = {}
    for p in sorted(Path(root).rglob("*.csv")):
        df = pd.read_csv(p, dtype=str)       # так же, как читают адаптеры
        res[str(p.relative_to(root))] = scan_frame(df, p.name)
    return res


def total(res: dict[str, dict]) -> dict:
    t: dict[str, int] = {}
    for r in res.values():
        for k, v in r.items():
            t[k] = t.get(k, 0) + v
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", action="store_true", help="сводная таблица, а не сырые csv")
    a = ap.parse_args()
    if a.canon:
        u = pd.read_parquet(paths.UNIFIED)
        print("сводная таблица:", scan_frame(u.drop(columns=["source_file"]), "canon"))
        return
    res = scan_dir(paths.RAW_DIR)
    for f, r in res.items():
        if any(r.values()):
            print(f"{f}: {r}")
    print("ИТОГО:", total(res))


if __name__ == "__main__":
    main()
