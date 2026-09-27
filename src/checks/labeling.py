#!/usr/bin/env python3
"""Проверка разметки тем для новой выгрузки — шаг сборки `nazar-build-data`.

    .venv/bin/python -m src.checks.labeling            # проверить data/unified.parquet
    .venv/bin/python -m src.checks.labeling --update   # принять эталон сопоставления

Регресс-тест tests/test_topic_coverage.py сравнивает число строк в каждой теме с
эталоном текущей выгрузки и падает на любой новой выгрузке: числа меняются всегда.
Он остаётся для правок src/topic_mapping.py на неизменных данных. Эта проверка —
для регулярной сборки: не абсолютные числа, а свойства разметки (CLAUDE.md, 0b,
препятствие 1).

Падает (код 1), если
  1) доля «прочего» внутри класса problem в каком-либо регионе выше OTHER_MAX —
     цель «меньше 15%» из раздела 5e;
  2) у категории, которая есть в эталоне, сменилась тема или класс обращения.
     Сравнивается сопоставление «категория → тема» целиком, поэтому ловится и
     симметричный обмен, который регресс-тест сумм пропускает (раздел 9).
  3) новая категория попала в класс info или system и уносит больше
     NEW_NONPROBLEM_MAX строк: исключение строк из жалоб — молчаливая ошибка,
     добавление — нет. Порог обоснован в CLAUDE.md, 0b, препятствие 1.
Не падает, а выводит списком на просмотр:
  4) остальные новые категории — их нет в эталоне; полный список пишется в
     data/labeling_review.csv (каталог data/ вне git).

Тема и класс — функции одной категории: на выгрузке 2026-09 у 961 категории нет
ни одной с двумя темами или классами. Поэтому эталон — словарь «категория →
(тема, класс)» без региона, и он годится и для нового региона.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from src import paths

DATA = paths.UNIFIED
BASE = paths.ROOT / "tests" / "topic_map_baseline.json"
REVIEW = paths.DATA_DIR / "labeling_review.csv"
OTHER_MAX = 0.15      # раздел 5e: «прочее» внутри problem меньше 15% по каждому региону
MIN_PROBLEM = 100     # регион с меньшим числом жалоб — доля «прочего» печатается, но не роняет
NEW_NONPROBLEM_MAX = 30   # 1% жалоб самого малого региона (Акмола 3 421) — CLAUDE.md, 0b
EMPTY = "<пусто>"
SHOW = 20             # сколько новых категорий печатать; остальные — в файле


def _masked(v: str) -> str:
    """Категория на экран: справочник, но при сдвиге полей туда попадали ПДн (раздел 1)."""
    from src.checks.person_names import person_hits, safe
    from src.checks.privacy import pii_hits
    if person_hits(v):
        return safe(v)
    if pii_hits(v):
        return " ".join(w[:2] + "*" * max(len(w) - 2, 0) for w in v.split())
    return v


def mapping(df: pd.DataFrame) -> dict[str, list[str]]:
    cat = df["category"].fillna(EMPTY)
    m = (df.assign(category=cat).groupby("category")
         .agg(topic=("topic", "first"), cls=("appeal_class", "first"),
              nt=("topic", "nunique"), nc=("appeal_class", "nunique")))
    bad = m[(m.nt > 1) | (m.nc > 1)]
    if len(bad):
        # Допущение модуля нарушено: тема зависит не только от категории.
        raise SystemExit(f"у {len(bad)} категорий больше одной темы или класса — "
                         "сопоставление «категория → тема» не определено, проверка неприменима")
    return {c: [r.topic, r.cls] for c, r in m.iterrows()}


def main() -> int:
    paths.require_source_marker(paths.SOURCE_MARK)
    from src.cli import BUILD_HINT, require
    require(DATA, "сводной таблицы обращений", BUILD_HINT)
    df = pd.read_parquet(DATA, columns=["region", "category", "topic", "appeal_class"])
    cur = mapping(df)

    if "--update" in sys.argv:
        BASE.write_text(json.dumps(dict(sorted(cur.items())), ensure_ascii=False, indent=0),
                        encoding="utf-8")
        print(f"эталон сопоставления обновлён: {BASE}, категорий {len(cur)}")
        return 0
    require(BASE, "эталона сопоставления «категория → тема»",
            "принять текущую разметку: .venv/bin/python -m src.checks.labeling --update")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    fail = False

    print(f"1) Доля «прочего» внутри жалоб по регионам (порог {OTHER_MAX:.0%})")
    for r, g in df[df.appeal_class == "problem"].groupby("region"):
        share = float((g.topic == "прочее").mean())
        flag = ""
        if share > OTHER_MAX:
            if len(g) >= MIN_PROBLEM:
                flag, fail = "  ВЫШЕ ПОРОГА", True
            else:
                flag = f"  выше порога, но жалоб меньше {MIN_PROBLEM} — не роняет"
        print(f"   {r:32s} {share:7.2%} из {len(g):>7,}{flag}".replace(",", " "))

    print("\n2) Прежние категории: тема и класс не сменились")
    changed = [(c, base[c], cur[c]) for c in cur if c in base and base[c] != cur[c]]
    if changed:
        fail = True
        n = df["category"].fillna(EMPTY).value_counts()
        for c, (ot, oc), (nt, nc) in sorted(changed, key=lambda x: -n.get(x[0], 0)):
            print(f"   СМЕНА  {_masked(c)[:60]:60s} {ot} / {oc} -> {nt} / {nc}  ({n.get(c, 0)} строк)")
    else:
        known = sum(c in base for c in cur)
        print(f"   без изменений: {known} категорий из эталона")

    new = [c for c in cur if c not in base]
    gone = len(base) - sum(c in base for c in cur)
    print(f"\n3) Новые категории — список на просмотр: {len(new)}"
          f" (из эталона в выгрузке нет: {gone})")
    if new:
        cat = df["category"].fillna(EMPTY)
        rev = (df[cat.isin(new)].assign(category=cat)
               .groupby(["category", "region"]).size().rename("строк").reset_index())
        rev["тема"] = rev.category.map(lambda c: cur[c][0])
        rev["класс"] = rev.category.map(lambda c: cur[c][1])
        rev = rev.sort_values("строк", ascending=False)
        REVIEW.parent.mkdir(parents=True, exist_ok=True)
        rev.to_csv(REVIEW, index=False)
        for r in rev.head(SHOW).itertuples():
            print(f"   {_masked(r.category)[:60]:60s} {r.region:30s} {r.строк:>7} -> {r.тема} / {r.класс}")
        if len(rev) > SHOW:
            print(f"   … ещё {len(rev) - SHOW}")
        print(f"   полный список: {REVIEW}. Просмотреть темы; если верны — "
              "принять эталон: --update")

        tot = rev.groupby("category").строк.sum()
        out = [(c, cur[c][1], int(tot[c])) for c in tot.index
               if cur[c][1] != "problem" and tot[c] > NEW_NONPROBLEM_MAX]
        print(f"\n4) Новые категории вне жалоб, больше {NEW_NONPROBLEM_MAX} строк — роняют сборку: {len(out)}")
        for c, cls, n in sorted(out, key=lambda x: -x[2]):
            fail = True
            print(f"   ВЫПАДАЕТ ИЗ ЖАЛОБ: «{_masked(c)[:70]}» — класс {cls}, {n} строк. "
                  f"Класс определяется по словам названия (src/topic_mapping.py, "
                  f"classify_appeal). Если это жалобы — поправить правило; если "
                  f"действительно {cls} — принять эталон: --update")

    if fail:
        print("\nПРОВАЛ: разметка изменилась там, где не должна была. Выше — что именно.")
        return 1
    print("\nОК: свойства разметки в норме.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
