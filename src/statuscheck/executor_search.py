"""Поиск структурированного поля исполнителя — шаг 0в раздела 5o CLAUDE.md.

    .venv/bin/python -m src.statuscheck.executor_search

Критерий записан до прогона: поле категориальное — не меньше 90% непустых
строк покрыты значениями с 20+ повторами; ни одно значение не срабатывает
на шаблоны ПДн (src/synth/checks.py: цифры, телефоны, адреса, ФИО и отчества);
по смыслу — организация-исполнитель, а не тема или услуга — решается ручным
просмотром верхних значений, которые печатаются только у колонок, прошедших
обе автоматические проверки.

Колонки ПДн и свободного текста (раздел 1) не просматриваются вовсе;
`contractor` Алматы исключён решением 5o (5m).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import paths
from src.statuscheck.frame import FRAME
from src.statuscheck.load import SPEC, _capture
from src.synth.checks import pii_hits

OUT = paths.REPORTS_DIR / "5o" / "executor_search.md"
MIN_REPEAT, COVERAGE = 20, 0.90
SKIP = {"full_name", "applicant_number", "operator", "xcoordinate", "ycoordinate",
        "appeal_address", "street", "com_exp", "result", "request_subject", "_source_file"}
SKIP_BY_REGION = {"Алматинская область": {"contractor"}}
# уже используемые поля: исполнитель канона, тема, статус, даты — просматриваются,
# но помечаются, чтобы было видно, что нового нашлось, а что уже известно
KNOWN_EXECUTOR = {"Карагандинская область": "executor_gov_org",
                  "Костанайская область": "organizationname",
                  "Туркестанская область": "organizationname",
                  "Восточно-Казахстанская область": "contractor"}


def scan(region: str) -> list[dict]:
    df, _ = _capture(SPEC[region]["loader"])
    rows = []
    for col in df.columns:
        if col in SKIP or col in SKIP_BY_REGION.get(region, set()):
            continue
        s = df[col]
        if not (s.dtype == object or str(s.dtype).startswith("str")):
            continue
        v = s.dropna().astype(str).str.strip()
        v = v[v != ""]
        if v.empty:
            continue
        # колонка дат или чисел — не исполнитель
        if pd.to_datetime(v.head(200), errors="coerce", format="mixed").notna().mean() > 0.9:
            continue
        vc = v.value_counts()
        coverage = float(vc[vc >= MIN_REPEAT].sum() / len(v))
        pii = sum(bool(pii_hits(x)) for x in vc.index)
        rows.append({"region": region, "column": col, "filled": len(v) / len(df),
                     "unique": int(len(vc)), "coverage": coverage, "pii_values": int(pii),
                     "passes": coverage >= COVERAGE and pii == 0,
                     "known": KNOWN_EXECUTOR.get(region) == col,
                     "top": list(vc.index[:6]) if coverage >= COVERAGE and pii == 0 else []})
    return rows


def render(rows: list[dict]) -> str:
    L = [FRAME, "", "# 5o, шаг 0в: поиск структурированного поля исполнителя", "",
         f"Критерий (записан до прогона): 90%+ непустых строк покрыты значениями с "
         f"{MIN_REPEAT}+ повторами; ни одного значения с шаблонами ПДн; по смыслу — "
         "организация. Колонки ПДн и свободного текста не просматривались; `contractor` "
         "Алматы исключён (5m).", "",
         "| Регион | Колонка | Заполнено | Значений | Покрытие частыми | Значений с ПДн | Прошла автопроверку |",
         "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['region']} | `{r['column']}`{' — исполнитель канона' if r['known'] else ''} | "
                 f"{r['filled']:.1%} | {r['unique']} | {r['coverage']:.1%} | {r['pii_values']} | "
                 f"{'да' if r['passes'] else 'нет'} |")
    L += ["", "## Верхние значения колонок, прошедших автопроверку — для ручного решения", ""]
    for r in rows:
        if r["passes"]:
            L.append(f"- **{r['region']}, `{r['column']}`:** " + "; ".join(r["top"]))
    return "\n".join(L) + "\n"


def main() -> None:
    rows = []
    for region in SPEC:
        rows += scan(region)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
