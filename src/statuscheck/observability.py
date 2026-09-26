"""Карта наблюдаемости раздела 5o — 7 регионов × 3 канала.

    .venv/bin/python -m src.statuscheck.observability

Обязательный выход прогона наравне с таблицей сигналов; если сигналов ноль,
показывается она. В каждой клетке — считается ли канал, почему нет (какого поля
или какой выгрузки не хватает) и сколько служб или тем попадает под порог.
Пороги — из постановки 5o: ячейка «регион × тема» годна при 5+ службах с 30+
жалобами; канал 1 — 100+ закрытий службы в годной ячейке; канал 2 — 30+ заявок
старше 60 дней; канал 3 — 3+ всплеска, где служба ведёт половину жалоб.
Пачки закрытий — reports/5o/batches.json (шаг 0б).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.statuscheck.frame import FRAME
from src.statuscheck.load import SPEC, load

OUT = Path("reports/5o")
CELL_SERVICES, CELL_COMPLAINTS, CH1_CLOSURES = 5, 30, 100
CH2_OLD_DAYS, CH2_MIN = 60, 30
CH3_SPIKES, CH3_SHARE = 3, 0.5
SHORT = {"Карагандинская область": "Караганда", "Костанайская область": "Костанай",
         "Туркестанская область": "Туркестан", "Восточно-Казахстанская область": "ВКО",
         "Алматинская область": "Алматы", "Акмолинская область": "Акмола",
         "Павлодарская область": "Павлодар"}
NO_EXECUTOR = {   # итог шага 0в — reports/5o/executor_search.md
    "Алматинская область": "нет поля исполнителя: `contractor` не берётся — в нём ПДн "
                           "(5m); другого нет — `service` это вид услуги с срабатываниями "
                           "ПДн, остальные поля — тема, статус, канал (шаг 0в)",
    "Акмолинская область": "нет поля исполнителя: организация только в свободном тексте "
                           "`request_subject` с ПДн; `current_project` — одно системное "
                           "значение на регион, `region_g_a` — район (шаг 0в)",
    "Павлодарская область": "нет поля исполнителя: `service_*`, `category_*`, "
                            "`request_type*` — справочники тем и типов, не организации "
                            "(шаг 0в)",
}
NO_CLOSURE = {
    "Карагандинская область": "нет факта закрытия: `updated_date` — дата последнего "
                              "изменения записи, не закрытия (раздел 3)",
    "Акмолинская область": "нет факта закрытия: есть только плановые даты "
                           "`planned_closing_date`, `completion_deadline`",
    "Павлодарская область": "нет даты закрытия: в выгрузке только дата загрузки "
                            "`sdu_load_date`",
}


def cells(d: pd.DataFrame) -> pd.Series:
    """Годные ячейки: тема → число служб с 30+ жалобами, если их 5+."""
    p = d[(d.cls == "problem") & d.executor.notna()]
    per = p.groupby(["topic", "executor"]).size()
    n = per[per >= CELL_COMPLAINTS].reset_index().groupby("topic").executor.nunique()
    return n[n >= CELL_SERVICES].sort_values(ascending=False)


def channel1(region, d, batch):
    if region in NO_CLOSURE:
        return "не считается", NO_CLOSURE[region], ""
    if d.executor.isna().all():
        return "не считается", NO_EXECUTOR[region], ""
    if batch and batch["branch"] == "большинство":
        return ("не считается", "отметку о закрытии ставят централизованно: пачки у "
                f"{batch['share_services_with_batches']:.0%} служб (шаг 0б)", "")
    ok = cells(d)
    p = d[(d.cls == "problem") & d.executor.notna() & d.closed.notna()
          & d.topic.isin(ok.index)]
    per = p.groupby(["topic", "executor"]).size()
    svc = per[per >= CH1_CLOSURES].reset_index().executor.nunique()
    note = (f"годных тем {len(ok)}: " + ", ".join(f"{t} ({n})" for t, n in ok.items())
            + f"; служб со 100+ закрытиями в них — {svc}")
    if batch:
        note += (f"; синхронных минут {batch['sync_minutes']}, закрытий в них "
                 f"{batch['sync_closures']} — удаляются")
    if len(ok) == 0:
        return "не считается", "ни одной темы с 5+ службами по 30+ жалоб", note
    if len(ok) == 1:
        return ("по службам — одна тема", "только одна тема набирает 5+ служб — сравнение "
                "служб есть лишь внутри неё", note)
    return "по службам", "исполнитель и время закрытия есть; пачки у меньшинства служб", note


def channel2(region, d):
    if d.status.isna().all():
        return "не считается", "нет колонки статуса", ""
    end = d.created.max()
    final = pd.to_numeric(d.final, errors="coerce").astype(float)
    open_old = (final == 0) & (d.created < end - pd.Timedelta(days=CH2_OLD_DAYS))
    last = d.created >= end - pd.Timedelta(days=7)
    closed_last = float(final[last].mean())
    no_date = int(((final == 1) & d.closed.isna()).sum()) if d.closed.notna().any() else None
    note = (f"незакрытых старше 60 дней на {end:%d.%m.%Y}: {int(open_old.sum())}; "
            f"закрыто среди поступивших за последнюю неделю: {closed_last:.1%}"
            + (f"; «закрыто» без даты закрытия: {no_date}" if no_date is not None else ""))
    if closed_last == 1.0 and open_old.sum() == 0:
        return ("не считается", "выгрузка только закрытыми заявками — нужна выгрузка вместе "
                "с незакрытыми (раздел 10, пункт 17)", note)
    if region == "Акмолинская область":
        note += ("; «Передано в службу » (64% строк) отнесено к незакрытым: для службы заявка "
                 "не закрыта, но для 109 это может быть конечная точка — сверить с заказчиком")
    if d.executor.isna().all():
        return "по региону", NO_EXECUTOR[region], note
    per = d[open_old].groupby("executor").size()
    svc = int((per >= CH2_MIN).sum())
    if svc < CELL_SERVICES:
        return ("не считается", f"незакрытых старше 60 дней слишком мало: служб с {CH2_MIN}+ "
                f"такими заявками {svc}, нужно 5+", note)
    return "по службам", "статус, дата поступления и исполнитель есть", note + f"; служб с 30+ — {svc}"


def channel3(region, d, ev):
    e = ev[ev["регион"] == region]
    if region in NO_CLOSURE:
        return "не считается", NO_CLOSURE[region] + f"; всплесков в регионе {len(e)}", ""
    p = d[d.cls == "problem"]
    n30, dom = 0, []
    for _, r in e.iterrows():
        day0 = pd.Timestamp(r["дата"]).normalize()
        m = ((p.topic == r["тема"]) & (p.created >= day0)
             & (p.created < day0 + pd.Timedelta(days=int(r["дней подряд"]))))
        n30 += int(m.sum() >= 30)
        if p.executor.notna().any() and m.any():
            vc = p.executor[m].value_counts(normalize=True)
            if vc.iloc[0] >= CH3_SHARE:
                dom.append(vc.index[0])
    note = f"всплесков {len(e)}, из них с 30+ жалобами {n30}"
    if p.executor.isna().all():
        return "по теме региона", NO_EXECUTOR[region], note
    per = pd.Series(dom).value_counts() if dom else pd.Series(dtype=int)
    svc = int((per >= CH3_SPIKES).sum())
    note += f"; служб с 3+ всплесками, где они ведут половину жалоб, — {svc}"
    return ("по теме региона, служб — единицы" if svc < CELL_SERVICES else "по службам",
            "всплеск — единица темы региона; службе он даёт отметку только при 3+ таких всплесках",
            note)


def build() -> list[dict]:
    from src.dashboard import load_events          # детектор — одна реализация (5i)
    _, _, ev = load_events()
    batches = {b["region"]: b for b in json.loads((OUT / "batches.json").read_text("utf-8"))}
    rows = []
    for region in SPEC:
        d = load(region)
        for ch, (state, reason, note) in (
                ("1. ритм закрытий", channel1(region, d, batches.get(region))),
                ("2. зависшие", channel2(region, d)),
                ("3. поток после закрытия", channel3(region, d, ev))):
            rows.append({"region": region, "channel": ch, "state": state,
                         "reason": reason, "numbers": note})
    return rows


def render(rows: list[dict]) -> str:
    mark = {"по службам": "по службам", "не считается": "—"}
    L = [FRAME, "", "# 5o: карта наблюдаемости — что на каком регионе считается", "",
         "Если сигналов ноль, показывается эта карта. Причина в каждой клетке говорит, "
         "какого поля или какой выгрузки не хватает.", "",
         "| Регион | Канал 1 — ритм закрытий | Канал 2 — зависшие | Канал 3 — поток после закрытия |",
         "|---|---|---|---|"]
    for region in SPEC:
        r = {x["channel"][0]: x for x in rows if x["region"] == region}
        L.append(f"| {SHORT[region]} | " + " | ".join(
            mark.get(r[c]["state"], r[c]["state"]) for c in "123") + " |")
    L += ["", "## Клетки подробно", "",
          "| Регион | Канал | Считается | Причина | Числа |", "|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {SHORT[x['region']]} | {x['channel']} | {x['state']} | "
                 f"{x['reason']} | {x['numbers'] or '—'} |")
    by = pd.DataFrame(rows).state.value_counts()
    L += ["", f"Итого клеток: {len(rows)}: " + " · ".join(f"«{k}» — {v}" for k, v in by.items()) + ".",
          "", "Пороги из постановки 5o: ячейка «регион × тема» годна при 5+ службах с 30+ "
          "жалобами; канал 1 — 100+ закрытий службы в годной ячейке; канал 2 — 30+ заявок "
          "старше 60 дней; канал 3 — 3+ всплеска, где служба ведёт половину жалоб. "
          "Шаг 0в выполнен: нового структурированного поля исполнителя ни в одном "
          "регионе не найдено (reports/5o/executor_search.md).", ""]
    return "\n".join(L)


def main() -> None:
    rows = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "observability.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (OUT / "observability.md").write_text(render(rows), encoding="utf-8")
    print(OUT / "observability.md")


if __name__ == "__main__":
    main()
