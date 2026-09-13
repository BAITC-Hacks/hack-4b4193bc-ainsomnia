#!/usr/bin/env python3
"""Сквозной сценарий: обращение → классификация → счётчики → сигнал → отчёт.

    .venv/bin/python -m src.demo                 # полный проход, ~90 секунд
    .venv/bin/python -m src.demo --pause 0       # без пауз, для проверки
    .venv/bin/python -m src.demo --no-export     # без сборки файлов

Эпизод настоящий: Павлодарская область, теплоснабжение, ноябрь 2022 — рост
обращений, совпавший по времени с аварией на ТЭЦ Экибастуза (совпадение по
данным не доказывается, вопрос заказчику, раздел 5f CLAUDE.md).

НИЧЕГО НЕ ИМИТИРУЕТСЯ. Все обращения читаются из data/unified.parquet вместе с
их настоящим временем и значением справочника. Тема и класс присваиваются живым
вызовом `src.topic_mapping`, всплески ищет живой `src.spikes`, отчёты собирает
живой `src.export`. Заглушек нет ни на одном шаге.

Детектор на каждом дне видит только те дни, которые уже «поступили»: ряд
обрезается по текущую дату. Это проверяется внутри самого прохода — результат
по дням сверяется с прогоном по всему ряду целиком.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.export import build_excel, build_pdf, check_export_pii
from src.spikes import (classify_seasonal, daily_counts, detect, gap_days,
                        with_duration)
from src.topic_mapping import classify_appeal, map_topic

DATA = Path("data/unified.parquet")
OUT = Path("reports/demo")
REGION = "Павлодарская область"
TOPIC = "теплоснабжение"
DAY_FROM = pd.Timestamp("2022-10-15")   # раньше эпизода: см. step_load, показ сигналов до него
DAY_TO = pd.Timestamp("2022-12-03")
W = 78
STREAM_BUDGET = 80       # секунд на проигрывание дней; пауза подстраивается под окно
MAX_PAUSE = 1.6          # быстрее следить за экраном всё равно не получится

BOLD, DIM, RED, GREEN, YELLOW, BLUE, OFF = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[0m")


def num(n):
    """Разделитель тысяч отдельной функцией: .replace(",", " ") по всей строке
    ответа съедает запятые перечисления — эта ошибка уже ловилась в src/nlq.py."""
    return f"{int(n):,}".replace(",", "\u00a0")


def boxed(lines, color=RED):
    """Рамка, выровненная по видимой длине строки, без учёта ANSI-кодов."""
    inner = W - 6
    print(f"\n{color}{BOLD}  ┌{'─' * inner}┐{OFF}")
    for ln in lines:
        print(f"{color}{BOLD}  │ {ln[:inner - 2].ljust(inner - 2)} │{OFF}")
    print(f"{color}{BOLD}  └{'─' * inner}┘{OFF}")


def head(title, color=BLUE):
    print(f"\n{color}{BOLD}{'━' * W}\n{title}\n{'━' * W}{OFF}")


def wait(seconds):
    if seconds:
        time.sleep(seconds)


def bar(n, scale):
    return "█" * max(0, min(int(n / scale), 42))


# ---------------------------------------------------------------- шаги
def step_load():
    head("ШАГ 0. Источник данных")
    df = pd.read_parquet(DATA, columns=["created_at", "region", "category", "topic",
                                        "appeal_class"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    print(f"  Файл {DATA}: {num(len(df))} обращений, "
          f"{df.region.nunique()} регионов")
    print(f"  Эпизод: {REGION}, тема «{TOPIC}», "
          f"{DAY_FROM:%d.%m.%Y} — {DAY_TO:%d.%m.%Y}")
    print(f"  {YELLOW}Показ начинается раньше эпизода: по этой теме детектор "
          f"поднимает сигналы\n  и до него, и это должно быть видно, а не скрыто "
          f"выбором стартовой даты.{OFF}")
    print(f"  {DIM}Ни одно число ниже не придумано: всё читается из этого файла{OFF}")
    print(f"\n  {YELLOW}Оговорка о границе показа.{OFF} Сборка канона из сырых CSV — "
          f"шаг пакетный:\n  адаптер разбирает файл региона целиком "
          f"(src/adapters/adapters.py), поштучно\n  его проиграть нельзя, и подменять "
          f"его имитацией мы не будем. В проходе ниже\n  обращения берутся уже из "
          f"канона, а живьём показана классификация:\n  тема и класс присваиваются "
          f"вызовом справочника на настоящем значении category.")
    return df


def step_stream(df, pause):
    """Проигрывание по дням: классификация, счётчики, детекция без заглядывания."""
    head("ШАГИ 1–3. Обращения поступают · классификация · детекция по мере поступления")
    problem = df[df.appeal_class == "problem"]
    daily, bounds = daily_counts(problem)
    blocked = gap_days(df, bounds)
    sl = daily[(daily.region == REGION) & (daily.topic == TOPIC)].sort_values("день")
    raw = df[(df.region == REGION) & (df.topic == TOPIC)]

    full = detect(sl, blocked=blocked).set_index("день")     # эталон для сверки
    print(f"  {'дата':>10s} {'обращений':>9s} {'медиана':>8s} {'порог':>7s}  "
          f"{'ряд':<42s} сигнал")
    first_signal, mismatches, shown = None, 0, False
    for day in pd.date_range(DAY_FROM, DAY_TO):
        today = raw[raw.created_at.dt.floor("D") == day]
        # Детектор видит только «поступившее»: ряд обрезан по текущий день
        inc = detect(sl[sl["день"] <= day], blocked=blocked).set_index("день")
        row = inc.loc[day]
        mismatches += bool(row["spike"]) != bool(full.loc[day, "spike"])

        if len(today) and not shown:
            shown = True
            ex = today.sort_values("created_at").iloc[0]
            print(f"\n  {DIM}первое обращение дня — как оно проходит справочник:{OFF}")
            print(f"    поступило {ex.created_at:%d.%m.%Y %H:%M:%S}, "
                  f"регион «{ex.region}»")
            print(f"    category «{ex.category}» "
                  f"{GREEN}→ тема «{map_topic(ex.category)}», "
                  f"класс «{classify_appeal(ex.category)}»{OFF}")
            print(f"    {DIM}вызваны map_topic() и classify_appeal() "
                  f"из src/topic_mapping.py{OFF}")
            # Сверка живого вызова с тем, что лежит в каноне, по всему эпизоду:
            # если справочник воспроизводит разметку, показ честен не только на
            # одной строке.
            ep = raw[(raw.created_at >= DAY_FROM)
                     & (raw.created_at < DAY_TO + pd.Timedelta(days=1))]
            same_t = (ep.category.map(map_topic) == ep.topic).sum()
            same_c = (ep.category.map(classify_appeal) == ep.appeal_class).sum()
            ok = same_t == len(ep) and same_c == len(ep)
            mark = f"{GREEN}совпадает{OFF}" if ok else f"{RED}РАСХОЖДЕНИЕ{OFF}"
            print(f"    {DIM}сверка по всему эпизоду: тема {same_t}/{len(ep)}, "
                  f"класс {same_c}/{len(ep)} — {mark}{DIM} с разметкой канона{OFF}\n")

        n = int(row["count"])
        med = "—" if pd.isna(row["median_w"]) else f"{row['median_w']:.1f}"
        thr = "—" if pd.isna(row["threshold"]) else f"{row['threshold']:.0f}"
        sig = ""
        if bool(row["spike"]):
            sig = f"{RED}{BOLD}◆ ВСПЛЕСК{OFF}"
            if first_signal is None:
                first_signal = day
        color = RED if row["spike"] else (YELLOW if n >= 50 else "")
        line = bar(n, 12).ljust(42)
        print(f"  {day:%d.%m.%Y} {n:9d} {med:>8s} {thr:>7s}  "
              + (f"{color}{line}{OFF}" if color else line) + f" {sig}")
        wait(pause)

        if first_signal == day:
            boxed([f"СИГНАЛ. {day:%d.%m.%Y}: {n} обращений при медиане окна "
                   f"{row['median_w']:.1f}",
                   f"Порог {row['threshold']:.0f} = медиана + 4·MAD "
                   f"по окну [t−28, t−1]",
                   f"Кратность ×{n / max(row['median_w'], 1):.1f}, "
                   f"прирост +{n - row['median_w']:.1f}"])
            wait(pause * 6)

    return sl, full, first_signal, mismatches, blocked


def step_no_lookahead(sl, first_signal, mismatches):
    head("ШАГ 3б. Проверка: детектор не заглядывал вперёд", GREEN)
    days = len(pd.date_range(DAY_FROM, DAY_TO))
    print(f"  Каждый день считался на ряде, обрезанном по этот день, и сверялся с "
          f"прогоном\n  по всему ряду целиком.")
    verdict = (f"{GREEN}совпадает{OFF}" if mismatches == 0
               else f"{RED}РАСХОЖДЕНИЙ {mismatches}{OFF}")
    print(f"  Дней проверено: {days}. Расхождений: {mismatches} — {verdict}")
    print(f"  {DIM}Окно медианы сдвинуто на [t−28, t−1]: день всплеска в собственный "
          f"фон не входит{OFF}")
    print(f"\n  Первый сигнал: {BOLD}{first_signal:%d.%m.%Y}{OFF}")


def step_feed(sl, blocked, pause):
    head("ШАГ 4а. Витрина: лента событий")
    from src.dashboard import feed_view
    det = detect(sl, blocked=blocked)
    ev = with_duration(det)
    ev = classify_seasonal(ev, det, {REGION: (sl["день"].min(), sl["день"].max())},
                           [REGION])
    ev["тип"] = ["без типа" if (x is None or pd.isna(x)) else
                 {"seasonal": "сезонный", "anomaly": "аномалия"}[x]
                 for x in ev["spike_type"]]
    win = ev[(ev["дата"] >= DAY_FROM) & (ev["дата"] <= DAY_TO)]
    view = feed_view(win.sort_values("прирост", ascending=False))
    # Колонка типа (сезонный/аномалия) в показе скрыта: при двух прошлых годах
    # метки внутри одного эпизода расходятся из-за попадания в единственный
    # всплеск прошлого года, и объяснять это на показе негде. В витрине колонка
    # остаётся — там под неё есть блок «Как это считается». См. 5f CLAUDE.md.
    shown_cols = [c for c in view.columns if c != "тип"]
    print(view[shown_cols].to_string(index=False))
    print(f"\n  {DIM}Та же таблица и в том же порядке, что в витрине: "
          f"feed_view() из src/dashboard.py{OFF}")
    print(f"  {DIM}Колонка типа события (сезонный / аномалия) в показе скрыта: "
          f"при двух прошлых\n  годах метки внутри одного эпизода расходятся — "
          f"объяснение в разделе 5f.{OFF}")
    print(f"  {DIM}Витрину целиком в терминале показать нельзя — это веб-страница. "
          f"Запуск:{OFF}")
    print(f"  {BOLD}.venv/bin/streamlit run src/dashboard.py{OFF}")
    wait(pause * 4)
    return view


def step_export(df, view):
    head("ШАГ 4б. Выгрузка отчёта")
    scope = df[(df.region == REGION) & (df.created_at >= DAY_FROM)
               & (df.created_at < DAY_TO + pd.Timedelta(days=1))]
    period = (DAY_FROM, DAY_TO)
    descr = [f"эпизод: {REGION}, тема «{TOPIC}»",
             "события: только этот срез и этот период"]
    OUT.mkdir(parents=True, exist_ok=True)
    xlsx = build_excel(scope, view, [REGION], period, descr)
    (OUT / "demo.xlsx").write_bytes(xlsx)
    print(f"  Excel: {OUT / 'demo.xlsx'} — {len(xlsx) / 1024:.0f} КБ")
    try:
        from src.dashboard import fig_dynamics, fig_topics
        figs = [("Динамика по месяцам", fig_dynamics(scope[scope.appeal_class == "problem"]
                                                     .assign(месяц=lambda x: x.created_at
                                                             .dt.to_period("M")
                                                             .dt.to_timestamp()))),
                ("Структура тем", fig_topics(scope[scope.appeal_class == "problem"]))]
        pdf = build_pdf(scope, view, [REGION], period, figs, descr)
        (OUT / "demo.pdf").write_bytes(pdf)
        print(f"  PDF:   {OUT / 'demo.pdf'} — {len(pdf) / 1024:.0f} КБ")
    except Exception as e:                       # noqa: BLE001 — показываем причину
        print(f"  {YELLOW}PDF не собран: {type(e).__name__}: {e}{OFF}")

    chk = check_export_pii(xlsx)
    bad = chk["нарушения"]
    mark = f"{GREEN}0 нарушений{OFF}" if not bad else f"{RED}{bad}{OFF}"
    print(f"\n  Проверка ПДн на готовом файле: {chk['листов']} листов, "
          f"{chk['ячеек']} ячеек, {chk['текстовых']} текстовых — {mark}")
    print(f"  {DIM}Шаблоны: телефон, 12 цифр подряд, адресные фрагменты, "
          f"имена запрещённых колонок{OFF}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pause", type=float, default=None,
                    help=f"пауза между днями, сек; по умолчанию "
                         f"{STREAM_BUDGET} с на всё окно, но не больше {MAX_PAUSE} с")
    ap.add_argument("--no-export", action="store_true", help="не собирать файлы")
    a = ap.parse_args()

    days = len(pd.date_range(DAY_FROM, DAY_TO))
    if a.pause is None:
        # Держим прогон в пределах двух минут независимо от длины окна показа:
        # проигрывание дней укладывается в STREAM_BUDGET, остальное — накладные.
        a.pause = min(MAX_PAUSE, STREAM_BUDGET / max(days, 1))
    started = time.time()
    print(f"\n{BOLD}СКВОЗНОЙ СЦЕНАРИЙ: обращение → тема → счётчики → сигнал → "
          f"отчёт{OFF}")
    print(f"{DIM}Дней в показе {days}, пауза {a.pause:.1f} с — "
          f"проигрывание уложится в {STREAM_BUDGET} с{OFF}")
    if not DATA.exists():
        print(f"{RED}Нет файла {DATA}. Сначала соберите его: "
              f".venv/bin/python -m src.adapters.adapters{OFF}")
        return 1

    df = step_load()
    wait(a.pause)
    sl, full, first, mism, blocked = step_stream(df, a.pause)
    if first is None:
        print(f"{RED}Сигнала в окне нет — сценарий показывать нечего.{OFF}")
        return 1
    step_no_lookahead(sl, first, mism)
    wait(a.pause)
    view = step_feed(sl, blocked, a.pause)
    clean = True if a.no_export else step_export(df, view)

    head("ИТОГ", GREEN)
    print(f"  Обращения прочитаны из выгрузки, тема и класс присвоены справочником,")
    print(f"  детектор поднял сигнал {BOLD}{first:%d.%m.%Y}{OFF} на "
          f"{len(pd.date_range(DAY_FROM, DAY_TO))} проигранных днях,")
    print(f"  расхождений с полным прогоном {mism}, отчёт собран"
          + ("" if a.no_export else f", ПДн в нём {'нет' if clean else 'НАЙДЕНЫ'}"))
    print(f"  Прогон занял {time.time() - started:.0f} с.")
    print(f"\n  {DIM}Что показать в терминале нельзя, и почему:{OFF}")
    print(f"  {DIM}  · витрина — веб-страница; данные её ленты выведены выше,{OFF}")
    print(f"  {DIM}    запуск: .venv/bin/streamlit run src/dashboard.py{OFF}")
    print(f"  {DIM}  · сборка канона адаптером — пакетный шаг по файлу целиком,{OFF}")
    print(f"  {DIM}    поштучно не проигрывается; заглушку не ставили{OFF}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
