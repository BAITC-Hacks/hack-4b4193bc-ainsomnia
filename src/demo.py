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
ROUTE_BUDGET = 35        # секунд на проход модулей 1–2 по тем же дням
MAX_PAUSE = 1.6          # быстрее следить за экраном всё равно не получится
SIMILAR_DAYS = 14        # окно поиска похожих обращений, суток назад
SIMILAR_SHOW = 5         # сколько ближайших похожих показывать
PRIORITY_MIN_BASE = 100  # меньше размеченных обращений по теме — доля ненадёжна
DUP_USELESS_SHARE = 0.5  # выше этой доли «повторов» проверка ничего не различает
# Канонические поля для проверки на повтор. created_at исключено намеренно:
# у двух обращений время не совпадает никогда, и проверка была бы пустой.
DUP_FIELDS = ["region", "category", "district", "executor", "status",
              "sla_breach", "topic", "appeal_class"]

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
                                        "appeal_class", "district", "executor",
                                        "status", "sla_breach"])
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


def priority_topics(df):
    """Темы, где доля просрочки исторически выше средней. Порог — из данных.

    Разметка просрочки (`sla_breach`) есть только у двух регионов, поэтому порог
    и доли считаются по ним. Это перенос между регионами, и он оговаривается
    в выводе: у самих этих двух регионов доли расходятся в разы."""
    lab = df[df.sla_breach.notna() & (df.appeal_class == "problem")]
    base = float(lab.sla_breach.mean())
    g = lab.groupby("topic").sla_breach.agg(["size", "mean"])
    hot = {t: float(r["mean"]) for t, r in g.iterrows()
           if r["mean"] > base and r["size"] >= PRIORITY_MIN_BASE}
    thin = {t: (int(r["size"]), float(r["mean"])) for t, r in g.iterrows()
            if r["mean"] > base and r["size"] < PRIORITY_MIN_BASE}
    return base, hot, thin, lab


def similar_appeals(pool, appeal):
    """Похожие: тот же регион и тема, тот же район если он заполнен, окно
    SIMILAR_DAYS суток до момента обращения. Сортировка — от ближайших по времени."""
    lo = appeal.created_at - pd.Timedelta(days=SIMILAR_DAYS)
    m = ((pool.created_at < appeal.created_at) & (pool.created_at >= lo)
         & (pool.region == appeal.region) & (pool.topic == appeal.topic))
    if pd.notna(appeal.district):
        m &= (pool.district == appeal.district)
    return pool[m].sort_values("created_at", ascending=False)


def duplicates_among(similar, appeal):
    """Повтор: совпадение по всем каноническим полям, кроме времени поступления."""
    if similar.empty:
        return similar
    # astype(object) до замены пропусков: в boolean-колонку (sla_breach)
    # строку положить нельзя, fillna на ней падает.
    block = similar[DUP_FIELDS].astype(object)
    key = block.mask(block.isna(), "∅").astype(str)
    want = pd.Series({c: "∅" if pd.isna(appeal[c]) else str(appeal[c])
                      for c in DUP_FIELDS})
    return similar[(key == want).all(axis=1)]


def step_modules_1_2(df, pause):
    """Модуль 1 — приём и маршрутизация. Модуль 2 — ассистент оператора."""
    head("МОДУЛЬ 1. Приём и маршрутизация · МОДУЛЬ 2. Ассистент оператора")
    base, hot, thin, lab = priority_topics(df)
    # Поток эпизода: тот же регион и та же тема, что и в модуле 3, иначе
    # сквозной сценарий распадается на два разных сюжета.
    pool = df[(df.region == REGION) & (df.topic == TOPIC)
              & (df.appeal_class == "problem")]
    ex_fill = float(df[df.region == REGION].executor.notna().mean())
    ex_all = float(df.executor.notna().mean())

    print(f"  {BOLD}Приоритет считается из данных, порог не подбирался.{OFF} "
          f"Средняя доля просрочки")
    print(f"  по размеченным обращениям — {base * 100:.2f}% ({num(len(lab))} обращений). "
          f"Тема приоритетна,")
    print(f"  если её доля выше средней при базе не меньше {PRIORITY_MIN_BASE} "
          f"размеченных обращений.")
    mark = (f"{RED}{BOLD}приоритетная{OFF} ({hot[TOPIC] * 100:.2f}%)" if TOPIC in hot
            else "не приоритетная")
    print(f"  Приоритетных тем: {len(hot)}. Тема «{TOPIC}» — {mark}")
    if thin:
        t0 = sorted(thin.items(), key=lambda kv: -kv[1][1])[0]
        print(f"  {DIM}Не засчитаны по малой базе: {len(thin)} тем, крупнейшая — "
              f"«{t0[0]}» {t0[1][1] * 100:.1f}% на {t0[1][0]} обращениях{OFF}")
    by_reg = lab.groupby("region").sla_breach.mean()
    print(f"  {YELLOW}Оговорка к приоритету.{OFF} Разметка просрочки есть только у "
          f"{len(by_reg)} регионов из {df.region.nunique()}:")
    print("  " + ", ".join(f"{r.replace(' область', '')} {v * 100:.2f}%"
                           for r, v in by_reg.items())
          + f" — разница в {by_reg.max() / by_reg.min():.1f} раза при одинаковой")
    print(f"  схеме данных и без объяснения. Для {REGION.replace(' область', '')} это "
          f"перенос чужой разметки.")

    print(f"\n  {YELLOW}Оговорка к маршрутизации.{OFF} Тема и класс определяются "
          f"справочником правил")
    print(f"  (src/topic_mapping.py), а не дообученной моделью. Классификатор по тексту")
    print(f"  обращения обучать не на чем: в выгрузке 55 строк текста на 1 063 216 "
          f"(раздел 5b).")
    if ex_fill == 0:
        print(f"\n  {RED}Ответственной службы у этого региона в данных нет.{OFF} Поле "
              f"executor заполнено")
        print(f"  у {REGION} в {ex_fill * 100:.2f}% строк (по всей таблице "
              f"{ex_all * 100:.1f}%). Подставлять")
        print(f"  службу неоткуда, и мы её не подставляем — показываем пропуск как пропуск.")

    print(f"\n  {DIM}Первое обращение каждого дня по теме «{TOPIC}»: путь от "
          f"поступления до службы.")
    print(f"  {DIM}Похожие ищутся за")
    print(f"  {SIMILAR_DAYS} суток до обращения; развёрнутый список — на первом дне и на "
          f"дне первого сигнала,")
    print(f"  дальше сводной строкой, чтобы проход остался в три минуты.{OFF}\n")

    expand_on = {DAY_FROM, pd.Timestamp("2022-10-28")}
    seen = dup_found = 0
    warned = False
    dup_share_max = 0.0
    for day in pd.date_range(DAY_FROM, DAY_TO):
        today = pool[pool.created_at.dt.floor("D") == day]
        if today.empty:
            continue
        a = today.sort_values("created_at").iloc[0]
        seen += 1
        topic, cls = map_topic(a.category), classify_appeal(a.category)
        pr = f" {RED}{BOLD}[приоритет]{OFF}" if topic in hot else ""
        svc = a.executor if pd.notna(a.executor) else f"{RED}поля нет{OFF}"
        print(f"  {day:%d.%m} {a.created_at:%H:%M:%S}  «{a.category}»  {GREEN}→{OFF} "
              f"{topic} · {cls}  {GREEN}→{OFF} служба: {svc}{pr}")
        sim = similar_appeals(pool, a)
        dups = duplicates_among(sim, a)
        dup_found += int(len(dups) > 0)
        if len(sim):
            dup_share_max = max(dup_share_max, len(dups) / len(sim))
        if day in expand_on:
            print(f"    {DIM}похожих за {SIMILAR_DAYS} суток: {len(sim)}; ближайшие:{OFF}")
            for _, r in sim.head(SIMILAR_SHOW).iterrows():
                dist = r.district if pd.notna(r.district) else "—"
                print(f"      {r.created_at:%d.%m.%Y %H:%M}  статус «{r.status}»  "
                      f"район: {dist}")
            verdict = (f"{RED}вероятный дубликат: {len(dups)}{OFF}" if len(dups)
                       else f"{GREEN}повторов нет{OFF}")
            print(f"    проверка на повтор по полям {', '.join(DUP_FIELDS)} "
                  f"(без времени): {verdict}")
            share = len(dups) / len(sim) if len(sim) else 0.0
            if share > DUP_USELESS_SHARE and not warned:
                warned = True
                filled = [c for c in DUP_FIELDS
                          if pool[c].notna().any()]
                print(f"    {RED}Проверка на повтор здесь ничего не различает:{OFF} "
                      f"{len(dups)} из {len(sim)} = {share * 100:.0f}%.")
                print(f"    У этого региона из восьми канонических полей заполнены "
                      f"только {len(filled)}:")
                print(f"    {', '.join(filled)} — они одинаковы у тысяч обращений.")
                print(f"    Чтобы отличать повтор, нужен адрес или идентификатор "
                      f"заявителя; и то и другое —")
                print(f"    персональные данные, в канон они не переносятся "
                      f"(раздел 1 CLAUDE.md). Вывод: без них")
                print(f"    дедупликация по метаданным невозможна, и выдавать эти "
                      f"числа за найденные")
                print(f"    повторы нельзя.")
        else:
            verdict = f"{RED}дубликат {len(dups)}{OFF}" if len(dups) else "повторов нет"
            print(f"    {DIM}похожих за {SIMILAR_DAYS} суток: {len(sim)}, {verdict}{OFF}")
        wait(pause)

    print(f"\n  {YELLOW}Оговорка к ассистенту.{OFF} Похожесть считается совпадением "
          f"метаданных —")
    print(f"  регион, тема, район, окно {SIMILAR_DAYS} суток, — а не дообученными "
          f"эмбеддингами:")
    print(f"  для них нужен текст обращения, которого в выгрузке нет.")
    return seen, dup_found, (TOPIC in hot), dup_share_max


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
    route_pause = min(a.pause, ROUTE_BUDGET / max(days, 1))
    print(f"\n{BOLD}СКВОЗНОЙ СЦЕНАРИЙ ПО ТРЁМ МОДУЛЯМ: приём и маршрутизация → "
          f"ассистент оператора → витрина{OFF}")
    print(f"{DIM}Дней в показе {days}; пауза {route_pause:.1f} с в модулях 1–2 и "
          f"{a.pause:.1f} с в модуле 3 —{OFF}")
    print(f"{DIM}проигрывание уложится в {ROUTE_BUDGET + STREAM_BUDGET} с{OFF}")
    if not DATA.exists():
        print(f"{RED}Нет файла {DATA}. Соберите его тремя шагами по порядку:{OFF}")
        print("  1. .venv/bin/python -m src.adapters.adapters")
        print("  2. .venv/bin/python -m src.adapters.build_unified")
        print("  3. .venv/bin/python -m src.topic_mapping")
        print("  Подробнее — раздел «Развёртывание с нуля» в CLAUDE.md")
        return 1

    df = step_load()
    wait(a.pause)
    routed, dup_days, prio, dup_share = step_modules_1_2(df, route_pause)
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
    print(f"  {BOLD}Модуль 1 — приём и маршрутизация:{OFF} путь показан для первого "
          f"обращения каждого")
    print(f"  из {routed} дней; тема и класс — живым вызовом справочника, приоритет — "
          f"по доле")
    print(f"  просрочки из данных. Ответственная служба не показана: поля executor "
          f"у региона нет.")
    print(f"  {BOLD}Модуль 2 — ассистент оператора:{OFF} похожие обращения найдены за "
          f"{SIMILAR_DAYS} суток")
    print(f"  по совпадению метаданных. Проверка на повтор в этом регионе "
          f"{RED}не работает{OFF}: до")
    print(f"  {dup_share * 100:.0f}% похожих совпадают по всем заполненным полям — "
          f"различать их нечем,")
    print(f"  адрес и заявитель в канон не переносятся как персональные данные.")
    print(f"  {BOLD}Модуль 3 — витрина и детектор:{OFF} ниже.\n")
    print(f"  Обращения прочитаны из выгрузки, тема и класс присвоены справочником,")
    print(f"  детектор поднял сигнал {BOLD}{first:%d.%m.%Y}{OFF} на "
          f"{len(pd.date_range(DAY_FROM, DAY_TO))} проигранных днях,")
    print(f"  расхождений с полным прогоном {mism}, отчёт собран"
          + ("" if a.no_export else f", ПДн в нём {'нет' if clean else 'НАЙДЕНЫ'}"))
    print(f"  Прогон занял {time.time() - started:.0f} с.")
    print(f"\n  {BOLD}Какие модули пройдены целиком, какие — на метаданных:{OFF}")
    print(f"  {GREEN}Целиком:{OFF} модуль 3 — детектор, витрина и выгрузка работают "
          f"на настоящих данных")
    print(f"    и настоящем коде, без допущений.")
    print(f"  {YELLOW}На метаданных вместо дообученных моделей:{OFF} модуль 1 "
          f"(маршрутизация по справочнику")
    print(f"    правил, а не по тексту обращения) и модуль 2 (похожесть по совпадению "
          f"полей,")
    print(f"    а не по эмбеддингам). Причина одна и та же: текста обращений в выгрузке "
          f"нет —")
    print(f"    55 строк на 1 063 216. Это ограничение данных, а не реализации.")
    print(f"\n  {DIM}Что показать в терминале нельзя, и почему:{OFF}")
    print(f"  {DIM}  · витрина — веб-страница; данные её ленты выведены выше,{OFF}")
    print(f"  {DIM}    запуск: .venv/bin/streamlit run src/dashboard.py{OFF}")
    print(f"  {DIM}  · сборка канона адаптером — пакетный шаг по файлу целиком,{OFF}")
    print(f"  {DIM}    поштучно не проигрывается; заглушку не ставили{OFF}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
