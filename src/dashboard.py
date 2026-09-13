#!/usr/bin/env python3
"""Витрина руководителя по обращениям 109.

    .venv/bin/streamlit run src/dashboard.py

Источник — data/unified.parquet (собирается src/adapters/build_unified.py
и размечается src/topic_mapping.py). Если не указано иное, работаем по
классу problem: info и system — справочный трафик и артефакты колл-центра,
включать их в аналитику нагрузки нельзя.

Функции fig_* не зависят от streamlit — их можно вызвать отдельно, чтобы
отрендерить графики в файл и проверить, что отрисовалось.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.export import build_excel, build_pdf
from src.spikes import (GAP_MIN, SEASONAL_REGIONS, classify_seasonal, daily_counts,
                        detect, gap_days, with_duration)

DATA = Path("data/unified.parquet")
CONTEXT_DAYS = 14        # сколько дней показывать до и после события
NEW_DAYS_DEFAULT = 7     # «требует внимания» — события за столько последних дней
TYPE_RU = {"seasonal": "сезонный", "anomaly": "аномалия", "без типа": "без типа"}
CLASS_RU = {"problem": "Городские проблемы", "info": "Справочные",
            "system": "Служебные"}
CLASS_COLOR = {"Городские проблемы": "#2E7D32", "Справочные": "#F9A825",
               "Служебные": "#9E9E9E"}
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн",
             "июл", "авг", "сен", "окт", "ноя", "дек"]


def load_data(path=DATA):
    df = pd.read_parquet(path, columns=["created_at", "region", "topic", "appeal_class"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["месяц"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------- события
def load_events(path=DATA):
    """Дневные ряды и лента всплесков. Параметры детектора — как в src.spikes.

    Возвращает (daily, det, ev): дневные счётчики по срезам, таблицу с флагом
    всплеска по дням и ленту событий с длительностью и типом. Считается один раз
    и кладётся в кэш: детектор идёт по всем срезам всех регионов.
    """
    full = pd.read_parquet(path, columns=["created_at", "region", "topic", "appeal_class"])
    full["created_at"] = pd.to_datetime(full["created_at"])
    problem = full[full.appeal_class == "problem"].copy()
    daily, bounds = daily_counts(problem)
    # Пропуски выгрузки ищутся по всем классам — см. gap_days в src/spikes.py
    det = detect(daily, blocked=gap_days(full, bounds))
    ev = classify_seasonal(with_duration(det), det, bounds, SEASONAL_REGIONS)
    # spike_type приходит как None либо NaN — оба значат «сравнивать не с чем»
    ev["тип"] = [TYPE_RU["без типа"] if (x is None or pd.isna(x)) else TYPE_RU.get(x, x)
                 for x in ev["spike_type"]]
    ev["конец"] = ev["дата"] + pd.to_timedelta(ev["дней подряд"] - 1, unit="D")
    return daily, det, ev


def region_last_day(daily):
    """Последний день данных по каждому региону — точка отсчёта «новизны».

    Отсчитывать от сегодняшней даты нельзя: выгрузка историческая и у регионов
    заканчивается в разное время (Караганда 2023-12, Павлодар 2026-07). От общей
    последней даты секция «требует внимания» показывала бы только Павлодар.
    """
    return daily.groupby("region")["день"].max()


def mark_new(ev, last_day, days):
    """Пометить события, попавшие в последние `days` дней СВОЕГО региона."""
    ev = ev.copy()
    edge = ev["регион"].map(last_day) - pd.Timedelta(days=days - 1)
    ev["новое"] = ev["конец"] >= edge
    return ev


def fig_event_series(daily, det, region, topic, day, before=CONTEXT_DAYS,
                     after=CONTEXT_DAYS):
    """Дневной ряд вокруг события с отмеченными днями всплеска.

    Тот же вид, что в разборе экибастузского эпизода: столбцы по дням, линия
    медианы окна, дни всплеска выделены цветом."""
    lo = pd.Timestamp(day) - pd.Timedelta(days=before)
    hi = pd.Timestamp(day) + pd.Timedelta(days=after)
    m = det[(det.region == region) & (det.topic == topic)
            & (det["день"] >= lo) & (det["день"] <= hi)].sort_values("день")
    if m.empty:
        return go.Figure().add_annotation(text="нет данных", showarrow=False)
    colors = ["#C62828" if sp else "#90A4AE" for sp in m["spike"]]
    fig = go.Figure()
    fig.add_bar(x=m["день"], y=m["count"], marker_color=colors, name="обращений",
                hovertemplate="%{x|%d.%m.%Y}<br>%{y} обращений<extra></extra>")
    fig.add_scatter(x=m["день"], y=m["median_w"], mode="lines", name="медиана окна",
                    line=dict(color="#1565C0", width=2, dash="dot"),
                    hovertemplate="%{x|%d.%m.%Y}<br>медиана %{y:.1f}<extra></extra>")
    fig.add_scatter(x=m["день"], y=m["threshold"], mode="lines", name="порог",
                    line=dict(color="#EF6C00", width=1),
                    hovertemplate="%{x|%d.%m.%Y}<br>порог %{y:.1f}<extra></extra>")
    fig.update_layout(title=f"{region} · {topic} · {pd.Timestamp(day):%d.%m.%Y}",
                      height=380, margin=dict(t=50, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


# ---------------------------------------------------------------- блок 2
def fig_structure(df):
    """Доли problem / info / system по регионам, столбцы стопкой."""
    g = (df.groupby(["region", "appeal_class"]).size()
           .rename("n").reset_index())
    g["доля"] = g["n"] / g.groupby("region")["n"].transform("sum")
    g["класс"] = g["appeal_class"].map(CLASS_RU)
    order = (g[g.appeal_class == "problem"]
             .sort_values("доля")["region"].tolist())
    order += [r for r in g["region"].unique() if r not in order]
    fig = px.bar(g, x="region", y="доля", color="класс",
                 category_orders={"region": order,
                                  "класс": list(CLASS_COLOR)},
                 color_discrete_map=CLASS_COLOR,
                 labels={"region": "", "доля": "Доля потока"},
                 custom_data=["n"])
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: "
                                    "%{y:.1%} (%{customdata[0]:,} обращений)"
                                    "<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%",
                      legend_title_text="", height=420,
                      margin=dict(t=30, b=80))
    return fig


# ---------------------------------------------------------------- блок 3
def fig_dynamics(df):
    """Обращения по месяцам, линия на регион. Только problem."""
    g = df.groupby(["месяц", "region"]).size().rename("обращений").reset_index()
    single = g["region"].nunique() == 1
    fig = px.line(g, x="месяц", y="обращений",
                  color=None if single else "region",
                  markers=True, labels={"месяц": "", "region": ""})
    if single:
        fig.update_traces(line_color="#1565C0", line_width=3,
                          fill="tozeroy", fillcolor="rgba(21,101,192,0.12)")
        fig.update_layout(title=f"{g['region'].iloc[0]} — обращения по месяцам")
    else:
        fig.update_layout(legend_title_text="")
    fig.update_layout(height=430, margin=dict(t=50, b=40),
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------- блок 4
def fig_topics(df):
    """Распределение по темам, столбцы по убыванию."""
    g = (df.groupby("topic").size().rename("обращений")
           .reset_index().sort_values("обращений", ascending=False))
    g["доля"] = g["обращений"] / g["обращений"].sum()
    fig = px.bar(g, x="topic", y="обращений", labels={"topic": ""},
                 custom_data=["доля"])
    fig.update_traces(marker_color="#1565C0",
                      hovertemplate="%{x}<br>%{y:,} обращений "
                                    "(%{customdata[0]:.1%})<extra></extra>")
    fig.update_layout(height=430, margin=dict(t=30, b=140),
                      xaxis={"categoryorder": "total descending"})
    return fig


# ---------------------------------------------------------------- секция событий
FEED_COLS = ["дата", "регион", "тема", "обращений", "медиана окна", "кратность",
             "прирост", "дней подряд", "тип"]


def feed_view(ev):
    """Лента в том виде, в каком она показывается и выгружается."""
    out = ev[FEED_COLS].copy()
    out["дата"] = out["дата"].dt.date
    out["медиана окна"] = out["медиана окна"].round(1)
    out["кратность"] = out["кратность"].round(1)
    out["прирост"] = out["прирост"].round(1)
    return out


def events_section(st, df):
    """Лента всплесков с фильтрами, секцией «требует внимания» и разбором события."""
    st.subheader("События детектора")
    daily, det, ev_all = st.cache_data(load_events)()
    last_day = region_last_day(daily)

    with st.expander("Как это считается", expanded=False):
        st.markdown(
            f"""Дневные счётчики по срезу «регион × тема», только городские проблемы.
Скользящая медиана и MAD по окну 28 суток, **окно сдвинуто на `[t−28, t−1]`** —
день всплеска в него не входит. Всплеск: `обращений > медиана + 4·MAD` и не
меньше 10 обращений за день.

Детекция **не работает** в двух случаях, и это сделано намеренно: первый
календарный месяц ряда региона и окно, задетое пропуском выгрузки
({GAP_MIN}+ дней подряд без обращений по региону). Нет окна — нет детекции.

Тип события: **сезонный** — в большинстве прошлых лет в те же даты ±10 дней
всплеск уже был; **аномалия** — прошлые годы наблюдались, но всплеск был в
меньшинстве из них; **без типа** — сравнивать не с чем. Метки разных регионов
несопоставимы по силе: за ними стоит разное число прошлых лет.""")

    # ---- фильтры
    f = st.columns([2, 2, 1.2, 1.2])
    regions = sorted(ev_all["регион"].unique())
    topics = sorted(ev_all["тема"].unique())
    sel_reg = f[0].multiselect("Регион", regions, default=regions, key="ev_reg")
    sel_top = f[1].multiselect("Тема", topics, default=topics, key="ev_top")
    types = ["аномалия", "сезонный", "без типа"]
    sel_type = f[2].multiselect("Тип", types, default=types, key="ev_type")
    min_ratio = f[3].number_input("Кратность от", min_value=1.0, value=1.0, step=0.5,
                                  key="ev_ratio")

    g = st.columns([3, 1.4, 1.6])
    dmin, dmax = ev_all["дата"].min().date(), ev_all["дата"].max().date()
    period = g[0].date_input("Период", (dmin, dmax), min_value=dmin, max_value=dmax,
                             key="ev_period")
    order = g[1].radio("Сортировать по", ["приросту", "кратности"], key="ev_order",
                       horizontal=True)
    new_days = g[2].number_input("«Новое» — сколько последних дней", min_value=1,
                                 max_value=90, value=NEW_DAYS_DEFAULT, key="ev_new")

    ev = ev_all[ev_all["регион"].isin(sel_reg) & ev_all["тема"].isin(sel_top)
                & ev_all["тип"].isin(sel_type) & (ev_all["кратность"] >= min_ratio)]
    if isinstance(period, (tuple, list)) and len(period) == 2:
        ev = ev[(ev["дата"] >= pd.Timestamp(period[0]))
                & (ev["дата"] <= pd.Timestamp(period[1]))]
    ev = mark_new(ev, last_day, int(new_days))
    sort_col = "прирост" if order == "приросту" else "кратность"
    ev = ev.sort_values(sort_col, ascending=False)
    descr = [f"события: регионов {len(sel_reg)} из {len(regions)}, "
             f"тем {len(sel_top)} из {len(topics)}",
             f"тип события: {', '.join(sel_type) if sel_type else '—'}",
             f"кратность не ниже {min_ratio:g}, сортировка по {order}"]

    # ---- требует внимания
    fresh = ev[ev["новое"]]
    st.markdown(f"#### Требует внимания — {len(fresh)}")
    st.caption(
        f"События последних {int(new_days)} дней. Отсчёт идёт от последнего дня данных "
        f"КАЖДОГО региона, а не от сегодняшней даты: выгрузка историческая и "
        f"заканчивается в разное время — "
        + ", ".join(f"{r.replace(' область', '')} {d:%d.%m.%Y}"
                    for r, d in last_day.items()) + ".")
    if fresh.empty:
        st.info("Новых событий под текущими фильтрами нет.")
    else:
        st.dataframe(feed_view(fresh), width="stretch", hide_index=True)

    # ---- вся лента
    st.markdown(f"#### Все события — {len(ev)}")
    if ev.empty:
        st.warning("Под выбранные фильтры не попало ни одного события.")
        return feed_view(ev), descr
    shown = feed_view(ev)
    sel = st.dataframe(shown, width="stretch", hide_index=True,
                       on_select="rerun", selection_mode="single-row", key="ev_table")
    rows = sel.selection.rows if hasattr(sel, "selection") else []
    if not rows:
        st.caption("Выберите строку, чтобы посмотреть ряд за две недели до и после "
                   "события.")
        return shown, descr
    r = ev.iloc[rows[0]]
    st.plotly_chart(fig_event_series(daily, det, r["регион"], r["тема"], r["дата"]),
                    width="stretch")
    st.caption(
        f"Пик {r['пик']:%d.%m.%Y} — {int(r['обращений'])} обращений при медиане окна "
        f"{r['медиана окна']:.1f}: кратность ×{r['кратность']:.1f}, прирост "
        f"+{r['прирост']:.1f}, держалось {int(r['дней подряд'])} дн. "
        f"Тип — {r['тип']}. Красным отмечены дни, которые детектор считает всплеском.")
    return shown, descr


# ---------------------------------------------------------------- выгрузка
def export_section(st, df, flt, events, sel_reg, sel_topic, all_topics, lo, hi, ev_descr):
    """Excel и PDF с тем же содержимым, что на экране. Ничего не пересчитывает."""
    st.subheader("Выгрузка отчётов")
    scope = df[df.region.isin(sel_reg) & (df.created_at >= lo) & (df.created_at < hi)]
    descr = [f"классы: все (сводка), только городские проблемы (темы и события)",
             f"темы: {len(sel_topic)} из {len(all_topics)}", *ev_descr]
    st.caption(
        "Выгружается ровно то, что показано выше при текущих фильтрах: сводка по "
        "регионам по всем классам, структура тем и лента событий. Числа не "
        "пересчитываются. Первый лист и первая страница — ограничения выгрузки; "
        "их текст берётся из CLAUDE.md, а не пишется здесь заново.")
    period = (lo, hi - pd.Timedelta(days=1))
    c = st.columns(2)
    if c[0].button("Собрать Excel", width="stretch"):
        data = build_excel(scope, events, sel_reg, period, descr)
        st.session_state["xlsx"] = data
    if c[1].button("Собрать PDF", width="stretch"):
        figs = [("Структура потока по регионам", fig_structure(scope)),
                ("Динамика по месяцам", fig_dynamics(flt)),
                ("Структура тем", fig_topics(flt))]
        st.session_state["pdf"] = build_pdf(scope, events, sel_reg, period, figs, descr)
    stamp = f"{lo:%Y%m%d}-{hi - pd.Timedelta(days=1):%Y%m%d}"
    if st.session_state.get("xlsx"):
        c[0].download_button("Скачать Excel", st.session_state["xlsx"],
                             file_name=f"obrashcheniya-109-{stamp}.xlsx", width="stretch",
                             mime="application/vnd.openxmlformats-officedocument."
                                  "spreadsheetml.sheet")
    if st.session_state.get("pdf"):
        c[1].download_button("Скачать PDF", st.session_state["pdf"],
                             file_name=f"obrashcheniya-109-{stamp}.pdf",
                             mime="application/pdf", width="stretch")


# ---------------------------------------------------------------- страница
def main():
    import streamlit as st

    st.set_page_config(page_title="Обращения 109 — витрина руководителя",
                       layout="wide")
    st.title("Обращения 109 — витрина руководителя")

    if not DATA.exists():
        st.error(f"Нет файла {DATA}. Соберите его: "
                 "`.venv/bin/python -m src.adapters.build_unified`, "
                 "затем `.venv/bin/python -m src.topic_mapping`.")
        st.stop()

    df = st.cache_data(load_data)()

    # ---------------- блок 1: сводка
    st.subheader("Сводка")
    vc = df["appeal_class"].value_counts()
    c = st.columns(6)
    c[0].metric("Всего обращений", f"{len(df):,}".replace(",", " "))
    c[1].metric("Городские проблемы", f"{vc.get('problem', 0):,}".replace(",", " "),
                f"{100*vc.get('problem', 0)/len(df):.1f}% потока")
    c[2].metric("Справочные", f"{vc.get('info', 0):,}".replace(",", " "),
                f"{100*vc.get('info', 0)/len(df):.1f}% потока")
    c[3].metric("Служебные", f"{vc.get('system', 0):,}".replace(",", " "),
                f"{100*vc.get('system', 0)/len(df):.1f}% потока")
    c[4].metric("Регионов", df["region"].nunique())
    c[5].metric("Период", f"{df.created_at.min():%m.%Y} — {df.created_at.max():%m.%Y}")

    st.divider()

    # ---------------- блок 1б: события детектора
    events, ev_descr = events_section(st, df)

    st.divider()

    # ---------------- блок 2: структура потока
    st.subheader("Структура потока по регионам")
    st.plotly_chart(fig_structure(df), width="stretch")
    st.caption(
        "Разброс доли не-problem от 0.2% до 67.8% — это разница в учётной "
        "политике регионов, а не в нагрузке. Костанай и Туркестан заводят "
        "карточку только на инцидент, ВКО и Павлодар регистрируют любой звонок."
    )

    st.divider()

    # ---------------- общие фильтры для блоков 3 и 4
    problem = df[df.appeal_class == "problem"]
    st.subheader("Фильтры")
    regions = sorted(problem["region"].unique())
    topics = sorted(problem["topic"].unique())
    f = st.columns([2, 2, 2])
    sel_reg = f[0].multiselect("Регион", regions, default=regions)
    # Границы берутся по ВСЕЙ таблице, а не по problem: этот же период уходит в
    # выгрузку, где сводка по регионам считается по всем классам. При границе по
    # problem три справочных обращения Павлодара за 2020-02-09 выпадали из сводки,
    # и она расходилась с эталоном на 3 строки.
    dmin, dmax = df.created_at.min().date(), df.created_at.max().date()
    sel_period = f[1].date_input("Период", (dmin, dmax),
                                 min_value=dmin, max_value=dmax)
    sel_topic = f[2].multiselect("Тема", topics, default=topics)

    flt = problem[problem.region.isin(sel_reg) & problem.topic.isin(sel_topic)]
    if isinstance(sel_period, (tuple, list)) and len(sel_period) == 2:
        a, b = (pd.Timestamp(sel_period[0]),
                pd.Timestamp(sel_period[1]) + pd.Timedelta(days=1))
        flt = flt[(flt.created_at >= a) & (flt.created_at < b)]

    st.caption(f"Под фильтр попало {len(flt):,} обращений класса «городские "
               f"проблемы»".replace(",", " "))

    if flt.empty:
        st.warning("Под выбранные фильтры не попало ни одного обращения.")
        st.stop()

    st.divider()

    # ---------------- блок 3: динамика
    st.subheader("Динамика по месяцам")
    st.plotly_chart(fig_dynamics(flt), width="stretch")

    st.divider()

    # ---------------- блок 4: структура тем
    title = "Структура тем"
    if len(sel_reg) == 1:
        title += f" — {sel_reg[0]}"
    st.subheader(title)
    st.plotly_chart(fig_topics(flt), width="stretch")

    st.divider()

    # ---------------- блок 5: выгрузка
    export_section(st, df, flt, events, sel_reg, sel_topic, topics, a, b, ev_descr)


if __name__ == "__main__":
    main()
