#!/usr/bin/env python3
"""Витрина руководителя по обращениям 109.

    PYTHONPATH=. .venv/bin/streamlit run src/dashboard.py

Из корня репозитория и только с PYTHONPATH=.: streamlit кладёт в sys.path каталог
скрипта (src/), а не корень, и без этого импорт src.* падает (CLAUDE.md, раздел 0).

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
from src.risk_view import risk_section
from src.spikes import (GAP_MIN, SEASONAL_REGIONS, classify_seasonal, daily_counts,
                        detect, gap_days, with_duration)

DATA = Path("data/unified.parquet")
CONTEXT_DAYS = 14        # сколько дней показывать до и после события
NEW_DAYS_DEFAULT = 7     # «требует внимания» — события за столько последних дней
# Подписи на экране — для руководителя, не для инженера. Технические имена
# (spike_type, appeal_class) остаются в данных и в блоках «Как это считается».
TYPE_RU = {"seasonal": "сезонное", "anomaly": "необычное",
           "без типа": "не с чем сравнить"}
CLASS_RU = {"problem": "Жалобы на городские проблемы", "info": "Справочные звонки",
            "system": "Служебные записи"}
CLASS_COLOR = {"Жалобы на городские проблемы": "#2E7D32", "Справочные звонки": "#F9A825",
               "Служебные записи": "#9E9E9E"}
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
    медианы окна («обычный уровень») и порога («граница всплеска»), дни
    всплеска выделены цветом."""
    lo = pd.Timestamp(day) - pd.Timedelta(days=before)
    hi = pd.Timestamp(day) + pd.Timedelta(days=after)
    m = det[(det.region == region) & (det.topic == topic)
            & (det["день"] >= lo) & (det["день"] <= hi)].sort_values("день")
    if m.empty:
        return go.Figure().add_annotation(text="нет данных", showarrow=False)
    colors = ["#C62828" if sp else "#90A4AE" for sp in m["spike"]]
    fig = go.Figure()
    fig.add_bar(x=m["день"], y=m["count"], marker_color=colors, name="жалоб за день",
                hovertemplate="%{x|%d.%m.%Y}<br>%{y} жалоб<extra></extra>")
    fig.add_scatter(x=m["день"], y=m["median_w"], mode="lines",
                    name="обычно в день (по прошлым 4 неделям)",
                    line=dict(color="#1565C0", width=2, dash="dot"),
                    hovertemplate="%{x|%d.%m.%Y}<br>обычно %{y:.1f}<extra></extra>")
    fig.add_scatter(x=m["день"], y=m["threshold"], mode="lines",
                    name="граница всплеска",
                    line=dict(color="#EF6C00", width=1),
                    hovertemplate="%{x|%d.%m.%Y}<br>граница %{y:.1f}<extra></extra>")
    fig.update_layout(title=f"{region} · {topic} · {pd.Timestamp(day):%d.%m.%Y}",
                      height=380, margin=dict(t=50, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


# ---------------------------------------------------------------- блок 2
def fig_structure(df):
    """Доли problem / info / system по регионам, столбцы стопкой.
    На экране: жалобы / справочные звонки / служебные записи (CLASS_RU)."""
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
                 labels={"region": "", "доля": "Доля всех обращений региона"},
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
    """Жалобы по месяцам, линия на регион. Только problem."""
    g = df.groupby(["месяц", "region"]).size().rename("обращений").reset_index()
    single = g["region"].nunique() == 1
    fig = px.line(g, x="месяц", y="обращений",
                  color=None if single else "region", markers=True,
                  labels={"месяц": "", "region": "", "обращений": "жалоб за месяц"})
    # без явного шаблона подсказка при пустой подписи оси начиналась с «=»
    fig.update_traces(hovertemplate=("" if single else "%{fullData.name}: ")
                      + "%{y:,} жалоб<extra></extra>")
    if single:
        fig.update_traces(line_color="#1565C0", line_width=3,
                          fill="tozeroy", fillcolor="rgba(21,101,192,0.12)")
        fig.update_layout(title=f"{g['region'].iloc[0]} — жалобы по месяцам")
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
    fig = px.bar(g, x="topic", y="обращений",
                 labels={"topic": "", "обращений": "жалоб"}, custom_data=["доля"])
    fig.update_traces(marker_color="#1565C0",
                      hovertemplate="%{x}<br>%{y:,} жалоб "
                                    "(%{customdata[0]:.1%})<extra></extra>")
    fig.update_layout(height=430, margin=dict(t=30, b=140),
                      xaxis={"categoryorder": "total descending"})
    return fig


# ---------------------------------------------------------------- секция событий
# Строка ленты — это СОБЫТИЕ, а не день: череда подряд идущих дней-всплесков
# сворачивается в одну запись. Поэтому «дата» — начало события, «обращений» —
# счётчик в ПИКОВЫЙ день, а «медиана окна» — фон на СТАРТЕ. Без этих подписей
# строка читается как «22.11 было 480 обращений», хотя 22.11 их было 36,
# а 480 пришлось на 30.11. Колонки названы так, чтобы это читалось само.
#
# Подписи — для руководителя: медиана окна показана как «обычно в день до
# начала», кратность — «во сколько раз больше обычного», прирост — «сверх
# обычного». Соответствие терминам — в блоке «Как это считается».
FEED_COLS = ["дата", "пик", "регион", "тема", "обращений", "медиана окна",
             "кратность", "прирост", "дней подряд", "тип"]
FEED_NAMES = {"дата": "начало", "пик": "день пика",
              "обращений": "жалоб в день пика",
              "медиана окна": "обычно в день до начала",
              "кратность": "во сколько раз больше обычного",
              "прирост": "жалоб сверх обычного",
              "дней подряд": "длилось дней",
              "тип": "характер"}


def feed_view(ev):
    """Лента в том виде, в каком она показывается и выгружается."""
    out = ev[FEED_COLS].copy()
    out["дата"] = out["дата"].dt.date
    out["пик"] = out["пик"].dt.date
    for c in ("медиана окна", "кратность", "прирост"):
        out[c] = out[c].round(1)
    return out.rename(columns=FEED_NAMES)


def event_caption(r):
    """Подпись под графиком события — отдельно от streamlit, чтобы её можно было
    проверить без клика по строке. Пик и фон — РАЗНЫЕ даты (5f, ограничение 2):
    кратность считается от фона на старте события, поэтому и в тексте фон
    привязан к началу всплеска, а не ко дню пика."""
    return (
        f"Всплеск начался {r['дата']:%d.%m.%Y} и отмечался {int(r['дней подряд'])} "
        f"дн. подряд. Самый тяжёлый день — {r['пик']:%d.%m.%Y}, жалоб за этот день: "
        f"{int(r['обращений'])}. Перед началом всплеска обычно было около "
        f"{r['медиана окна']:.1f} в день, то есть в день пика жалоб было в "
        f"{r['кратность']:.1f} раза больше обычного — на {r['прирост']:.1f} сверх "
        f"обычного. Характер — {r['тип']}. Красным отмечены дни всплеска: жалоб выше "
        f"оранжевой линии и не меньше 10 за день. Синяя пунктирная линия — сколько "
        f"обычно бывало в день за предыдущие четыре недели.")


def events_section(st, df):
    """Лента всплесков с фильтрами, секцией «требует внимания» и разбором события."""
    st.subheader("Всплески жалоб")
    st.markdown(
        "**Что это значит.** Всплеск — дни, когда жалоб на одну тему в одном регионе "
        "стало резко больше, чем обычно бывало в предыдущие четыре недели. Всплеск "
        "показывает, **куда посмотреть в первую очередь**; предсказанием аварии он "
        "не является.")
    daily, det, ev_all = st.cache_data(load_events)()
    last_day = region_last_day(daily)

    with st.expander("Как это считается", expanded=False):
        st.markdown(
            f"""Дневные счётчики по срезу «регион × тема», только класс `problem`
(жалобы на городские проблемы). Скользящая медиана и MAD по окну 28 суток,
**окно сдвинуто на `[t−28, t−1]`** — день всплеска в него не входит. Всплеск:
`обращений > медиана + 4·MAD` и не меньше 10 обращений за день.

Колонки ленты и их технические имена:
- **обычно в день до начала** — медиана окна на СТАРТЕ события;
- **во сколько раз больше обычного** — кратность: жалоб в день пика / max(медиана
  на старте, 1). При медиане меньше 1 делитель равен 1;
- **жалоб сверх обычного** — прирост: жалоб в день пика − медиана на старте;
- **день пика** и **начало** — разные даты: пик сравнивается с фоном на старте.

На графике события: синяя пунктирная линия — медиана окна на каждый день,
оранжевая — порог `медиана + 4·MAD`.

Детекция **не работает** в двух случаях, и это сделано намеренно: первый
календарный месяц ряда региона и окно, задетое пропуском выгрузки
({GAP_MIN}+ дней подряд без обращений по региону). Нет окна — нет детекции.

Характер события (`spike_type`): **сезонное** (`seasonal`) — в большинстве
прошлых лет в те же даты ±10 дней всплеск уже был; **необычное** (`anomaly`) —
прошлые годы наблюдались, но всплеск был в меньшинстве из них; **не с чем
сравнить** — прошлых лет в данных нет. Метки разных регионов несопоставимы по
силе: за ними стоит разное число прошлых лет. Разбор — раздел 5f CLAUDE.md.""")

    # ---- фильтры
    f = st.columns([2, 2, 1.2, 1.2])
    regions = sorted(ev_all["регион"].unique())
    topics = sorted(ev_all["тема"].unique())
    sel_reg = f[0].multiselect("Регион", regions, default=regions, key="ev_reg")
    sel_top = f[1].multiselect("Тема", topics, default=topics, key="ev_top")
    types = [TYPE_RU[k] for k in ("anomaly", "seasonal", "без типа")]
    sel_type = f[2].multiselect("Характер", types, default=types, key="ev_type",
                                help="Сезонное — в прошлые годы в эти же даты тоже "
                                     "был всплеск. Необычное — обычно в эти даты его "
                                     "не было. Не с чем сравнить — данных за прошлые "
                                     "годы нет.")
    min_ratio = f[3].number_input("Жалоб больше обычного хотя бы во столько раз",
                                  min_value=1.0,
                                  value=1.0, step=0.5, key="ev_ratio",
                                  help="1 — показать все всплески.")

    g = st.columns([3, 1.4, 1.6])
    dmin, dmax = ev_all["дата"].min().date(), ev_all["дата"].max().date()
    period = g[0].date_input("Период", (dmin, dmax), min_value=dmin, max_value=dmax,
                             key="ev_period")
    orders = {"по числу жалоб сверх обычного": "прирост",
              "по тому, во сколько раз больше обычного": "кратность"}
    order = g[1].radio("Сначала показывать", list(orders), key="ev_order")
    new_days = g[2].number_input("Свежими считать всплески за последние, дней",
                                 min_value=1, max_value=90, value=NEW_DAYS_DEFAULT,
                                 key="ev_new")

    ev = ev_all[ev_all["регион"].isin(sel_reg) & ev_all["тема"].isin(sel_top)
                & ev_all["тип"].isin(sel_type) & (ev_all["кратность"] >= min_ratio)]
    if isinstance(period, (tuple, list)) and len(period) == 2:
        ev = ev[(ev["дата"] >= pd.Timestamp(period[0]))
                & (ev["дата"] <= pd.Timestamp(period[1]))]
    ev = mark_new(ev, last_day, int(new_days))
    ev = ev.sort_values(orders[order], ascending=False)
    descr = [f"всплески: регионов {len(sel_reg)} из {len(regions)}, "
             f"тем {len(sel_top)} из {len(topics)}",
             f"характер всплеска: {', '.join(sel_type) if sel_type else '—'}",
             f"больше обычного не менее чем в {min_ratio:g} раза, "
             f"порядок — {order}"]

    # ---- требует внимания
    fresh = ev[ev["новое"]]
    st.markdown(f"#### Требует внимания — {len(fresh):,}".replace(",", " "))
    st.caption(
        f"Всплески за последние {int(new_days)} дней данных. Данные регионов "
        f"заканчиваются в разное время, поэтому «последние дни» у каждого региона "
        f"свои, а не от сегодняшней даты: "
        + ", ".join(f"{r.replace(' область', '')} — до {d:%d.%m.%Y}"
                    for r, d in last_day.items()) + ".")
    if fresh.empty:
        st.info("Свежих всплесков при выбранных фильтрах нет.")
    else:
        st.dataframe(feed_view(fresh), width="stretch", hide_index=True)

    # ---- вся лента
    st.markdown(f"#### Все всплески — {len(ev):,}".replace(",", " "))
    st.caption("Как читать строку: «обычно в день до начала» — сколько жалоб было в "
               "обычный день перед всплеском; «во сколько раз больше обычного» и "
               "«жалоб сверх обычного» — насколько день пика превысил этот обычный "
               "уровень. Характер «сезонное» — в эти даты такое уже бывало в "
               "большинстве прошлых лет. Где прошлых лет в данных один-два, эта "
               "отметка ненадёжна и может смениться у соседних дней одного эпизода.")
    if ev.empty:
        st.warning("Под выбранные фильтры не попало ни одного всплеска.")
        return feed_view(ev), descr
    shown = feed_view(ev)
    sel = st.dataframe(shown, width="stretch", hide_index=True,
                       on_select="rerun", selection_mode="single-row", key="ev_table")
    rows = sel.selection.rows if hasattr(sel, "selection") else []
    if not rows:
        st.caption("Нажмите на строку — ниже появится график жалоб по дням за две "
                   "недели до и после всплеска.")
        return shown, descr
    r = ev.iloc[rows[0]]
    st.plotly_chart(fig_event_series(daily, det, r["регион"], r["тема"], r["дата"]),
                    width="stretch")
    st.caption(event_caption(r))
    return shown, descr


# ---------------------------------------------------------------- выгрузка
def export_section(st, df, flt, events, sel_reg, sel_topic, all_topics, lo, hi, ev_descr):
    """Excel и PDF с тем же содержимым, что на экране. Ничего не пересчитывает."""
    st.subheader("Выгрузка отчётов")
    scope = df[df.region.isin(sel_reg) & (df.created_at >= lo) & (df.created_at < hi)]
    descr = ["сводка по регионам — все обращения; темы и всплески — только жалобы "
             "на городские проблемы",
             f"темы: {len(sel_topic)} из {len(all_topics)}", *ev_descr]
    st.caption(
        "В файл попадает ровно то, что показано на экране при текущих фильтрах: "
        "сводка по регионам, темы жалоб и список всплесков. Числа не пересчитываются. "
        "На первом листе Excel и первой странице PDF — оговорки, без которых цифры "
        "легко понять неправильно, и список выбранных фильтров.")
    with st.expander("Как это считается", expanded=False):
        st.markdown(
            "Сводка по регионам считается по всем классам обращения (`problem`, "
            "`info`, `system`), темы и события — только по `problem`. Текст "
            "ограничений берётся из CLAUDE.md, раздел 5d, по якорным фразам "
            "(`LIMIT_ANCHORS` в `src/export.py`), а не пишется здесь заново; "
            "поэтому вторая оговорка в файле содержит техническое условие "
            "`appeal_class == 'problem'`. Excel и PDF строятся из одной функции "
            "`export_frames`; готовый Excel проверяется на ПДн `check_export_pii`.")
    period = (lo, hi - pd.Timedelta(days=1))
    c = st.columns(2)
    if c[0].button("Собрать Excel", width="stretch"):
        data = build_excel(scope, events, sel_reg, period, descr)
        st.session_state["xlsx"] = data
    if c[1].button("Собрать PDF", width="stretch"):
        figs = [("Из чего состоит поток обращений по регионам", fig_structure(scope)),
                ("Жалобы по месяцам", fig_dynamics(flt)),
                ("О чём жалуются", fig_topics(flt))]
        # PDF зависит от reportlab и от системного шрифта с кириллицей. Обе
        # причины отказа внешние по отношению к данным, поэтому объясняем их,
        # а не показываем трейсбек: Excel при этом остаётся доступен.
        try:
            st.session_state["pdf"] = build_pdf(scope, events, sel_reg, period,
                                                figs, descr)
        except ImportError:
            st.error("PDF не собран: не установлен `reportlab`. "
                     "Установите его — `uv pip install reportlab==5.0.1` — "
                     "или выгрузите Excel, он не требует дополнительных пакетов.")
        except RuntimeError as e:
            st.error(f"PDF не собран: {e}. Excel при этом доступен.")
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

    st.set_page_config(page_title="Обращения в 109 — обзор для руководителя",
                       layout="wide")
    st.title("Обращения в 109 — обзор для руководителя")

    if not DATA.exists():
        st.error(
            f"Нет файла {DATA}. Соберите его тремя шагами по порядку:\n\n"
            "1. `.venv/bin/python -m src.adapters.adapters`\n"
            "2. `.venv/bin/python -m src.adapters.build_unified`\n"
            "3. `.venv/bin/python -m src.topic_mapping`\n\n"
            "Подробнее — раздел «Развёртывание с нуля» в CLAUDE.md.")
        st.stop()

    df = st.cache_data(load_data)()

    # ---------------- блок 1: сводка
    st.subheader("Сводка")
    vc = df["appeal_class"].value_counts()
    c = st.columns(6)
    c[0].metric("Всего обращений", f"{len(df):,}".replace(",", " "))
    for i, k in enumerate(("problem", "info", "system"), start=1):
        c[i].metric(CLASS_RU[k], f"{vc.get(k, 0):,}".replace(",", " "),
                    f"{100*vc.get(k, 0)/len(df):.1f}% всех обращений",
                    delta_color="off")
    c[4].metric("Регионов", df["region"].nunique())
    c[5].metric("Период", f"{df.created_at.min():%m.%Y} — {df.created_at.max():%m.%Y}")
    st.markdown(
        f"**Что это значит.** Работу городским службам дают только жалобы на "
        f"городские проблемы — {100*vc.get('problem', 0)/len(df):.1f}% всех "
        f"обращений. Справочные звонки — просьбы о справке: контакты служб, "
        f"справочная информация, выборы, карантин. Служебные записи — сброшенные "
        f"звонки, переадресации на 102, 103, 112, благодарности. Всплески, динамика и "
        f"темы ниже считаются только по жалобам. Период у регионов разный: Павлодар — "
        f"с 2020 года, Акмола — только с июля 2025.")
    with st.expander("Как это считается", expanded=False):
        st.markdown(
            "Класс обращения — колонка `appeal_class`: `problem` (жалобы на "
            "городские проблемы), `info` (справочные), `system` (служебные). "
            "Присваивается по теме из справочника региона правилами "
            "`src/topic_mapping.py`, раздел 5e CLAUDE.md. Регионы записывают разное: "
            "одни заводят карточку только на проблему, другие — на любой звонок "
            "(раздел 5d). Окна данных по регионам — раздел 5c.")

    st.divider()

    # ---------------- блок 1б: события детектора
    events, ev_descr = events_section(st, df)

    st.divider()

    # ---------------- блок 2: структура потока
    st.subheader("Из чего состоит поток обращений по регионам")
    st.plotly_chart(fig_structure(df), width="stretch")
    st.markdown(
        "**Что это значит.** Доля записей, которые не являются жалобами, — от 0.2% в "
        "Туркестанской области до 67.8% в ВКО. Это разница в том, что регион "
        "записывает как обращение, а не в нагрузке: Костанай и Туркестан заводят "
        "карточку только на обращение по проблеме, ВКО и Павлодар — на любой звонок. "
        "Поэтому сравнивать регионы по общему числу обращений нельзя — только по "
        "жалобам на городские проблемы.")
    with st.expander("Как это считается", expanded=False):
        st.markdown(
            "Доля «не-problem» = (`info` + `system`) / все строки региона. Точные "
            "крайние значения: Туркестан 0.1748%, ВКО 67.8408% — разброс в 388 раз "
            "(по округлённым 0.2 и 67.8 выходит 339, так не считать). Разбор по "
            "регионам — раздел 5d CLAUDE.md.")

    st.divider()

    # ---------------- общие фильтры для блоков 3 и 4
    problem = df[df.appeal_class == "problem"]
    st.subheader("Фильтры для графиков ниже")
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

    st.caption(f"Жалоб на городские проблемы под фильтром: {len(flt):,}. "
               .replace(",", " ")
               + "Справочные звонки и служебные записи в графики ниже не входят.")

    if flt.empty:
        st.warning("Под выбранные фильтры не попало ни одного обращения.")
        st.stop()

    st.divider()

    # ---------------- блок 3: динамика
    st.subheader("Жалобы по месяцам")
    st.plotly_chart(fig_dynamics(flt), width="stretch")
    st.markdown(
        "**Что это значит.** Каждая линия — один регион. Сравнивать высоту линий "
        "разных регионов нельзя: регионы разного размера и по-разному ведут учёт. "
        "Смотреть стоит на изменения внутри одной линии. Первый и последний месяц "
        "региона могут быть неполными. Провал у Туркестанской области в первой "
        "половине 2025 года — за эти месяцы в выгрузке нет данных, а не нет жалоб.")
    with st.expander("Как это считается", expanded=False):
        st.markdown(
            "Счётчик строк класса `problem` по календарным месяцам (`created_at`). "
            "У Туркестана в выгрузке нет обращений с 2025-01-01 по 2025-06-29 — "
            "180 дней, 27.4% окна региона (раздел 10, пункт 7). Месяцы без строк "
            "на графике не рисуются, и линия соединяет соседние точки.")

    st.divider()

    # ---------------- блок 4: структура тем
    title = "О чём жалуются"
    if len(sel_reg) == 1:
        title += f" — {sel_reg[0]}"
    st.subheader(title)
    st.plotly_chart(fig_topics(flt), width="stretch")
    st.markdown(
        "**Что это значит.** Какие темы дают больше всего жалоб при выбранных "
        "фильтрах. «Прочее» — жалобы, которым не нашлось места в общем списке из 14 "
        "тем. У ВКО в 2023 году менялся порядок учёта тем, поэтому её темы до и после "
        "2023 года между собой не сравнивать.")
    with st.expander("Как это считается", expanded=False):
        st.markdown(
            "Тема (`topic`) присваивается по значению справочника региона правилами "
            "`src/topic_mapping.py` (раздел 5e CLAUDE.md); только класс `problem`. "
            "«Прочее» внутри `problem` по всей таблице — 1.67%. Смена состава потока "
            "ВКО по кварталам и темам — раздел 5g и раздел 10, пункт 8.")

    st.divider()

    # ---------------- блок 5: выгрузка
    export_section(st, df, flt, events, sel_reg, sel_topic, topics, a, b, ev_descr)

    st.divider()

    # ---------------- блок 6: риск просрочки (только Караганда, из reports/)
    risk_section(st)


if __name__ == "__main__":
    main()
