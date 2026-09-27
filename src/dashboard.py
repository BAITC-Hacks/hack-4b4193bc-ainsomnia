#!/usr/bin/env python3
"""Витрина руководителя по обращениям 109.

    .venv/bin/nazar-dashboard

Команда появляется после `uv pip install --python .venv/bin/python -e .` и сама
переходит в корень проекта. Пакет src установлен, поэтому PYTHONPATH не нужен
(CLAUDE.md, раздел 0).

Источник — data/unified.parquet (собирается src/adapters/build_unified.py
и размечается src/topic_mapping.py). Если не указано иное, работаем по
классу problem: info и system — справочный трафик и артефакты колл-центра,
включать их в аналитику нагрузки нельзя.

Функции fig_* не зависят от streamlit — их можно вызвать отдельно, чтобы
отрендерить графики в файл и проверить, что отрисовалось.
"""
import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import paths
from src.ui.theme import apply_theme, chart
from src.ui.components import masthead, fake_banner, executive_summary, empty_state, info_callout, number
from src.export import PdfUnavailable, build_excel, build_pdf
from src.risk_view import risk_section
from src.event_service import load_events, mark_new, region_last_day
from src.spikes import (GAP_MIN, SEASONAL_REGIONS, classify_seasonal, daily_counts,
                        detect, gap_days, with_duration)

DATA = paths.UNIFIED
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


def load_data(path=DATA, revision=None):
    paths.require_source_marker(Path(path).parent / "SOURCE")
    df = pd.read_parquet(path, columns=["created_at", "region", "topic", "appeal_class"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["месяц"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------- события

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


def days_ru(n):
    """«1 день», «3 дня», «7 дней» — число в поле «свежие» задаёт пользователь."""
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} день"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} дня"
    return f"{n} дней"


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
    st.subheader("Всплески жалоб", anchor="vspleski")
    st.markdown(
        "**Что это значит.** Всплеск — дни, когда жалоб на одну тему в одном регионе "
        "стало резко больше, чем обычно бывало в предыдущие четыре недели. Всплеск "
        "показывает, **куда посмотреть в первую очередь**; предсказанием аварии он "
        "не является.")
    daily, det, ev_all = st.cache_data(load_events)(revision=data_revision())
    last_day = region_last_day(daily)

    with st.expander("Как это считается", expanded=False):
        st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
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

    # UI filters only: the service result and detector are unchanged.
    if ev_all.empty:
        empty_state(st, "В этой версии данных всплесков нет.")
        return feed_view(ev_all), [], {"fresh": ev_all, "last_day": last_day,
                                      "new_days": NEW_DAYS_DEFAULT, "order": "", "regions": []}
    def reset_events():
        for key in ('ev_reg','ev_top','ev_type','ev_ratio','ev_period','ev_order','ev_new','event_choice'):
            st.session_state.pop(key, None)
    with st.expander("Фильтры событий", expanded=False):
        st.button("Сбросить фильтры событий", on_click=reset_events)
        f = st.columns([2, 2, 2])
        regions = sorted(ev_all["регион"].unique())
        topics = sorted(ev_all["тема"].unique())
        sel_reg = f[0].multiselect("Регион", regions, default=regions, key="ev_reg")
        sel_top = f[1].multiselect("Тема", topics, default=topics, key="ev_top")
        dmin, dmax = ev_all["дата"].min().date(), ev_all["дата"].max().date()
        period = f[2].date_input("Период", (dmin, dmax), min_value=dmin, max_value=dmax, key="ev_period")
        st.caption("Дополнительно")
        g = st.columns(4)
        types = [TYPE_RU[k] for k in ("anomaly", "seasonal", "без типа")]
        sel_type = g[0].multiselect("Характер", types, default=types, key="ev_type")
        min_ratio = g[1].number_input("Минимальная кратность", min_value=1.,value=1.,step=.5,key="ev_ratio")
        orders = {"по числу жалоб сверх обычного": "прирост",
                  "по тому, во сколько раз больше обычного": "кратность"}
        order = g[2].radio("Сначала показывать", list(orders), key="ev_order")
        new_days = g[3].number_input("Свежими считать всплески за последние, дней",min_value=1,max_value=90,
                                    value=NEW_DAYS_DEFAULT,key="ev_new")
    st.caption(f"Применено: регионов {len(sel_reg)} · тем {len(sel_top)} · период "
               + (" — ".join(f"{d:%d.%m.%Y}" for d in period) if isinstance(period,(tuple,list)) else str(period)))

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
    # то же, что видно в секции, — для карточек шапки (они рисуются выше, в контейнер)
    info = {"fresh": fresh, "last_day": last_day, "new_days": int(new_days),
            "order": order, "regions": sel_reg}
    st.markdown(f"#### Требует внимания — {len(fresh):,}".replace(",", " "))
    st.caption(
        f"Всплески за последние {days_ru(new_days)} данных. Данные регионов "
        f"заканчиваются в разное время, поэтому «последние дни» у каждого региона "
        f"свои, а не от сегодняшней даты: "
        + ", ".join(f"{r.replace(' область', '')} — до {d:%d.%m.%Y}"
                    for r, d in last_day.items()) + ".")
    if fresh.empty:
        st.info("Свежих всплесков при выбранных фильтрах нет.")
    else:
        from src.ui.events import event_cards
        event_cards(st, fresh.head(3))
        if len(fresh) > 3:
            with st.expander(f"Все свежие события ({len(fresh)})"):
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
        return feed_view(ev), descr, info
    shown = feed_view(ev)
    with st.expander("Полная таблица событий"):
        st.dataframe(shown, width="stretch", hide_index=True)
    options = list(range(len(ev)))
    chosen = st.selectbox("Открыть событие", options, key="event_choice",
                          format_func=lambda i: f"{ev.iloc[i]['дата']:%d.%m.%Y} · {ev.iloc[i]['регион']} · {ev.iloc[i]['тема']}")
    r = ev.iloc[chosen]
    from src.ui.events import event_cards
    event_cards(st, ev.iloc[[chosen]])
    st.plotly_chart(chart(fig_event_series(daily, det, r["регион"], r["тема"], r["дата"])),
                    width="stretch")
    st.caption(event_caption(r))
    return shown, descr, info


# ---------------------------------------------------------------- шапка
def _times(x):
    """42.0 -> «42», 8.2 -> «8.2»: только запись числа, значение то же."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _card(col, label, value, note, link=None, size="2.2rem"):
    """Карточка шапки: подпись, крупное значение, пояснение, ссылка вниз."""
    box = col.container(border=True)
    box.markdown(
        f"<div style='font-size:0.95rem;opacity:0.75'>{html.escape(label)}</div>"
        f"<div style='font-size:{size};font-weight:700;line-height:1.25'>"
        f"{html.escape(value)}</div>", unsafe_allow_html=True)
    box.caption(note)
    if link:
        box.markdown(link)


def top_cards(st, df, info, work):
    """«Что сейчас важно»: три карточки и строка о том, что это за система.

    Числа не считаются заново: первая и вторая карточки — это секция «Требует
    внимания» при текущих фильтрах ленты, третья — рабочая точка блока риска
    (20% потока), которую возвращает risk_section."""
    fresh, last_day = info["fresh"], info["last_day"]
    c = st.columns(3)
    by_reg = fresh["регион"].str.replace(" область", "").value_counts()
    first, last = last_day.idxmin(), last_day.idxmax()
    _card(c[0], "Всплесков требуют внимания", f"{len(fresh):,}".replace(",", " "),
          (f"За последние {days_ru(info['new_days'])} данных каждого региона"
           + (": " + ", ".join(f"{r} — {n}" for r, n in by_reg.items()) if len(fresh)
              else "") + ". Даты у регионов разные, потому что выгрузка историческая "
           f"и заканчивается в разное время: от {last_day[first]:%d.%m.%Y} "
           f"({first.replace(' область', '')}) до {last_day[last]:%d.%m.%Y} "
           f"({last.replace(' область', '')})."),
          "[Открыть список ↓](#vspleski)")
    # Вторая карточка — только регион с самыми свежими данными среди выбранных в
    # ленте. Максимум по всем регионам давал декабрь 2023 у Караганды: «последние
    # дни» у неё — это конец её выгрузки, а не сегодняшний день.
    sel = [r for r in info["regions"] if r in last_day.index]
    if not sel:
        _card(c[1], "Самый сильный в самом свежем регионе", "Регионы не выбраны",
              "В фильтре ленты не выбран ни один регион.", size="1.4rem")
    else:
        newest = last_day[sel].idxmax()
        short = newest.replace(" область", "")
        own = fresh[fresh["регион"] == newest]
        if own.empty:
            _card(c[1], "Самый сильный в самом свежем регионе",
                  f"{short}: свежих всплесков нет",
                  f"Самые свежие данные — у региона «{newest}», по "
                  f"{last_day[newest]:%d.%m.%Y}. За последние {days_ru(info['new_days'])} "
                  f"его данных всплесков при выбранных фильтрах нет. Более старые "
                  f"события других регионов — в ленте.",
                  "[Открыть ленту ↓](#vspleski)", size="1.4rem")
        else:
            r = own.iloc[0]
            _card(c[1], "Самый сильный в самом свежем регионе",
                  f"{short}: {r['тема']} — жалоб в {_times(r['кратность'])} раза "
                  f"больше обычного",
                  f"Данные региона — по {last_day[newest]:%d.%m.%Y}, это самые свежие "
                  f"в выгрузке. День пика {r['пик']:%d.%m.%Y}: {int(r['обращений'])} "
                  f"жалоб при обычных ~{r['медиана окна']:.1f} в день. Характер — "
                  f"{r['тип']}. Первый из всплесков региона в списке «Требует "
                  f"внимания», порядок — {info['order']}.",
                  "[Открыть в ленте ↓](#vspleski)", size="1.4rem")
    if work is None:
        _card(c[2], "Очередь на проверку — Караганда", "нет данных",
              "Нет готового результата модели риска — что сделать, написано в "
              "блоке риска ниже.", "[К блоку риска ↓](#risk)")
    else:
        _card(c[2], "Очередь на проверку риска просрочки — только Караганда",
              f"{work['n']:,}".replace(",", " "),
              f"{work['share']:.0%} обращений с самым высоким риском не уложиться в "
              f"{work['sla_days']} суток (срок — допущение команды). Это очередь на "
              f"проверку, а не прогноз по всей базе. Из них действительно просрочены "
              f"{work['precision']:.0%}, при выборе наугад было бы {work['base']:.0%}. "
              f"Период проверки модели — обращения Карагандинской области с "
              f"{pd.Timestamp(work['cutoff']):%d.%m.%Y}, новее данных нет.",
              "[К блоку риска ↓](#risk)")
    names = ", ".join(r.replace(" область", "") for r in sorted(df["region"].unique()))
    st.markdown(
        f"Обзор обращений граждан в единый контакт-центр 109: всплески жалоб, риск "
        f"просрочки, состав потока. Данные — историческая выгрузка по "
        f"{df['region'].nunique()} областям ({names}), у каждой свой период; в сумме "
        f"— с {df.created_at.min():%m.%Y} по {df.created_at.max():%m.%Y}.")


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
        st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            "Сводка по регионам считается по всем классам обращения (`problem`, "
            "`info`, `system`), темы и события — только по `problem`. Текст "
            "ограничений берётся из CLAUDE.md, раздел 5d, по якорным фразам "
            "(`LIMIT_ANCHORS` в `src/export.py`), а не пишется здесь заново; "
            "править формулировку — только в самом 5d. Excel и PDF строятся из одной "
            "функции "
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
        # PDF зависит от reportlab, системного шрифта с кириллицей и браузера
        # для картинок графиков. Все три причины внешние по отношению к данным,
        # поэтому объясняем их, а не показываем трейсбек: Excel от них не зависит.
        skipped = []
        try:
            st.session_state["pdf"] = build_pdf(scope, events, sel_reg, period,
                                                figs, descr, skipped=skipped)
        except PdfUnavailable as e:
            st.session_state.pop("pdf", None)
            st.error(f"PDF не собран: {e}. Excel при этом доступен.")
        if skipped:
            st.warning(f"PDF собран, но без графиков ({len(skipped)} из {len(figs)}): "
                       f"{skipped[0][1]}. Таблицы и оговорки в нём есть; в самом "
                       "PDF на месте графика стоит та же причина.")
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
REGION_GEN = {"Восточно-Казахстанская область": "ВКО"}
REGION_SHORT = {"Туркестанская область": "Туркестан", "Костанайская область": "Костанай",
                "Акмолинская область": "Акмола", "Алматинская область": "Алматы",
                "Павлодарская область": "Павлодар", "Карагандинская область": "Караганда",
                "Восточно-Казахстанская область": "ВКО"}


def flow_note(df):
    """Пояснение к структуре потока: крайние доли не-жалоб считаются из данных.

    До 2026-09-27 числа были зашиты в текст — доли настоящей выгрузки печатались и на
    поддельной (5p), и печатались бы на любой новой выгрузке."""
    share = (df.appeal_class != "problem").groupby(df.region).mean() * 100
    lo, hi = share.idxmin(), share.idxmax()
    gen = lambda r: REGION_GEN.get(r, r.replace("ская область", "ской области"))
    short = lambda r: REGION_SHORT.get(r, r)
    note = (f"**Что это значит.** Доля записей, которые не являются жалобами, — от "
            f"{share[lo]:.1f}% в {gen(lo)} до {share[hi]:.1f}% в {gen(hi)}. Это разница в том, "
            "что регион записывает как обращение, а не в нагрузке: Костанай и Туркестан заводят "
            "карточку только на обращение по проблеме, ВКО и Павлодар — на любой звонок. "
            "Поэтому сравнивать регионы по общему числу обращений нельзя — только по "
            "жалобам на городские проблемы.")
    ratio = (f" — разброс в {share[hi] / share[lo]:.0f} раз (по округлённым "
             f"{share[lo]:.1f} и {share[hi]:.1f} выходит "
             f"{round(share[hi], 1) / round(share[lo], 1):.0f}, так не считать)"
             if share[lo] > 0 and round(share[lo], 1) > 0 else "")
    how = ("Доля «не-problem» = (`info` + `system`) / все строки региона. Точные "
           f"крайние значения: {short(lo)} {share[lo]:.4f}%, {short(hi)} {share[hi]:.4f}%{ratio}. "
           "Разбор по регионам — раздел 5d CLAUDE.md.")
    return note, how


def other_share(df):
    """Доля «прочего» внутри жалоб по всей таблице, % — считается, а не пишется в текст."""
    p = df[df.appeal_class == "problem"]
    return float((p.topic == "прочее").mean() * 100) if len(p) else 0.0


GEN = {"Туркестанская область": "Туркестанской области"}
SHORT_GEN = {"Туркестанская область": "Туркестана"}


def _days_word(n):
    if n % 10 == 1 and n % 100 != 11:
        return "день"
    return "дня" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "дней"


def gap_notes(df):
    """Пропуски выгрузки (7+ дней подряд без обращений по региону целиком — как у
    детектора, 5f) для пояснений к динамике. До 2026-09-27 провал Туркестана был
    зашит в текст."""
    from src.spikes import GAP_MIN
    short, long = [], []
    for reg, g in df.groupby("region"):
        d = g.created_at.dt.normalize()
        rng = pd.date_range(d.min(), d.max())
        miss = rng.difference(d.unique())
        if not len(miss):
            continue
        runs = (pd.Series(miss).diff() != pd.Timedelta(days=1)).cumsum()
        for _, r in pd.Series(miss).groupby(runs.values):
            if len(r) < GAP_MIN:
                continue
            a, b = r.iloc[0], r.iloc[-1]
            half = "первой" if b.month <= 6 else "второй"
            name = GEN.get(reg, reg.replace("ская область", "ской области"))
            sname = SHORT_GEN.get(reg, name)
            short.append(f" Провал у {name} в {half} половине {b.year} года — за эти месяцы "
                         "в выгрузке нет данных, а не нет жалоб.")
            long.append(f"У {sname} в выгрузке нет обращений с {a:%Y-%m-%d} по {b:%Y-%m-%d} — "
                        f"{len(r)} {_days_word(len(r))}, {100 * len(r) / len(rng):.1f}% окна региона (раздел 10, "
                        "пункт 7). ")
    return "".join(short), "".join(long)


def data_revision():
    stat = DATA.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def main():
    import streamlit as st

    st.set_page_config(page_title="Обращения в 109 — обзор для руководителя",
                       layout="wide", initial_sidebar_state="collapsed")
    presentation = st.session_state.get("presentation", False)
    apply_theme(st, presentation)
    if paths.RUNTIME:
        if not paths.RELEASE_ERROR:
            from src.release_store import verify_release
            try:
                verify_release(paths.RUNTIME, paths.RELEASE_ID, paths.SOURCE)
            except (ValueError, OSError, KeyError, TypeError):
                st.error('BLOCKED — закреплённый release повреждён; выполните nazar-health и проверенный rollback.')
                st.stop()
        if paths.RELEASE_ERROR:
            st.error(paths.RELEASE_ERROR)
            st.stop()
        st.caption(f"Закреплённый release: {paths.RELEASE_ID}. После успешного refresh/rollback "
                   "перезапустите процесс витрины для перехода на CURRENT.")
    try:
        is_fake = paths.data_is_fake() if DATA.exists() else paths.FAKE
    except paths.SourceError as exc:
        st.error(str(exc))
        st.stop()
    if is_fake:
        fake_banner(st)

    if not DATA.exists():
        st.error("Нет данных для витрины. Подготовьте проверенную выгрузку.")
        with st.expander("Для администратора"):
            st.code(".venv/bin/nazar-build-data", language="bash")
            st.caption("Источник должен быть выбран явно. Порядок сборки — в разделе 0 CLAUDE.md.")
        st.stop()

    try:
        df = st.cache_data(load_data)(revision=data_revision())
        if df.empty or df.created_at.isna().any():
            raise ValueError("invalid canonical dates")
    except (OSError, ValueError, KeyError):
        st.error("Данные витрины повреждены или несовместимы со схемой. "
                 "Повторите проверенную сборку nazar-build-data; аналитика не показана.")
        st.stop()

    # A mismatched release must not feed the analytical sections.
    from src.health_view import health_section
    from src.data_health import load as load_health
    _, blocked = load_health()
    if blocked:
        health_section(st)
        st.stop()

    from src.operations import snapshot, revision
    from src.operations_view import action_section, planning_section, closure_section, brief_section
    current_revision = revision()
    if st.session_state.get('brief_revision') != current_revision:
        st.session_state.pop('brief_html', None)
        st.session_state.pop('brief_pdf', None)
        st.session_state['brief_revision'] = current_revision
    current = st.cache_data(snapshot)(revision_key=current_revision)
    if not current["available"]:
        st.error("Операционная аналитика недоступна; проверьте согласованность сборки.")
        st.stop()
    masthead(st, df, current["source"])
    _, controls = st.columns([4, 1])
    controls.toggle("Режим презентации", key="presentation", help="Меняет только оформление, не данные и фильтры.")
    with st.sidebar:
        st.subheader("О данных")
        st.write("Источник: " + current['source'].upper())
        st.caption("Очередь внимания охватывает все регионы. Фильтры событий и аналитики независимы.")
        st.caption("Свежесть определяется концом каждой выгрузки. Обновление на сегодня не предполагается.")
    operational, planning, quality, analytics, reports = st.tabs([
        "Оперативно", "Планирование", "Контроль", "Аналитика", "Отчёты"])
    with operational:
        executive_summary(st, current)
        action_section(st, current)
        from src.ui.health import health_overview
        health_overview(st, current['health'], compact=True)
        events, ev_descr, info = events_section(st, df)
        risk_section(st)

    with planning:
        planning_section(st, current)
    with quality:
        closure_section(st, current["closure"])
        health_section(st)

    with analytics:
        # ---------------- 4. структура потока
        st.subheader("Из чего состоит поток обращений по регионам")
        st.plotly_chart(chart(fig_structure(df)), width="stretch")
        note, how = flow_note(df)
        st.markdown(note)
        with st.expander("Как это считается", expanded=False):
            st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
            st.markdown(how)
        st.divider()

        # ---------------- 5. темы; фильтры здесь же — они действуют на темы,
        # динамику и выгрузку
        problem = df[df.appeal_class == "problem"]
        title = st.empty()                 # заголовок зависит от выбранного региона
        st.markdown("**Фильтры** — действуют на темы, динамику по месяцам и выгрузку "
                    "отчётов.")
        regions = sorted(problem["region"].unique())
        topics = sorted(problem["topic"].unique())
        with st.expander("Фильтры аналитики и отчётов"):
            def reset_analytics():
                for key in ('an_region', 'an_period', 'an_topic'):
                    st.session_state.pop(key, None)
            st.button('Сбросить фильтры аналитики', on_click=reset_analytics)
            f = st.columns([2, 2, 2])
            sel_reg = f[0].multiselect("Регион", regions, default=regions, key="an_region")
            # Границы берутся по ВСЕЙ таблице, а не по problem: этот же период уходит в
            # выгрузку, где сводка по регионам считается по всем классам. При границе по
            # problem три справочных обращения Павлодара за 2020-02-09 выпадали из сводки,
            # и она расходилась с эталоном на 3 строки.
            dmin, dmax = df.created_at.min().date(), df.created_at.max().date()
            sel_period = f[1].date_input("Период", (dmin, dmax),
                                         min_value=dmin, max_value=dmax, key="an_period")
            sel_topic = f[2].multiselect("Тема", topics, default=topics, key="an_topic")
        st.caption(f"Применено: регионов {len(sel_reg)} · тем {len(sel_topic)} · "
                   + (" — ".join(f"{d:%d.%m.%Y}" for d in sel_period) if isinstance(sel_period,(tuple,list)) else str(sel_period)))
        title.subheader("О чём жалуются" + (f" — {sel_reg[0]}" if len(sel_reg) == 1 else ""))

        # Пока в календаре выбрана только начальная дата, date_input отдаёт одну
        # дату. Раньше a и b тогда не определялись, и выгрузка падала с
        # UnboundLocalError (найдено 2026-09-25) — до выбора второй даты берём весь период.
        a, b = pd.Timestamp(dmin), pd.Timestamp(dmax) + pd.Timedelta(days=1)
        flt = problem[problem.region.isin(sel_reg) & problem.topic.isin(sel_topic)]
        if isinstance(sel_period, (tuple, list)) and len(sel_period) == 2:
            a, b = (pd.Timestamp(sel_period[0]),
                    pd.Timestamp(sel_period[1]) + pd.Timedelta(days=1))
            flt = flt[(flt.created_at >= a) & (flt.created_at < b)]

        st.caption(f"Жалоб на городские проблемы под фильтром: {len(flt):,}. "
                   .replace(",", " ")
                   + "Справочные звонки и служебные записи сюда не входят.")
        if flt.empty:
            st.warning("Под выбранные фильтры не попало ни одного обращения.")
        else:
            st.plotly_chart(chart(fig_topics(flt)), width="stretch")
        st.markdown(
            "**Что это значит.** Какие темы дают больше всего жалоб при выбранных "
            "фильтрах. «Прочее» — жалобы, которым не нашлось места в общем списке из 14 "
            "тем. У ВКО в 2023 году менялся порядок учёта тем, поэтому её темы до и после "
            "2023 года между собой не сравнивать.")
        with st.expander("Как это считается", expanded=False):
            st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
            st.markdown(
                "Тема (`topic`) присваивается по значению справочника региона правилами "
                "`src/topic_mapping.py` (раздел 5e CLAUDE.md); только класс `problem`. "
                f"«Прочее» внутри `problem` по всей таблице — {other_share(df):.2f}%. Смена состава потока "
                "ВКО по кварталам и темам — раздел 5g и раздел 10, пункт 8.")
        st.divider()

        # ---------------- 6. динамика — свёрнута: семь линий сразу не читаются
        st.subheader("Жалобы по месяцам")
        st.caption("График свёрнут: в нём по линии на регион, и все регионы сразу читаются "
                   "плохо. Удобнее выбрать один регион в фильтрах выше.")
        with st.expander("Показать график по месяцам", expanded=False):
            if flt.empty:
                st.info("Под выбранные фильтры не попало ни одного обращения.")
            else:
                st.plotly_chart(chart(fig_dynamics(flt)), width="stretch")
            st.markdown(
                "**Что это значит.** Каждая линия — один регион. Сравнивать высоту линий "
                "разных регионов нельзя: регионы разного размера и по-разному ведут учёт. "
                "Смотреть стоит на изменения внутри одной линии. Первый и последний месяц "
                "региона могут быть неполными." + gap_notes(df)[0])
        with st.expander("Как это считается", expanded=False):
            st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
            st.markdown(
                "Счётчик строк класса `problem` по календарным месяцам (`created_at`). "
                + gap_notes(df)[1] + "Месяцы без строк "
                "на графике не рисуются, и линия соединяет соседние точки.")
        st.divider()

        # ---------------- 7. общие цифры — от фильтров не зависят
        summary_section(st, df)
        st.divider()

    with reports:
        st.caption(f"Источник: {current['source'].upper()} · Последняя дата данных: {df.created_at.max():%d.%m.%Y}. У регионов разные окна.")
        brief_section(st, current)
        # ---------------- 8. выгрузка
        if flt.empty:
            st.subheader("Выгрузка отчётов")
            st.info("Выгрузка недоступна: под выбранные фильтры не попало ни одного "
                    "обращения.")
        else:
            export_section(st, df, flt, events, sel_reg, sel_topic, topics, a, b, ev_descr)

def summary_section(st, df):
    """Шесть общих чисел — бывшая «Сводка», теперь ниже, под «Общие цифры»."""
    st.subheader("Общие цифры")
    from src.export import summary_counts
    vc = summary_counts(df)['class_counts']
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
        f"темы выше считаются только по жалобам. Период у регионов разный: Павлодар — "
        f"с 2020 года, Акмола — только с июля 2025.")
    with st.expander("Как это считается", expanded=False):
        st.markdown('<span class="nazar-help-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            "Класс обращения — колонка `appeal_class`: `problem` (жалобы на "
            "городские проблемы), `info` (справочные), `system` (служебные). "
            "Присваивается по теме из справочника региона правилами "
            "`src/topic_mapping.py`, раздел 5e CLAUDE.md. Регионы записывают разное: "
            "одни заводят карточку только на проблему, другие — на любой звонок "
            "(раздел 5d). Окна данных по регионам — раздел 5c. Эти числа не зависят "
            "от фильтров страницы.")


if __name__ == "__main__":
    main()
