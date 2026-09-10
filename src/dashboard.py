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

DATA = Path("data/unified.parquet")
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

    # ---------------- блок 2: структура потока
    st.subheader("Структура потока по регионам")
    st.plotly_chart(fig_structure(df), use_container_width=True)
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
    dmin, dmax = problem.created_at.min().date(), problem.created_at.max().date()
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
    st.plotly_chart(fig_dynamics(flt), use_container_width=True)

    st.divider()

    # ---------------- блок 4: структура тем
    title = "Структура тем"
    if len(sel_reg) == 1:
        title += f" — {sel_reg[0]}"
    st.subheader(title)
    st.plotly_chart(fig_topics(flt), use_container_width=True)


if __name__ == "__main__":
    main()
