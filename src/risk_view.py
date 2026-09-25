#!/usr/bin/env python3
"""Блок витрины «Риск просрочки» (раздел 4b CLAUDE.md).

Ничего не пересчитывает: берёт готовые reports/predictions.csv (тест модели
train.py, построчно, в git не коммитится) и reports/metrics.json. Здесь только
сортировка, отбор верхней доли потока и доли по отобранному.

Модель обучена и проверена ТОЛЬКО на Карагандинской области. Поэтому блок
показывает только её и говорит об этом прямо.

Функции без streamlit (load_risk, top_share, headline) можно вызвать отдельно
и сверить с разделом 4b: precision в топ-10% обязана совпасть с 0.7628.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.topic_mapping import map_topic

PRED = Path("reports/predictions.csv")
METRICS = Path("reports/metrics.json")
SHARES = (0.10, 0.20, 0.30)


def load_risk(path=PRED):
    """Тест модели, отсортированный по предсказанному риску, сверху вниз."""
    p = pd.read_csv(path)
    out = pd.DataFrame({
        "дата": pd.to_datetime(p["created_date"]).dt.strftime("%d.%m.%Y %H:%M"),
        # В Караганде region — город, а не область (раздел 6).
        "город": p["region"],
        # Тема Караганды лежит в sub_category (раздел 5c) — тот же слой тем,
        # что во всей витрине.
        "тема": p["sub_category"].map(lambda v: map_topic(v) if pd.notna(v) else "прочее"),
        # Служба: executor_gov_org из модели убран (4b), а category в Караганде —
        # это и есть организация-исполнитель, не тема (раздел 3).
        "служба": p["category"].fillna("—"),
        "риск, %": (p["y_prob"] * 100).round(1),
        "_y": p["y_true"].astype(int),
        "_p": p["y_prob"],
    })
    return out.sort_values("_p", ascending=False, kind="mergesort").reset_index(drop=True)


def top_share(risk, share):
    """Верхняя доля потока по риску: сколько отобрано, сколько из них просрочено
    на самом деле и сколько было бы при случайном отборе того же размера."""
    n = int(round(share * len(risk)))
    top = risk.head(n)
    base = float(risk["_y"].mean())
    hit = int(top["_y"].sum())
    return {"n": n, "overdue": hit, "precision": hit / n if n else 0.0,
            "random_overdue": base * n, "base": base, "top": top}


def headline(path=METRICS):
    """ROC-AUC и PR-AUC основной модели, база и справочник — из metrics.json."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    main = m["models"][m["main_model"]]["at_tuned"]
    base = m["models"]["dummy"]["at_0.5"]["pr_auc"]          # = доля класса 1 в тесте
    sub = m["models"]["baseline_subcat"]["at_0.5"]["pr_auc"]
    return {"model": m["main_model"], "roc_auc": main["roc_auc"], "pr_auc": main["pr_auc"],
            "base": base, "subcat_pr_auc": sub,
            # Та же доля, что в 4b: (справочник − база) / (модель − база).
            "subcat_closes": (sub - base) / (main["pr_auc"] - base),
            "sla_days": m["data"]["sla_days"], "sla_source": m["data"]["sla_source"],
            "test_n": m["data"]["test_n"], "cutoff": m["data"]["cutoff"]}


def risk_section(st):
    st.subheader("Риск просрочки — только Карагандинская область")
    if not (PRED.exists() and METRICS.exists()):
        st.info(f"Нет {PRED} или {METRICS}. Их создаёт `.venv/bin/python train.py` "
                "(раздел 0 CLAUDE.md); predictions.csv в git не хранится.")
        return
    h = headline()
    risk = load_risk()
    st.warning(
        f"**Модель обучена и проверена только на Карагандинской области.** Ниже — её "
        f"тестовая выборка: {h['test_n']:,} обращений, поступивших с {h['cutoff']}. "
        "К другим регионам модель не применяется.".replace(",", " "))

    share = st.radio("Доля потока сверху по риску", SHARES, horizontal=True,
                     format_func=lambda s: f"{s:.0%}")
    sel = top_share(risk, share)
    top10 = top_share(risk, 0.10)

    left, right = st.columns([3, 1])
    left.dataframe(sel["top"][["дата", "город", "тема", "служба", "риск, %"]],
                   hide_index=True, width="stretch", height=420)
    left.markdown(
        f"В выборке **{sel['n']:,}** обращений; действительно просрочены "
        f"**{sel['overdue']:,} — {sel['precision']:.1%}**. При случайном отборе "
        f"того же размера было бы около {sel['random_overdue']:,.0f} "
        f"({sel['base']:.1%}).".replace(",", " "))
    right.metric("ROC-AUC", f"{h['roc_auc']:.4f}")
    right.metric("PR-AUC", f"{h['pr_auc']:.4f}", f"база {h['base']:.4f}",
                 delta_color="off")
    right.metric("Precision в топ-10%", f"{top10['precision']:.4f}",
                 f"база {h['base']:.4f}, в {top10['precision'] / h['base']:.2f} раза выше",
                 delta_color="off")

    st.caption(
        f"**Порог просрочки {h['sla_days']} суток — допущение команды, не норматив** "
        f"(источник: {h['sla_source']}); фактические нормативы в Костанае и Туркестане — "
        "1–7 суток (раздел 3 CLAUDE.md).  \n"
        "**Модель обучена только на Караганде** — к другим регионам не переносится.  \n"
        f"**Справочник по подкатегориям закрывает большую часть результата:** "
        f"{h['subcat_closes']:.0%} прироста PR-AUC над базой ({h['base']:.4f} → "
        f"{h['subcat_pr_auc']:.4f}) даёт таблица долей просрочки по подкатегориям без "
        f"обучения; модель добавляет сверху до {h['pr_auc']:.4f}.  \n"
        "**«Риск, %» — порядок, а не вероятность:** модель систематически переоценивает "
        "риск в середине и вверху шкалы (калибровка, раздел 4b), поэтому число годится "
        "для очерёдности проверки, но не для утверждения «просрочится с вероятностью N%».")
