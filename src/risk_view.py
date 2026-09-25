#!/usr/bin/env python3
"""Блок витрины «Риск просрочки» (раздел 4b CLAUDE.md).

Ничего не пересчитывает: берёт готовые reports/predictions.csv (тест модели
train.py, построчно, в git не коммитится) и reports/metrics.json. Здесь только
сортировка, отбор верхней доли потока и доли по отобранному.

Модель обучена и проверена ТОЛЬКО на Карагандинской области. Поэтому блок
показывает только её и говорит об этом прямо.

Функции без streamlit (load_risk, top_share, headline) можно вызвать отдельно
и сверить с разделом 4b: precision в топ-10% обязана совпасть с 0.7628.

Подписи на экране — для руководителя: ROC-AUC, PR-AUC и калибровка убраны в
блок «Как это считается», «риск, %» показан как оценка очерёдности.
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
        # y_prob × 100. Не «%»: вероятность не откалибрована (4b) и годится
        # только как очерёдность, поэтому подпись не обещает процент.
        "оценка риска (из 100)": (p["y_prob"] * 100).round(1),
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
            "test_n": m["data"]["test_n"], "cutoff": m["data"]["cutoff"],
            "train_share": m["data"]["train_share_y1"],
            "test_share": m["data"]["test_share_y1"],
            # отмечено моделью на её пороге с train (4b): TP + FP, не пересчёт
            "threshold": main["threshold"], "flagged": main["TP"] + main["FP"],
            "flag_precision": main["precision_1"]}


def _n(x, fmt=","):
    """Число с пробелом между тысячами. Только само число: replace по всей строке
    съедал бы запятые в тексте."""
    return f"{x:{fmt}}".replace(",", " ")


def risk_section(st):
    st.subheader("Риск просрочки — только Карагандинская область", anchor="risk")
    if not (PRED.exists() and METRICS.exists()):
        st.info(f"Нет {PRED} или {METRICS}. Их создаёт `.venv/bin/python train.py` "
                "(раздел 0 CLAUDE.md); predictions.csv в git не хранится.")
        return
    h = headline()
    risk = load_risk()
    cutoff = pd.Timestamp(h["cutoff"])
    st.warning(
        f"**Прогноз построен и проверен только на Карагандинской области** и к другим "
        f"регионам не применяется. Ниже — {_n(h['test_n'])} обращений, поступивших с "
        f"{cutoff:%d.%m.%Y}. При обучении модель их не видела, а чем они закончились, "
        "уже известно, — поэтому точность прогноза можно проверить.")

    share = st.radio("Сколько обращений проверять первыми — самые рискованные",
                     SHARES, horizontal=True, format_func=lambda s: f"{s:.0%} потока")
    sel = top_share(risk, share)
    top10 = top_share(risk, 0.10)

    left, right = st.columns([3, 1])
    left.dataframe(sel["top"][["дата", "город", "тема", "служба",
                               "оценка риска (из 100)"]],
                   hide_index=True, width="stretch", height=420)
    left.markdown(
        f"Самых рискованных обращений в списке: **{_n(sel['n'])}**. Из них "
        f"действительно просрочены **{_n(sel['overdue'])} — {sel['precision']:.1%}**. "
        f"При выборе наугад "
        f"того же числа просроченных было бы около {_n(sel['random_overdue'], ',.0f')} "
        f"({sel['base']:.1%}).  \n**Что это значит:** проверяя по этому списку, а не "
        f"наугад, на просрочку попадают в {sel['precision'] / sel['base']:.2f} раза "
        f"чаще.")
    right.metric("Из 10% самых рискованных просрочены", f"{top10['precision']:.1%}",
                 f"наугад — {h['base']:.1%}", delta_color="off")
    right.caption(f"В {top10['precision'] / h['base']:.2f} раза чаще, чем при выборе "
                  "наугад.")

    st.markdown(
        f"**«Просрочено» здесь — обрабатывалось дольше {h['sla_days']} суток.** Это "
        "допущение команды, а не утверждённый норматив: в Костанае и Туркестане "
        "фактические сроки — 1–7 суток. Даты закрытия в данных Караганды нет, срок "
        "считается до последнего изменения записи.  \n"
        "**Большую часть результата даёт простая таблица без модели** — как часто "
        "просрочиваются обращения каждой подкатегории. Она обеспечивает около "
        f"{h['subcat_closes']:.0%} выигрыша над выбором наугад; модель добавляет "
        "остальное.  \n"
        "**Оценка риска — очерёдность, а не вероятность.** Она годится, чтобы решить, "
        "что проверять первым, но не для фразы «просрочится с вероятностью N%»: в "
        "средней и верхней части шкалы модель риск завышает.")

    with st.expander("Как это считается", expanded=False):
        st.markdown(
            f"Основная модель — `{h['model']}` (логистическая регрессия), раздел 4b "
            f"CLAUDE.md. Тест — `created_date >= {h['cutoff']}`, {_n(h['test_n'])} "
            f"строк; таргет `updated_date − created_date > {h['sla_days']}` суток "
            f"(источник порога: {h['sla_source']}). `updated_date` — дата последнего "
            "изменения записи, не закрытия (раздел 3).\n\n"
            f"- ROC-AUC {h['roc_auc']:.4f}; PR-AUC {h['pr_auc']:.4f} при базе "
            f"{h['base']:.4f} — доле просроченных в тесте.\n"
            f"- Precision в топ-10%: {top10['precision']:.4f} — в "
            f"{top10['precision'] / h['base']:.2f} раза выше базы.\n"
            f"- Справочник по `sub_category` без обучения: PR-AUC "
            f"{h['subcat_pr_auc']:.4f}. Он закрывает {h['subcat_closes']:.1%} "
            "прироста PR-AUC модели над базой: (справочник − база) / (модель − база).\n"
            "- «Оценка риска» = `y_prob × 100`. Калибровка: систематическая "
            "переоценка риска в середине и вверху шкалы — следствие сдвига доли "
            f"просрочек между train ({h['train_share']:.1%}) и test "
            f"({h['test_share']:.1%}). Подгонкой под тест не правится.")
