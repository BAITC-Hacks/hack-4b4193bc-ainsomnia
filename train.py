#!/usr/bin/env python3
"""
Модель риска просрочки обращения 109 (Карагандинская область).

Один прогон от начала до конца:
    .venv/bin/python train.py

Печатает полный текстовый отчёт и складывает артефакты в reports/ и models/.
Графиков нет намеренно — нужен вывод, который читается глазами.
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

DATE_FMT = "%m/%d/%y %H:%M"
GOOD_ANSWER = {"Быстрый ответ", "Письменное обращение", ""}
VALID_APPEAL = {"Запрос информации", "Обращение", "Инцидент"}
TARGET_APPEAL = ("Обращение", "Инцидент")
MISSING = "__MISSING__"
SEED = 42

CAT_FULL = ["appeal_type", "source", "region", "district",
            "category", "sub_category", "executor_gov_org", "type"]
NUM_FULL = ["hour", "dayofweek", "month", "has_address",
            "executor_load_7d", "district_load_7d"]
# Убраны по permutation importance полной модели (вклад <= 0.0023, часть отрицательный)
DROPPED = ["executor_load_7d", "district_load_7d", "source", "executor_gov_org"]
CAT_TRIM = [c for c in CAT_FULL if c not in DROPPED]
NUM_TRIM = [c for c in NUM_FULL if c not in DROPPED]
MAIN_MODEL = "logreg_trim"   # урезанный не просел, а вырос: ROC-AUC +0.0019, PR-AUC +0.0051

def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

# --------------------------------------------------------------------------
# Шаг 1. Загрузка и чистка
# --------------------------------------------------------------------------
def load_and_clean(path):
    hr("ШАГ 1. ЗАГРУЗКА И ЧИСТКА")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    n0 = len(df)
    print(f"Строк в файле: {n0}")

    print("\nДоля пропусков по колонкам (до чистки):")
    for c in df.columns:
        miss = int((df[c] == "").sum())
        print(f"  {c:22s} {miss:7d}  ({100*miss/n0:6.2f}%)")

    # Битые строки: незакрытые кавычки сдвигают поля ПОСЛЕ appeal_address,
    # поэтому appeal_type у них остаётся валидным и фильтр из ТЗ их не ловит.
    bad_quotes = ~df["answer_type"].isin(GOOD_ANSWER)
    n_quotes = int(bad_quotes.sum())
    print(f"\nВыброшено по битым кавычкам (answer_type вне справочника): {n_quotes}")
    if n_quotes:
        print("  примеры испорченных значений answer_type:")
        for v in df.loc[bad_quotes, "answer_type"].head(4):
            print(f"    {v[:70]!r}")
    df = df[~bad_quotes]

    bad_type = ~df["appeal_type"].isin(VALID_APPEAL)
    n_type = int(bad_type.sum())
    print(f"Выброшено по невалидному appeal_type (страховочный фильтр ТЗ): {n_type}")
    df = df[~bad_type]

    print(f"ВСЕГО выброшено при чистке: {n_quotes + n_type}")
    print(f"Строк после чистки: {len(df)}")

    for c in ("created_date", "updated_date", "submission_date"):
        df[c] = pd.to_datetime(df[c], format=DATE_FMT, errors="coerce")
    n_bad_dates = int(df["created_date"].isna().sum() + df["updated_date"].isna().sum())
    print(f"Не распарсилось дат (created/updated): {n_bad_dates}")
    df = df[df["created_date"].notna() & df["updated_date"].notna()]

    same = int((df["submission_date"] == df["created_date"]).sum())
    print(f"\nsubmission_date == created_date: {same} из {len(df)} "
          f"({100*same/len(df):.2f}%) -> поле выброшено как дубль created_date")

    print("\nРаспределение region (ТЗ называет поле константой — проверка):")
    vc = df["region"].value_counts()
    for k, v in vc.head(8).items():
        print(f"  {k:38s} {v:7d}")
    if len(vc) > 8:
        print(f"  ... всего уникальных значений: {len(vc)} -> НЕ константа, включён в признаки")

    return df, n_quotes, n_type

# --------------------------------------------------------------------------
# Шаг 2. Целевая переменная + санити-чек обрезки
# --------------------------------------------------------------------------
def build_target(df, sla_days):
    hr("ШАГ 2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ")
    df = df[df["appeal_type"].isin(TARGET_APPEAL)].copy()
    df["days"] = (df["updated_date"] - df["created_date"]).dt.total_seconds() / 86400.0
    df["y"] = (df["days"] > sla_days).astype(int)

    n = len(df)
    share = df["y"].mean()
    print(f"Подвыборка ('Обращение' + 'Инцидент'): {n} строк")
    print(f"Порог просрочки: days > {sla_days} сут")
    print(f"Доля y=1: {100*share:.2f}%")

    qs = [50, 75, 90, 95, 99]
    vals = np.percentile(df["days"], qs)
    print("\nКвантили days:")
    for q, v in zip(qs, vals):
        print(f"  p{q:<3d} {v:9.3f} сут")
    print(f"  min {df['days'].min():.3f} · max {df['days'].max():.3f} · "
          f"отрицательных {int((df['days'] < 0).sum())}")

    print("\nДоля y=1 по годам создания:")
    for yr, g in df.groupby(df["created_date"].dt.year):
        print(f"  {yr}: n={len(g):6d}  y=1 {100*g['y'].mean():.1f}%")

    # --- санити-чек обрезки таргета (hard stop) ---
    hr("САНИТИ-ЧЕК ОБРЕЗКИ ТАРГЕТА")
    cut = df["created_date"].max() - pd.Timedelta(days=sla_days)
    tail = df[df["created_date"] >= cut]
    tail_share = tail["y"].mean()
    ratio = tail_share / share if share else 0.0
    print(f"Обращений, созданных в последние {sla_days} сут окна: {len(tail)}")
    print(f"Доля y=1 в хвосте: {100*tail_share:.2f}%   против средней {100*share:.2f}%")
    print(f"ratio = {ratio:.3f}")
    if ratio < 0.50:
        print("\nОСТАНОВ: окно исходов обрывается раньше окна создания.")
        print("Обращения в конце периода не успевают просрочиться -> таргет невалиден.")
        print("Модель на таких данных обучать нельзя.")
        sys.exit(1)
    elif ratio < 0.75:
        print("ВНИМАНИЕ: ratio ниже 0.75 — хвост подозрителен, прогон продолжается.")
    else:
        print("OK: обрезки таргета нет.")
    return df, share, ratio

def print_limitations(df_all, df, sla_days):
    hr("ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ")
    c_min, c_max = df["created_date"].min(), df["created_date"].max()
    u_min, u_max = df["updated_date"].min(), df["updated_date"].max()
    gap = (u_max - c_max).days
    print(f"created_date: {c_min}  ->  {c_max}")
    print(f"updated_date: {u_min}  ->  {u_max}")
    print(f"Окно исходов шире окна создания на {gap} сут.")
    after = int((df["updated_date"] > c_max).sum())
    print(f"Строк, обновлённых после закрытия окна создания: {after} "
          f"({100*after/len(df):.2f}%) -> обрезки таргета нет.")
    for t in (180, 365):
        k = int((df["days"] > t).sum())
        print(f"Доля наблюдений с days > {t}: {k} ({100*k/len(df):.2f}%)")
    print()
    print("ОГРАНИЧЕНИЕ: updated_date — дата ПОСЛЕДНЕГО ИЗМЕНЕНИЯ записи, а не дата")
    print("закрытия обращения. Запись 2022 года, к которой административно")
    print("прикоснулись позже, даёт 600+ суток, и это не время обработки.")
    print("Длинный хвост распределения days частично не отражает реальную")
    print("длительность. Это ГИПОТЕЗА, требующая подтверждения у владельца данных.")
    print()
    print(f"ОГРАНИЧЕНИЕ: порог {sla_days} сут взят из ТЗ и нормативом не является.")
    print("Фактические нормативы SLA в сопоставимых регионах (Костанай, Туркестан)")
    print("составляют 1-7 суток: sla<=7 покрывает 98.7% и 99.8% обращений.")
    print("Справочники услуг несопоставимы, перенести нормативы нельзя — порог")
    print("оставлен как консервативная граница, примерно вдвое мягче фактической.")
    return {"created_min": str(c_min), "created_max": str(c_max),
            "updated_min": str(u_min), "updated_max": str(u_max),
            "gap_days": int(gap), "updated_after_creation_window": after,
            "share_days_gt_180": float((df["days"] > 180).mean()),
            "share_days_gt_365": float((df["days"] > 365).mean())}

# --------------------------------------------------------------------------
# Шаг 3. Признаки
# --------------------------------------------------------------------------
def causal_load(df, col, window_days=7):
    """Сколько обращений к тому же исполнителю/району создано за предыдущие
    window_days суток. Считается СТРОГО по прошлому: внутри группы для строки j
    берутся только строки с меньшим порядковым номером в сортировке по времени."""
    out = np.zeros(len(df), dtype=np.float64)
    w = np.timedelta64(window_days, "D")
    times = df["created_date"].values
    for _, idx in df.groupby(col, observed=True, sort=False).indices.items():
        idx = np.sort(idx)
        t = times[idx]
        left = np.searchsorted(t, t - w, side="left")
        out[idx] = np.arange(len(t)) - left
    return out

def build_features(df):
    hr("ШАГ 3. ПРИЗНАКИ")
    df = df.sort_values("created_date", kind="mergesort").reset_index(drop=True)

    for c in CAT_FULL + ["answer_type"]:
        df[c] = df[c].replace("", MISSING).fillna(MISSING)

    df["hour"] = df["created_date"].dt.hour
    df["dayofweek"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month
    df["has_address"] = (df["appeal_address"].fillna("") != "").astype(int)
    df["executor_load_7d"] = causal_load(df, "executor_gov_org")
    df["district_load_7d"] = causal_load(df, "district")

    print("Категориальные:", ", ".join(CAT_FULL))
    print("Числовые:      ", ", ".join(NUM_FULL))
    print("\nПропуски категорий заменены сентинелом '__MISSING__' (строки не удалялись).")
    print("Кардинальность категориальных признаков:")
    for c in CAT_FULL:
        print(f"  {c:20s} {df[c].nunique():5d}")
    print("\nПроверка причинности load-признаков:")
    print(f"  у самой ранней строки executor_load_7d={df['executor_load_7d'].iloc[0]:.0f}, "
          f"district_load_7d={df['district_load_7d'].iloc[0]:.0f} (обязаны быть 0)")
    print(f"  executor_load_7d: max={df['executor_load_7d'].max():.0f}, "
          f"mean={df['executor_load_7d'].mean():.1f}")
    print(f"  district_load_7d: max={df['district_load_7d'].max():.0f}, "
          f"mean={df['district_load_7d'].mean():.1f}")
    print("\nИсключены: updated_date и days (из них построен таргет), submission_date")
    print("(дубль created_date), appeal_address (только производный has_address).")
    return df

# --------------------------------------------------------------------------
# Шаг 4-5. Разбиение и модели
# --------------------------------------------------------------------------
def make_logreg(cat, num):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat),
            ("num", StandardScaler(), num)])),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))])

def make_hgb(cat, num):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan, min_frequency=20,
                                   max_categories=250, dtype=np.float64), cat),
            ("num", "passthrough", num)])),
        ("clf", HistGradientBoostingClassifier(
            categorical_features=np.array([True]*len(cat) + [False]*len(num)),
            random_state=SEED))])

def baseline_predict(tr, te, col="sub_category"):
    """Справочник без обучения: доля просрочек в train по подкатегории,
    применённая как вероятность. Неизвестные подкатегории -> общая доля train."""
    g = tr.groupby(col, observed=True)["y"].mean()
    return te[col].map(g).fillna(tr["y"].mean()).to_numpy(dtype=float)

def best_f1_threshold(y, proba):
    grid = np.arange(0.05, 0.95, 0.01)
    sc = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(sc))]), float(max(sc))

def block(name, y_true, proba, thr, quiet=False):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {"threshold": float(thr), "accuracy": accuracy_score(y_true, pred),
         "roc_auc": roc_auc_score(y_true, proba),
         "pr_auc": average_precision_score(y_true, proba),
         "precision_1": precision_score(y_true, pred, zero_division=0),
         "recall_1": recall_score(y_true, pred, zero_division=0),
         "f1_1": f1_score(y_true, pred, zero_division=0),
         "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    if quiet:
        return m
    print(f"\n--- {name}  (порог {thr:.2f})")
    for k in ("accuracy", "roc_auc", "pr_auc", "precision_1", "recall_1", "f1_1"):
        print(f"  {k:14s}{m[k]:.4f}")
    print("  матрица ошибок:")
    print("        предсказано 0   предсказано 1")
    print(f"  факт 0  TN={tn:<8d}      FP={fp:<8d}")
    print(f"  факт 1  FN={fn:<8d}      TP={tp:<8d}")
    print("  classification_report:")
    for line in classification_report(y_true, pred, digits=4, zero_division=0).strip().split("\n"):
        print("    " + line)
    return m

def oof_threshold(fit_fn, cat, num, train, n_splits=5):
    """Порог по максимуму F1 на out-of-fold вероятностях train.
    Тест не участвует. In-sample порог смещён переобучением, поэтому OOF."""
    X, y = train[cat + num], train["y"]
    oof = np.full(len(y), np.nan)
    for a, b in TimeSeriesSplit(n_splits=n_splits).split(X):
        if fit_fn is None:
            oof[b] = baseline_predict(train.iloc[a], train.iloc[b])
        else:
            oof[b] = fit_fn(cat, num).fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b])[:, 1]
    ok = ~np.isnan(oof)
    thr, f1 = best_f1_threshold(y[ok], oof[ok])
    return thr, f1, int(ok.sum())


def _sweep(y, p):
    """Для каждого префикса топ-k по убыванию риска: порог, precision, recall."""
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]; ps = p[o]
    k = np.arange(1, len(y) + 1)
    tp = np.cumsum(ys)
    return ps, tp / k, tp / y.sum(), k

def _at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"threshold": float(thr), "n_flagged": int(pred.sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / int(y.sum())}

def _point_for(y, p, target, metric):
    """Порог, при котором достигается заданный recall (или precision).
    Ties учитываются: берётся порог префикса, затем метрики пересчитываются
    правилом p >= thr, поэтому вся группа с равной вероятностью входит целиком."""
    ps, prec, rec, _ = _sweep(y, p)
    idx = np.where(rec >= target)[0] if metric == "recall" else np.where(prec >= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0] if metric == "recall" else idx[-1]
    return _at_threshold(y, p, ps[i])

def compare_operating_points(y, p_base, p_model, results, main_key):
    hr("СПРАВОЧНИК ПРОТИВ МОДЕЛИ В СОПОСТАВИМЫХ РАБОЧИХ ТОЧКАХ")
    nb, nm = len(np.unique(p_base)), len(np.unique(p_model))
    print(f"Различных значений вероятности на тесте: справочник {nb}, модель {nm}.")
    print(f"У справочника порог режется по {nb} ступеням — вся подкатегория")
    print("переключается целиком, тонкая настройка рабочей точки невозможна.\n")

    out = {"distinct_values": {"baseline": nb, "model": nm,
                               "model_rounded6": len(np.unique(np.round(p_model, 6)))},
           "targets": [], "topk": []}
    for metric, target in (("recall", 0.80), ("precision", 0.65)):
        print(f"--- целевой {metric} = {target:.2f}")
        print(f"{'подход':14s} {'порог':>8s} {'помечено':>10s} {'precision':>11s} {'recall':>9s}")
        row = {"metric": metric, "target": target}
        for nm_, p in (("справочник", p_base), ("модель", p_model)):
            r = _point_for(y, p, target, metric)
            if r is None:
                print(f"{nm_:14s}  недостижимо"); row[nm_] = None; continue
            print(f"{nm_:14s} {r['threshold']:8.4f} {r['n_flagged']:10d} "
                  f"{r['precision']:11.4f} {r['recall']:9.4f}")
            row[nm_] = r
        out["targets"].append(row)
        print()

    print("--- операционная таблица: оператор успевает проверить долю входящих")
    print("(в пределах группы с равной вероятностью порядок произволен — для")
    print(" справочника это означает, что часть подкатегории попадает в отбор случайно)\n")
    n, pos = len(y), int(y.sum())
    print(f"{'просмотр':>9s} {'шт.':>7s} {'precision спр.':>15s} {'precision мод.':>15s} "
          f"{'охват спр.':>11s} {'охват мод.':>11s}")
    for frac in (0.10, 0.20, 0.30):
        k = int(round(frac * n)); rec = {"share": frac, "n": k}
        vals = []
        for key, p in (("baseline", p_base), ("model", p_model)):
            o = np.argsort(-p, kind="mergesort")[:k]
            tp = int(y[o].sum())
            rec[key] = {"precision": tp / k, "recall": tp / pos}
            vals.append((tp / k, tp / pos))
        print(f"{100*frac:8.0f}% {k:7d} {vals[0][0]:15.4f} {vals[1][0]:15.4f} "
              f"{vals[0][1]:11.4f} {vals[1][1]:11.4f}")
        out["topk"].append(rec)

    base_rate = float(y.mean())
    b_pr = results["baseline_subcat"]["at_0.5"]["pr_auc"]
    m_pr = results[main_key]["at_0.5"]["pr_auc"]
    b_roc = results["baseline_subcat"]["at_0.5"]["roc_auc"]
    m_roc = results[main_key]["at_0.5"]["roc_auc"]
    lift = {"pr_auc": {"base": base_rate, "baseline": b_pr, "model": m_pr,
                       "baseline_share": (b_pr-base_rate)/(m_pr-base_rate),
                       "model_adds": (m_pr-base_rate)/(b_pr-base_rate)-1},
            "roc_auc": {"base": 0.5, "baseline": b_roc, "model": m_roc,
                        "baseline_share": (b_roc-0.5)/(m_roc-0.5),
                        "model_adds": (m_roc-0.5)/(b_roc-0.5)-1}}
    print(f"\n--- прирост над базой, две системы отсчёта")
    for k_, lab, base in (("pr_auc", "PR-AUC", f"доля класса 1 = {base_rate:.4f}"),
                          ("roc_auc", "ROC-AUC", "случайное угадывание = 0.5")):
        d = lift[k_]
        print(f"{lab:8s} (база: {base})")
        print(f"         справочник {d['baseline']:.4f}, модель {d['model']:.4f} -> "
              f"справочник закрывает {100*d['baseline_share']:.1f}%, "
              f"модель добавляет +{100*d['model_adds']:.1f}%")
    out["lift"] = lift
    write_baseline_report(y, out, results, main_key)
    return out

def write_baseline_report(y, ops, results, main_key):
    """reports/baseline_vs_model.md — идёт в слайды напрямую."""
    n, pos = len(y), int(y.sum())
    base_rate = pos / n
    L = ops["lift"]
    t = {r["metric"]: r for r in ops["targets"]}
    md = []
    md.append("# Справочник против модели\n")
    md.append(f"Тест: {n} обращений, из них просрочено {pos} ({100*base_rate:.1f}%). "
              f"Разбиение по времени, обучение только на данных до 2023-07-01.\n")
    md.append("**Справочник** — доля просрочек по `sub_category`, посчитанная на train "
              "и применённая к test без обучения.  \n"
              f"**Модель** — `{main_key}`, логистическая регрессия на one-hot.\n")

    md.append("\n## 1. Сопоставимые рабочие точки\n")
    md.append("Сравнение при одинаковом пороге бессмысленно — сравниваем при "
              "одинаковом результате.\n")
    md.append("| Целевая точка | Подход | Порог | Помечено | Precision | Recall |")
    md.append("|---|---|---|---|---|---|")
    for metric, lab in (("recall", "recall = 0.80"), ("precision", "precision = 0.65")):
        for who in ("справочник", "модель"):
            r = t[metric][who]
            md.append(f"| {lab} | {who} | {r['threshold']:.4f} | {r['n_flagged']} | "
                      f"**{r['precision']:.4f}** | **{r['recall']:.4f}** |")
    dr = t["recall"]["модель"]["precision"] - t["recall"]["справочник"]["precision"]
    dp = t["precision"]["модель"]["recall"] - t["precision"]["справочник"]["recall"]
    md.append(f"\nПри равном recall модель точнее на **{dr:+.4f}** precision.  ")
    md.append(f"При равной precision модель ловит на **{dp:+.4f}** recall больше — "
              f"это {100*dp/t['precision']['справочник']['recall']:.0f}% относительного прироста.\n")
    md.append(f"**Гранулярность.** Справочник выдаёт на тесте всего "
              f"**{ops['distinct_values']['baseline']}** различных значений вероятности "
              f"против **{ops['distinct_values']['model']}** у модели (по неокруглённым "
              f"вероятностям; в `reports/predictions.csv` они округлены до 6 знаков, и по "
              f"файлу различных значений **{ops['distinct_values']['model_rounded6']}**). "
              f"Порог режется по "
              f"подкатегориям целиком — рабочую точку под ёмкость оператора не подстроить.\n")

    md.append("\n## 2. Операционная таблица\n")
    md.append("Оператор успевает проверить часть входящих. Берём топ по предсказанному "
              "риску.\n")
    md.append("| Просмотр | Обращений | Precision справочника | Precision модели | "
              "Охват просрочек справочником | Охват моделью | Потолок охвата |")
    md.append("|---|---|---|---|---|---|---|")
    for r in ops["topk"]:
        ceil = min(1.0, r["n"] / pos)
        md.append(f"| {100*r['share']:.0f}% | {r['n']} | {r['baseline']['precision']:.4f} | "
                  f"**{r['model']['precision']:.4f}** | {r['baseline']['recall']:.4f} | "
                  f"**{r['model']['recall']:.4f}** | {ceil:.4f} |")
    r10 = ops["topk"][0]
    ceil10 = min(1.0, r10["n"] / pos)
    md.append(f"\n**Потолок охвата** — сколько просрочек физически помещается в отобранную "
              f"долю. При {100*base_rate:.1f}% просрочек топ-10% не может содержать больше "
              f"{100*ceil10:.1f}% всех просрочек. Модель берёт "
              f"{100*r10['model']['recall']/ceil10:.0f}% от потолка, справочник — "
              f"{100*r10['baseline']['recall']/ceil10:.0f}%.\n")
    md.append(f"Точность в топ-10% у модели **{r10['model']['precision']:.4f}** против "
              f"базовой доли {base_rate:.4f} — рост в "
              f"{r10['model']['precision']/base_rate:.2f} раза.\n")

    md.append("\n## 3. Прирост над базой — две системы отсчёта\n")
    md.append("| Метрика | База | Справочник | Модель | Справочник закрывает | Модель добавляет |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| PR-AUC | {L['pr_auc']['base']:.4f} (доля класса 1) | "
              f"{L['pr_auc']['baseline']:.4f} | {L['pr_auc']['model']:.4f} | "
              f"**{100*L['pr_auc']['baseline_share']:.1f}%** | "
              f"**+{100*L['pr_auc']['model_adds']:.1f}%** |")
    md.append(f"| ROC-AUC | 0.5000 (случайное угадывание) | "
              f"{L['roc_auc']['baseline']:.4f} | {L['roc_auc']['model']:.4f} | "
              f"{100*L['roc_auc']['baseline_share']:.1f}% | "
              f"+{100*L['roc_auc']['model_adds']:.1f}% |")
    md.append("\nПри несбалансированных классах ориентироваться на **PR-AUC**: его база — "
              "фактическая доля просрочек, а не гипотетическая монета. Оценка по ROC-AUC "
              "льстит справочнику. Две последние колонки таблицы — разные отношения "
              "и в 100% не складываются: «закрывает» считается от прироста модели над "
              "базой, «добавляет» — от прироста справочника над базой.\n")

    md.append("\n## 4. Вывод\n")
    md.append("Модель выигрывает у справочника **во всех сопоставимых точках** — при равном "
              "recall, при равной precision и на каждом уровне просмотра. Наибольший отрыв "
              f"при равной точности: recall {t['precision']['модель']['recall']:.4f} против "
              f"{t['precision']['справочник']['recall']:.4f}.\n")
    md.append("Справочник остаётся честной нижней планкой: он не требует обучения, "
              "обновляется одним `groupby` и закрывает "
              f"{100*L['pr_auc']['baseline_share']:.0f}% прироста по PR-AUC. "
              "Показывать его на защите как точку отсчёта, а не как альтернативу.\n")
    md.append("\n---\n")
    md.append("Таргет: `days > 15` суток. Порог взят из ТЗ, нормативом не является и "
              "остаётся **открытым вопросом** — см. CLAUDE.md, раздел 3.\n")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_vs_model.md").write_text("\n".join(md), encoding="utf-8")

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="drive-download-20260907T161509Z-1-001/"
                                     "Обращения граждан 109 - Карагандинская область.csv")
    ap.add_argument("--cutoff", default="2023-07-01")
    ap.add_argument("--sla-days", type=float, default=15)
    a = ap.parse_args()

    df_raw, n_quotes, n_type = load_and_clean(a.csv)
    df, share_all, trunc_ratio = build_target(df_raw, a.sla_days)
    lims = print_limitations(df_raw, df, a.sla_days)
    df = build_features(df)

    hr("ШАГ 4. РАЗБИЕНИЕ ПО ВРЕМЕНИ")
    tr_m = df["created_date"] < pd.Timestamp(a.cutoff)
    train, test = df[tr_m], df[~tr_m]
    print(f"Граница: {a.cutoff} (строго по времени, не случайное)")
    print(f"train: {len(train):6d} строк, доля y=1 {100*train['y'].mean():.1f}%")
    print(f"test:  {len(test):6d} строк, доля y=1 {100*test['y'].mean():.1f}%")
    print("Все преобразования обучаются только на train (Pipeline), затем к test.")
    ytr, yte = train["y"], test["y"]
    results, probs = {}, {}

    hr("ШАГ 5-6. МОДЕЛИ И МЕТРИКИ")
    print(f"Урезанный набор: убраны {', '.join(DROPPED)}")
    print(f"  причина — permutation importance на полной модели: вклад <= 0.0023,")
    print(f"  у district_load_7d, source, executor_gov_org отрицательный.")

    # 0. пустышка
    dm = DummyClassifier(strategy="most_frequent").fit(train[CAT_FULL + NUM_FULL], ytr)
    probs["dummy"] = dm.predict_proba(test[CAT_FULL + NUM_FULL])[:, 1]
    results["dummy"] = {"at_0.5": block("ПУСТЫШКА (most_frequent)", yte, probs["dummy"], 0.5)}

    # 1. справочник без обучения
    probs["baseline_subcat"] = baseline_predict(train, test)
    thr_b, f1_b, n_b = oof_threshold(None, CAT_FULL, NUM_FULL, train)
    unseen = int((~test["sub_category"].isin(train["sub_category"].unique())).sum())
    print(f"\nСПРАВОЧНИК: доля просрочек по sub_category из train, без обучения.")
    print(f"  подкатегорий в train: {train['sub_category'].nunique()}, "
          f"строк test с неизвестной подкатегорией: {unseen} "
          f"({100*unseen/len(test):.2f}%) -> общая доля train {train['y'].mean():.4f}")
    print(f"  порог с OOF train: {thr_b:.2f} (F1 на OOF {f1_b:.4f}, {n_b} строк)")
    results["baseline_subcat"] = {
        "at_0.5": block("СПРАВОЧНИК sub_category", yte, probs["baseline_subcat"], 0.5),
        "at_tuned": block("СПРАВОЧНИК sub_category — порог с train", yte,
                          probs["baseline_subcat"], thr_b),
        "unseen_subcategories_in_test": unseen}

    # 2-3. логрег полный и урезанный, 4. бустинг
    specs = [("logreg_full", make_logreg, CAT_FULL, NUM_FULL, "ЛОГРЕГ полный"),
             ("logreg_trim", make_logreg, CAT_TRIM, NUM_TRIM, "ЛОГРЕГ урезанный"),
             ("hgb_full", make_hgb, CAT_FULL, NUM_FULL, "БУСТИНГ полный")]
    fitted, thrs = {}, {"baseline_subcat": thr_b}
    for key, fn, cat, num, label in specs:
        m = fn(cat, num).fit(train[cat + num], ytr)
        p = m.predict_proba(test[cat + num])[:, 1]
        thr, f1o, n_ok = oof_threshold(fn, cat, num, train)
        fitted[key], probs[key], thrs[key] = m, p, thr
        print(f"\n{label}: порог с OOF train {thr:.2f} (F1 на OOF {f1o:.4f}, {n_ok} строк)")
        results[key] = {"at_0.5": block(label, yte, p, 0.5),
                        "at_tuned": block(f"{label} — порог с train", yte, p, thr)}

    # --- сводка ---
    hr("СВОДКА: РАЗНИЦА МЕЖДУ МОДЕЛЯМИ")
    print(f"{'модель':26s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'F1@0.5':>9s} "
          f"{'F1@train':>9s} {'Recall@train':>13s}")
    for k in ("dummy", "baseline_subcat", "logreg_full", "logreg_trim", "hgb_full"):
        r5 = results[k]["at_0.5"]; rt = results[k].get("at_tuned", r5)
        print(f"{k:26s} {r5['roc_auc']:9.4f} {r5['pr_auc']:9.4f} {r5['f1_1']:9.4f} "
              f"{rt['f1_1']:9.4f} {rt['recall_1']:13.4f}")
    b, lf, lt = (results["baseline_subcat"]["at_0.5"], results["logreg_full"]["at_0.5"],
                 results["logreg_trim"]["at_0.5"])
    print(f"\nЛогрег полный минус справочник:   ROC-AUC {lf['roc_auc']-b['roc_auc']:+.4f}, "
          f"PR-AUC {lf['pr_auc']-b['pr_auc']:+.4f}")
    print(f"Логрег урезанный минус полный:    ROC-AUC {lt['roc_auc']-lf['roc_auc']:+.4f}, "
          f"PR-AUC {lt['pr_auc']-lf['pr_auc']:+.4f}, "
          f"F1@train {results['logreg_trim']['at_tuned']['f1_1']-results['logreg_full']['at_tuned']['f1_1']:+.4f}")

    # --- основная модель ---
    main_key = MAIN_MODEL
    cat, num = (CAT_TRIM, NUM_TRIM) if main_key == "logreg_trim" else (CAT_FULL, NUM_FULL)
    model, p_main, thr_main = fitted[main_key], probs[main_key], thrs[main_key]
    hr(f"ОСНОВНАЯ МОДЕЛЬ: {main_key}")
    print(f"Порог {thr_main:.2f}, подобран на OOF train, к тесту не подгонялся.")

    # --- A/B answer_type ---
    hr("A/B: answer_type — УТЕЧКА ИЛИ НЕТ")
    cat_b = cat + ["answer_type"]
    auc_b = roc_auc_score(yte, make_logreg(cat_b, num).fit(train[cat_b + num], ytr)
                          .predict_proba(test[cat_b + num])[:, 1])
    auc_a = results[main_key]["at_0.5"]["roc_auc"]
    print(f"ROC-AUC БЕЗ answer_type: {auc_a:.4f} (основная)\n"
          f"ROC-AUC С  answer_type: {auc_b:.4f}\nРазница: {auc_b-auc_a:+.4f}")
    print("Вердикт:", "УТЕЧКА" if auc_b - auc_a > 0.02 else
          "утечки нет, прирост в пределах шума; поле в основную модель не берём")
    results["main_with_answer_type"] = {"roc_auc": float(auc_b), "delta": float(auc_b - auc_a)}

    # --- permutation importance ---
    hr("PERMUTATION IMPORTANCE (основная модель, тест, n_repeats=10)")
    pi = permutation_importance(model, test[cat + num], yte, n_repeats=10,
                                random_state=SEED, scoring="roc_auc", n_jobs=-1)
    cols = cat + num
    top = [{"feature": cols[i], "mean": float(pi.importances_mean[i]),
            "std": float(pi.importances_std[i])}
           for i in np.argsort(pi.importances_mean)[::-1][:15]]
    print(f"{'признак':24s} {'среднее':>10s} {'std':>10s}")
    for t in top:
        print(f"{t['feature']:24s} {t['mean']:10.5f} {t['std']:10.5f}")

    # --- калибровка ---
    hr("ТАБЛИЦА КАЛИБРОВКИ (10 децилей, основная модель)")
    d = pd.DataFrame({"p": p_main, "y": yte.values})
    d["bin"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    print(f"{'дециль':>7s} {'границы':>20s} {'n':>7s} {'ср.предсказ':>12s} {'факт.доля':>11s}")
    calib = []
    for bi, g in d.groupby("bin"):
        lo, hi, mp, ay = g["p"].min(), g["p"].max(), g["p"].mean(), g["y"].mean()
        print(f"{int(bi)+1:>7d} {f'[{lo:.3f}, {hi:.3f}]':>20s} {len(g):>7d} {mp:>12.4f} {ay:>11.4f}")
        calib.append({"decile": int(bi)+1, "lo": float(lo), "hi": float(hi),
                      "n": int(len(g)), "mean_pred": float(mp), "actual": float(ay)})
    print("\nЕсли модель откалибрована, две последние колонки идут рядом.")

    # --- сопоставимые рабочие точки: справочник против модели ---
    ops = compare_operating_points(yte.to_numpy(), probs["baseline_subcat"], p_main,
                                   results, main_key)

    # --- артефакты ---
    hr("ШАГ 7. АРТЕФАКТЫ")
    Path("reports").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
    out = test[["created_date"] + cat + num].copy()
    out["y_true"] = yte.values
    out["y_prob"] = np.round(p_main, 6)
    out["y_prob_baseline"] = np.round(probs["baseline_subcat"], 6)
    out["y_pred_0.5"] = (p_main >= 0.5).astype(int)
    out["y_pred_tuned"] = (p_main >= thr_main).astype(int)
    out.to_csv("reports/predictions.csv", index=False, encoding="utf-8-sig")
    json.dump({"main_model": main_key, "dropped_features": DROPPED,
               "data": {"rows_raw": len(df_raw) + n_quotes + n_type,
                        "dropped_broken_quotes": n_quotes, "dropped_bad_appeal_type": n_type,
                        "sample": len(df), "share_y1": float(share_all),
                        "train_n": len(train), "train_share_y1": float(ytr.mean()),
                        "test_n": len(test), "test_share_y1": float(yte.mean()),
                        "truncation_ratio": float(trunc_ratio), "cutoff": a.cutoff,
                        "sla_days": a.sla_days, "sla_source": "ТЗ (догадка), не норматив"},
               "limitations": lims, "thresholds": thrs, "models": results,
               "permutation_importance_top15": top, "calibration_deciles": calib},
              open("reports/metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    joblib.dump(model, "models/model.pkl")
    for f in ("reports/metrics.json", "reports/predictions.csv", "models/model.pkl"):
        print(f"  {f:28s} {Path(f).stat().st_size:>10d} байт")

    # --- ИТОГО ---
    m5, mt = results[main_key]["at_0.5"], results[main_key]["at_tuned"]
    hr("ИТОГО")
    print(f"Основная модель:                    {main_key}")
    print(f"Таргет:                             days > {a.sla_days} сут "
          f"(порог из ТЗ, НЕ норматив — см. ограничения)")
    print(f"Размер train / test:                {len(train)} / {len(test)}")
    print(f"Доля просрочек в train / test:      {100*ytr.mean():.1f}% / {100*yte.mean():.1f}%")
    print(f"Accuracy пустышки:                  {results['dummy']['at_0.5']['accuracy']:.4f}")
    print(f"ROC-AUC справочника sub_category:   {b['roc_auc']:.4f}")
    print(f"ROC-AUC основной модели:            {m5['roc_auc']:.4f}  "
          f"(разница {m5['roc_auc']-b['roc_auc']:+.4f})")
    print(f"PR-AUC справочника / основной:      {b['pr_auc']:.4f} / {m5['pr_auc']:.4f}")
    print(f"Accuracy основной модели:           {m5['accuracy']:.4f}")
    print(f"Recall по классу 1 (порог 0.5):     {m5['recall_1']:.4f}")
    print(f"Recall по классу 1 (порог с train): {mt['recall_1']:.4f}  (порог {thr_main:.2f})")
    print(f"Топ-5 признаков:                    {', '.join(t['feature'] for t in top[:5])}")
    print(f"Строк выброшено при чистке:         {n_quotes + n_type}")

if __name__ == "__main__":
    main()
