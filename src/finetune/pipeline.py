#!/usr/bin/env python3
"""Контракт входа, разбиение, базовые линии, метрики (раздел 5k CLAUDE.md).

Здесь НЕТ torch: всё в этом файле работает на голом sklearn и проверяется
отдельно от обучения. Обучение — в src/finetune/train.py.

Пайплайн не знает, какой корпус ему дали. Он знает только имена колонок.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

RARE_LABEL = "__rare__"


def num(n):
    return f"{int(n):,}".replace(",", " ")


def md_table(df, floatfmt="{:.3f}"):
    """Markdown-таблица без tabulate: лишняя зависимость ради трёх таблиц
    в отчёте не окупается, а requirements-ml.txt и так тяжёлый."""
    if not len(df):
        return "_пусто_"
    def cell(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- контракт
def load_csv(path, text_col, label_col, id_col=None, time_col=None,
             split_col=None):
    """CSV -> таблица с каноническими именами. Метка остаётся СТРОКОЙ."""
    df = pd.read_csv(path)
    need = {text_col, label_col} | {c for c in (id_col, time_col, split_col) if c}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: нет колонок {sorted(missing)}; "
                         f"есть {list(df.columns)}")
    out = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "label": df[label_col].astype(str).str.strip(),
    })
    out["id"] = df[id_col].astype(str) if id_col else [str(i) for i in range(len(df))]
    out["time"] = pd.to_datetime(df[time_col], errors="coerce") if time_col else pd.NaT
    out["split"] = df[split_col].astype(str) if split_col else None
    empty = int((out.text.str.len() == 0).sum())
    if empty:
        out = out[out.text.str.len() > 0].reset_index(drop=True)
    return out, empty


def dedupe(df, log):
    """Точные повторы текста — до разбиения, а не после.

    Два разных случая, и путать их нельзя:
      * повтор внутри одного сплита — оставляем одну строку;
      * один и тот же текст в train и в оценочном сплите — это утечка,
        и убираем его ИЗ TRAIN, чтобы оценочные сплиты остались такими,
        какими их опубликовал автор корпуса.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["split", "text"], keep="first").copy()
    within = before - len(df)

    leaked = 0
    if df.split.notna().all() and df.split.nunique() > 1:
        eval_texts = set(df.loc[df.split != "train", "text"])
        mask = (df.split == "train") & df.text.isin(eval_texts)
        leaked = int(mask.sum())
        df = df[~mask].copy()
    log(f"  дубли: внутри сплита снято {num(within)}, "
        f"пересечение train с оценкой снято {num(leaked)}")
    return df.reset_index(drop=True), within, leaked


def apply_rare(df, min_count, policy, log):
    """Редкие классы — правило, а не решение по ходу."""
    counts = df.label.value_counts()
    rare = sorted(counts[counts < min_count].index)
    if not rare:
        log(f"  редкие классы: нет (порог {min_count})")
        return df, []
    rows = int(counts[rare].sum())
    if policy == "drop":
        df = df[~df.label.isin(rare)].reset_index(drop=True)
        log(f"  редкие классы: отброшено {len(rare)} классов / {num(rows)} строк "
            f"(порог {min_count}): {', '.join(rare[:6])}"
            + (" …" if len(rare) > 6 else ""))
    else:
        df = df.copy()
        df.loc[df.label.isin(rare), "label"] = RARE_LABEL
        log(f"  редкие классы: слито в «{RARE_LABEL}» {len(rare)} классов / "
            f"{num(rows)} строк (порог {min_count})")
    return df, rare


def split_data(df, mode, seed, log, val_size=0.15, test_size=0.15):
    """Готовое разбиение корпуса, иначе стратифицированное или временное."""
    if df.split.notna().all() and df.split.nunique() > 1:
        log(f"  разбиение: собственное разбиение корпуса "
            f"({', '.join(sorted(df.split.unique()))})")
        return {s: g.reset_index(drop=True) for s, g in df.groupby("split")}

    if mode == "temporal":
        if df.time.isna().any():
            raise ValueError("--split-mode temporal требует --time-col без пропусков")
        df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
        n = len(df)
        a, b = int(n * (1 - val_size - test_size)), int(n * (1 - test_size))
        parts = {"train": df.iloc[:a], "validation": df.iloc[a:b], "test": df.iloc[b:]}
        log(f"  разбиение: временное, границы "
            f"{parts['validation'].time.min():%Y-%m-%d} и "
            f"{parts['test'].time.min():%Y-%m-%d}")
        return {k: v.reset_index(drop=True) for k, v in parts.items()}

    rest, test = train_test_split(df, test_size=test_size, random_state=seed,
                                  stratify=df.label)
    tr, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                               random_state=seed, stratify=rest.label)
    log(f"  разбиение: стратифицированное, seed {seed}")
    return {k: v.reset_index(drop=True) for k, v in
            (("train", tr), ("validation", val), ("test", test))}


def label_map(parts):
    """Индексы назначаются по классам ОБУЧАЮЩЕЙ выборки, отсортированным."""
    labels = sorted(parts["train"].label.unique())
    return {lab: i for i, lab in enumerate(labels)}


# ---------------------------------------------------------------- метрики
def evaluate(y_true, y_pred, names):
    """macro-F1 основная: при перекосе классов accuracy льстит (см. 4b)."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "classes_true": int(len(set(y_true))),
        "classes_pred": int(len(set(y_pred))),
    }


def per_class(y_true, y_pred, names):
    rep = classification_report(y_true, y_pred, labels=list(range(len(names))),
                                target_names=names, output_dict=True,
                                zero_division=0)
    rows = [{"класс": k, "precision": v["precision"], "recall": v["recall"],
             "f1": v["f1-score"], "support": int(v["support"])}
            for k, v in rep.items() if k in names]
    return pd.DataFrame(rows).sort_values("f1")


def confusions(y_true, y_pred, names, top=10):
    bad = [(names[t], names[p]) for t, p in zip(y_true, y_pred) if t != p]
    if not bad:
        return pd.DataFrame(columns=["истинный", "предсказан", "раз"])
    s = pd.Series(bad).value_counts().head(top)
    return pd.DataFrame([{"истинный": a, "предсказан": b, "раз": int(n)}
                         for (a, b), n in s.items()])


# ---------------------------------------------------------------- базовые линии
def baseline_most_frequent(train, test, lmap):
    """Нижняя граница: всегда самый частый класс обучающей выборки."""
    top = train.label.value_counts().idxmax()
    y_pred = np.full(len(test), lmap[top])
    return y_pred, top


C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
C_ARBITRARY = 4.0          # значение, взятое наугад в первом прогоне


def baseline_tfidf(train, val, test, lmap, seed, log, grid=C_GRID):
    """Честная точка отсчёта. Дообучение, не обошедшее её, ничего не стоит.

    Базовой линии полагается **то же усилие, что и модели**: C подбирается
    по валидации той же сеткой, какой у модели подбирался lr. Иначе сравнение
    меряет не качество подходов, а то, кому из них уделили внимание.

    Слова и символьные n-граммы вместе: на русском морфология съедает
    словарь, и словесный tf-idf в одиночку занижал бы планку — а занижать
    планку под свою модель здесь нельзя (та же логика, что справочник
    sub_category в разделе 4b).

    Обучается ТОЛЬКО на train, как и модель: дообучить базовую линию ещё и
    на валидации значило бы дать ей больше данных и сравнить несравнимое."""
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])
    Xtr = feats.fit_transform(train.text)
    Xva, Xte = feats.transform(val.text), feats.transform(test.text)
    ytr, yva = train.label.map(lmap), val.label.map(lmap)

    search, best_c, best_val, arb_pred = [], None, -1.0, None
    for c in grid:
        clf = LogisticRegression(max_iter=2000, C=c, random_state=seed)
        clf.fit(Xtr, ytr)
        v = float(f1_score(yva, clf.predict(Xva), average="macro", zero_division=0))
        search.append({"C": c, "val_macro_f1": v})
        log(f"    C={c:<5g} val macro-F1 {v:.4f}")
        if c == C_ARBITRARY:
            arb_pred = clf.predict(Xte)
        if v > best_val:
            best_c, best_val, best_pred = c, v, clf.predict(Xte)
    log(f"    выбрано C={best_c:g} по валидации (тест не участвовал)")
    return best_pred, arb_pred, {"C": best_c, "val_macro_f1": best_val,
                                 "grid": search}
