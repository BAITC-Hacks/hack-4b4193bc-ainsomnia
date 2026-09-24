#!/usr/bin/env python3
"""Поиск похожих: кодирование, метрики, базовые линии (раздел 5l CLAUDE.md).

Метрики и правило решения зафиксированы в CLAUDE.md ДО кода и здесь только
реализованы: основная — macro precision@10, справочно precision@1 и hit@10.
«Похоже» на этапе отладки — совпадение метки корпуса.

Индекс — только train («ранее поступившие»), запросы — validation или test.
Тест не входит ни в индекс, ни в обучение.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

E5_PREFIX = "query: "      # сверено с карточкой e5: симметричная задача — query: с обеих сторон
TOPK = 10
C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


# ---------------------------------------------------------------- кодирование
def pool(out, mask, how):
    """mean — как у e5 (average pooling в карточке); cls — как у rubert-tiny2."""
    if how == "cls":
        return out.last_hidden_state[:, 0]
    m = mask.unsqueeze(-1).float()
    return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode(model, tok, texts, device, prefix="", pooling="mean", max_len=64,
           batch=128, train=False):
    """Тексты -> L2-нормированные эмбеддинги. train=True — с градиентом."""
    ctx = torch.enable_grad() if train else torch.no_grad()
    out = []
    with ctx:
        for i in range(0, len(texts), batch):
            enc = tok([prefix + t for t in texts[i:i + batch]], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            z = pool(model(**enc), enc["attention_mask"], pooling)
            z = torch.nn.functional.normalize(z, dim=-1)
            out.append(z if train else z.float().cpu())
    return torch.cat(out) if train else torch.cat(out).numpy()


# ---------------------------------------------------------------- метрики
def neighbours(q, idx, k=TOPK):
    """Индексы top-k соседей по косинусу (векторы уже нормированы) и их близость."""
    sim = q @ idx.T
    if hasattr(sim, "toarray"):
        sim = sim.toarray()
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    s = np.take_along_axis(sim, top, 1)
    order = np.argsort(-s, axis=1)
    return np.take_along_axis(top, order, 1), np.take_along_axis(s, order, 1)


def per_query(top, y_idx, y_q):
    """Построчные величины, из которых собираются все метрики и бутстрэп."""
    same = (y_idx[top] == y_q[:, None])
    return {"p10": same.mean(1), "p1": same[:, 0].astype(float),
            "hit10": same.any(1).astype(float)}


def macro(v, y_q):
    """Среднее по классам запроса: каждый класс весит одинаково, как macro-F1."""
    return float(np.mean([v[y_q == c].mean() for c in np.unique(y_q)]))


def summarize(pq, y_q):
    return {"macro_p10": macro(pq["p10"], y_q), "macro_p1": macro(pq["p1"], y_q),
            "macro_hit10": macro(pq["hit10"], y_q),
            "micro_p10": float(pq["p10"].mean()), "n_queries": int(len(y_q))}


def random_level(y_idx, y_q):
    """Ожидаемая precision у случайного поиска: доля класса запроса в индексе."""
    share = {c: float(np.mean(y_idx == c)) for c in np.unique(y_idx)}
    v = np.array([share.get(c, 0.0) for c in y_q])
    return {"macro_p10": macro(v, y_q), "micro_p10": float(v.mean())}


def paired_bootstrap(a, b, y_q, n=1000, seed=0):
    """95% интервал для разности macro(a) − macro(b) по ОДНИМ И ТЕМ ЖЕ запросам.

    Парный: на каждом повторе обе величины считаются на одной выборке
    запросов, поэтому общий для обоих методов шум запросов вычитается."""
    rng = np.random.default_rng(seed)
    n_q = len(y_q)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, n_q, n_q)
        d[i] = macro(a[s], y_q[s]) - macro(b[s], y_q[s])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- базовые линии
def _tfidf():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])


def baseline_tfidf(tr, qs):
    """TF-IDF как в модуле 1 (слова + символьные n-граммы), косинус."""
    f = _tfidf()
    idx = normalize(f.fit_transform(tr.text))
    return idx, {k: normalize(f.transform(q.text)) for k, q in qs.items()}


def baseline_label_probs(tr, qs, y_tr, y_val, y_idx, seed, log):
    """Опора на метках: вероятности классов TF-IDF + логрега как вектор.

    Видит метки, как и дообученная модель, — поэтому сравнение с ней главное.
    C подбирается по той же метрике, по которой всё оценивается (macro p@10 на
    валидации), а не по macro-F1: базовой линии даётся то же усилие."""
    f = _tfidf()
    X = f.fit_transform(tr.text)
    Xq = {k: f.transform(q.text) for k, q in qs.items()}
    best = (None, -1.0, None)
    grid = []
    for c in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=c, random_state=seed).fit(X, y_tr)
        idx = normalize(clf.predict_proba(X))
        v = normalize(clf.predict_proba(Xq["validation"]))
        top, _ = neighbours(v, idx)
        m = macro(per_query(top, y_idx, y_val)["p10"], y_val)
        grid.append({"C": c, "val_macro_p10": m})
        log(f"    C={c:<5g} val macro p@10 {m:.4f}")
        if m > best[1]:
            best = (c, m, clf)
    c, m, clf = best
    log(f"    выбрано C={c:g} по валидации (тест не участвовал)")
    return (normalize(clf.predict_proba(X)),
            {k: normalize(clf.predict_proba(x)) for k, x in Xq.items()},
            {"C": c, "val_macro_p10": m, "grid": grid})


def baseline_frozen(name, tr, qs, device, prefix, pooling, max_len):
    """Готовая модель без дообучения."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    enc = lambda d: encode(model, tok, d.text.tolist(), device, prefix, pooling, max_len)
    return enc(tr), {k: enc(q) for k, q in qs.items()}
