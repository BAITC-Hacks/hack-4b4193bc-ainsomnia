"""Проба «отличи генераторы» (раздел 5n, проверка 3).

    .venv/bin/python -m src.synth.probe

TF-IDF + логрег учится отличать тексты A (модель по листу заданий) от C
(шаблоны), отдельно по языкам, на пятикратной перекрёстной проверке. Признаки
те же, что у базовой линии модуля 1 (src/finetune/pipeline.py). Фолды — по
группам: происшествия и почти-дубли A и почти-дубли C целиком в одном фолде,
иначе проба узнавала бы соседа по группе, а не генератор.

Почти идеальное различение значит, что у генераторов свой стиль, и разрыв
качества A → C нельзя читать только как перенос на новые формулировки той же
темы: часть его — смена стиля записи.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline

from src.synth.split import effective_size, groups

FIX = Path("tests/fixtures/synth")
SEED = 42
FOLDS = 5
LANGS = {"русский": ({"ru"}, {"ru"}), "казахский": ({"kk", "kk-ru"}, {"kk"})}


def _features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])


def _read(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIX / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run() -> dict:
    a_rows, c_rows = _read("corpus_a.jsonl"), _read("templates_c.jsonl")
    c_group = {}
    for gi, g in enumerate(groups(c_rows)[0]):
        for i in g:
            c_group[c_rows[i]["id"]] = f"C{gi}"
    out = {}
    for lang, (a_langs, c_langs) in LANGS.items():
        a = [r for r in a_rows if r["lang"] in a_langs]
        c = [r for r in c_rows if r["lang"] in c_langs]
        texts = [r["text"] for r in a + c]
        y = np.array([0] * len(a) + [1] * len(c))
        grp = np.array([f"A{r['group']}" for r in a] + [c_group[r["id"]] for r in c])
        proba = np.zeros(len(y))
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for tr, te in cv.split(texts, y, grp):
            model = make_pipeline(_features(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED))
            model.fit([texts[i] for i in tr], y[tr])
            proba[te] = model.predict_proba([texts[i] for i in te])[:, 1]
        out[lang] = {
            "a": len(a), "c": len(c),
            "c_effective": effective_size(r["text"] for r in c),
            "roc_auc": float(roc_auc_score(y, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y, proba >= 0.5)),
            "errors": int(((proba >= 0.5) != y).sum()),
        }
    return out


def main() -> None:
    for lang, m in run().items():
        print(f"{lang}: A {m['a']} против C {m['c']} · ROC-AUC {m['roc_auc']:.4f} · "
              f"сбалансированная точность {m['balanced_accuracy']:.4f} · "
              f"ошибок {m['errors']} из {m['a'] + m['c']}")


if __name__ == "__main__":
    main()
