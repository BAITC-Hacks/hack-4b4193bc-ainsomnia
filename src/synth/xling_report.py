"""Отчёт о переносе между языками (раздел 5n, проверка 6).

    .venv/bin/python -m src.synth.xling_report [--status …synth-xling-….status]

Прогоны берутся ровно из статус-файла scripts/xling_chain.sh. Для сравнения —
основная модель модуля 1, обученная на обоих языках (прогоны из статус-файла
основной цепочки): она отделяет потерю от меньшего объёма данных от потери
из-за другого языка. TF-IDF переобучается на том же исходном языке с тем же C,
что выбрала валидация исходного языка, и применяется к обоим тестам.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.finetune.pipeline import C_GRID

from src.synth import probe
from src.synth.module1_report import chain_runs, macro_f1
from src.synth.preamble import PREAMBLE

LOGS = Path("analysis/finetune_logs")
X = Path("data/synth/xling")
OUT = Path("reports/synth/xling.md")
SOURCES = {"ru": "только русский", "kk": "только казахский (kk + kk-ru)"}


def xling_runs(status: Path) -> dict[str, dict[int, str]]:
    text = status.read_text(encoding="utf-8")
    if "ГОТОВО" not in text:
        raise SystemExit(f"{status}: цепочка не закончена — отчёт не собирается")
    runs: dict[str, dict[int, str]] = {"ru": {}, "kk": {}}
    for m in re.finditer(r"конец x(ru|kk)-s(\d+): (\S+) ", text):
        runs[m.group(1)][int(m.group(2))] = m.group(3)
    if any(set(v) != set(SEEDS) for v in runs.values()):
        raise SystemExit(f"{status}: нужно по 5 seed на направление, есть "
                         f"{ {k: sorted(v) for k, v in runs.items()} }")
    return runs


def preds(src: str, run: str, test_lang: str) -> pd.Series:
    if test_lang == src:
        p = pd.read_csv(Path("models/finetune") / run / "predictions.csv")
    else:
        p = pd.read_csv(Path("models/synth/xling") / run / f"test_{test_lang}.csv")
    return p.set_index("id")["y_pred"]


def tests() -> dict[str, pd.DataFrame]:
    ru = pd.read_csv(X / "test_ru.csv")
    kk = pd.read_csv(X / "test_kk.csv")
    return {"русский": ru, "казахский (kk + kk-ru)": kk,
            "чистый казахский (kk)": kk[kk.lang == "kk"]}


def test_lang(name: str) -> str:
    return "ru" if name == "русский" else "kk"


def tfidf(src: str, C: float, frames) -> dict[str, float]:
    d = pd.read_csv(X / f"{src}.csv")
    tr = d[d.split == "train"]
    f = probe._features()
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(f.fit_transform(tr.text), tr.label)
    return {name: macro_f1(fr.label, clf.predict(f.transform(fr.text)))
            for name, fr in frames.items()}


SEEDS = (42, 43, 44, 45, 46)


def char_features():
    """Только символьные n-граммы 3–5 внутри слов: они переносятся между языками
    с общими корнями и заимствованиями, слова — нет (поправка 2026-09-27)."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)


def word_features():
    """Словная часть прежней смешанной базы, без символьных признаков."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def tfidf_baseline(src: str, feats, frames) -> tuple[float, dict[str, float]]:
    """C выбирается только по validation; lbfgs не использует random_state.

    Один детерминированный прогон. Тесты читаются моделью лишь после выбора C.
    """
    d = pd.read_csv(X / f"{src}.csv")
    tr, va = d[d.split == "train"], d[d.split == "validation"]
    f = feats()
    Xtr = f.fit_transform(tr.text)
    Xva = f.transform(va.text)
    best = (-1.0, None, None)
    for c in C_GRID:
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=c).fit(Xtr, tr.label)
        score = macro_f1(va.label, clf.predict(Xva))
        if score > best[0]:
            best = (score, c, clf)
    return best[1], {name: macro_f1(fr.label, best[2].predict(f.transform(fr.text)))
                     for name, fr in frames.items()}


def summary(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "std": float(v.std(ddof=1)), "per_seed": v.tolist()}


def comparison(vals, baseline):
    st = summary(vals)
    delta = st["mean"] - baseline
    spread = st["max"] - st["min"]
    clear = delta > max(2 * st["std"], spread) and st["min"] > baseline
    return delta, ("разница больше измеренного разброса" if clear
                   else "неразличимы в пределах измеренного шума")


def fmt(vals) -> str:
    v = np.array(vals)
    return f"{v.mean():.4f} ({v.min():.4f}–{v.max():.4f}), s={v.std(ddof=1):.4f}"


def render(xstatus: Path, mstatus: Path, summary_path: Path | None = None) -> str:
    runs = xling_runs(xstatus)
    main_runs, _ = chain_runs(mstatus)
    frames = tests()
    L = [PREAMBLE, "", "# Модуль 1: перенос между языками", "",
         "Единственная проверка двуязычности, которая у проекта будет: ТЗ требует "
         "казахский, а реальных текстов нет и не будет (раздел 10, пункт 6). "
         "Казахская часть синтетики носителем языка не проверялась.", "",
         "**Протокол.** XLM-R base, lr 5e-5 из основного прогона модуля 1, 15 эпох, "
         "эпоха по валидации **исходного** языка, пять seed на направление. Тест — "
         "тексты теста A на каждом языке; группы разбиения те же, поэтому ни одно "
         "происшествие не встречается и в обучении, и в тесте.", ""]

    sizes = {src: pd.read_csv(X / f"{src}.csv").groupby("split").size().to_dict()
             for src in SOURCES}
    base, tf = {}, {}
    for src in SOURCES:
        m = json.loads((Path("reports/finetune") / runs[src][42] / "metrics.json")
                       .read_text(encoding="utf-8"))
        base[src] = m["baselines"]
        tf[src] = tfidf(src, m["baselines"]["tfidf_search"]["C"], frames)
        own = "русский" if src == "ru" else "казахский (kk + kk-ru)"
        if abs(tf[src][own] - m["baselines"]["tfidf_logreg"]["macro_f1"]) > 1e-9:
            raise AssertionError(f"пересчёт TF-IDF ({src}) не совпал с базовой линией")
    wc, ch, word = {}, {}, {}
    for src in SOURCES:
        c_wc, wc[src] = tfidf_baseline(src, probe._features, frames)
        c_ch, ch[src] = tfidf_baseline(src, char_features, frames)
        if c_wc != base[src]["tfidf_search"]["C"] or any(
                abs(wc[src][n] - tf[src][n]) > 1e-9 for n in frames):
            raise AssertionError(f"повтор TF-IDF слова+символы ({src}) не совпал с базовой линией")
        wc[src + "_C"], ch[src + "_C"] = c_wc, c_ch
        word[src + "_C"], word[src] = tfidf_baseline(src, word_features, frames)
    L += [f"Объёмы: русский — train {sizes['ru']['train']}, validation "
          f"{sizes['ru']['validation']}; казахский — train {sizes['kk']['train']}, "
          f"validation {sizes['kk']['validation']}. Казахской валидации мало — у "
          "редких классов по два текста, — поэтому выбор эпохи в казахском "
          "направлении шумный.", ""]

    # таблица: строки — тестовый язык, колонки — чем обучено
    head = ["| Тест | Текстов | Обучение на обоих языках | Только русский | "
            "Только казахский | Слова+символы: только русский | Слова+символы: только казахский |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    L += ["## macro-F1 по языку теста", "",
          "У моделей — среднее, минимум–максимум и выборочное std по пяти seed. Ячейки, где язык обучения "
          "совпадает с языком теста, — перенос не проверяют; остальные — "
          "проверяют.", ""] + head
    both, per_seed = {}, {}
    for name, fr in frames.items():
        y, ids = fr.label.to_numpy(), fr.id
        tl = test_lang(name)
        b = [macro_f1(y, pd.read_csv(Path("models/finetune") / r / "predictions.csv")
                      .set_index("id")["y_pred"].loc[ids].to_numpy())
             for r in main_runs.values()]
        cols = {src: [macro_f1(y, preds(src, r, tl).loc[ids].to_numpy())
                      for _, r in sorted(runs[src].items())] for src in SOURCES}
        both[name] = (float(np.mean(b)), {s: float(np.mean(v)) for s, v in cols.items()})
        per_seed[name] = cols
        L.append(f"| {name} | {len(fr)} | {fmt(b)} | {fmt(cols['ru'])} | "
                 f"{fmt(cols['kk'])} | {tf['ru'][name]:.4f} | {tf['kk'][name]:.4f} |")
    L += ["", f"Случайный уровень 1/15 = {1 / 15:.4f}; самый частый класс — "
          f"{base['ru']['most_frequent']['macro_f1']:.4f} на русском тесте, "
          f"{base['kk']['most_frequent']['macro_f1']:.4f} на казахском.", ""]

    L += ["## XLM-R, словная и символьная базовые линии", "",
          "Все числа ниже — на синтетическом корпусе 5n, не на настоящих обращениях. "
          "Словная база почти не знает слов другого языка. Символьная char_wb 3–5 "
          "проверяет перенос общих фрагментов. Прежняя смешанная база сохранена "
          "справочно; она не называлась и не считается чисто словной.", "",
          "Для всех TF-IDF: одна сетка C " + ", ".join(f"{c:g}" for c in C_GRID) +
          "; выбор по validation исходного языка, векторизатор и классификатор "
          "обучаются только на train. После выбора C — один итоговый test для "
          "каждого среза. Solver lbfgs: детерминированный результат, random_state "
          "не используется; искусственных пяти повторов и std у баз нет.", "",
          "| Исходный язык | C word | C char_wb | C слова+символы | train | validation |",
          "|---|---:|---:|---:|---:|---:|"]
    for src in SOURCES:
        L.append(f"| {src} | {word[src + '_C']:g} | {ch[src + '_C']:g} | "
                 f"{wc[src + '_C']:g} | {sizes[src]['train']} | {sizes[src]['validation']} |")
    L += ["", "| Обучение | Тест | n test | XLM-R mean (min–max), sample std | "
          "word TF-IDF | char_wb 3–5 | слова+символы | Δ XLM-R − char_wb | Вывод |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for src, lab in SOURCES.items():
        for name, frame in frames.items():
            vals = per_seed[name][src]
            delta, verdict = comparison(vals, ch[src][name])
            L.append(f"| {lab} | {name} | {len(frame)} | {fmt(vals)} | "
                     f"{word[src][name]:.4f} | {ch[src][name]:.4f} | {wc[src][name]:.4f} | "
                     f"{delta:+.4f} | {verdict} |")
    L += ["", "Разница называется превышающей разброс только если она больше "
          "и двух выборочных std, и полного размаха пяти seed, а худший seed "
          "выше char_wb. Иначе: «неразличимы в пределах измеренного шума». "
          "Это описательное сравнение, не проверка качества на настоящих обращениях.", "",
          "### Значения по seed и арифметика", "",
          "Порядок seed: 42, 43, 44, 45, 46. mean = сумма / 5; "
          "sample std = sqrt(сумма квадратов отклонений от mean / 4).", ""]
    for src, name in (("ru", "казахский (kk + kk-ru)"), ("kk", "русский")):
        vals = per_seed[name][src]
        st = summary(vals)
        ss = sum((v - st["mean"]) ** 2 for v in vals)
        L.append(f"- {src} → {test_lang(name)}: " + ", ".join(f"{v:.10f}" for v in vals)
                 + f"; сумма {sum(vals):.10f}; mean {st['mean']:.10f}; "
                 f"Σ(x−mean)² {ss:.10f}; std {st['std']:.10f}.")
    L.append("")

    ru_b, ru_c = both["русский"]
    kk_b, kk_c = both["казахский (kk + kk-ru)"]
    pk_b, pk_c = both["чистый казахский (kk)"]
    L += ["## Что это значит", "",
          f"- **Русский → казахский:** модель, не видевшая ни одного казахского "
          f"текста, даёт на казахском тесте {kk_c['ru']:.4f} — на синтетическом корпусе 5n, "
          "не на настоящих обращениях (на чистом казахском "
          f"{pk_c['ru']:.4f}); обученная на казахском — {kk_c['kk']:.4f}; на обоих "
          f"языках — {kk_b:.4f}. TF-IDF с русского на казахский — "
          f"{tf['ru']['казахский (kk + kk-ru)']:.4f} со словами и символами, "
          f"{np.mean(ch['ru']['казахский (kk + kk-ru)']):.4f} только по символам.",
          f"- **Казахский → русский:** {ru_c['kk']:.4f} на русском тесте — на синтетическом корпусе 5n, не на настоящих обращениях; обученная "
          f"на русском — {ru_c['ru']:.4f}; на обоих — {ru_b:.4f}. TF-IDF с "
          f"казахского на русский — {tf['kk']['русский']:.4f} со словами и символами, "
          f"{np.mean(ch['kk']['русский']):.4f} только по символам.",
          "- Разность «обучение на обоих» − «только на языке теста» — вклад второго "
          "языка и большего объёма вместе; «только на языке теста» − «только на "
          "другом языке» — цена переноса при сопоставимом объёме по направлению.",
          "", "## Оговорки", "",
          "1. **Оба языка написаны одним генератором по одному листу заданий:** "
          "ситуации, места и опорные значения общие. Перенос здесь проще, чем "
          "будет между реальными русскими и казахскими обращениями.",
          "2. **kk-ru — смешанная речь с русскими вставками:** в строке «казахский» "
          "перенос с русского частично не перенос. Поэтому отдельно показан чистый "
          "казахский.",
          "3. **lr взят из основного прогона**, где валидация была смешанной: "
          "целевой язык влиял на выбор lr, но не на выбор эпохи.",
          "4. Казахский носителем языка не проверялся; ошибки генератора в "
          "казахском могут и облегчать, и затруднять перенос.", ""]
    report = "\n".join(L)
    if not report.startswith(PREAMBLE):
        raise AssertionError("отчёт по синтетике обязан начинаться преамбулой")
    if summary_path is not None:
        summary_path.write_text(json.dumps({
            "scope": "synthetic 5n only", "seeds": list(SEEDS), "sizes": sizes,
            "test_sizes": {name: len(fr) for name, fr in frames.items()},
            "word": word, "char_wb": ch, "word_char": wc,
            "xlmr": {name: {src: summary(v) for src, v in cols.items()}
                     for name, cols in per_seed.items()},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status")
    ap.add_argument("--main-status")
    a = ap.parse_args()
    xs = Path(a.status) if a.status else sorted(LOGS.glob("synth-xling-*.status"))[-1]
    ms = (Path(a.main_status) if a.main_status
          else sorted(LOGS.glob("synth-chain-*.status"))[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(xs, ms, OUT.with_suffix(".json")), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
