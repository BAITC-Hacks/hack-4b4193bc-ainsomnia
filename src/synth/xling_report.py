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
from sklearn.linear_model import LogisticRegression

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
    if any(len(v) != 5 for v in runs.values()):
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


def fmt(vals) -> str:
    v = np.array(vals)
    return f"{v.mean():.4f} ({v.min():.4f}–{v.max():.4f})"


def render(xstatus: Path, mstatus: Path) -> str:
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
    L += [f"Объёмы: русский — train {sizes['ru']['train']}, validation "
          f"{sizes['ru']['validation']}; казахский — train {sizes['kk']['train']}, "
          f"validation {sizes['kk']['validation']}. Казахской валидации мало — у "
          "редких классов по два текста, — поэтому выбор эпохи в казахском "
          "направлении шумный.", ""]

    # таблица: строки — тестовый язык, колонки — чем обучено
    head = ["| Тест | Текстов | Обучение на обоих языках | Только русский | "
            "Только казахский | TF-IDF: только русский | TF-IDF: только казахский |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    L += ["## macro-F1 по языку теста", "",
          "У моделей — среднее по пяти seed и диапазон. Ячейки, где язык обучения "
          "совпадает с языком теста, — перенос не проверяют; остальные — "
          "проверяют.", ""] + head
    both = {}
    for name, fr in frames.items():
        y, ids = fr.label.to_numpy(), fr.id
        tl = test_lang(name)
        b = [macro_f1(y, pd.read_csv(Path("models/finetune") / r / "predictions.csv")
                      .set_index("id")["y_pred"].loc[ids].to_numpy())
             for r in main_runs.values()]
        cols = {src: [macro_f1(y, preds(src, r, tl).loc[ids].to_numpy())
                      for r in runs[src].values()] for src in SOURCES}
        both[name] = (float(np.mean(b)), {s: float(np.mean(v)) for s, v in cols.items()})
        L.append(f"| {name} | {len(fr)} | {fmt(b)} | {fmt(cols['ru'])} | "
                 f"{fmt(cols['kk'])} | {tf['ru'][name]:.4f} | {tf['kk'][name]:.4f} |")
    L += ["", f"Случайный уровень 1/15 = {1 / 15:.4f}; самый частый класс — "
          f"{base['ru']['most_frequent']['macro_f1']:.4f} на русском тесте, "
          f"{base['kk']['most_frequent']['macro_f1']:.4f} на казахском.", ""]

    ru_b, ru_c = both["русский"]
    kk_b, kk_c = both["казахский (kk + kk-ru)"]
    pk_b, pk_c = both["чистый казахский (kk)"]
    L += ["## Что это значит", "",
          f"- **Русский → казахский:** модель, не видевшая ни одного казахского "
          f"текста, даёт на казахском тесте {kk_c['ru']:.4f} (на чистом казахском "
          f"{pk_c['ru']:.4f}); обученная на казахском — {kk_c['kk']:.4f}; на обоих "
          f"языках — {kk_b:.4f}. TF-IDF с русского на казахский — "
          f"{tf['ru']['казахский (kk + kk-ru)']:.4f}.",
          f"- **Казахский → русский:** {ru_c['kk']:.4f} на русском тесте; обученная "
          f"на русском — {ru_c['ru']:.4f}; на обоих — {ru_b:.4f}. TF-IDF с "
          f"казахского на русский — {tf['kk']['русский']:.4f}.",
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
    OUT.write_text(render(xs, ms), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
