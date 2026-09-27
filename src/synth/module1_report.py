"""Сводный отчёт модуля 1 на синтетическом корпусе (раздел 5n).

    .venv/bin/python -m src.synth.module1_report [--status analysis/finetune_logs/synth-chain-….status]

Прогоны берутся ровно из статус-файла цепочки scripts/finetune_synth_chain.sh,
а не по маске каталогов: иначе при повторном запуске в сводку попали бы старые
прогоны тех же seed (урок 5l). Порядок отчёта задан:
  1) рамка 5n — первой строкой, до любых цифр;
  2) проба «отличи A от C» — до метрик модели: всё остальное читается с её учётом;
  3) разброс по пяти seed, срезы по наборам и языкам, перемешанные метки;
     рядом с каждым числом на C — эффективный объём;
  4) сравнение с TF-IDF: что результат 0.8229 на тесте A говорит о корпусе;
  5) C — односторонний тест.
Маскированные наборы, перенос между языками и прочие проверки 5n сюда не входят.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.synth import probe
from src.synth.preamble import PREAMBLE
from src.synth.split import effective_size

LOGS = Path("analysis/finetune_logs")
DATA = Path("data/synth")
OUT = Path("reports/synth/module1.md")
LANG = {"ru": "русский", "kk": "казахский", "kk-ru": "казахский"}
SETS = (("test_a", "тест A"), ("seed_holdout", "отложенные значения справочника"),
        ("test_c", "шаблоны C"))


def chain_runs(status: Path) -> tuple[dict[str, str], str]:
    runs = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        m = re.match(r"конец (\S+): (\S+) ", line)
        if m:
            runs[m.group(1)] = m.group(2)
    if "ГОТОВО" not in status.read_text(encoding="utf-8"):
        raise SystemExit(f"{status}: цепочка не закончена — отчёт не собирается")
    seeds = {k: v for k, v in runs.items() if re.fullmatch(r"s\d+", k)}
    if len(seeds) != 5 or "shuf" not in runs:
        raise SystemExit(f"{status}: нужны 5 seed и прогон shuf, найдено {sorted(runs)}")
    return seeds, runs["shuf"]


def macro_f1(y, p) -> float:
    labels = sorted(set(y))
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def set_frame(name: str) -> pd.DataFrame:
    if name == "test_a":
        d = pd.read_csv(DATA / "train_eval.csv")
        return d[d.split == "test"].reset_index(drop=True)
    return pd.read_csv(DATA / f"{name}.csv")


def model_preds(run: str, name: str) -> pd.Series:
    if name == "test_a":
        p = pd.read_csv(Path("models/finetune") / run / "predictions.csv")
    else:
        p = pd.read_csv(Path("models/synth/module1") / run / f"{name}.csv")
    return p.set_index("id")["y_pred"]


def tfidf_preds(C: float) -> dict[str, pd.Series]:
    d = pd.read_csv(DATA / "train_eval.csv")
    tr = d[d.split == "train"]
    f = probe._features()
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(f.fit_transform(tr.text), tr.label)
    out = {}
    for name, _ in SETS:
        s = set_frame(name)
        out[name] = pd.Series(clf.predict(f.transform(s.text)), index=s.id)
    return out


def slices(frame: pd.DataFrame):
    yield "все", frame
    lang = frame.lang.map(LANG)
    for key in ("русский", "казахский"):
        yield key, frame[lang == key]


def eff(name: str, frame: pd.DataFrame) -> str:
    if name != "test_c":
        return f"{len(frame)}"
    return f"{len(frame)}, **эффективно {effective_size(frame.text)}**"


def gap_rows(seeds, base_mf):
    vals = [m["test"]["macro_f1"] for m in seeds.values()]
    diff = np.array(vals) - base_mf
    sd = float(np.std(vals, ddof=1))
    return vals, diff, sd


def render(status: Path) -> str:
    seed_runs, shuf_run = chain_runs(status)
    metrics = {k: json.loads((Path("reports/finetune") / r / "metrics.json")
                             .read_text(encoding="utf-8")) for k, r in seed_runs.items()}
    shuf = json.loads((Path("reports/finetune") / shuf_run / "metrics.json")
                      .read_text(encoding="utf-8"))
    m42 = metrics["s42"]
    base = m42["baselines"]
    C = base["tfidf_search"]["C"]
    tf = tfidf_preds(C)
    frames = {name: set_frame(name) for name, _ in SETS}
    if abs(macro_f1(frames["test_a"].label, tf["test_a"].loc[frames["test_a"].id])
           - base["tfidf_logreg"]["macro_f1"]) > 1e-9:
        raise AssertionError("пересчёт TF-IDF не совпал с базовой линией прогона s42")
    pr = probe.run()

    L = [PREAMBLE, "", "# Модуль 1 на синтетическом корпусе: XLM-R, пять seed", ""]

    # 2. проба
    ru, kk = pr["русский"], pr["казахский"]
    L += ["## Сначала: проба «отличи A от C»", "",
          f"**TF-IDF с логрегом отличает A от C без единой ошибки — ROC-AUC "
          f"{ru['roc_auc']:.4f} на русском и {kk['roc_auc']:.4f} на казахском, "
          f"ошибок {ru['errors']} из {ru['a'] + ru['c']} и {kk['errors']} из "
          f"{kk['a'] + kk['c']}; значит, разрыв A → C нельзя читать как чистую "
          "меру переноса на новые формулировки той же темы: в нём неразделимо "
          "смешаны смена формулировки и полная смена стиля записи.**", "",
          "| Язык | Текстов A | Текстов C | ROC-AUC | Сбалансированная точность | Ошибок |",
          "|---|---:|---|---:|---:|---:|"]
    for lang, m in pr.items():
        L.append(f"| {lang} | {m['a']} | {m['c']}, эффективно {m['c_effective']} | {m['roc_auc']:.4f} | "
                 f"{m['balanced_accuracy']:.4f} | {m['errors']} |")
    L += ["", "Пятикратная проверка по группам: происшествия и почти-дубли A и "
          "почти-дубли C целиком в одном фолде. Признаки — те же, что у базовой "
          "линии TF-IDF модуля 1; логрег без подбора `C`. Казахский A включает "
          "смешанную речь (kk-ru).", ""]

    # 3. модель
    vals, diff, sd = gap_rows(metrics, base["tfidf_logreg"]["macro_f1"])
    mean = float(np.mean(vals))
    L += ["## Модель: разброс по пяти seed на тесте A", "",
          f"XLM-R base, `max_len` 128, lr выбран по валидации на seed 42: сетка "
          + ", ".join(f"{r['lr']:g} → {r['val_macro_f1']:.4f}" for r in m42["lr_search"])
          + f" (по {m42['lr_search'][0]['probe_epochs']} эпох на пробу). Лучший lr "
          f"{m42['lr']:g} лёг на край исходной сетки, проба за краем хуже — оптимум "
          "накрыт. Остальные seed — с тем же lr, эпоха из 15 по валидации.", "",
          "| seed | Эпоха | macro-F1 | accuracy | к TF-IDF |", "|---|---:|---:|---:|---:|"]
    for (k, m), d in zip(metrics.items(), diff):
        L.append(f"| {k[1:]} | {m['best_epoch']} из 15 | {m['test']['macro_f1']:.4f} | "
                 f"{m['test']['accuracy']:.4f} | {d:+.4f} |")
    L += ["",
          f"Среднее **{mean:.4f}**, min {min(vals):.4f}, max {max(vals):.4f}, размах "
          f"{max(vals) - min(vals):.4f}, выборочное ст. откл. **{sd:.4f}**. "
          f"Базовые линии на тех же {m42['test']['n']} текстах: TF-IDF + логрег "
          f"(`C={C:g}` по валидации) **{base['tfidf_logreg']['macro_f1']:.4f}**, "
          f"самый частый класс {base['most_frequent']['macro_f1']:.4f}. "
          f"Разрыв с TF-IDF: в среднем {float(diff.mean()):+.4f} = "
          f"{float(diff.mean()) / sd:.1f} ст. откл. разброса по seed; худший seed "
          f"{float(diff.min()):+.4f}. "
          + ("Все пять seed выше TF-IDF." if (diff > 0).all() else
             f"Ниже TF-IDF {(diff <= 0).sum()} seed из 5."), ""]

    # срезы
    L += ["## Срезы по наборам и языкам", "",
          "macro-F1; у модели — среднее по пяти seed и диапазон. Для C в колонке "
          "«Текстов» — эффективный объём после схлопывания почти-дублей по порогу "
          "0.8: числа на C относятся к нему, а не к числу строк.", "",
          "| Набор | Язык | Текстов | TF-IDF | XLM-R, среднее | XLM-R, min–max |",
          "|---|---|---|---:|---:|---:|"]
    for name, title in SETS:
        preds = {k: model_preds(r, name) for k, r in seed_runs.items()}
        for lang, fr in slices(frames[name]):
            y = fr.label.to_numpy()
            t = macro_f1(y, tf[name].loc[fr.id].to_numpy())
            ms = [macro_f1(y, p.loc[fr.id].to_numpy()) for p in preds.values()]
            L.append(f"| {title} | {lang} | {eff(name, fr)} | {t:.4f} | "
                     f"{np.mean(ms):.4f} | {min(ms):.4f}–{max(ms):.4f} |")
    L += ["", "Казахский включает смешанную речь (kk-ru): на тесте A её 27 текстов, "
          "отдельно не показана. Отложенные значения справочника — тексты по "
          "опорным значениям, которых не было в обучении (классы с тремя и более "
          "значениями).", "",
          "**Разрыв с TF-IDF по наборам** (все языки вместе):", "",
          "| Набор | Текстов | TF-IDF | XLM-R, среднее | Разрыв, среднее | Разрыв, худший seed |",
          "|---|---|---:|---:|---:|---:|"]
    gaps = {}
    for name, title in SETS:
        fr = frames[name]
        y = fr.label.to_numpy()
        t = macro_f1(y, tf[name].loc[fr.id].to_numpy())
        ms = np.array([macro_f1(y, model_preds(r, name).loc[fr.id].to_numpy())
                       for r in seed_runs.values()])
        gaps[name] = (t, ms)
        L.append(f"| {title} | {eff(name, fr)} | {t:.4f} | {ms.mean():.4f} | "
                 f"{ms.mean() - t:+.4f} | {ms.min() - t:+.4f} |")
    t_h, m_h = gaps["seed_holdout"]
    t_c, m_c = gaps["test_c"]
    L += ["",
          "- **Отложенные значения справочника:** "
          + (f"худший seed ниже TF-IDF ({m_h.min() - t_h:+.4f}), средний разрыв "
             f"{m_h.mean() - t_h:+.4f} — на новых опорных значениях модель от "
             "TF-IDF не отличить."
             if m_h.min() <= t_h else
             f"все seed выше TF-IDF, средний разрыв {m_h.mean() - t_h:+.4f}."),
          f"- **Шаблоны C** (эффективно {effective_size(frames['test_c'].text)} из "
          f"{len(frames['test_c'])}): сама модель теряет от A к C "
          f"{float(np.mean(vals)) - m_c.mean():.4f} ({float(np.mean(vals)):.4f} → "
          f"{m_c.mean():.4f}), и по пробе это падение не разделить на смену "
          "формулировки и смену стиля. Разрыв с TF-IDF "
          f"{m_c.mean() - t_c:+.4f} — больше, "
          f"чем на тесте A. С учётом пробы это ожидаемо: TF-IDF держится за слова и "
          "n-граммы стиля A, а C написан другим стилем, который проба отличает без "
          "ошибок. Выигрыш на C согласуется с устойчивостью модели к смене стиля, но "
          "знания тем он не доказывает, и держится на трёх ситуациях на класс и язык "
          "(раздел о C ниже)."]

    # перемешанные метки
    st = shuf["test"]
    L += ["", "## Перемешанные метки", "",
          f"Тот же lr {shuf['lr']:g} и seed 42, метки train перемешаны: тест A "
          f"macro-F1 **{st['macro_f1']:.4f}** при случайном уровне 1/15 = "
          f"{shuf['random_macro_f1']:.4f}, accuracy {st['accuracy']:.4f}. "
          f"Ниже 1/15 — потому что модель схлопывается: на тесте она выдаёт "
          f"{st['classes_pred']} класса из 15. "
          # Порог — вдвое выше случайного уровня: ниже него обучение на
          # перемешанных метках ничего не выучило (в 5k: 0.0151 при 0.0169).
          + ("Качество падает до случайного — утечки меток в конвейере нет."
             if st["macro_f1"] < 2 * shuf["random_macro_f1"] else
             "**Качество не упало до случайного уровня — это надо разобрать до "
             "любых выводов.**"), ""]

    # 4. сравнение с TF-IDF
    d = pd.read_csv(DATA / "train_eval.csv")
    te = frames["test_a"].assign(pred=tf["test_a"].loc[frames["test_a"].id].to_numpy())
    acc = lambda s: float((s.label == s.pred).mean())
    ex, nex = te[te.explicit], te[~te.explicit]
    bd, nbd = te[te.border.notna()], te[te.border.isna()]
    tr = d[d.split == "train"]
    rtr, rest = train_test_split(d, train_size=len(tr), stratify=d.label, random_state=42)
    rte = rest.sample(n=len(te), random_state=42)
    f2 = probe._features()
    c2 = LogisticRegression(max_iter=2000, C=C, random_state=42).fit(
        f2.fit_transform(rtr.text), rtr.label)
    rnd = macro_f1(rte.label, c2.predict(f2.transform(rte.text)))
    sib = int(rte.incident.isin(set(rtr.incident.dropna())).sum())
    noisy = te[te.noise_typo | te.noise_no_punct | te.noise_kk_letters]
    clean = te[~(te.noise_typo | te.noise_no_punct | te.noise_kk_letters)]
    L += ["## Сравнение с TF-IDF: на тесте A — 0.8229, а не около 1.0", "",
          "Постановка 5n ждала высоких чисел: «модель воспроизводит генератор». "
          f"Но TF-IDF с логрегом берёт на тесте A {base['tfidf_logreg']['macro_f1']:.4f}. "
          "**Это хорошая новость о корпусе:** тему нельзя прочитать по нескольким "
          "словам или по шаблону, и между дешёвым методом и потолком остаётся место, "
          "где сравнение моделей что-то значит — при TF-IDF около 1.0 любая модель "
          "упёрлась бы в тот же потолок, и сравнивать было бы нечего. Что это "
          "обеспечило — по числам, а не по намерениям:", "",
          f"1. **Опорное значение и название темы дословно не повторяются**, кроме "
          f"около 10% текстов с явным названием; нарушение ловится скриптом. На явных "
          f"текстах TF-IDF угадывает {acc(ex):.3f} (n={len(ex)}), на остальных "
          f"{acc(nex):.3f} (n={len(nex)}).",
          f"2. **Защита разбиения группами.** Тексты одного происшествия и почти-"
          f"дубли не расходятся по частям. На случайном разбиении тех же размеров без "
          f"групп у {sib} из {len(rte)} тестовых текстов «брат» по происшествию "
          f"оказывается в train, и TF-IDF поднимается до {rnd:.4f} — "
          f"на {rnd - base['tfidf_logreg']['macro_f1']:+.4f}.",
          f"3. **Около 10% пограничных текстов** между легко путаемыми темами, метка — "
          f"по главной проблеме: на них {acc(bd):.3f} ({len(bd)}) против "
          f"{acc(nbd):.3f} ({len(nbd)}). Это заложенная неоднозначность, а не "
          "трудность корпуса: метка здесь — решение по главной проблеме, поэтому "
          "эта часть расстояния до 1.0 внесена намеренно.",
          "4. Разнообразие листа заданий — регистр, роль, ситуация, длина, место, "
          "дата — работает вместе с пунктами 1–2; отдельной меры его вклада нет.",
          f"5. **Шум вклада не дал:** на текстах с опечатками, без пунктуации или с "
          f"заменой казахских букв TF-IDF угадывает {acc(noisy):.3f} ({len(noisy)}) "
          f"против {acc(clean):.3f} ({len(clean)}) на чистых — символьные n-граммы "
          "к такому шуму устойчивы. Приписывать ему 0.8229 нельзя.", ""]

    # 5. C
    ceff = effective_size(frames["test_c"].text)
    L += ["## C — односторонний тест", "",
          f"C — 1 200 строк, но {ceff} различных текстов, и по содержанию всего "
          "**3 ситуации на класс и язык**. Поэтому тест односторонний: **провал на C "
          "информативен** — модель не узнала тему в чужом стиле даже на простой "
          "формулировке; **успех на C — слабое свидетельство**, потому что он "
          "подтверждён на трёх ситуациях, а не на разнообразии обращений. Слот "
          "длительности в шаблоны не добавляем: он поднял бы число различных строк "
          "(симуляция — 897 вместо 514), но не число ситуаций — счётчик вырос бы, "
          "разнообразие нет.", "",
          "## Не входит в этот отчёт", "",
          "Перенос между языками, сводка по маскированию ключевых слов и остальные "
          "проверки 5n не запускались — отложены до просмотра этого отчёта. "
          "Предсказания моделей на маскированных наборах цепочка посчитала, но "
          "здесь они не сводятся.", ""]

    report = "\n".join(L)
    if not report.startswith(PREAMBLE):
        raise AssertionError("отчёт по синтетике обязан начинаться преамбулой")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status")
    a = ap.parse_args()
    status = Path(a.status) if a.status else sorted(LOGS.glob("synth-chain-*.status"))[-1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(status), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
