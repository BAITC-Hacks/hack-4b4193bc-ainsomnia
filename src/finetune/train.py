#!/usr/bin/env python3
"""Дообучение классификатора тем на произвольном CSV (раздел 5k CLAUDE.md).

    .venv/bin/python -m src.finetune.train --csv data/external/massive.csv \
        --text-col text --label-col label --split-col split --id-col id

Пайплайн не знает, какой корпус ему дали: только имена колонок. Приведение
корпуса к контракту — src/finetune/fetch.py, разбиение и метрики —
src/finetune/pipeline.py, обучение — здесь.

Считаем в fp32: на Apple Silicon (MPS) смешанная точность ненадёжна, и
--fp16/--bf16 здесь нет намеренно, а не по недосмотру.

Артефакты: веса, токенизатор, label_map.json, run.json и построчные
predictions.csv — в models/finetune/<прогон>/ (каталог в .gitignore);
агрегатные metrics.json и report.md — в reports/finetune/<прогон>/.
Разделение не косметическое: строка на пример коммититься не может (раздел 1).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.finetune import pipeline as P
from src.synth.preamble import require_flag, with_preamble

MODELS = Path("models/finetune")
REPORTS = Path("reports/finetune")
SPEED_AFTER = 100          # шагов, после которых печатаем фактическую скорость


def log(msg=""):
    print(msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def pick_device(force_cpu):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(tok, texts, labels, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"],
                         torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device),
                       attention_mask=mask.to(device)).logits
        out.append(logits.float().cpu())
    logits = torch.cat(out)
    prob = torch.softmax(logits, dim=1)
    return logits.argmax(1).numpy(), prob.numpy()


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--id-col")
    ap.add_argument("--time-col")
    ap.add_argument("--split-col")
    ap.add_argument("--model", default="cointegrated/rubert-tiny2")
    ap.add_argument("--split-mode", choices=["stratified", "temporal"],
                    default="stratified")
    ap.add_argument("--min-class-count", type=int, default=10)
    ap.add_argument("--rare-policy", choices=["drop", "merge"], default="drop")
    ap.add_argument("--lr-grid", default="",
                    help="через запятую: короткие пробные прогоны на каждом "
                         "значении, выбор по валидации, тест не участвует")
    ap.add_argument("--lr-probe-epochs", type=int, default=3)
    ap.add_argument("--lr-edge", type=float,
                    help="если лучший lr оказался наибольшим в сетке, "
                         "дополнительно пробуется это значение: оптимум на "
                         "краю сетки значит, что сетка его может не накрывать")
    ap.add_argument("--baselines-from",
                    help="metrics.json прошлого прогона: базовые линии взять "
                         "оттуда, а не считать заново (сверяется по составу "
                         "классов и размерам выборок)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="умышленная поломка: метки train перемешиваются, "
                         "macro-F1 обязан упасть к 1/числа классов")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="отчёт начинается обязательной оговоркой о синтетике")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    require_flag(a.csv, a.synthetic)

    started = time.time()
    set_seed(a.seed)
    device = pick_device(a.cpu)
    run = (f"{Path(a.csv).stem}-{a.model.split('/')[-1]}"
           f"{'-shuffled' if a.shuffle_labels else ''}"
           f"{('-' + a.tag) if a.tag else ''}-{datetime.now():%Y%m%d-%H%M}")

    log(f"\n{'=' * 78}\nДООБУЧЕНИЕ: {run}\n{'=' * 78}")
    log(f"Модель {a.model} · устройство {device.type} · fp32 · seed {a.seed}")

    # ---------------------------------------------------------- данные
    log("\nДАННЫЕ")
    df, empty = P.load_csv(a.csv, a.text_col, a.label_col, a.id_col,
                           a.time_col, a.split_col)
    log(f"  {a.csv}: {P.num(len(df))} строк, {df.label.nunique()} классов"
        + (f"; пустых текстов снято {empty}" if empty else ""))
    df, dup_within, dup_leak = P.dedupe(df, log)
    df, rare = P.apply_rare(df, a.min_class_count, a.rare_policy, log)
    parts = P.split_data(df, a.split_mode, a.seed, log)

    lmap = P.label_map(parts)
    names = [k for k, _ in sorted(lmap.items(), key=lambda kv: kv[1])]
    unseen = 0
    for k in list(parts):
        n0 = len(parts[k])
        parts[k] = parts[k][parts[k].label.isin(lmap)].reset_index(drop=True)
        unseen += n0 - len(parts[k])
    if unseen:
        log(f"  снято {unseen} строк оценочных сплитов с классами, "
            f"которых нет в train")
    for k in ("train", "validation", "test"):
        p = parts[k]
        log(f"  {k:<11s} {P.num(len(p)):>8s} строк, {p.label.nunique():>3} классов")
    log(f"  классов в обучении: {len(names)}; "
        f"случайное угадывание macro-F1 ≈ {1 / len(names):.4f}")

    if a.shuffle_labels:
        rng = np.random.default_rng(a.seed)
        parts["train"] = parts["train"].copy()
        parts["train"]["label"] = rng.permutation(parts["train"]["label"].values)
        log(f"  {'!' * 3} МЕТКИ TRAIN ПЕРЕМЕШАНЫ — это проверка на утечку")

    y = {k: v.label.map(lmap).to_numpy() for k, v in parts.items()}

    # ---------------------------------------------------------- базовые линии
    log("\nБАЗОВЫЕ ЛИНИИ (на тесте)")
    t0 = time.time()
    if a.baselines_from:
        # Переиспользование допустимо только при совпадении выборок: базовая
        # линия, посчитанная на другом разбиении, сравнению не подлежит.
        prev = json.loads(Path(a.baselines_from).read_text(encoding="utf-8"))
        prev_rows = prev["baselines"]["most_frequent"]["n"]
        if prev["classes"] != len(names) or prev_rows != len(parts["test"]):
            raise ValueError(
                f"{a.baselines_from}: там {prev['classes']} классов и "
                f"{prev_rows} строк теста, здесь {len(names)} и "
                f"{len(parts['test'])} — базовые линии несопоставимы")
        if bool(prev.get("shuffle_labels")) != a.shuffle_labels:
            raise ValueError(f"{a.baselines_from}: другой режим меток")
        b = prev["baselines"]
        mf, tf, tf_arb, tf_search = (b["most_frequent"], b["tfidf_logreg"],
                                     b["tfidf_logreg_arbitrary_C"],
                                     b["tfidf_search"])
        log(f"  взяты из {a.baselines_from} (прогон {prev['run']}), "
            f"не пересчитывались")
        log(f"  самый частый класс: macro-F1 {mf['macro_f1']:.4f}")
        log(f"  TF-IDF при C={tf_search['C']:g}: macro-F1 {tf['macro_f1']:.4f}, "
            f"accuracy {tf['accuracy']:.4f}")
    else:
        mf_pred, mf_label = P.baseline_most_frequent(parts["train"],
                                                     parts["test"], lmap)
        mf = P.evaluate(y["test"], mf_pred, names)
        log(f"  самый частый класс («{mf_label}»): macro-F1 {mf['macro_f1']:.4f}, "
            f"accuracy {mf['accuracy']:.4f}")
        log("  TF-IDF + логрег, подбор C по валидации:")
        tfidf_pred, arb_pred, tf_search = P.baseline_tfidf(
            parts["train"], parts["validation"], parts["test"], lmap, a.seed, log)
        tf = P.evaluate(y["test"], tfidf_pred, names)
        tf_arb = P.evaluate(y["test"], arb_pred, names) if arb_pred is not None \
            else None
        if tf_arb:
            log(f"  TF-IDF при C={P.C_ARBITRARY:g} (наугад): macro-F1 "
                f"{tf_arb['macro_f1']:.4f}")
        log(f"  TF-IDF при C={tf_search['C']:g} (по валидации): macro-F1 "
            f"{tf['macro_f1']:.4f}, accuracy {tf['accuracy']:.4f}  "
            f"({time.time() - t0:.0f} с)")

    metrics = {"run": run, "model": a.model, "csv": a.csv,
               "synthetic": a.synthetic,
               "shuffle_labels": a.shuffle_labels,
               "classes": len(names), "random_macro_f1": 1 / len(names),
               "baselines": {"most_frequent": mf, "tfidf_logreg": tf,
                             "tfidf_logreg_arbitrary_C": tf_arb,
                             "tfidf_search": tf_search}}

    if a.baselines_only:
        log("\n--baselines-only: обучение пропущено")
        return 0

    # ---------------------------------------------------------- обучение
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    log("\nОБУЧЕНИЕ")
    tok = AutoTokenizer.from_pretrained(a.model)
    loaders = {k: DataLoader(encode(tok, parts[k].text, y[k], a.max_len),
                             batch_size=a.batch_size, shuffle=(k == "train"))
               for k in parts}

    def fit(lr, epochs, quiet=False):
        """Одно обучение с нуля. Возвращает модель, историю и лучшую эпоху.

        Лучшая эпоха выбирается ПО ВАЛИДАЦИИ, а не берётся последняя: val
        macro-F1 выходит на плато и дальше колеблется, пока train loss падает
        к нулю. Сохранять последнюю — значит отдавать в артефакт переобученную
        модель по чистой случайности того, где остановили счётчик эпох."""
        set_seed(a.seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, num_labels=len(names)).to(device)
        total = len(loaders["train"]) * epochs
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda(total, int(0.1 * total)))
        best = {"epoch": 0, "macro_f1": -1.0, "state": None}
        history, step, t0f = [], 0, time.time()
        for ep in range(1, epochs + 1):
            model.train()
            run_loss, seen = 0.0, 0
            for ids, mask, lab in loaders["train"]:
                out = model(input_ids=ids.to(device),
                            attention_mask=mask.to(device), labels=lab.to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                run_loss += out.loss.detach().item() * len(lab)
                seen += len(lab)
                step += 1
                if step == SPEED_AFTER and not quiet:
                    sp = SPEED_AFTER / (time.time() - t0f)
                    log(f"  скорость на {SPEED_AFTER} шагах: {sp:.1f} шаг/с — "
                        f"оценка всего обучения {total / sp / 60:.1f} мин")
            vp, _ = predict(model, loaders["validation"], device)
            vm = P.evaluate(y["validation"], vp, names)
            history.append({"epoch": ep, "train_loss": run_loss / seen,
                            "val_macro_f1": vm["macro_f1"],
                            "val_accuracy": vm["accuracy"]})
            mark = ""
            if vm["macro_f1"] > best["macro_f1"]:
                best = {"epoch": ep, "macro_f1": vm["macro_f1"],
                        "state": {k: v.detach().to("cpu").clone()
                                  for k, v in model.state_dict().items()}}
                mark = "  ← лучшая"
            log(f"  {'проба ' if quiet else ''}эпоха {ep}/{epochs}: "
                f"loss {run_loss / seen:.4f} · val macro-F1 {vm['macro_f1']:.4f} "
                f"· val acc {vm['accuracy']:.4f} · {time.time() - t0f:.0f} с{mark}")
        return model, history, best, time.time() - t0f

    lr_search = []
    PARTIAL = Path("analysis/finetune_logs") / f"{run}-lr_search.json"
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    if a.lr_grid:
        # Короткие пробы на каждом lr, выбор по валидации. Тест не участвует.
        grid = [float(x) for x in a.lr_grid.split(",")]
        log(f"  подбор lr по валидации: {a.lr_probe_epochs} эпох на каждое "
            f"из {len(grid)} значений")
        def probe(lr, note=""):
            log(f"  -- lr {lr:g}{note}")
            _, _, pb, sec = fit(lr, a.lr_probe_epochs, quiet=True)
            lr_search.append({"lr": lr, "val_macro_f1": pb["macro_f1"],
                              "probe_epochs": a.lr_probe_epochs,
                              "seconds": round(sec, 1)})
            # Промежуточный итог на диск после каждой пробы: прогон долгий,
            # и обрыв не должен уносить уже посчитанное.
            PARTIAL.write_text(json.dumps(lr_search, indent=2), encoding="utf-8")

        for lr in grid:
            probe(lr)
        top = max(lr_search, key=lambda r: r["val_macro_f1"])["lr"]
        if a.lr_edge and top == max(grid):
            log(f"  лучший lr {top:g} — на краю сетки, сетка может не накрывать "
                f"оптимум")
            probe(a.lr_edge, " (расширение сетки за край)")
        a.lr = max(lr_search, key=lambda r: r["val_macro_f1"])["lr"]
        log(f"  выбрано lr={a.lr:g} по валидации (тест не участвовал)")

    model, history, best, train_sec = fit(a.lr, a.epochs)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  параметров {n_par / 1e6:.1f} млн · max_len {a.max_len} · "
        f"батч {a.batch_size} · lr {a.lr} · эпох {a.epochs}")
    model.load_state_dict(best["state"])
    model.to(device)
    log(f"  восстановлена эпоха {best['epoch']} из {a.epochs} "
        f"(val macro-F1 {best['macro_f1']:.4f}); "
        f"последняя эпоха давала {history[-1]['val_macro_f1']:.4f}")

    # ---------------------------------------------------------- оценка
    log("\nТЕСТ")
    tp, tprob = predict(model, loaders["test"], device)
    tm = P.evaluate(y["test"], tp, names)
    log(f"  macro-F1 {tm['macro_f1']:.4f} · micro-F1 {tm['micro_f1']:.4f} · "
        f"accuracy {tm['accuracy']:.4f}")
    log(f"  против TF-IDF: {tm['macro_f1'] - tf['macro_f1']:+.4f} macro-F1; "
        f"против частого класса: {tm['macro_f1'] - mf['macro_f1']:+.4f}")

    pc = P.per_class(y["test"], tp, names)
    conf = P.confusions(y["test"], tp, names)
    log(f"  худшие классы по F1: "
        + ", ".join(f"{r['класс']} {r['f1']:.2f}" for _, r in pc.head(3).iterrows()))

    metrics.update({"model_params": int(n_par), "device": device.type,
                    "history": history, "test": tm,
                    "best_epoch": best["epoch"], "lr": a.lr,
                    "lr_search": lr_search,
                    "best_val_macro_f1": best["macro_f1"],
                    "last_val_macro_f1": history[-1]["val_macro_f1"],
                    "train_seconds": round(train_sec, 1),
                    "gain_over_tfidf": tm["macro_f1"] - tf["macro_f1"],
                    "gain_over_most_frequent": tm["macro_f1"] - mf["macro_f1"]})

    # ---------------------------------------------------------- артефакты
    mdir, rdir = MODELS / run, REPORTS / run
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    (mdir / "label_map.json").write_text(
        json.dumps(lmap, ensure_ascii=False, indent=2), encoding="utf-8")
    run_info = vars(a) | {
        "run": run, "device": device.type, "python": platform.python_version(),
        "torch": torch.__version__, "classes": len(names),
        "rows": {k: len(v) for k, v in parts.items()},
        "dup_within_split": dup_within, "dup_train_eval": dup_leak,
        "rare_classes": rare, "empty_texts": empty,
        "started": datetime.now().isoformat(timespec="seconds")}
    (mdir / "run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        "id": parts["test"].id, "text": parts["test"].text,
        "y_true": [names[i] for i in y["test"]],
        "y_pred": [names[i] for i in tp],
        "y_prob": np.round(tprob.max(axis=1), 6),
    }).to_csv(mdir / "predictions.csv", index=False)

    (rdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rdir / "report.md", run, a, names, metrics, pc, conf, run_info)

    log(f"\nАРТЕФАКТЫ\n  {mdir}/ — веса, токенизатор, label_map.json, run.json, "
        f"predictions.csv\n  {rdir}/ — metrics.json, report.md (агрегаты, "
        f"их можно коммитить)")
    log(f"\nПрогон занял {time.time() - started:.0f} с "
        f"(обучение {train_sec:.0f} с)")
    return 0


def write_report(path, run, a, names, m, pc, conf, info):
    """Отчёт — только агрегаты. Текстов примеров здесь нет намеренно: на
    реальных обращениях они содержали бы ПДн, и путь вывода должен быть
    одинаковым на внешнем корпусе и на настоящих данных."""
    t, tf, mf = m["test"], m["baselines"]["tfidf_logreg"], \
        m["baselines"]["most_frequent"]
    t_tf, srch = tf, m["baselines"]["tfidf_search"]
    arb, arb_c = m["baselines"]["tfidf_logreg_arbitrary_C"], P.C_ARBITRARY
    L = [f"# Дообучение: {run}\n",
         f"Корпус `{a.csv}`, модель `{a.model}`, устройство {m['device']}, "
         f"fp32, seed {a.seed}.",
         f"Классов {m['classes']}, случайное угадывание macro-F1 "
         f"≈ {m['random_macro_f1']:.4f}.\n",
         "## Выборки\n",
         "| Сплит | Строк |", "|---|---|"]
    L += [f"| {k} | {P.num(v)} |" for k, v in info["rows"].items()]
    L += [f"\nСнято дублей: внутри сплита {info['dup_within_split']}, "
          f"пересечение train с оценкой {info['dup_train_eval']}.\n",
          "## Результат на тесте\n",
          "| Подход | macro-F1 | micro-F1 | accuracy |", "|---|---|---|---|",
          f"| Самый частый класс | {mf['macro_f1']:.4f} | {mf['micro_f1']:.4f} "
          f"| {mf['accuracy']:.4f} |",
          f"| TF-IDF + логрег | {tf['macro_f1']:.4f} | {tf['micro_f1']:.4f} "
          f"| {tf['accuracy']:.4f} |",
          f"| **Дообученная модель** | **{t['macro_f1']:.4f}** | "
          f"{t['micro_f1']:.4f} | {t['accuracy']:.4f} |",
          f"\nРазница с TF-IDF {m['gain_over_tfidf']:+.4f} macro-F1, "
          f"с частым классом {m['gain_over_most_frequent']:+.4f}. Это один "
          f"прогон: сравнивать с разбросом по seed (раздел 5k CLAUDE.md), "
          f"а не с нулём.\n",
          "Базовой линии дано то же усилие, что модели: `C` подобран по "
          "валидации, тест в подборе не участвовал.\n",
          "| C | val macro-F1 |", "|---|---|"]
    L += [f"| {g['C']:g}{' ← выбрано' if g['C'] == srch['C'] else ''} "
          f"| {g['val_macro_f1']:.4f} |" for g in srch["grid"]]
    if arb:
        L += [f"\nПри взятом наугад `C={arb_c:g}` тест давал "
              f"{arb['macro_f1']:.4f} macro-F1 — разница с подобранным "
              f"{t_tf['macro_f1'] - arb['macro_f1']:+.4f}.\n"]
    if m.get("lr_search"):
        L += ["\n## Подбор learning rate по валидации\n",
              f"Короткие пробы по {m['lr_search'][0]['probe_epochs']} эпохи "
              f"на каждое значение, тест не участвовал.\n",
              "Пробы по три эпохи смещают выбор в пользу больших lr, потому "
              "что малым нужно больше шагов.\n",
              "| lr | val macro-F1 |", "|---|---|"]
        L += [f"| {g['lr']:g}{' ← выбрано' if g['lr'] == m['lr'] else ''} "
              f"| {g['val_macro_f1']:.4f} |" for g in m["lr_search"]]
    L += ["\n## По эпохам\n",
          f"Лучшая эпоха выбрана по валидации: **{m['best_epoch']}** из "
          f"{a.epochs} (val macro-F1 {m['best_val_macro_f1']:.4f}); последняя "
          f"эпоха давала {m['last_val_macro_f1']:.4f}. В артефакт сохранена "
          f"лучшая, а не последняя.\n",
          "| Эпоха | train loss | val macro-F1 | val accuracy |",
          "|---|---|---|---|"]
    L += [f"| {h['epoch']}{' ←' if h['epoch'] == m['best_epoch'] else ''} "
          f"| {h['train_loss']:.4f} | {h['val_macro_f1']:.4f} "
          f"| {h['val_accuracy']:.4f} |" for h in m["history"]]
    L += ["\n## Худшие десять классов по F1\n",
          P.md_table(pc.head(10)),
          "\n## Частые ошибки\n",
          P.md_table(conf),
          f"\n\nОбучение {m['train_seconds']:.0f} с.\n"]
    if not a.synthetic:
        L.append("**Числа этого отчёта относятся к внешнему корпусу и ничего не "
                 "говорят о качестве на обращениях граждан.** Корпус взят для "
                 "отладки пайплайна (раздел 5k CLAUDE.md).\n")
    path.write_text(with_preamble("\n".join(L), a.synthetic), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
