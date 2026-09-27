#!/usr/bin/env python3
"""Модуль 2: дообучение эмбеддингов для поиска похожих (раздел 5l CLAUDE.md).

    # 1. базовые линии, все четыре
    .venv/bin/python -m src.embed.train --csv data/external/massive.csv \
        --text-col text --label-col label --split-col split --id-col id --baselines-only
    # 2–3. подбор lr с правилом края, выбор эпохи по валидации, финал
    .venv/bin/python -m src.embed.train ... --lr-grid 1e-5,2e-5,5e-5,1e-4 --lr-edge 2e-4 \
        --baselines-from reports/embed/<прогон базовых линий>/metrics.json

Контракт входа — тот же, что у модуля 1: CSV и имена колонок флагами; разбор,
дедупликация и разбиение берутся из src/finetune/pipeline.py, не дублируются.

Обучение — supervised contrastive на батчах P классов × K примеров: все тексты
той же метки в батче положительные. Выборка P × K берёт каждый класс с равной
вероятностью, поэтому крупные интенты не забивают градиент парами.
Матрица словаря по умолчанию заморожена: так дешевле, и эмбеддинги казахских
токенов не сдвигаются при обучении на русском.

fp32, без --fp16/--bf16 — на MPS смешанная точность ненадёжна.
Построчное (эмбеддинги, соседи, величины по запросам) — в models/embed/,
в git не идёт; агрегаты — в reports/embed/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.embed import retrieval as R
from src.finetune import pipeline as P
from src.finetune.train import lr_lambda, pick_device, set_seed
from src.synth.preamble import require_flag, with_preamble

MODELS, REPORTS = Path("models/embed"), Path("reports/embed")
E5 = "intfloat/multilingual-e5-small"
TINY = "cointegrated/rubert-tiny2"
SPEED_AFTER = 100


def log(msg=""):
    print(msg, flush=True)


def load(a):
    df, empty = P.load_csv(a.csv, a.text_col, a.label_col, a.id_col, a.time_col,
                           a.split_col)
    log(f"  {a.csv}: {P.num(len(df))} строк, {df.label.nunique()} классов")
    df, _, _ = P.dedupe(df, log)
    df, _ = P.apply_rare(df, a.min_class_count, "drop", log)
    parts = P.split_data(df, a.split_mode, a.seed, log)
    lmap = P.label_map(parts)
    for k in list(parts):
        parts[k] = parts[k][parts[k].label.isin(lmap)].reset_index(drop=True)
    y = {k: v.label.map(lmap).to_numpy() for k, v in parts.items()}
    for k in ("train", "validation", "test"):
        log(f"  {k:<11s} {P.num(len(parts[k])):>8s} строк, "
            f"{parts[k].label.nunique():>3} классов")
    return parts, y, lmap


def evaluate(idx, q, y_idx, y_q):
    top, sim = R.neighbours(q, idx)
    pq = R.per_query(top, y_idx, y_q)
    return R.summarize(pq, y_q), pq, top, sim


# ---------------------------------------------------------------- базовые линии
def run_baselines(a, parts, y, device, run):
    tr, qs = parts["train"], {"validation": parts["validation"], "test": parts["test"]}
    out, perq = {}, {}

    def record(name, idx, q):
        res = {}
        for split in ("validation", "test"):
            m, pq, _, _ = evaluate(idx, q[split], y["train"], y[split])
            res[split] = m
            if split == "test":
                perq[name] = pq["p10"]
        out[name] = res
        log(f"  {name:<22s} val {res['validation']['macro_p10']:.4f} · "
            f"тест {res['test']['macro_p10']:.4f} (macro p@10)")

    t0 = time.time()
    log("  TF-IDF, косинус")
    record("tfidf", *R.baseline_tfidf(tr, qs))
    log("  опора на метках: вероятности TF-IDF + логрега, подбор C по валидации")
    idx, q, search = R.baseline_label_probs(tr, qs, y["train"], y["validation"],
                                            y["train"], a.seed, log)
    record("label_probs", idx, q)
    out["label_probs"]["search"] = search
    log(f"  {E5} без дообучения, префикс «{R.E5_PREFIX}», mean pooling")
    record("e5_frozen", *R.baseline_frozen(E5, tr, qs, device, R.E5_PREFIX, "mean",
                                           a.max_len))
    log(f"  {TINY} без дообучения, CLS pooling (справочно)")
    record("tiny2_frozen", *R.baseline_frozen(TINY, tr, qs, device, "", "cls",
                                              a.max_len))
    out["random"] = {s: R.random_level(y["train"], y[s]) for s in ("validation", "test")}
    log(f"  случайный уровень: тест macro {out['random']['test']['macro_p10']:.4f}, "
        f"по запросам {out['random']['test']['micro_p10']:.4f}  ({time.time()-t0:.0f} с)")

    mdir, rdir = MODELS / run, REPORTS / run
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    np.savez(mdir / "perquery_test.npz", y_q=y["test"],
             ids=parts["test"].id.to_numpy(), **perq)
    metrics = {"run": run, "kind": "baselines", "csv": a.csv,
               "synthetic": a.synthetic,
               "classes": int(len(np.unique(y["train"]))),
               "n_test": int(len(y["test"])), "baselines": out,
               "perquery": str(mdir / "perquery_test.npz")}
    (rdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    rows = [f"| {n} | {out[n]['validation']['macro_p10']:.4f} | "
            f"{out[n]['test']['macro_p10']:.4f} | {out[n]['test']['macro_p1']:.4f} | "
            f"{out[n]['test']['macro_hit10']:.4f} |"
            for n in ("tfidf", "label_probs", "e5_frozen", "tiny2_frozen")]
    rnd = out["random"]["test"]["macro_p10"]
    (rdir / "report.md").write_text(with_preamble("\n".join([
        f"# Базовые линии поиска похожих: {run}\n",
        f"Корпус `{a.csv}`, индекс — train ({P.num(len(y['train']))}), запросы — "
        f"validation и test. Похоже = та же метка. Основная метрика — macro "
        f"precision@10 (раздел 5l CLAUDE.md).\n",
        "| Базовая линия | val macro p@10 | тест macro p@10 | тест macro p@1 | "
        "тест macro hit@10 |", "|---|---|---|---|---|", *rows,
        f"\nСлучайный уровень на тесте: macro {rnd:.4f}.\n",
        f"Опора на метках: C={search['C']:g} выбран по валидации.\n"]), a.synthetic),
        encoding="utf-8")
    log(f"\n  {rdir}/ — metrics.json, report.md;  {mdir}/ — величины по запросам")


# ---------------------------------------------------------------- дообучение
def supcon(z, labels, tau):
    """Supervised contrastive: для каждого текста все тексты той же метки в
    батче — положительные, остальные — отрицательные, сам текст исключён."""
    sim = z @ z.T / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def pk_batches(y, p, k, steps, rng):
    """P классов × K примеров, каждый класс с равной вероятностью."""
    by = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    cls = np.array(list(by))
    for _ in range(steps):
        pick = rng.choice(cls, size=min(p, len(cls)), replace=False)
        yield np.concatenate([rng.choice(by[c], size=k, replace=len(by[c]) < k)
                              for c in pick])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for f in ("--csv", "--text-col", "--label-col"):
        ap.add_argument(f, required=True)
    for f in ("--id-col", "--time-col", "--split-col", "--baselines-from",
              "--lr-edge", "--lr-search-from"):
        ap.add_argument(f)
    ap.add_argument("--split-mode", choices=["stratified", "temporal"],
                    default="stratified")
    ap.add_argument("--min-class-count", type=int, default=10)
    ap.add_argument("--model", default=E5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-grid", default="")
    ap.add_argument("--lr-probe-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--p", type=int, default=16, help="классов в батче")
    ap.add_argument("--k", type=int, default=4, help="примеров класса в батче")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--unfreeze-vocab", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="отчёт начинается обязательной оговоркой о синтетике")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    require_flag(a.csv, a.synthetic)

    started = time.time()
    set_seed(a.seed)
    device = pick_device(False)
    run = (f"{Path(a.csv).stem}-{'baselines' if a.baselines_only else a.model.split('/')[-1]}"
           f"{('-' + a.tag) if a.tag else ''}-{datetime.now():%Y%m%d-%H%M}")
    log(f"\n{'=' * 78}\nПОИСК ПОХОЖИХ: {run}\n{'=' * 78}")
    log(f"устройство {device.type} · fp32 · seed {a.seed}\n\nДАННЫЕ")
    parts, y, lmap = load(a)

    if a.baselines_only:
        log("\nБАЗОВЫЕ ЛИНИИ")
        run_baselines(a, parts, y, device, run)
        log(f"\nПрогон занял {time.time() - started:.0f} с")
        return 0

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_text = parts["train"].text.tolist()
    n_cls = len(np.unique(y["train"]))
    if a.p > n_cls:
        # pk_batches и так берёт не больше классов, чем есть; фиксируем
        # фактическое P, чтобы лог, run.json и отчёт не называли 16×K
        # там, где батч на деле 15×K.
        log(f"  классов в train {n_cls} < P={a.p}: батч {n_cls}×{a.k}")
        a.p = n_cls
    steps = max(1, len(tr_text) // (a.p * a.k))

    def fit(lr, epochs, quiet=False):
        set_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        model = AutoModel.from_pretrained(a.model).to(device)
        if not a.unfreeze_vocab:
            model.get_input_embeddings().weight.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        total = steps * epochs
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, int(0.1 * total)))
        ylab = torch.tensor(y["train"], device=device)
        best = {"epoch": 0, "val": -1.0, "state": None}
        hist, step, t0 = [], 0, time.time()
        for ep in range(1, epochs + 1):
            model.train()
            run_loss = 0.0
            for b in pk_batches(y["train"], a.p, a.k, steps, rng):
                z = R.encode(model, tok, [tr_text[i] for i in b], device, R.E5_PREFIX,
                             "mean", a.max_len, batch=len(b), train=True)
                loss = supcon(z, ylab[b], a.tau)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                run_loss += loss.detach().item()
                step += 1
                if step == SPEED_AFTER and not quiet:
                    sp = SPEED_AFTER / (time.time() - t0)
                    log(f"  скорость на {SPEED_AFTER} шагах: {sp:.1f} шаг/с — "
                        f"оценка всего обучения {total / sp / 60:.1f} мин")
            model.eval()
            enc = lambda d: R.encode(model, tok, d.text.tolist(), device, R.E5_PREFIX,
                                     "mean", a.max_len)
            vm, _, _, _ = evaluate(enc(parts["train"]), enc(parts["validation"]),
                                   y["train"], y["validation"])
            hist.append({"epoch": ep, "train_loss": run_loss / steps,
                         "val_macro_p10": vm["macro_p10"]})
            mark = ""
            if vm["macro_p10"] > best["val"]:
                best = {"epoch": ep, "val": vm["macro_p10"],
                        "state": {k: v.detach().to("cpu").clone()
                                  for k, v in model.state_dict().items()}}
                mark = "  ← лучшая"
            log(f"  {'проба ' if quiet else ''}эпоха {ep}/{epochs}: loss "
                f"{run_loss / steps:.4f} · val macro p@10 {vm['macro_p10']:.4f} · "
                f"{time.time() - t0:.0f} с{mark}")
        return model, hist, best, time.time() - t0

    log(f"\nОБУЧЕНИЕ  {a.model} · батч {a.p}×{a.k} · τ {a.tau} · "
        f"словарь {'обучается' if a.unfreeze_vocab else 'заморожен'} · "
        f"шагов на эпоху {steps}")
    lr_search = []
    if a.lr_search_from:
        # Пробы lr, посчитанные в прерванном процессе: сетку не повторяем, а
        # берём сохранённое после каждой пробы. Выбор тот же — по валидации.
        lr_search = json.loads(Path(a.lr_search_from).read_text(encoding="utf-8"))
        a.lr = max(lr_search, key=lambda r: r["val_macro_p10"])["lr"]
        log(f"  пробы lr взяты из {a.lr_search_from}: " + ", ".join(
            f"{r['lr']:g} → {r['val_macro_p10']:.4f}" for r in lr_search))
        log(f"  выбрано lr={a.lr:g} по валидации (тест не участвовал)")
    elif a.lr_grid:
        partial = Path("analysis/finetune_logs") / f"{run}-lr_search.json"
        partial.parent.mkdir(parents=True, exist_ok=True)

        def probe(lr, note=""):
            log(f"  -- lr {lr:g}{note}")
            _, _, pb, sec = fit(lr, a.lr_probe_epochs, quiet=True)
            lr_search.append({"lr": lr, "val_macro_p10": pb["val"],
                              "probe_epochs": a.lr_probe_epochs, "seconds": round(sec, 1)})
            partial.write_text(json.dumps(lr_search, indent=2), encoding="utf-8")

        grid = [float(x) for x in a.lr_grid.split(",")]
        for lr in grid:
            probe(lr)
        top = max(lr_search, key=lambda r: r["val_macro_p10"])["lr"]
        if a.lr_edge and top == max(grid):
            log(f"  лучший lr {top:g} — на краю сетки, сетка может не накрывать оптимум")
            probe(float(a.lr_edge), " (расширение сетки за край)")
        a.lr = max(lr_search, key=lambda r: r["val_macro_p10"])["lr"]
        log(f"  выбрано lr={a.lr:g} по валидации (тест не участвовал)")

    model, hist, best, train_sec = fit(a.lr, a.epochs)
    model.load_state_dict(best["state"])
    model.to(device).eval()
    log(f"  восстановлена эпоха {best['epoch']} из {a.epochs} (val macro p@10 "
        f"{best['val']:.4f}); последняя давала {hist[-1]['val_macro_p10']:.4f}")

    enc = lambda d: R.encode(model, tok, d.text.tolist(), device, R.E5_PREFIX, "mean",
                             a.max_len)
    idx = enc(parts["train"])
    tm, pq, top, sim = evaluate(idx, enc(parts["test"]), y["train"], y["test"])
    log(f"\nТЕСТ  macro p@10 {tm['macro_p10']:.4f} · macro p@1 {tm['macro_p1']:.4f} · "
        f"macro hit@10 {tm['macro_hit10']:.4f}")

    base = None
    if a.baselines_from:
        base = json.loads(Path(a.baselines_from).read_text(encoding="utf-8"))
        if base["n_test"] != len(y["test"]) or base["classes"] != len(lmap):
            raise ValueError(f"{a.baselines_from}: другие выборки — несопоставимо")
        for n in ("e5_frozen", "label_probs", "tfidf"):
            log(f"  разница с {n:<12s} {tm['macro_p10'] - base['baselines'][n]['test']['macro_p10']:+.4f}")

    mdir, rdir = MODELS / run, REPORTS / run
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    cfg = {"model": a.model, "prefix": R.E5_PREFIX, "pooling": "mean",
           "normalize": True, "max_len": a.max_len, "run": run}
    (mdir / "embed_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    np.save(mdir / "index.npy", idx)
    np.savez(mdir / "perquery_test.npz", y_q=y["test"], ids=parts["test"].id.to_numpy(),
             finetuned=pq["p10"])
    ids_tr = parts["train"].id.to_numpy()
    pd.DataFrame({"query_id": parts["test"].id,
                  "neighbours": [" ".join(ids_tr[r]) for r in top],
                  "scores": [" ".join(f"{s:.4f}" for s in r) for r in sim]}
                 ).to_csv(mdir / "neighbours.csv", index=False)
    metrics = {"run": run, "kind": "finetuned", "model": a.model, "seed": a.seed,
               "synthetic": a.synthetic,
               "lr": a.lr, "lr_search": lr_search, "epochs": a.epochs,
               "best_epoch": best["epoch"], "best_val_macro_p10": best["val"],
               "history": hist, "test": tm, "train_seconds": round(train_sec, 1),
               "p": a.p, "k": a.k, "tau": a.tau, "unfreeze_vocab": a.unfreeze_vocab,
               "baselines_from": a.baselines_from,
               "perquery": str(mdir / "perquery_test.npz")}
    (rdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    write_report(rdir / "report.md", metrics, base)
    log(f"\nАРТЕФАКТЫ\n  {mdir}/ — веса, индекс, соседи, величины по запросам (не в git)"
        f"\n  {rdir}/ — metrics.json, report.md")
    log(f"\nПрогон занял {time.time() - started:.0f} с (обучение {train_sec:.0f} с)")
    return 0


def write_report(path, m, base):
    t = m["test"]
    L = [f"# Поиск похожих, дообучение: {m['run']}\n",
         f"Модель `{m['model']}`, seed {m['seed']}, батч {m['p']}×{m['k']}, τ {m['tau']}, "
         f"словарь {'обучался' if m['unfreeze_vocab'] else 'заморожен'}. Основная "
         f"метрика — macro precision@10 (раздел 5l CLAUDE.md).\n",
         "| Подход | тест macro p@10 | macro p@1 | macro hit@10 |", "|---|---|---|---|"]
    if base:
        for n, title in (("tfidf", "TF-IDF"), ("label_probs", "Опора на метках"),
                         ("e5_frozen", "e5 без дообучения"),
                         ("tiny2_frozen", "rubert-tiny2 без дообучения")):
            b = base["baselines"][n]["test"]
            L.append(f"| {title} | {b['macro_p10']:.4f} | {b['macro_p1']:.4f} | "
                     f"{b['macro_hit10']:.4f} |")
    L.append(f"| **Дообученная e5** | **{t['macro_p10']:.4f}** | {t['macro_p1']:.4f} | "
             f"{t['macro_hit10']:.4f} |")
    L.append("\nЭто один прогон: вывод «обходит» делается только по правилу трёх "
             "условий на пяти seed (раздел 5l).\n")
    if m["lr_search"]:
        L += ["## Подбор lr\n", "Пробы по три эпохи смещают выбор в пользу больших lr, "
              "потому что малым нужно больше шагов.\n", "| lr | val macro p@10 |",
              "|---|---|"]
        L += [f"| {g['lr']:g}{' ← выбрано' if g['lr'] == m['lr'] else ''} | "
              f"{g['val_macro_p10']:.4f} |" for g in m["lr_search"]]
    L += [f"\n## По эпохам\n", f"Выбрана эпоха {m['best_epoch']} из {m['epochs']} по "
          f"валидации.\n", "| Эпоха | loss | val macro p@10 |", "|---|---|---|"]
    L += [f"| {h['epoch']}{' ←' if h['epoch'] == m['best_epoch'] else ''} | "
          f"{h['train_loss']:.4f} | {h['val_macro_p10']:.4f} |" for h in m["history"]]
    L.append(f"\nОбучение {m['train_seconds']:.0f} с по настенным часам.\n")
    path.write_text(with_preamble("\n".join(L), m["synthetic"]), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
