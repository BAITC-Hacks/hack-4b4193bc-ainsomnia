"""Применить сохранённый классификатор к новому CSV (раздел 5k/5n).

    .venv/bin/python -m src.finetune.predict \
      --model-dir models/finetune/<прогон> --csv data/synth/test_c.csv \
      --text-col text --id-col id --label-col label --synthetic \
      --output models/synth/classifier_c.csv --report reports/synth/classifier_c.md

Метки загружаются только из label_map.json сохранённой модели. Построчный
результат идёт в models/, агрегаты — в reports/. Свободные тексты не печатаются.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finetune import pipeline as P
from src.finetune.train import encode, pick_device, predict
from src.synth.preamble import (PREAMBLE, c_caveat, is_template_set,
                                require_flag, with_preamble)
from src.synth.split import effective_size


def run(args) -> dict:
    require_flag(args.csv, args.synthetic)
    model_dir = Path(args.model_dir)
    label_map = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    if sorted(label_map.values()) != list(range(len(names))):
        raise ValueError("label_map.json должен содержать непрерывные индексы с нуля")
    config = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    df = pd.read_csv(args.csv)
    required = {args.text_col} | {x for x in (args.id_col, args.label_col) if x}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv}: нет колонок {sorted(missing)}")
    texts = df[args.text_col].astype(str).str.strip()
    if texts.eq("").any() or df[args.text_col].isna().any():
        raise ValueError("в CSV есть пустые тексты")
    ids = df[args.id_col].astype(str) if args.id_col else pd.Series(range(len(df)))
    if ids.duplicated().any():
        raise ValueError("повторные id в CSV")
    truth = None
    if args.label_col:
        truth = df[args.label_col].astype(str)
        unknown = sorted(set(truth) - set(label_map))
        if unknown:
            raise ValueError(f"истинные метки отсутствуют в модели: {unknown}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    device = pick_device(args.cpu)
    model.to(device)
    loader = DataLoader(encode(tok, texts, np.zeros(len(texts), dtype=int),
                               config["max_len"]), batch_size=args.batch_size)
    pred_idx, probs = predict(model, loader, device)
    pred = [names[i] for i in pred_idx]
    output = Path(args.output)
    if output.suffix != ".csv" or not output.resolve().is_relative_to(
            Path("models").resolve()):
        raise ValueError("построчный результат должен быть CSV внутри models/")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "y_pred": pred,
                  "score": np.round(probs.max(axis=1), 6)}).to_csv(output, index=False)
    result = {"model_dir": str(model_dir), "csv": args.csv, "n": len(df),
              "classes_model": len(names), "output": str(output)}
    caveat = ""
    if is_template_set(ids):
        result["effective_texts"] = effective_size(texts)
        caveat = c_caveat(result["effective_texts"], len(df))
        result["effective_note"] = caveat
    if args.label_col:
        y_true = truth.map(label_map).to_numpy()
        result["metrics"] = P.evaluate(y_true, pred_idx, names)
        result["per_class"] = P.per_class(y_true, pred_idx, names).to_dict("records")
    if args.report:
        if not args.label_col:
            raise ValueError("--report требует --label-col")
        report = Path(args.report)
        if report.suffix != ".md":
            raise ValueError("--report должен заканчиваться на .md")
        report.parent.mkdir(parents=True, exist_ok=True)
        m = result["metrics"]
        body = "\n".join([
            f"# Оценка сохранённого классификатора: {Path(args.csv).stem}",
            "",
            f"Модель: `{model_dir}`. Текстов: {len(df)}; классов в тесте: "
            f"{m['classes_true']}.",
            "",
            f"macro-F1 {m['macro_f1']:.4f}; accuracy {m['accuracy']:.4f}; "
            f"micro-F1 {m['micro_f1']:.4f}"
            + (f" — {caveat}." if caveat else "."),
            "",
            "Это одна оценка сохранённой модели; сравнения подходов и разброса "
            "по seed она не заменяет.",
            "",
        ])
        report.write_text(with_preamble(body, args.synthetic), encoding="utf-8")
        metrics_path = report.with_suffix(".json")
        metrics_path.write_text(json.dumps(
            ({"scope": PREAMBLE} if args.synthetic else {}) | result,
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["report"] = str(report)
        result["metrics_json"] = str(metrics_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--id-col")
    ap.add_argument("--label-col")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    result = run(args)
    print(f"применено: {result['n']} текстов; выход: {result['output']}")
    if "metrics" in result:
        print(f"macro-F1: {result['metrics']['macro_f1']:.4f}"
              + (f" — {result['effective_note']}" if "effective_note" in result else ""))


if __name__ == "__main__":
    main()
