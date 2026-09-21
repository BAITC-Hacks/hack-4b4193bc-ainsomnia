#!/usr/bin/env python3
"""Внешние корпуса -> CSV контрактного вида (раздел 5k CLAUDE.md).

    .venv/bin/python -m src.finetune.fetch massive
    .venv/bin/python -m src.finetune.fetch grnti

Выход: data/external/<корпус>.csv с колонками id, text, label, split.
Каталог под .gitignore: корпуса скачиваются, а не хранятся в репозитории.

Пайплайн обучения об этих корпусах НЕ знает: он читает произвольный CSV по
именам колонок. Здесь и только здесь лежит знание о том, как устроен каждый
источник — та же схема, что у адаптеров регионов в src/adapters/.

Метка кладётся СТРОКОЙ, а не индексом: кодирование в индексы — дело пайплайна,
иначе при смене корпуса классы молча переедут.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT = Path("data/external")

# corpus -> (hub id, config, колонка текста, колонка метки, сплиты)
SOURCES = {
    "massive": ("mteb/amazon_massive_intent", "ru", "text", "label",
                {"train": "train", "validation": "validation", "test": "test"}),
    "grnti": ("ai-forever/ru-scibench-grnti-classification", "default",
              "text", "label_text", {"train": "train", "test": "test"}),
}


def fetch(name):
    from datasets import load_dataset
    hub_id, config, text_col, label_col, splits = SOURCES[name]
    print(f"{name}: {hub_id} (конфиг {config})")
    frames = []
    for split, hub_split in splits.items():
        ds = load_dataset(hub_id, config, split=hub_split)
        df = ds.to_pandas()
        missing = {text_col, label_col} - set(df.columns)
        if missing:
            raise ValueError(f"{hub_id}/{hub_split}: нет колонок {missing}; "
                             f"есть {list(df.columns)}")
        out = pd.DataFrame({"text": df[text_col].astype(str),
                            "label": df[label_col].astype(str),
                            "split": split})
        print(f"  {split:<10s} {len(out):>7,} строк, {out.label.nunique():>3} классов"
              .replace(",", " "))
        frames.append(out)
    all_ = pd.concat(frames, ignore_index=True)
    all_.insert(0, "id", [f"{name}-{i}" for i in range(len(all_))])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    all_.to_csv(path, index=False)
    ln = all_.text.str.len()
    print(f"  -> {path}: {len(all_):,} строк, {all_.label.nunique()} классов, "
          f"длина текста медиана {ln.median():.0f}, 95-й процентиль "
          f"{ln.quantile(0.95):.0f}, максимум {ln.max()}".replace(",", " "))
    return path


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SOURCES:
        print(f"Использование: python -m src.finetune.fetch "
              f"{{{'|'.join(SOURCES)}}}")
        return 1
    fetch(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
