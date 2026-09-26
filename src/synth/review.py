"""Фиксированная выборка для ручного чтения 200 A и 100 C (раздел 5n).

    .venv/bin/python -m src.synth.review 0    # строки 0–24
    .venv/bin/python -m src.synth.review 25   # строки 25–49
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.topic_mapping import TOPICS

ROOT = Path("tests/fixtures/synth")
SEED = 42
BATCH = 25


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample() -> list[dict]:
    rng = np.random.default_rng(SEED)
    a = _load(ROOT / "corpus_a.jsonl")
    c = _load(ROOT / "templates_c.jsonl")
    selected = []
    for topic_i, topic in enumerate(TOPICS):
        for source, rows, quotas in (
            ("A", a, {"ru": 7, "kk": 4, "kk-ru": 2}),
            ("C", c, {"ru": 4, "kk": 2}),
        ):
            if source == "A" and topic_i < 5:
                quotas["ru"] += 1
            if source == "C" and topic_i < 10:
                quotas["kk"] += 1
            for lang, count in quotas.items():
                pool = [r for r in rows if r["label"] == topic and r["lang"] == lang]
                indices = rng.choice(len(pool), size=count, replace=False)
                selected.extend(pool[int(i)] for i in indices)
    assert len([r for r in selected if r["id"].startswith("A-")]) == 200
    assert len([r for r in selected if r["id"].startswith("C-")]) == 100
    return selected


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("использование: python -m src.synth.review START")
    start = int(sys.argv[1])
    rows = sample()
    for r in rows[start:start + BATCH]:
        print(f"{r['id']} | {r['label']} | {r['lang']}")
        print(r["text"])
        print()


if __name__ == "__main__":
    main()
