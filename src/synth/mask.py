"""Диагностика опоры на ключевые слова: маскирование тестов (5n)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.topic_mapping import RULES, norm

ROOT = Path("tests/fixtures/synth")
OUT = Path("data/synth")
WORD = re.compile(r"[^\W_]+", re.UNICODE)
TOKEN = "[МАСКА]"


def _stems():
    result = set()
    for _, keys in RULES:
        for key in keys:
            for part in WORD.findall(norm(key.removeprefix("^"))):
                if len(part) >= 3:
                    result.add(part)
    return result


STEMS = _stems()


def mask(text: str, seed: str | None) -> tuple[str, int]:
    words = {w for w in WORD.findall(norm(seed or "")) if len(w) >= 4}
    spans = []
    for m in WORD.finditer(text):
        word = norm(m.group())
        if word in words or any((stem in word if len(stem) >= 4 else stem == word)
                                for stem in STEMS):
            spans.append((m.start(), m.end()))
    out, cursor = [], 0
    for start, end in spans:
        out.extend((text[cursor:start], TOKEN))
        cursor = end
    out.append(text[cursor:])
    return "".join(out), len(spans)


def main() -> None:
    a = [json.loads(line) for line in (ROOT / "corpus_a.jsonl").read_text(
        encoding="utf-8").splitlines()]
    c = [json.loads(line) for line in (ROOT / "templates_c.jsonl").read_text(
        encoding="utf-8").splitlines()]
    counts = {}
    for name, rows in (
        ("test_a", [r for r in a if r["split"] == "test"]),
        ("seed_holdout", [r for r in a if r["split"] == "seed_holdout"]),
        ("test_c", c),
    ):
        output = []
        masked = 0
        for row in rows:
            text, count = mask(row["text"], row.get("seed"))
            output.append({"id": row["id"], "text": text, "label": row["label"],
                           "masked_words": count})
            masked += count
        pd.DataFrame(output).to_csv(OUT / f"{name}_masked.csv", index=False)
        counts[name] = {"rows": len(rows), "masked_words": masked,
                        "rows_without_mask": sum(r["masked_words"] == 0 for r in output)}
    print(counts)


if __name__ == "__main__":
    main()
