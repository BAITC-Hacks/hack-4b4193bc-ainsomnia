"""Фиксированный шум синтетического корпуса A, раздел 5n CLAUDE.md."""
from __future__ import annotations

import re

import numpy as np

SEED = 42
TYPO_SHARE = 0.20
NO_PUNCT_SHARE = 0.20
KK_REPLACE_SHARE = 0.40

WORD = re.compile(r"\b[а-яёәғқңөұүһі]{5,}\b")
PUNCT = re.compile(r"[.,!?;:«»„“”()]")
KK_TO_RU = str.maketrans({
    "қ": "к", "ө": "о", "ү": "у", "ұ": "у", "ң": "н",
    "ғ": "г", "ә": "а", "і": "и",
    "Қ": "К", "Ө": "О", "Ү": "У", "Ұ": "У", "Ң": "Н",
    "Ғ": "Г", "Ә": "А", "І": "И",
})
KK_LETTERS = set("қөүұңғәіҚӨҮҰҢҒӘІ")


def _choose(rng: np.random.Generator, ids: list[str], share: float) -> set[str]:
    count = round(len(ids) * share)
    return set(rng.choice(ids, count, replace=False).tolist()) if count else set()


def _typo(text: str, rng: np.random.Generator) -> str:
    words = [m for m in WORD.finditer(text)
             if any(a != b for a, b in zip(m.group(), m.group()[1:]))]
    if not words:
        raise ValueError("нет слова для опечатки")
    m = words[int(rng.integers(len(words)))]
    word = m.group()
    positions = [i for i in range(1, len(word) - 1) if word[i] != word[i + 1]]
    if not positions:
        positions = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
    i = int(rng.choice(positions))
    changed = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return text[:m.start()] + changed + text[m.end():]


def apply_noise(rows: list[dict]) -> list[dict]:
    """Возвращает копии строк с текстом после шума и тремя булевыми флагами."""
    from src.synth.checks import pii_hits

    rng = np.random.default_rng(SEED)
    allowed = {w for r in rows for s in (r["place"], r["street"] or "")
               for w in re.split(r"[\s,.-]+", s) if w}
    residents = [r["id"] for r in rows if r["register"] == "житель"]
    operators = [r["id"] for r in rows if r["register"] == "оператор"]
    kk_residents = [r["id"] for r in rows if r["register"] == "житель"
                    and r["lang"].startswith("kk")]
    kk_eligible = [r["id"] for r in rows if r["id"] in set(kk_residents)
                   and KK_LETTERS.intersection(r["text"])
                   and not pii_hits(r["text"].translate(KK_TO_RU), allowed)]
    kk_count = round(len(kk_residents) * KK_REPLACE_SHARE)
    if len(kk_eligible) < kk_count:
        raise ValueError(f"для замены казахских букв годятся только {len(kk_eligible)} "
                         f"из {kk_count} требуемых текстов")
    typo_ids = _choose(rng, residents, TYPO_SHARE)
    no_punct_ids = _choose(rng, operators, NO_PUNCT_SHARE)
    kk_ids = set(rng.choice(kk_eligible, kk_count, replace=False).tolist())
    out = []
    for row in rows:
        r = row.copy()
        for _ in range(100):
            text = r["text"]
            if r["id"] in typo_ids:
                text = _typo(text, rng)
            if r["id"] in no_punct_ids:
                text = " ".join(PUNCT.sub(" ", text).split())
            if r["id"] in kk_ids:
                if not KK_LETTERS.intersection(text):
                    raise ValueError(f"нет казахских букв для замены: {r['id']}")
                text = text.translate(KK_TO_RU)
            if not pii_hits(text, allowed):
                break
        else:
            raise ValueError(f"шум не прошёл проверку ПДн: {r['id']}")
        r["text"] = text
        r["noise_typo"] = r["id"] in typo_ids
        r["noise_no_punct"] = r["id"] in no_punct_ids
        r["noise_kk_letters"] = r["id"] in kk_ids
        out.append(r)
    return out
