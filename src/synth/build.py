"""Сборка проверенного корпуса и CSV для существующих конвейеров, раздел 5n.

    .venv/bin/python -m src.synth.build

Сборка строго требует все 3 150 строк A и 1 200 строк C. Исходные лист
заданий и рабочие пачки A остаются в data/synth/; в git уходит только
проверенный синтетический корпус tests/fixtures/synth/.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.synth import checks
from src.synth import privacy
from src.synth.noise import apply_noise
from src.synth.preamble import PREAMBLE
from src.synth.split import assign, groups
from src.topic_mapping import TOPICS

OUT = Path("tests/fixtures/synth")
DATA = Path("data/synth")
EXPECTED_A = 3150
EXPECTED_C = 1200
# Claude Code написал первые 3 045 строк; Codex завершил класс 14 и переписал
# одну почти-дублирующую строку, чтобы не переносить происшествие в seed_holdout.
CODEX_A_IDS = {"A-09-177"} | {f"A-14-{i:03d}" for i in range(105, 210)}
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    tasks = checks.load_tasks()
    raw, duplicates = checks.load_raw()
    missing = sorted(set(tasks) - set(raw))
    extra = {w for r in tasks.values() for s in (r["place"], r["street"] or "")
             for w in re.split(r"[\s,.-]+", s) if w}
    problems = checks.check(raw, tasks, extra)
    if len(tasks) != EXPECTED_A or len(raw) != EXPECTED_A or duplicates or missing or problems:
        raise ValueError(
            f"A не готов: заданий {len(tasks)}, текстов {len(raw)}, "
            f"повторных id {len(duplicates)}, пропусков {len(missing)}, "
            f"ошибок проверки {len(problems)}"
        )
    source = [{**tasks[i], "text": raw[i], "generator": "session",
               "author": "Codex" if i in CODEX_A_IDS else "Claude Code"}
              for i in sorted(tasks)]
    noisy = apply_noise(source)
    if any(checks.pii_hits(r["text"], extra) for r in noisy):
        raise ValueError("ПДн после внесения шума")
    a_rows, split_summary = assign(noisy)

    c_path = OUT / "templates_c.jsonl"
    c_rows = _read_jsonl(c_path)
    if len(c_rows) != EXPECTED_C:
        raise ValueError(f"C не готов: {len(c_rows)} вместо {EXPECTED_C}")
    if len({r["id"] for r in c_rows}) != EXPECTED_C:
        raise ValueError("повторные id в C")
    if len({r["text"] for r in c_rows}) != EXPECTED_C:
        raise ValueError("повторные тексты в C")
    if any(checks.pii_hits(r["text"]) for r in c_rows):
        raise ValueError("ПДн в C")
    expected_counts = {(topic, lang): 40 for topic in TOPICS for lang in ("ru", "kk")}
    actual_counts = Counter((r["label"], r["lang"]) for r in c_rows)
    if actual_counts != expected_counts:
        raise ValueError("нарушены объёмы классов или языков C")
    c_groups, c_near_pairs = groups(c_rows)
    if any(len({(c_rows[i]["label"], c_rows[i]["lang"]) for i in g}) > 1
           for g in c_groups):
        raise ValueError("почти-дубли C смешивают темы или языки — "
                         "запись о причине повторов в описи неверна")
    c_effective = {
        "near_duplicate_pairs": c_near_pairs,
        "effective_texts": len(c_groups),
        "effective_by_language": dict(sorted(Counter(
            c_rows[g[0]]["lang"] for g in c_groups).items())),
        "cause": "по построению: 3 ситуации × 4 рамки на класс и язык, в трёх "
                 "рамках из четырёх меняется только город; группы не смешивают "
                 "разные ситуации или рамки",
    }
    combined_groups, combined_near_pairs = groups(a_rows + c_rows)
    cross_groups = sum(
        any(i < len(a_rows) for i in group)
        and any(i >= len(a_rows) for i in group)
        for group in combined_groups
    )
    if cross_groups:
        raise ValueError(f"почти-дубли A и C: {cross_groups} групп")
    overlap = privacy.check(a_rows + c_rows)
    if overlap["overlap_count"]:
        raise ValueError(f"общих цепочек из 8 слов с реальным текстом: "
                         f"{overlap['overlap_count']} синтетических строк")

    OUT.mkdir(parents=True, exist_ok=True)
    a_path = OUT / "corpus_a.jsonl"
    _write_jsonl(a_path, a_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(a_rows)
    frame[frame.split.isin(("train", "validation", "test"))].to_csv(
        DATA / "train_eval.csv", index=False)
    frame[frame.split == "seed_holdout"].to_csv(
        DATA / "seed_holdout.csv", index=False)
    pd.DataFrame(c_rows).to_csv(DATA / "test_c.csv", index=False)

    counts = defaultdict(dict)
    for (label, lang), n in Counter((r["label"], r["lang"]) for r in a_rows).items():
        counts[label][lang] = n
    noise = {
        "typo": {"count": sum(r["noise_typo"] for r in a_rows),
                 "denominator": sum(r["register"] == "житель" for r in a_rows)},
        "no_punct": {"count": sum(r["noise_no_punct"] for r in a_rows),
                     "denominator": sum(r["register"] == "оператор" for r in a_rows)},
        "kk_letters": {
            "count": sum(r["noise_kk_letters"] for r in a_rows),
            "denominator": sum(r["register"] == "житель"
                               and r["lang"].startswith("kk") for r in a_rows)},
    }
    hashes = {"corpus_a.jsonl": _sha256(a_path),
              "templates_c.jsonl": _sha256(c_path)}
    previous = OUT / "manifest.json"
    old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    review = (old.get("privacy", {}).get("manual_review", "не проведён")
              if old.get("sha256") == hashes else "не проведён")
    manifest = {
        "scope": PREAMBLE,
        "seed": 42,
        "classes": TOPICS,
        "a_rows": len(a_rows),
        "c_rows": len(c_rows),
        "authors": {"A": {"Claude Code": len(a_rows) - len(CODEX_A_IDS),
                          "Codex": len(CODEX_A_IDS)},
                    "C": {"Codex": len(c_rows)}},
        "a_by_class_language": dict(sorted(counts.items())),
        "c_per_class_language": 40,
        "c_near_duplicates": c_effective,
        "noise": noise,
        "split": split_summary,
        "a_c_near_duplicate_groups": cross_groups,
        "combined_near_duplicate_pairs": combined_near_pairs,
        "privacy": {
            "automated_hits": 0,
            "removed_and_rewritten": None,
            "removal_count_note": "Пачки A были написаны до счётчика удалений; число неизвестно.",
            "manual_review": review,
            "real_text_overlap_8_words": overlap["overlap_count"],
            "real_values_scanned": overlap["real_values_scanned"],
        },
        "sha256": hashes,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    try:
        manifest = build()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"A: {manifest['a_rows']} · C: {manifest['c_rows']}")
    print("разбиение:", manifest["split"]["split_counts"])
    print("шум:", manifest["noise"])
    print("Проверки автоматические: 0 срабатываний ПДн")


if __name__ == "__main__":
    main()
