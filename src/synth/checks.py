"""Проверки синтетического корпуса — раздел 5n CLAUDE.md.

    .venv/bin/python -m src.synth.checks            # полный корпус генератора A
    .venv/bin/python -m src.synth.checks --partial  # во время написания пачек

Что проверяется у каждого текста:
  * ПДн — шаблоны src/export.py (телефон, 12 цифр, адрес «дом … кв») плюс 10–11
    цифр подряд, почта, ссылки, «ИИН»/«ЖСН»; имена по словарю, отчества по
    окончаниям, слова с фамильными окончаниями вне белого списка (улицы и места
    из листа заданий, известные слова);
  * опорное значение и название темы дословно — только если в задании нет
    флага explicit. Однословные значения («Лифт», «Пожар») не проверяются:
    без этого слова текст о лифте не написать;
  * длина — в своём диапазоне с допуском;
  * пропущенные и лишние id, точные дубли.

Срабатывание ПДн — текст удаляется и пишется заново (5n), не правится.
Синтетические тексты можно печатать; совпадения ПДн всё равно маскируются.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.checks.privacy import pii_hits
from src.topic_mapping import norm

RAW = Path("data/synth/raw_a")
TASKS = Path("data/synth/tasks_a.jsonl")
LEN_RANGE = {"короткий": (40, 110), "средний": (110, 260), "длинный": (260, 520)}
LEN_SLACK = 0.25                 # допуск: генератор не считает символы точно

def seed_copy(text, task, class_name):
    """Опорное значение или название темы дословно — нарушение без explicit."""
    if task["explicit"]:
        return None
    t = norm(text)
    for phrase in (task["seed"], class_name):
        p = norm(phrase)
        if len(p.split()) >= 2 and p in t:
            return phrase
    return None


def load_tasks():
    return {r["id"]: r for r in map(json.loads, TASKS.open(encoding="utf-8"))}


def load_raw():
    out, duplicates = {}, []
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["id"] in out:
                    duplicates.append(r["id"])
                out[r["id"]] = r["text"].strip()
    return out, duplicates


def check(raw, tasks, allow_extra=()):
    """Словарь id -> список проблем. allow_extra — места/улицы в белый список."""
    problems, seen = {}, {}
    for i, text in raw.items():
        t = tasks.get(i)
        if t is None:
            problems.setdefault(i, []).append("id нет в листе заданий")
            continue
        p = [f"ПДн {h}" for h in pii_hits(text, allow_extra)]
        c = seed_copy(text, t, t["label"])
        if c:
            p.append(f"дословно «{c}»")
        lo, hi = LEN_RANGE[t["length"]]
        if not lo * (1 - LEN_SLACK) <= len(text) <= hi * (1 + LEN_SLACK):
            p.append(f"длина {len(text)} вне {t['length']} {lo}–{hi}")
        key = norm(text)
        if key in seen:
            p.append(f"дубль {seen[key]}")
        seen[key] = i
        if p:
            problems[i] = p
    return problems


def main():
    tasks = load_tasks()
    raw, duplicates = load_raw()
    extra = {w for r in tasks.values() for s in (r["place"], r["street"] or "")
             for w in re.split(r"[\s,.-]+", s) if w}
    probs = check(raw, tasks, extra)
    partial = sys.argv[1:] == ["--partial"]
    if sys.argv[1:] and not partial:
        raise SystemExit("использование: python -m src.synth.checks [--partial]")
    done = {i[:4] for i in raw}
    missing = [i for i in tasks if i not in raw and (not partial or i[:4] in done)]
    print(f"написано {len(raw)} из {len(tasks)}; пропущено: {len(missing)}; повторных id: {len(duplicates)}")
    if missing:
        print("  пропуски:", ", ".join(missing[:20]))
    if duplicates:
        print("  повторные id:", ", ".join(duplicates[:20]))
    kinds = {}
    for v in probs.values():
        for x in v:
            kinds[x.split(":")[0].split(" «")[0]] = kinds.get(x.split(":")[0].split(" «")[0], 0) + 1
    print("проблем по видам:", kinds or "нет")
    for i, v in list(probs.items())[:60]:
        print(f"  {i}: {'; '.join(v)}")
    return int(bool((missing and not partial) or duplicates or probs))


if __name__ == "__main__":
    sys.exit(main())
