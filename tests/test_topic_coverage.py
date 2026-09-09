#!/usr/bin/env python3
"""Регресс-тест покрытия тем.

    .venv/bin/python -m tests.test_topic_coverage            # проверить
    .venv/bin/python -m tests.test_topic_coverage --update   # принять новый эталон

Падает, если доля «прочего» внутри класса problem выросла хотя бы
по одному региону. Эталон — tests/topic_coverage_baseline.json.

Смысл: правки маппинга должны ловиться автоматически. Регрессия с
потерянным ключом «дорожн» (Костанай 0.7% -> 17.4%) была найдена глазами,
второй раз так везти не должно.
"""
import json, sys
from pathlib import Path
import pandas as pd

BASE = Path("tests/topic_coverage_baseline.json")
TOL = 0.001  # 0.1 п.п. — шум округления, не регресс

def measure():
    src = Path("data/unified.parquet")
    if not src.exists():
        sys.exit("нет data/unified.parquet — сначала src/adapters/build_unified.py "
                 "и src/topic_mapping.py")
    df = pd.read_parquet(src, columns=["region", "appeal_class", "topic"])
    out = {}
    for r, g in df.groupby("region", sort=False):
        pr = g[g.appeal_class == "problem"]
        out[r] = round(float((pr.topic == "прочее").mean()) if len(pr) else 0.0, 6)
    return out

def main():
    cur = measure()
    if "--update" in sys.argv:
        BASE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"эталон обновлён: {BASE}")
        for r, v in cur.items():
            print(f"  {r:32s} {100*v:6.2f}%")
        return 0
    if not BASE.exists():
        sys.exit(f"нет эталона {BASE} — запустите с --update")
    old = json.loads(BASE.read_text(encoding="utf-8"))

    bad, new_regions = [], []
    print(f"{'регион':32s} {'эталон':>8s} {'сейчас':>8s} {'дельта':>9s}")
    for r, v in cur.items():
        if r not in old:
            new_regions.append(r)
            print(f"{r:32s} {'—':>8s} {100*v:7.2f}%  новый регион")
            continue
        d = v - old[r]
        flag = ""
        if d > TOL:
            bad.append((r, old[r], v, d)); flag = "  РЕГРЕСС"
        elif d < -TOL:
            flag = "  улучшение"
        print(f"{r:32s} {100*old[r]:7.2f}% {100*v:7.2f}% {100*d:+8.2f}п.п.{flag}")

    gone = set(old) - set(cur)
    if gone:
        print(f"\nИсчезли регионы: {', '.join(sorted(gone))}")
    if bad:
        print(f"\nПРОВАЛ: доля «прочего» выросла по {len(bad)} регион(ам):")
        for r, o, n, d in bad:
            print(f"  {r}: {100*o:.2f}% -> {100*n:.2f}% (+{100*d:.2f} п.п.)")
        print("\nЕсли рост осознанный — примите эталон: --update")
        return 1
    print("\nОК: регресса нет.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
