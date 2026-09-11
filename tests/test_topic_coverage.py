#!/usr/bin/env python3
"""Регресс-тест покрытия тем.

    .venv/bin/python -m tests.test_topic_coverage            # проверить
    .venv/bin/python -m tests.test_topic_coverage --update   # принять новый эталон

Падает в двух случаях:
  1) доля «прочего» внутри класса problem выросла хотя бы по одному региону;
  2) число строк в любой из тем (класс problem, вся таблица) изменилось
     больше чем на TOPIC_TOL строк — то есть строки переехали между
     темами. Такой переезд может быть и исправлением, и регрессией;
     тест не различает, а заставляет посмотреть и принять осознанно.
Эталон — tests/topic_coverage_baseline.json.

Смысл: правки маппинга должны ловиться автоматически. Две регрессии
были найдены глазами: потерянный ключ «дорожн» (Костанай 0.7% -> 17.4%)
и «Прорыв трубопровода» — 428 строк, ушедших из воды в электроснабжение
через ключ «провод». Вторую первая версия теста не видела: «прочее» не
изменилось, строки переехали между двумя настоящими темами.
"""
import json, sys
from pathlib import Path
import pandas as pd

BASE = Path("tests/topic_coverage_baseline.json")
TOL = 0.001      # 0.1 п.п. — шум округления, не регресс
TOPIC_TOL = 0     # переезд строк между темами — любой, без допуска

def measure():
    src = Path("data/unified.parquet")
    if not src.exists():
        sys.exit("нет data/unified.parquet — сначала src/adapters/build_unified.py "
                 "и src/topic_mapping.py")
    df = pd.read_parquet(src, columns=["region", "appeal_class", "topic"])
    other = {}
    for r, g in df.groupby("region", sort=False):
        pr = g[g.appeal_class == "problem"]
        other[r] = round(float((pr.topic == "прочее").mean()) if len(pr) else 0.0, 6)
    topics = {t: int(n) for t, n in
              df[df.appeal_class == "problem"].topic.value_counts().items()}
    return {"other_share": other, "topic_counts": topics}

def main():
    cur = measure()
    if "--update" in sys.argv:
        BASE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"эталон обновлён: {BASE}")
        for r, v in cur["other_share"].items():
            print(f"  прочее  {r:32s} {100*v:6.2f}%")
        for t, n in sorted(cur["topic_counts"].items(), key=lambda x: -x[1]):
            print(f"  тема    {t:36s} {n:8d}")
        return 0
    if not BASE.exists():
        sys.exit(f"нет эталона {BASE} — запустите с --update")
    old = json.loads(BASE.read_text(encoding="utf-8"))
    if "other_share" not in old:                     # эталон старого формата
        old = {"other_share": old, "topic_counts": {}}
    fail = False

    print("1) Доля «прочего» внутри problem по регионам")
    print(f"{'регион':32s} {'эталон':>8s} {'сейчас':>8s} {'дельта':>9s}")
    for r, v in cur["other_share"].items():
        o = old["other_share"].get(r)
        if o is None:
            print(f"{r:32s} {'—':>8s} {100*v:7.2f}%  новый регион"); continue
        d = v - o; flag = ""
        if d > TOL: flag, fail = "  РЕГРЕСС", True
        elif d < -TOL: flag = "  улучшение"
        print(f"{r:32s} {100*o:7.2f}% {100*v:7.2f}% {100*d:+8.2f}п.п.{flag}")

    print("\n2) Число строк по темам (problem)")
    if not old["topic_counts"]:
        print("   в эталоне нет разбивки по темам — примите эталон через --update")
    else:
        moved = []
        for t in sorted(set(cur["topic_counts"]) | set(old["topic_counts"])):
            o, n = old["topic_counts"].get(t, 0), cur["topic_counts"].get(t, 0)
            if abs(n - o) > TOPIC_TOL:
                moved.append((t, o, n))
        if moved:
            fail = True
            print(f"{'тема':36s} {'эталон':>8s} {'сейчас':>8s} {'разница':>8s}")
            for t, o, n in sorted(moved, key=lambda x: -abs(x[2] - x[1])):
                print(f"{t:36s} {o:8d} {n:8d} {n-o:+8d}  ПЕРЕЕЗД")
        else:
            print("   без изменений")

    if fail:
        print("\nПРОВАЛ. Если изменения осознанные и проверены — примите эталон: --update")
        return 1
    print("\nОК: регресса нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
