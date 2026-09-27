#!/usr/bin/env python3
"""Сборка data/unified.parquet из порегиональных parquet.

    .venv/bin/python -m src.adapters.build_unified

Предварительно должен быть выполнен src/adapters/adapters.py, который
кладёт data/by_region/<регион>.parquet.

Свободный текст и ПДн в сводную таблицу не переносятся — перед записью
выполняется проверка, и при обнаружении запрещённой колонки сборка падает.
"""
import sys
from pathlib import Path
import pandas as pd
from src import paths

BANNED = {"com_exp", "result", "request_subject", "appeal_address", "street",
          "full_name", "applicant_number", "operator", "xcoordinate", "ycoordinate"}
SRC = paths.BY_REGION
OUT = paths.UNIFIED

def main():
    paths.require_writable()
    paths.require_source_marker(paths.SOURCE_MARK)
    parts = sorted(SRC.glob("*.parquet"))
    if not parts:
        sys.exit(f"нет файлов в {SRC} — сначала запустите src/adapters/adapters.py")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    leak = BANNED & set(df.columns)
    if leak:
        sys.exit(f"ОСТАНОВ: в сводной таблице запрещённые колонки: {sorted(leak)}")

    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{OUT} — {len(df)} строк, {OUT.stat().st_size/1e6:.1f} МБ\n")
    print(f"{'регион':32s} {'строк':>8s}  диапазон created_at        пусто category")
    for r, g in df.groupby("region", sort=False):
        print(f"{r:32s} {len(g):8d}  {str(g.created_at.min())[:10]} → "
              f"{str(g.created_at.max())[:10]}  {100*g.category.isna().mean():8.2f}%")
    print(f"\nИТОГО {len(df)}")
    print("\nдоля пустых в обязательных полях:")
    for c in ("created_at", "region", "category"):
        print(f"  {c:12s} {100*df[c].isna().mean():.3f}%")
    print("заполненность опциональных:")
    for c in ("district", "executor", "status", "sla_breach"):
        print(f"  {c:12s} {100*df[c].notna().mean():5.1f}%")

if __name__ == "__main__":
    main()
