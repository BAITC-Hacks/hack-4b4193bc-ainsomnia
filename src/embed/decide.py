#!/usr/bin/env python3
"""Правило трёх условий (раздел 5l CLAUDE.md) — зафиксировано до прогонов.

    .venv/bin/python -m src.embed.decide \
        --baselines reports/embed/<базовые линии>/metrics.json \
        --runs "reports/embed/massive-multilingual-e5-small-*/metrics.json"

Дообучение ОБХОДИТ базовую линию, только если выполнены все три условия:
  1. среднее macro p@10 по seed выше неё больше чем на 2 ст. откл. разброса;
  2. худший seed тоже выше неё;
  3. 95% интервал парного бутстрэпа (seed 42 минус базовая линия, по одним и
     тем же запросам теста) не содержит нуля.
Иначе — неразличимы в пределах шума. Применяется к двум сравнениям:
против e5 без дообучения и против опоры на метках; второе главное.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from src.embed.retrieval import paired_bootstrap

OUT = Path("reports/embed")
COMPARE = (("label_probs", "опора на метках"), ("e5_frozen", "e5 без дообучения"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--runs", required=True, help="glob по metrics.json прогонов seed")
    ap.add_argument("--anchor-seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    base = json.loads(Path(a.baselines).read_text(encoding="utf-8"))
    bq = np.load(base["perquery"], allow_pickle=True)
    runs = [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in sorted(glob.glob(a.runs))]
    runs = [r for r in runs if r.get("kind") == "finetuned"]
    seeds = sorted(r["seed"] for r in runs)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"повтор seed среди прогонов: {seeds} — по правилу по одному на seed")
    if len(runs) < 2:
        raise ValueError("для разброса по seed нужно хотя бы два прогона")
    anchor = next(r for r in runs if r["seed"] == a.anchor_seed)
    fq = np.load(anchor["perquery"], allow_pickle=True)
    if not np.array_equal(fq["ids"], bq["ids"]):
        raise ValueError("запросы прогона и базовых линий не совпадают — парный "
                         "бутстрэп невозможен")
    y_q = bq["y_q"]
    v = [r["test"]["macro_p10"] for r in runs]
    mean, sd = st.mean(v), st.stdev(v)

    lines = [f"Прогонов {len(v)}, seed {', '.join(map(str, seeds))}: macro p@10 "
             f"среднее {mean:.4f}, min {min(v):.4f}, max {max(v):.4f}, "
             f"ст. откл. {sd:.4f}"]
    verdicts = {}
    for key, title in COMPARE:
        b = base["baselines"][key]["test"]["macro_p10"]
        c1 = mean - b > 2 * sd
        c2 = min(v) > b
        lo, hi = paired_bootstrap(fq["finetuned"], bq[key], y_q, n=a.n_boot)
        c3 = lo > 0
        ok = c1 and c2 and c3
        verdicts[key] = {"baseline": b, "gap_mean": mean - b, "gap_in_sd":
                         (mean - b) / sd if sd else None, "worst_gap": min(v) - b,
                         "boot_ci": [lo, hi], "c1": c1, "c2": c2, "c3": c3,
                         "beats": ok}
        lines += ["", f"ПРОТИВ: {title} ({b:.4f})",
                  f"  1. среднее выше больше чем на 2 ст. откл.: разрыв {mean - b:+.4f}, "
                  f"порог {2 * sd:.4f} — {'да' if c1 else 'НЕТ'}",
                  f"  2. худший seed выше: {min(v) - b:+.4f} — {'да' if c2 else 'НЕТ'}",
                  f"  3. парный бутстрэп seed {a.anchor_seed}, 95%: [{lo:+.4f}; {hi:+.4f}]"
                  f" — {'ноль вне интервала' if c3 else 'НОЛЬ ВНУТРИ'}",
                  f"  ВЕРДИКТ: {'ОБХОДИТ' if ok else 'неразличимы в пределах шума'}"]
    print("\n".join(lines))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(json.dumps(
        {"runs": [r["run"] for r in runs], "seeds": seeds, "macro_p10": v, "mean": mean,
         "sd": sd, "baselines_run": base["run"], "verdicts": verdicts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "decision.md").write_text(
        "# Правило трёх условий — решение\n\n```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
