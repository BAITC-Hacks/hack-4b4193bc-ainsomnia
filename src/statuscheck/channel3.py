"""Канал 3 раздела 5o — поток после закрытия.

    .venv/bin/python -m src.statuscheck.channel3

Это длительность повышенного потока, а не длительность аварии; расхождение
с закрытиями не доказывает отписки.

Для каждого всплеска (src.spikes через src.dashboard.load_events — одна
реализация): жалобы темы региона, поступившие в дни всплеска; дата, к которой
закрыто 80% из них; сколько дней после неё дневной поток темы выше фона; сколько
жалоб сверх фона за этот хвост. Фон — медиана окна на старте всплеска.
Службе всплеск засчитывается, если она ведёт половину его жалоб; отметка службе —
при 3+ таких всплесках и 5+ таких служб в теме региона.

Проверки на шум зафиксированы до расчёта (2026-09-27):
  сдвиг дат закрытия — 200 раз даты закрытия всплеска сдвигаются на случайные
  ±1…14 дней; если медианная ранговая корреляция фактических хвостов со
  сдвинутыми не меньше 0.9, порядок всплесков от закрытий не зависит — шум;
  те же даты год назад — при 10+ всплесках с прошлым годом: хвосты не длиннее
  прошлогодних (Манн–Уитни, односторонний, p ≥ 0.05) — шум. Где проверить
  нельзя — канал не засчитан: непроверенное не считается пройденным.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from src import paths
from src.statuscheck.frame import FRAME
from src.statuscheck.load import load
from src.statuscheck.observability import CELL_SERVICES, CH3_SHARE, CH3_SPIKES

OUT = paths.REPORTS_DIR / "5o"
REGIONS = ("Костанайская область", "Туркестанская область",
           "Восточно-Казахстанская область", "Алматинская область")
CLOSED_MIN, PLACEBO, SHIFT, RHO_NOISE, LASTYEAR_MIN, SEED = 0.8, 200, 14, 0.9, 10, 42
WORDING = ("Это длительность повышенного потока, а не длительность аварии; "
           "расхождение с закрытиями не доказывает отписки.")


def tail(series: pd.Series, start: pd.Timestamp, bg: float) -> tuple[int, float, bool]:
    """Дни от start до первого дня с потоком не выше фона; жалобы сверх фона."""
    s = series[series.index >= start]
    below = s[s <= bg]
    if below.empty:
        return len(s), float((s - bg).clip(lower=0).sum()), True
    end = below.index[0]
    part = s[s.index < end]
    return int((end - start).days), float((part - bg).clip(lower=0).sum()), False


def spikes(region: str, det: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    d = load(region)
    d = d[d.cls == "problem"]
    series = {t: g.set_index("день")["count"] for t, g in det[det.region == region].groupby("topic")}
    rows = []
    for _, r in ev[ev["регион"] == region].iterrows():
        day0, day1 = pd.Timestamp(r["дата"]), pd.Timestamp(r["конец"])
        m = (d.topic == r["тема"]) & (d.created.dt.normalize() >= day0) & \
            (d.created.dt.normalize() <= day1)
        g = d[m]
        if g.empty or g.closed.notna().mean() < CLOSED_MIN:
            continue
        d80 = g.closed.quantile(0.8).normalize()
        dom = (g.executor.value_counts(normalize=True) if g.executor.notna().any()
               else pd.Series(dtype=float))
        t, x, cens = tail(series[r["тема"]], d80, float(r["медиана окна"]))
        rows.append({"topic": r["тема"], "start": day0, "end": day1, "n": len(g),
                     "bg": float(r["медиана окна"]), "d80": d80, "tail_days": t, "excess": x,
                     "censored": cens,
                     "dominant": dom.index[0] if len(dom) and dom.iloc[0] >= CH3_SHARE else None,
                     "_closed": g.closed.to_numpy()})
    return pd.DataFrame(rows)


def placebo_shift(sp: pd.DataFrame, det: pd.DataFrame, region: str) -> dict:
    series = {t: g.set_index("день")["count"] for t, g in det[det.region == region].groupby("topic")}
    rng = np.random.default_rng(SEED)
    rhos = []
    for _ in range(PLACEBO):
        pt = []
        for r in sp.itertuples():
            off = int(rng.choice(np.r_[-SHIFT:0, 1:SHIFT + 1]))
            pt.append(tail(series[r.topic], r.d80 + pd.Timedelta(days=off), r.bg)[0])
        if len(set(pt)) > 1 and sp.tail_days.nunique() > 1:
            rhos.append(spearmanr(sp.tail_days, pt).statistic)
    med = float(np.median(rhos)) if rhos else None
    return {"median_rho": med, "noise": med is None or med >= RHO_NOISE}


def last_year(sp: pd.DataFrame, det: pd.DataFrame, region: str) -> dict:
    series = {t: g.set_index("день")["count"] for t, g in det[det.region == region].groupby("topic")}
    base = []
    for r in sp.itertuples():
        s = series[r.topic]
        a0 = r.start - pd.Timedelta(days=365)
        win = s[(s.index >= a0 - pd.Timedelta(days=28)) & (s.index < a0)]
        if len(win) < 28 or s.index.min() > a0 - pd.Timedelta(days=28):
            continue
        base.append((r.tail_days, tail(s, r.d80 - pd.Timedelta(days=365), float(win.median()))[0]))
    if len(base) < LASTYEAR_MIN:
        return {"spikes": len(base), "p": None, "noise": True, "checked": False}
    now, prev = zip(*base)
    p = float(mannwhitneyu(now, prev, alternative="greater").pvalue)
    return {"spikes": len(base), "p": p, "noise": p >= 0.05, "checked": True}


def run() -> list[dict]:
    from src.dashboard import load_events
    _, det, ev = load_events()
    out = []
    for region in REGIONS:
        sp = spikes(region, det, ev)
        dom = sp.dominant.dropna().value_counts()
        by_topic = (sp.dropna(subset=["dominant"]).groupby(["topic", "dominant"]).size()
                    .reset_index(name="k"))
        rankable = by_topic[by_topic.k >= CH3_SPIKES].groupby("topic").dominant.nunique()
        res = {"region": region, "spikes": int(len(sp)),
               "tail_median": float(sp.tail_days.median()) if len(sp) else None,
               "tail_p75": float(sp.tail_days.quantile(.75)) if len(sp) else None,
               "tail_max": int(sp.tail_days.max()) if len(sp) else None,
               "excess_median": float(sp.excess.median()) if len(sp) else None,
               "censored": int(sp.censored.sum()) if len(sp) else 0,
               "services_3plus": int((dom >= CH3_SPIKES).sum()),
               "topics_with_5_services": int((rankable >= CELL_SERVICES).sum()),
               "shift": placebo_shift(sp, det, region) if len(sp) > 2 else {"noise": True},
               "last_year": last_year(sp, det, region)}
        res["counted_region"] = not (res["shift"]["noise"] or res["last_year"]["noise"])
        res["counted_services"] = res["counted_region"] and res["topics_with_5_services"] > 0
        out.append(res)
    return out


def render(results: list[dict]) -> str:
    L = [FRAME, "", "# 5o, канал 3: поток после закрытия", "", f"**{WORDING}**", "",
         "| | " + " | ".join(r["region"] for r in results) + " |",
         "|---|" + "---:|" * len(results)]

    def row(title, f):
        L.append(f"| {title} | " + " | ".join(f(r) for r in results) + " |")

    row("Всплесков с закрытыми жалобами", lambda r: str(r["spikes"]))
    row("Хвост после 80% закрытий, дней: медиана / 75-й / макс",
        lambda r: f"{r['tail_median']:.0f} / {r['tail_p75']:.0f} / {r['tail_max']}"
        if r["tail_median"] is not None else "—")
    row("Жалоб сверх фона за хвост, медиана",
        lambda r: f"{r['excess_median']:.0f}" if r["excess_median"] is not None else "—")
    row("Хвост не закончился до конца ряда", lambda r: str(r["censored"]))
    row("Сдвиг дат закрытия: медианная ρ (шум при ≥ 0.9)",
        lambda r: (f"{r['shift']['median_rho']:.2f}" if r["shift"].get("median_rho") is not None
                   else "—") + (" — шум" if r["shift"]["noise"] else ""))
    row("Те же даты год назад: всплесков / p",
        lambda r: (f"{r['last_year']['spikes']} / {r['last_year']['p']:.3f}"
                   if r["last_year"]["checked"] else f"{r['last_year']['spikes']} — не проверяется")
        + (" — шум" if r["last_year"]["noise"] else ""))
    row("**Засчитан по теме региона**", lambda r: "**да**" if r["counted_region"] else "**нет**")
    row("Служб с 3+ всплесками, где они ведут половину жалоб", lambda r: str(r["services_3plus"]))
    row("Тем, где таких служб 5+", lambda r: str(r["topics_with_5_services"]))
    row("**Засчитан по службам**", lambda r: "**да**" if r["counted_services"] else "**нет**")
    L += ["", "Непроверенное не считается пройденным: где прошлого года нет или всплесков с "
          "ним меньше 10, канал не засчитан.", ""]
    return "\n".join(L)


def main() -> None:
    results = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "channel3.json").write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                                  default=str), encoding="utf-8")
    (OUT / "channel3.md").write_text(render(results), encoding="utf-8")
    for r in results:
        print(f"{r['region']}: всплесков {r['spikes']}, хвост медиана {r['tail_median']}, "
              f"сдвиг {r['shift']}, год назад {r['last_year']}, по теме {r['counted_region']}, "
              f"по службам {r['counted_services']}")


if __name__ == "__main__":
    main()
