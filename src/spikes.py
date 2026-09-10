#!/usr/bin/env python3
"""Детектор всплесков обращений.

    .venv/bin/python -m src.spikes [--k 4] [--min-count 10] [--window 28]

Логика:
  дневные счётчики по срезу регион x тема, только класс problem;
  скользящая медиана и MAD по окну W суток, ОКНО СДВИНУТО — оно
  покрывает [t-W, t-1] и не включает сам день t;
  всплеск: count[t] > median + K*MAD  И  count[t] >= MIN_COUNT.

Дни без обращений дозаполняются нулями внутри окна наблюдения каждого
региона. Без этого «28 дней» на редкой теме растянулось бы на месяцы и
медиана считалась бы по несопоставимому периоду.

Кратность = count / max(median, 1). Медиана окна часто равна нулю
(тема обычно молчит), деление на ноль заменено единицей — сама медиана
печатается рядом, поэтому ничего не скрыто.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA = Path("data/unified.parquet")
DEEP_DIVE_REGIONS = ["Павлодарская область", "Карагандинская область"]


def daily_counts(df):
    """Дневные счётчики регион x тема с дозаполнением нулей."""
    df = df.copy()
    df["день"] = df["created_at"].dt.floor("D")
    g = df.groupby(["region", "topic", "день"]).size().rename("count").reset_index()
    out = []
    for reg, gr in g.groupby("region", sort=False):
        lo, hi = gr["день"].min(), gr["день"].max()
        cal = pd.date_range(lo, hi, freq="D")
        for topic, gt in gr.groupby("topic", sort=False):
            s = gt.set_index("день")["count"].reindex(cal, fill_value=0)
            out.append(pd.DataFrame({"region": reg, "topic": topic,
                                     "день": cal, "count": s.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def detect(daily, k=4.0, min_count=10, window=28):
    """Всплески по сдвинутому окну. Возвращает строки с флагом spike."""
    res = []
    for (reg, topic), g in daily.groupby(["region", "topic"], sort=False):
        g = g.sort_values("день").reset_index(drop=True)
        y = g["count"].to_numpy(dtype=float)
        n = len(y)
        med = np.full(n, np.nan)
        mad = np.full(n, np.nan)
        if n > window:
            # окно [t-window, t-1]: берём y[0:n-window] ... т.е. срез до t-1
            win = sliding_window_view(y[:-1], window)      # строка i = y[i:i+window]
            m = np.median(win, axis=1)                     # относится к t = i+window
            a = np.median(np.abs(win - m[:, None]), axis=1)
            med[window:] = m
            mad[window:] = a
        g["median_w"] = med
        g["mad_w"] = mad
        g["threshold"] = med + k * mad
        g["spike"] = (y > g["threshold"]) & (y >= min_count) & np.isfinite(med)
        res.append(g)
    return pd.concat(res, ignore_index=True)


def with_duration(det):
    """Длительность: сколько дней подряд держался всплеск в этом срезе."""
    rows = []
    for (reg, topic), g in det.groupby(["region", "topic"], sort=False):
        g = g.sort_values("день").reset_index(drop=True)
        sp = g["spike"].to_numpy()
        grp = (sp != np.r_[False, sp[:-1]]).cumsum()
        g["_run"] = grp
        for _, run in g[g.spike].groupby("_run"):
            dur = len(run)
            first = run.iloc[0]
            peak = run.loc[run["count"].idxmax()]
            rows.append({
                "дата": first["день"], "регион": reg, "тема": topic,
                "обращений": int(peak["count"]),
                "медиана окна": float(first["median_w"]),
                "MAD": float(first["mad_w"]),
                "порог": float(first["threshold"]),
                "кратность": peak["count"] / max(first["median_w"], 1.0),
                "дней подряд": dur,
                "пик": peak["день"]})
    if not rows:
        return pd.DataFrame(columns=["дата", "регион", "тема", "обращений",
                                     "медиана окна", "MAD", "порог",
                                     "кратность", "дней подряд", "пик"])
    return pd.DataFrame(rows).sort_values("дата").reset_index(drop=True)


def deep_dive(daily, ev, before=14, after=14):
    m = daily[(daily.region == ev["регион"]) & (daily.topic == ev["тема"])]
    lo = ev["пик"] - pd.Timedelta(days=before)
    hi = ev["пик"] + pd.Timedelta(days=after)
    w = m[(m["день"] >= lo) & (m["день"] <= hi)].sort_values("день")
    print(f"\n{'='*78}")
    print(f"{ev['регион']} — «{ev['тема']}»")
    print(f"пик {ev['пик']:%Y-%m-%d}: {ev['обращений']} обращений при медиане "
          f"{ev['медиана окна']:.1f} — рост в {ev['кратность']:.1f} раза, "
          f"держалось {ev['дней подряд']} дн.")
    print(f"{'='*78}")
    mx = max(w["count"].max(), 1)
    for _, r in w.iterrows():
        c = int(r["count"])
        bar = "#" * int(round(40 * c / mx))
        mark = "  <-- ВСПЛЕСК" if r["день"] == ev["пик"] else ""
        print(f"  {r['день']:%Y-%m-%d} {c:6d} |{bar:<40}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--window", type=int, default=28)
    a = ap.parse_args()

    df = pd.read_parquet(DATA, columns=["created_at", "region", "topic", "appeal_class"])
    df = df[df.appeal_class == "problem"].copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    print(f"Класс problem: {len(df)} обращений, {df.region.nunique()} регионов")
    print(f"Параметры: K={a.k}, минимум обращений в день={a.min_count}, "
          f"окно={a.window} сут (сдвинуто, день всплеска в окно не входит)\n")

    daily = daily_counts(df)
    print(f"Дневных точек регион x тема x день: {len(daily)}")
    det = detect(daily, a.k, a.min_count, a.window)
    ev = with_duration(det)
    print(f"Найдено всплесков (событий, а не дней): {len(ev)}\n")

    # ---------- 1. таблица всех всплесков
    print("=" * 78)
    print("1. ВСЕ НАЙДЕННЫЕ ВСПЛЕСКИ")
    print("=" * 78)
    show = ev.copy()
    show["дата"] = show["дата"].dt.strftime("%Y-%m-%d")
    show["кратность"] = show["кратность"].round(1)
    show["медиана окна"] = show["медиана окна"].round(1)
    cols = ["дата", "регион", "тема", "обращений", "медиана окна", "кратность", "дней подряд"]
    with pd.option_context("display.max_rows", 400, "display.width", 200,
                           "display.max_colwidth", 34):
        print(show[cols].to_string(index=False))

    # ---------- 2. по регионам
    print("\n" + "=" * 78)
    print("2. ВСПЛЕСКОВ ПО РЕГИОНАМ")
    print("=" * 78)
    span = daily.groupby("region")["день"].agg(["min", "max"])
    span["лет"] = (span["max"] - span["min"]).dt.days / 365.25
    cnt = ev.groupby("регион").size().rename("всплесков")
    tab = span.join(cnt).fillna({"всплесков": 0})
    tab["в год"] = tab["всплесков"] / tab["лет"]
    print(f"{'регион':32s} {'период':>23s} {'лет':>5s} {'всплесков':>10s} {'в год':>8s}")
    hot = []
    for r, row in tab.iterrows():
        flag = ""
        if row["в год"] > 200:
            flag = "  <-- >200/год"
            hot.append(r)
        print(f"{r:32s} {row['min']:%Y-%m-%d}—{row['max']:%Y-%m-%d} "
              f"{row['лет']:5.1f} {int(row['всплесков']):10d} {row['в год']:8.0f}{flag}")

    # ---------- 3. топ-10
    print("\n" + "=" * 78)
    print("3. ТОП-10 ПО КРАТНОСТИ")
    print("=" * 78)
    top = ev.sort_values(["кратность", "обращений"], ascending=False).head(10)
    t = top.copy()
    t["дата"] = t["дата"].dt.strftime("%Y-%m-%d")
    t["кратность"] = t["кратность"].round(1)
    t["медиана окна"] = t["медиана окна"].round(1)
    with pd.option_context("display.width", 200, "display.max_colwidth", 34):
        print(t[cols].to_string(index=False))

    # ---------- проверка на разумность
    print("\n" + "=" * 78)
    print("ПРОВЕРКА НА РАЗУМНОСТЬ ПОРОГА")
    print("=" * 78)
    deg = int((ev["MAD"] == 0).sum())
    print(f"Всплесков с MAD=0 (окно без разброса, порог вырождается "
          f"в медиану): {deg} из {len(ev)} ({100*deg/max(len(ev),1):.0f}%)")
    if hot:
        print(f"\nПОРОГ ЗАНИЖЕН: больше 200 всплесков на регион в год — "
              f"{', '.join(hot)}.")
        print("Распределение кратностей:")
    else:
        print("\nНи по одному региону не превышено 200 всплесков в год — "
              "порог выглядит адекватным.")
        print("Распределение кратностей для контроля:")
    q = ev["кратность"].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    for kq in ["min", "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        print(f"    {kq:>4s}  x{q[kq]:.1f}")
    bins = [0, 2, 3, 4, 6, 10, 20, np.inf]
    lab = ["<2", "2-3", "3-4", "4-6", "6-10", "10-20", ">20"]
    hist = pd.cut(ev["кратность"], bins, labels=lab).value_counts().reindex(lab)
    print("    гистограмма:")
    for l, v in hist.items():
        print(f"      x{l:<6s} {int(v):5d}  {'#'*int(40*v/max(hist.max(),1))}")

    # ---------- 4. разбор трёх сильнейших
    print("\n" + "=" * 78)
    print("4. РАЗБОР ТРЁХ СИЛЬНЕЙШИХ (Павлодар и Караганда)")
    print("=" * 78)
    pool = ev[ev["регион"].isin(DEEP_DIVE_REGIONS)]
    pool = pool.sort_values(["кратность", "обращений"], ascending=False).head(3)
    if pool.empty:
        print("В этих регионах всплесков не найдено.")
    for _, e in pool.iterrows():
        deep_dive(daily, e)


if __name__ == "__main__":
    main()
