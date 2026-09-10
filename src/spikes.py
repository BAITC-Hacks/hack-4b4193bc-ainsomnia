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
Прирост = пик − медиана окна на старте события. Две шкалы ранжирования
нужны потому, что кратность выводит наверх скачки с нулевого фона, а
абсолютный прирост — крупные события на высоком фоне.

Первый календарный месяц ряда каждого региона выбрасывается из ряда
целиком — и как объект детекции, и из окон: это выход системы на режим.
Если оставить его в окнах, медиана следующих 28 суток будет занижена.

spike_type (только регионы из SEASONAL_REGIONS):
  seasonal — в предыдущие годы в те же календарные даты (±season_tol дней)
             по этому же срезу регион x тема тоже был всплеск;
  anomaly  — окно прошлого года наблюдалось полностью, всплеска не было;
  пусто    — сравнивать не с чем: регион вне списка или у всплеска нет
             наблюдаемого прошлогоднего окна (первый год ряда).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA = Path("data/unified.parquet")
DEEP_DIVE_REGIONS = ["Павлодарская область", "Карагандинская область"]
SEASONAL_REGIONS = ["Павлодарская область", "Карагандинская область"]


def daily_counts(df, skip_first_month=True):
    """Дневные счётчики регион x тема с дозаполнением нулей.
    Возвращает (таблица, {регион: (первый день, последний день)})."""
    df = df.copy()
    df["день"] = df["created_at"].dt.floor("D")
    g = df.groupby(["region", "topic", "день"]).size().rename("count").reset_index()
    out, bounds = [], {}
    for reg, gr in g.groupby("region", sort=False):
        lo, hi = gr["день"].min(), gr["день"].max()
        if skip_first_month:
            lo = (lo.to_period("M") + 1).to_timestamp()
        bounds[reg] = (lo, hi)
        cal = pd.date_range(lo, hi, freq="D")
        for topic, gt in gr.groupby("topic", sort=False):
            s_ = gt.set_index("день")["count"].reindex(cal, fill_value=0)
            out.append(pd.DataFrame({"region": reg, "topic": topic,
                                     "день": cal, "count": s_.to_numpy()}))
    return pd.concat(out, ignore_index=True), bounds


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
                "прирост": peak["count"] - first["median_w"],
                "дней подряд": dur,
                "пик": peak["день"]})
    if not rows:
        return pd.DataFrame(columns=["дата", "регион", "тема", "обращений",
                                     "медиана окна", "MAD", "порог",
                                     "кратность", "прирост", "дней подряд", "пик"])
    return pd.DataFrame(rows).sort_values("дата").reset_index(drop=True)


def classify_seasonal(ev, det, bounds, regions, tol=10):
    """seasonal / anomaly / пусто. Сравнение только с ПРЕДЫДУЩИМИ годами."""
    days = {k: np.sort(g["день"].to_numpy())
            for k, g in det[det.spike].groupby(["region", "topic"])}
    td = pd.Timedelta(days=tol)
    out = []
    for _, e in ev.iterrows():
        reg, top, d = e["регион"], e["тема"], e["дата"]
        if reg not in regions:
            out.append(None)
            continue
        lo = bounds[reg][0]
        arr = days.get((reg, top), np.array([], dtype="datetime64[ns]"))
        observed = found = False
        k = 1
        while True:
            a = d - pd.DateOffset(years=k)
            if a + td < lo:
                break
            w_lo, w_hi = max(a - td, lo), a + td
            hit = np.any((arr >= np.datetime64(w_lo)) & (arr <= np.datetime64(w_hi)))
            if hit:
                found = observed = True
            elif a - td >= lo:          # окно наблюдалось полностью
                observed = True
            k += 1
        out.append("seasonal" if found else ("anomaly" if observed else None))
    ev = ev.copy()
    ev["spike_type"] = out
    return ev


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


def _fmt(t, cols):
    t = t.copy()
    t["дата"] = t["дата"].dt.strftime("%Y-%m-%d")
    for c in ("кратность", "прирост", "медиана окна"):
        if c in t:
            t[c] = t[c].round(1)
    if "spike_type" in t:
        t["spike_type"] = t["spike_type"].fillna("—")
    with pd.option_context("display.max_rows", 2000, "display.width", 220,
                           "display.max_colwidth", 34):
        return t[cols].to_string(index=False)


def print_series(daily, det, region, topic, d_from, d_to):
    m = daily[(daily.region == region) & (daily.topic == topic) &
              (daily["день"] >= d_from) & (daily["день"] <= d_to)].sort_values("день")
    sp = det[(det.region == region) & (det.topic == topic)].set_index("день")
    print(f"\n{region}, «{topic}», {d_from:%d.%m.%Y} — {d_to:%d.%m.%Y}, класс problem")
    print(f"{'дата':>10s} {'обращений':>10s} {'медиана окна':>13s}  всплеск")
    for _, r in m.iterrows():
        row = sp.loc[r["день"]] if r["день"] in sp.index else None
        med = "" if row is None or pd.isna(row["median_w"]) else f"{row['median_w']:.1f}"
        flag = "да" if row is not None and bool(row["spike"]) else ""
        print(f"{r['день']:%Y-%m-%d} {int(r['count']):10d} {med:>13s}  {flag}")
    print(f"{'итого':>10s} {int(m['count'].sum()):10d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--season-tol", type=int, default=10,
                    help="допуск по календарной дате для seasonal, дней")
    ap.add_argument("--keep-first-month", action="store_true",
                    help="не выбрасывать первый календарный месяц ряда")
    ap.add_argument("--series", nargs=4, metavar=("РЕГИОН", "ТЕМА", "С", "ПО"),
                    help="вывести дневной ряд по срезу и выйти")
    a = ap.parse_args()

    df = pd.read_parquet(DATA, columns=["created_at", "region", "topic", "appeal_class"])
    df = df[df.appeal_class == "problem"].copy()
    df["created_at"] = pd.to_datetime(df["created_at"])

    daily, bounds = daily_counts(df, skip_first_month=not a.keep_first_month)
    det = detect(daily, a.k, a.min_count, a.window)

    if a.series:
        reg, top, f, t = a.series
        print_series(daily, det, reg, top, pd.Timestamp(f), pd.Timestamp(t))
        return

    print(f"Класс problem: {len(df)} обращений, {df.region.nunique()} регионов")
    print(f"Параметры: K={a.k}, минимум {a.min_count} обращений в день, окно {a.window} сут "
          f"(сдвинуто), допуск сезонности ±{a.season_tol} дн.\n")

    ev = with_duration(det)
    ev = classify_seasonal(ev, det, bounds, SEASONAL_REGIONS, a.season_tol)

    # ---------- отсечение первого месяца
    print("=" * 78)
    print("ОТСЕЧЕНИЕ ПЕРВОГО КАЛЕНДАРНОГО МЕСЯЦА РЯДА")
    print("=" * 78)
    if a.keep_first_month:
        print("Отключено флагом --keep-first-month.")
    else:
        d0, _ = daily_counts(df, skip_first_month=False)
        ev0 = with_duration(detect(d0, a.k, a.min_count, a.window))
        c0 = ev0.groupby("регион").size(); c1 = ev.groupby("регион").size()
        print(f"{'регион':32s} {'выброшен месяц':>15s} {'было':>6s} {'стало':>6s} {'отсечено':>9s}")
        for r in sorted(bounds):
            fm = (bounds[r][0].to_period("M") - 1).strftime("%m.%Y")
            b, n = int(c0.get(r, 0)), int(c1.get(r, 0))
            print(f"{r:32s} {fm:>15s} {b:6d} {n:6d} {b-n:9d}")
        print(f"{'ИТОГО':32s} {'':>15s} {len(ev0):6d} {len(ev):6d} {len(ev0)-len(ev):9d}")
        print("Отсечённое включает не только всплески внутри выброшенного месяца, но и")
        print("всплески следующих недель, чьё окно опиралось на заниженный стартовый фон.")

    # ---------- 1. все всплески
    print("\n" + "=" * 78)
    print(f"1. ВСЕ НАЙДЕННЫЕ ВСПЛЕСКИ: {len(ev)}")
    print("=" * 78)
    cols = ["дата", "регион", "тема", "обращений", "медиана окна",
            "кратность", "прирост", "дней подряд", "spike_type"]
    print(_fmt(ev, cols))

    # ---------- 2. по регионам
    print("\n" + "=" * 78)
    print("2. ВСПЛЕСКОВ ПО РЕГИОНАМ")
    print("=" * 78)
    print(f"{'регион':32s} {'лет':>5s} {'всего':>6s} {'в год':>6s} "
          f"{'seasonal':>9s} {'anomaly':>8s} {'без типа':>9s}")
    hot = []
    for r in sorted(bounds):
        lo, hi = bounds[r]
        yrs = (hi - lo).days / 365.25
        e = ev[ev["регион"] == r]
        py = len(e) / yrs if yrs else 0
        if py > 200:
            hot.append(r)
        print(f"{r:32s} {yrs:5.1f} {len(e):6d} {py:6.0f} "
              f"{int((e.spike_type == 'seasonal').sum()):9d} "
              f"{int((e.spike_type == 'anomaly').sum()):8d} "
              f"{int(e.spike_type.isna().sum()):9d}")

    # ---------- 3. топ-10 по двум шкалам
    c10 = ["дата", "регион", "тема", "обращений", "медиана окна",
           "кратность", "прирост", "дней подряд", "spike_type"]
    print("\n" + "=" * 78)
    print("3а. ТОП-10 ПО КРАТНОСТИ (все типы)")
    print("=" * 78)
    print(_fmt(ev.sort_values(["кратность", "обращений"], ascending=False).head(10), c10))
    print("\n" + "=" * 78)
    print("3б. ТОП-10 ПО АБСОЛЮТНОМУ ПРИРОСТУ (все типы)")
    print("=" * 78)
    print(_fmt(ev.sort_values(["прирост", "кратность"], ascending=False).head(10), c10))

    # ---------- 4. топ-10 anomaly
    an = ev[ev.spike_type == "anomaly"]
    print("\n" + "=" * 78)
    print(f"4а. ТОП-10 ANOMALY ПО КРАТНОСТИ (всего anomaly: {len(an)})")
    print("=" * 78)
    print(_fmt(an.sort_values(["кратность", "обращений"], ascending=False).head(10), c10))
    print("\n" + "=" * 78)
    print("4б. ТОП-10 ANOMALY ПО АБСОЛЮТНОМУ ПРИРОСТУ")
    print("=" * 78)
    print(_fmt(an.sort_values(["прирост", "кратность"], ascending=False).head(10), c10))

    # ---------- проверка разумности
    print("\n" + "=" * 78)
    print("ПРОВЕРКА НА РАЗУМНОСТЬ ПОРОГА")
    print("=" * 78)
    deg = int((ev["MAD"] == 0).sum())
    print(f"Всплесков с MAD=0 (порог вырождается в медиану): {deg} из {len(ev)} "
          f"({100*deg/max(len(ev),1):.1f}%)")
    print("Больше 200 всплесков в год: " + (", ".join(hot) if hot else "ни одного региона"))
    q = ev["кратность"].quantile([.5, .9, .99])
    print(f"Кратность: медиана x{q[.5]:.1f}, 90% x{q[.9]:.1f}, 99% x{q[.99]:.1f}")
    qa = ev["прирост"].quantile([.5, .9, .99])
    print(f"Прирост:   медиана +{qa[.5]:.0f}, 90% +{qa[.9]:.0f}, 99% +{qa[.99]:.0f}")

    # ---------- 5. разбор трёх сильнейших anomaly
    print("\n" + "=" * 78)
    print("5. РАЗБОР ТРЁХ СИЛЬНЕЙШИХ ANOMALY ПО КРАТНОСТИ")
    print("=" * 78)
    for _, e in an.sort_values(["кратность", "обращений"], ascending=False).head(3).iterrows():
        deep_dive(daily, e)


if __name__ == "__main__":
    main()
