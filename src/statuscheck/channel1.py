"""Канал 1 раздела 5o — ритм закрытий: пачки старых заявок сверх ожидаемого.

    .venv/bin/python -m src.statuscheck.channel1 [--reps 1000]

Всё задано в CLAUDE.md, 5o, до прогона:
  пачка — 5+ закрытий одной службы в окне 10 минут;
  в меру идут только пачки, где заявки поступили в 3+ разных дня и медианный
  возраст выше медианы ячейки (сброс очереди, а не одна авария);
  мера службы в теме — доля её закрытий в таких пачках минус ожидаемая: время
  суток каждого закрытия берётся случайно из собственного распределения службы,
  день и число закрытий за день сохраняются (1 000 повторов);
  ранжирование — службы внутри темы региона: годная ячейка (5+ служб с 30+
  жалобами), служба со 100+ закрытиями в ячейке, 5+ таких служб;
  худшая четверть — верхние ceil(n/4) по мере.
Проверки на шум: (а) доля пачек, синхронных по минуте с пачками 3+ других
служб, больше трети; (б) ранги первой и второй половины периода не связаны —
бутстрэп-интервал ранговой корреляции содержит ноль; (в) при перемешивании
меток служб внутри «регион × тема × день» разброс не уже фактического —
больше 5% перемешиваний дают разброс не меньше фактического (порог
зафиксирован до расчёта, 2026-09-26). Провал любой — канал в регионе не
засчитан. В Туркестане синхронные минуты шага 0б удаляются, и отдельно
считается условие (б) шага 0б — меняет ли их удаление худшую четверть.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import paths
from src.statuscheck.batches import sync_mask
from src.statuscheck.frame import FRAME
from src.statuscheck.load import load
from src.statuscheck.observability import CELL_COMPLAINTS, CELL_SERVICES, CH1_CLOSURES
from src.checks.person_names import person_hits, safe as person_safe
from src.synth.checks import pii_hits

OUT = paths.REPORTS_DIR / "5o"
WINDOW, BURST, DAYS, SEED = 600, 5, 3, 42
REGIONS = ("Костанайская область", "Туркестанская область", "Восточно-Казахстанская область")


def prepare(region: str) -> tuple[pd.DataFrame, list[str], int]:
    d = load(region)
    d = d[d.closed.notna() & d.executor.notna()].copy().reset_index(drop=True)
    minute = ((d.closed.dt.floor("min") - d.closed.min().floor("min"))
              .dt.total_seconds() // 60).to_numpy()
    sync = sync_mask(minute, d.executor.to_numpy())
    d = d[~sync & (d.cls == "problem")].copy()
    d["day"] = d.closed.dt.normalize()
    d["cday"] = d.created.dt.normalize()
    per = d.groupby(["topic", "executor"]).size()
    n = per[per >= CELL_COMPLAINTS].reset_index().groupby("topic").executor.nunique()
    topics = sorted(n[n >= CELL_SERVICES].index)
    med = d.groupby("topic").apply(lambda g: (g.closed - g.created).dt.total_seconds().median())
    d["cell_age"] = d.topic.map(med).astype(float)
    return d.sort_values(["executor", "closed"]).reset_index(drop=True), topics, int(sync.sum())


def qualifying(svc: np.ndarray, t: np.ndarray, created: np.ndarray, cday: np.ndarray,
               cell_age: np.ndarray) -> np.ndarray:
    """Какие закрытия лежат в пачке «сброса очереди». Строки отсортированы по
    службе и времени; t, created — секунды."""
    n = len(t)
    order = np.lexsort((t, svc))
    t, svc_o = t[order], svc[order]
    key = svc_o.astype(np.int64) * 10**11 + t
    end = np.searchsorted(key, key + WINDOW, side="left")
    cnt = end - np.arange(n)
    diff = np.zeros(n + 1, dtype=np.int64)
    starts = np.flatnonzero(cnt >= BURST)
    np.add.at(diff, starts, 1)
    np.add.at(diff, end[starts], -1)
    inb = np.cumsum(diff[:-1]) > 0
    # пачки — связные куски отмеченных закрытий одной службы
    new = inb & ~np.r_[False, inb[:-1] & (svc_o[1:] == svc_o[:-1])
                       & (t[1:] - t[:-1] <= WINDOW)]
    bid = np.where(inb, np.cumsum(new), 0)
    f = pd.DataFrame({"b": bid, "cday": cday[order],
                      "ratio": (t - created[order]) / cell_age[order]})[inb]
    g = f.groupby("b").agg(days=("cday", "nunique"), ratio=("ratio", "median"))
    good = g.index[(g.days >= DAYS) & (g.ratio > 1)]
    out = np.zeros(n, dtype=bool)
    out[order] = np.isin(bid, good)
    return out


def secs(s: pd.Series) -> np.ndarray:
    """Секунды эпохи независимо от внутренней единицы: в pandas 3 даты хранятся
    в микросекундах, и деление на 10**9 давало отрезки по 1 000 секунд."""
    return s.astype("datetime64[s]").astype("int64").to_numpy()


def arrays(d: pd.DataFrame):
    return (d.executor.astype("category").cat.codes.to_numpy(),
            secs(d.closed), secs(d.created), secs(d.cday) // 86400,
            np.maximum(d.cell_age.to_numpy(), 1.0))   # медиана 0 с у ВКО: «старше 0»


def expected(d: pd.DataFrame, reps: int, seed: int = SEED) -> np.ndarray:
    """Средняя доля в пачках при случайном времени суток из распределения службы."""
    svc, t, created, cday, cell_age = arrays(d)
    day0 = secs(d.day)
    tod = t - day0
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(svc == s) for s in np.unique(svc)]
    acc = np.zeros(len(t))
    for _ in range(reps):
        new = np.empty_like(tod)
        for idx in idx_by:
            new[idx] = rng.choice(tod[idx], size=len(idx), replace=True)
        acc += qualifying(svc, day0 + new, created, cday, cell_age)
    return acc / reps


def measure(d: pd.DataFrame, topics: list[str], reps: int) -> pd.DataFrame:
    svc, t, created, cday, cell_age = arrays(d)
    d = d.assign(obs=qualifying(svc, t, created, cday, cell_age),
                 exp=expected(d, reps) if reps else 0.0)
    m = (d[d.topic.isin(topics)].groupby(["topic", "executor"])
         .agg(n=("obs", "size"), obs=("obs", "mean"), exp=("exp", "mean")).reset_index())
    m["excess"] = m.obs - m.exp
    return m[m.n >= CH1_CLOSURES]


def worst(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, g in m.groupby("topic"):
        if len(g) < CELL_SERVICES:
            continue
        k = math.ceil(len(g) / 4)
        top = set(g.nlargest(k, "excess").executor)
        rows.append(g.assign(worst=g.executor.isin(top), peers=len(g)))
    return pd.concat(rows) if rows else m.iloc[0:0].assign(worst=False, peers=0)


def split_half(d, topics, reps) -> dict:
    mid = d.closed.min() + (d.closed.max() - d.closed.min()) / 2
    a = measure(d[d.closed < mid], topics, reps).set_index(["topic", "executor"]).excess
    b = measure(d[d.closed >= mid], topics, reps).set_index(["topic", "executor"]).excess
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    both.columns = ["a", "b"]
    if len(both) < 5:
        return {"pairs": len(both), "rho": None, "ci": None, "noise": True}
    rho = float(spearmanr(both.a, both.b).statistic)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        s = both.iloc[rng.integers(0, len(both), len(both))]
        if s.a.nunique() > 1 and s.b.nunique() > 1:
            boots.append(spearmanr(s.a, s.b).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"pairs": len(both), "rho": rho, "ci": [float(lo), float(hi)],
            "noise": bool(lo <= 0 <= hi)}


def placebo(d, topics, m, reps=200) -> dict:
    """Метки служб перемешаны внутри «тема × день»: разброс наблюдаемой доли."""
    svc, t, created, cday, cell_age = arrays(d)
    base = d.assign(obs=qualifying(svc, t, created, cday, cell_age))
    actual = (base[base.topic.isin(topics)].groupby(["topic", "executor"]).obs.mean()
              .loc[list(m.set_index(["topic", "executor"]).index)].groupby("topic").std().mean())
    rng = np.random.default_rng(SEED)
    wider = 0
    for _ in range(reps):
        p = d.copy()
        p["executor"] = p.groupby(["topic", "day"]).executor.transform(
            lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_numpy())
        p = p.sort_values(["executor", "closed"]).reset_index(drop=True)
        ps, pt, pc, pd_, pa = arrays(p)
        p["obs"] = qualifying(ps, pt, pc, pd_, pa)
        g = p[p.topic.isin(topics)].groupby(["topic", "executor"]).obs.agg(["mean", "size"])
        g = g[g["size"] >= CH1_CLOSURES]
        spread = g["mean"].groupby("topic").std().mean()
        wider += int(spread >= actual)
    return {"actual_spread": float(actual), "reps": reps, "not_narrower": wider,
            "noise": bool(wider / reps > 0.05)}


def sync_bursts(d: pd.DataFrame) -> float:
    """Доля пачек, синхронных по минуте с пачками 3+ других служб."""
    svc, t, created, cday, cell_age = arrays(d)
    q = qualifying(svc, t, created, cday, cell_age)
    b = d[q].assign(minute=d[q].closed.dt.floor("min"))
    if b.empty:
        return 0.0
    other = b.groupby("minute").executor.nunique()
    return float((b.minute.map(other) >= 4).mean())


def safe_name(name: str) -> str:
    """Название службы — маской, если в нём имя человека (раздел 1): имена в
    тексте (pii_hits) и название-человек вида «ИП Фамилия» (person_names)."""
    if person_hits(name):
        return person_safe(name)
    if pii_hits(name):
        return " ".join(w[:2] + "*" * max(len(w) - 2, 0) for w in str(name).split())
    return name


def run(region: str, reps: int) -> dict:
    d, topics, removed = prepare(region)
    m = measure(d, topics, reps)
    w = worst(m)
    res = {"region": region, "topics": topics, "removed_sync_closures": removed,
           "ranked_services": int(w.executor.nunique()), "ranked_pairs": int(len(w)),
           "worst_pairs": int(w.worst.sum()),
           "burst_share_obs": float(m.obs.mean()) if len(m) else None,
           "burst_share_exp": float(m.exp.mean()) if len(m) else None,
           "check_a_sync_bursts": sync_bursts(d),
           "check_b_split_half": split_half(d, topics, reps),
           "check_c_placebo": placebo(d, topics, w) if len(w) else None}
    res["check_a_noise"] = res["check_a_sync_bursts"] > 1 / 3
    res["counted"] = not (res["check_a_noise"] or res["check_b_split_half"]["noise"]
                          or (res["check_c_placebo"] or {}).get("noise", True))
    if region == "Туркестанская область":
        raw = load(region)
        raw = raw[raw.closed.notna() & raw.executor.notna() & (raw.cls == "problem")].copy()
        raw["day"], raw["cday"] = raw.closed.dt.normalize(), raw.created.dt.normalize()
        med = raw.groupby("topic").apply(lambda g: (g.closed - g.created).dt.total_seconds().median())
        raw["cell_age"] = raw.topic.map(med).astype(float)
        raw = raw.sort_values(["executor", "closed"]).reset_index(drop=True)
        w_all = worst(measure(raw, topics, reps))
        a = set(map(tuple, w_all[w_all.worst][["topic", "executor"]].to_numpy()))
        b = set(map(tuple, w[w.worst][["topic", "executor"]].to_numpy()))
        res["step0b_condition_b"] = bool(a != b)
    res["table"] = [{"topic": r.topic, "service": safe_name(r.executor), "n": int(r.n),
                     "obs": round(float(r.obs), 4), "exp": round(float(r.exp), 4),
                     "excess": round(float(r.excess), 4), "peers": int(r.peers),
                     "worst": bool(r.worst)} for r in w.itertuples()]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    results = [run(r, a.reps) for r in REGIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "channel1.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    for r in results:
        sh = r["check_b_split_half"]
        pl = r["check_c_placebo"]
        print(f"{r['region']}: тем {len(r['topics'])}, служб в ранжировании {r['ranked_services']}, "
              f"пар {r['ranked_pairs']}, в худшей четверти {r['worst_pairs']}; "
              f"доля в пачках {r['burst_share_obs']:.2%} при ожидаемой {r['burst_share_exp']:.2%}; "
              f"(а) синхронных пачек {r['check_a_sync_bursts']:.1%}; "
              f"(б) ρ={sh['rho']} ДИ {sh['ci']} пар {sh['pairs']}; "
              f"(в) не уже фактического {pl['not_narrower'] if pl else '—'} из {pl['reps'] if pl else '—'}; "
              f"ЗАСЧИТАН: {r['counted']}"
              + (f"; условие (б) шага 0б: {r['step0b_condition_b']}" if "step0b_condition_b" in r else ""))


if __name__ == "__main__":
    main()
