"""Пачки закрытий — шаг 0б раздела 5o CLAUDE.md: служба или оператор 109.

    .venv/bin/python -m src.statuscheck.batches [--perm 1000]

Все пороги зафиксированы в CLAUDE.md до прогона («Поправки до прогона»):
  синхронная минута — 3+ разных служб закрыли в ней по 3+ заявки;
  знаменатель — службы региона со 100+ закрытиями;
  служба с пачками — 10%+ её закрытий в синхронных минутах;
  большинство — больше 50% знаменателя: finishdate ставится централизованно,
  канал 1 в регионе не считается ни по одной службе;
  условие (а) — доля закрытий региона в синхронных минутах выше 99-го
  процентиля нулевой модели: блок закрытий службы за день целиком переносится
  на случайный другой её рабочий день (1 000 перестановок).
Условие (б) требует меры канала 1 и считается вместе с каналом 1 — только если
в регионе меньшинство.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.statuscheck.frame import FRAME
from src.statuscheck.load import load

OUT = paths.REPORTS_DIR / "5o"
SYNC_SERVICES, SYNC_PER_SERVICE = 3, 3
MIN_CLOSURES, SERVICE_SHARE, MAJORITY = 100, 0.10, 0.50
REGIONS = ("Туркестанская область", "Костанайская область")
WITH_TIME = ("Костанайская область", "Туркестанская область",
             "Восточно-Казахстанская область", "Алматинская область")


def _frame(region: str) -> pd.DataFrame:
    d = load(region)
    d = d[d.closed.notna() & d.executor.notna()].copy()
    d["minute"] = d.closed.dt.floor("min")
    d["day"] = d.closed.dt.normalize()
    d["age"] = (d.closed - d.created).dt.total_seconds() / 86400
    return d.reset_index(drop=True)


def sync_mask(minute_key: np.ndarray, service: np.ndarray) -> np.ndarray:
    """Какие закрытия попали в синхронную минуту (все закрытия этой минуты)."""
    pair = pd.DataFrame({"m": minute_key, "s": service})
    per = pair.groupby(["m", "s"]).size()
    heavy = per[per >= SYNC_PER_SERVICE].reset_index().groupby("m").s.nunique()
    sync_minutes = heavy[heavy >= SYNC_SERVICES].index
    return np.isin(minute_key, sync_minutes)


def null_shares(d: pd.DataFrame, n: int, seed: int = 42) -> np.ndarray:
    """Доля закрытий в синхронных минутах при перестановке дней внутри службы."""
    rng = np.random.default_rng(seed)
    svc = d.executor.astype("category").cat.codes.to_numpy()
    day = ((d.day - d.day.min()).dt.days).to_numpy()
    tod = (d.minute - d.day).dt.total_seconds().to_numpy() // 60
    groups = {}
    for s in np.unique(svc):
        idx = np.flatnonzero(svc == s)
        days = np.unique(day[idx])
        groups[s] = (idx, day[idx], days)
    out = np.empty(n)
    for k in range(n):
        new_day = np.empty_like(day)
        for s, (idx, dd, days) in groups.items():
            perm = dict(zip(days, rng.permutation(days)))
            new_day[idx] = [perm[x] for x in dd]
        key = new_day * 1440 + tod
        out[k] = sync_mask(key, svc).mean()
    return out


def analyse(region: str, perms: int) -> dict:
    d = _frame(region)
    key = ((d.minute - d.minute.min()).dt.total_seconds() // 60).to_numpy()
    d["sync"] = sync_mask(key, d.executor.to_numpy())
    per_service = d.groupby("executor").agg(n=("sync", "size"), sync=("sync", "mean"))
    denom = per_service[per_service.n >= MIN_CLOSURES]
    with_batches = int((denom.sync >= SERVICE_SHARE).sum())
    share_services = with_batches / len(denom) if len(denom) else float("nan")
    per_minute = d.groupby("minute").size()
    single = d[d.groupby(["minute", "executor"]).executor.transform("size") == 1]
    last_days = d.closed.dt.day >= (d.closed.dt.days_in_month - 1)
    res = {
        "region": region,
        "closures": len(d),
        "services": int(d.executor.nunique()),
        "denominator_services": int(len(denom)),
        "services_with_batches": with_batches,
        "share_services_with_batches": share_services,
        "branch": "большинство" if share_services > MAJORITY else "меньшинство",
        "sync_minutes": int(d.loc[d.sync, "minute"].nunique()),
        "sync_closures": int(d.sync.sum()),
        "sync_share": float(d.sync.mean()),
        "services_in_sync": int(d.loc[d.sync, "executor"].nunique()),
        "per_minute": {"median": float(per_minute.median()),
                       "p99": float(per_minute.quantile(.99)), "max": int(per_minute.max())},
        "hour_all": d.closed.dt.hour.value_counts(normalize=True).sort_index().round(4).to_dict(),
        "hour_sync": d.loc[d.sync].closed.dt.hour.value_counts(normalize=True).sort_index().round(4).to_dict(),
        "after_17_all": float((d.closed.dt.hour >= 17).mean()),
        "after_17_sync": float((d.loc[d.sync].closed.dt.hour >= 17).mean()) if d.sync.any() else None,
        "month_end_all": float(last_days.mean()),
        "month_end_sync": float(last_days[d.sync].mean()) if d.sync.any() else None,
        "age_median_all": float(d.age.median()),
        "age_median_sync": float(d.loc[d.sync, "age"].median()) if d.sync.any() else None,
        "age_median_single": float(single.age.median()),
    }
    if perms:
        null = null_shares(d, perms)
        res["null_p99"] = float(np.quantile(null, .99))
        res["null_median"] = float(np.median(null))
        res["condition_a"] = bool(res["sync_share"] > res["null_p99"])
    return res


def render(results: list[dict]) -> str:
    L = [FRAME, "", "# 5o, шаг 0б: пачки закрытий — служба или оператор 109", "",
         "Синхронная минута — 3+ разных службы закрыли в ней по 3+ заявки. Служба с "
         "пачками — у которой 10%+ закрытий в синхронных минутах; знаменатель — службы "
         "со 100+ закрытиями. **Больше 50% — большинство: отметку о закрытии в регионе "
         "ставят централизованно, канал 1 по службам не считается.** Пороги записаны в "
         "CLAUDE.md (5o) до прогона.", "",
         "| | " + " | ".join(r["region"] for r in results) + " |",
         "|---|" + "---:|" * len(results)]

    def row(title, f):
        L.append(f"| {title} | " + " | ".join(f(r) for r in results) + " |")

    row("Закрытий с исполнителем", lambda r: f"{r['closures']:,}".replace(",", " "))
    row("Служб со 100+ закрытиями", lambda r: str(r["denominator_services"]))
    row("**из них с пачками (10%+ в синхронных минутах)**",
        lambda r: f"**{r['services_with_batches']} — {r['share_services_with_batches']:.1%}**")
    row("**Развилка**", lambda r: f"**{r['branch']}**")
    row("Синхронных минут", lambda r: str(r["sync_minutes"]))
    row("Закрытий в них, доля региона",
        lambda r: f"{r['sync_closures']} — {r['sync_share']:.2%}")
    row("Служб, участвовавших хоть раз", lambda r: str(r["services_in_sync"]))
    row("Условие (а): доля выше 99-го процентиля нулевой модели",
        lambda r: (f"{'да' if r['condition_a'] else 'нет'} (модель: медиана "
                   f"{r['null_median']:.2%}, 99-й {r['null_p99']:.2%})")
        if "condition_a" in r else "—")
    row("Закрытий в минуте: медиана / 99-й / макс",
        lambda r: f"{r['per_minute']['median']:.0f} / {r['per_minute']['p99']:.0f} / "
                  f"{r['per_minute']['max']}")
    row("После 17:00 — все / в синхронных минутах",
        lambda r: f"{r['after_17_all']:.1%} / " + (f"{r['after_17_sync']:.1%}"
                                                  if r["after_17_sync"] is not None else "—"))
    row("Последние 2 дня месяца — все / в синхронных",
        lambda r: f"{r['month_end_all']:.1%} / " + (f"{r['month_end_sync']:.1%}"
                                                   if r["month_end_sync"] is not None else "—"))
    row("Медианный возраст заявки, сут: все / в пачке / одиночное закрытие",
        lambda r: f"{r['age_median_all']:.1f} / " + (f"{r['age_median_sync']:.1f}"
                  if r["age_median_sync"] is not None else "—") + f" / {r['age_median_single']:.1f}")
    L += ["", "«Одиночное закрытие» — служба закрыла в эту минуту ровно одну заявку.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    a = ap.parse_args()
    results = [analyse(r, a.perm) for r in a.regions]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batches.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (OUT / "batches.md").write_text(render(results), encoding="utf-8")
    for r in results:
        print(f"{r['region']}: служб с пачками {r['services_with_batches']} из "
              f"{r['denominator_services']} ({r['share_services_with_batches']:.1%}) — "
              f"{r['branch']}; закрытий в синхронных минутах {r['sync_share']:.2%}"
              + (f"; условие (а) {'да' if r.get('condition_a') else 'нет'}" if 'condition_a' in r else ""))


if __name__ == "__main__":
    main()
