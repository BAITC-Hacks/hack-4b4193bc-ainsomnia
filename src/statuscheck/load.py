"""Закрытия и статусы из сырых выгрузок — шаг 0а раздела 5o CLAUDE.md.

Дата закрытия в канон не входит (5c), поэтому она берётся из сырья. Строки —
ровно те, что адаптеры кладут в канон: та же дедупликация, разбор дат и отсев
сдвига полей. Для этого на время загрузки перехватывается src.adapters.adapters
._build_canon — сами адаптеры не меняются, канон не пересобирается.

Из сырья берутся только нужные поля. Колонки ПДн и свободного текста
(раздел 1) не читаются дальше перехвата. `contractor` Алматы не берётся:
частота не защищает от ПДн исполнителя (5m).

Выход — таблица на регион: created, closed, status, final, executor, topic, cls.
Кэш — data/5o/<регион>.parquet (data/ под .gitignore).
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd

import src.adapters.adapters as A
from src import paths
from src.topic_mapping import classify_appeal, map_topic

CACHE = paths.DATA_DIR / "5o"

# Что брать из сырья. closed — только ФАКТ закрытия: плановая дата Акмолы и
# updated_date Караганды (дата последнего изменения записи, раздел 3) — не факт.
SPEC = {
    "Карагандинская область": dict(loader="load_karaganda", topic="sub_category",
                                   executor="executor_gov_org", closed=None, status=None),
    "Костанайская область": dict(loader="load_kostanay", topic="servicelevel1",
                                 executor="organizationname",
                                 closed=("finishdate", "ISO8601"), status="status"),
    "Туркестанская область": dict(loader="load_turkestan", topic="servicelevel1",
                                  executor="organizationname",
                                  closed=("finishdate", "ISO8601"), status="status"),
    "Восточно-Казахстанская область": dict(loader="load_vko", topic="category",
                                           executor="contractor",
                                           closed=("closing_date", "%d.%m.%Y %H:%M:%S"),
                                           status="status"),
    "Алматинская область": dict(loader="load_almaty", topic="category", executor=None,
                                closed=("closing_date", "%d.%m.%Y %H:%M:%S"),
                                status="status"),
    "Акмолинская область": dict(loader="load_akmola", topic="direction", executor=None,
                                closed=None, status="status"),
    "Павлодарская область": dict(loader="load_pavlodar", topic="category_name",
                                 executor=None, closed=None, status="status"),
}

# Финальные статусы — явный список на регион. Незнакомый статус роняет
# загрузку, а не угадывается (5o, точка отсчёта канала 2).
FINAL = {
    "Костанайская область": {"закрыто": True, "закрыто инициатором": True},
    "Туркестанская область": {"закрыто": True, "закрыто инициатором": True},
    "Восточно-Казахстанская область": {"Закрыто": True, "В работе": False},
    "Алматинская область": {"Закрыто": True, "В работе": False},
    "Акмолинская область": {"Выполнено": True, "Отклонено": True,
                            "Передано в службу ": False, "В работе": False,
                            "Новый": False},
    "Павлодарская область": {"CLOSED": True, "PROCESSING": False,
                             "WAITING_ORGANIZATION": False},
}


def _capture(loader: str) -> tuple[pd.DataFrame, pd.Series]:
    got = {}
    orig = A._build_canon

    def spy(df_valid, parsed_valid, region, **kw):
        got["df"], got["created"] = df_valid, parsed_valid
        return orig(df_valid, parsed_valid, region, **kw)

    A._build_canon = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            getattr(A, loader)()
    finally:
        A._build_canon = orig
    return got["df"], got["created"]


def _build(region: str) -> pd.DataFrame:
    s = SPEC[region]
    df, created = _capture(s["loader"])
    cat = df[s["topic"]]
    out = pd.DataFrame({
        "region": region,
        "created": pd.to_datetime(created.values),
        "topic": [map_topic(v) if pd.notna(v) else "прочее" for v in cat],
        "cls": [classify_appeal(v) if pd.notna(v) else "problem" for v in cat],
        "executor": df[s["executor"]].values if s["executor"] else None,
    })
    if s["closed"]:
        col, fmt = s["closed"]
        out["closed"] = pd.to_datetime(df[col].values, format=fmt, errors="coerce")
    else:
        out["closed"] = pd.NaT
    if s["status"]:
        status = df[s["status"]].values
        unknown = set(pd.unique(status)) - set(FINAL[region])
        if unknown:
            raise ValueError(f"{region}: незнакомые статусы {sorted(map(str, unknown))} — "
                             "добавьте их в FINAL явно, угадывать нельзя")
        out["status"] = status
        out["final"] = [FINAL[region][v] for v in status]
    else:
        out["status"] = None
        out["final"] = pd.NA
    return out


def load(region: str, refresh: bool = False) -> pd.DataFrame:
    """Таблица закрытий региона; кэш в data/5o/."""
    path = CACHE / f"{region}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    out = _build(region)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def load_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {r: load(r, refresh) for r in SPEC}


if __name__ == "__main__":
    for region, d in load_all(refresh=True).items():
        print(f"{region}: {len(d)} строк; исполнитель {d.executor.notna().mean():.2%}; "
              f"факт закрытия {d.closed.notna().mean():.2%}; "
              f"финальный статус {pd.to_numeric(d.final, errors='coerce').mean():.2%}")
