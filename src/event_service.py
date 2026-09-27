"""Shared event feed; no UI dependency and no second detector."""
from pathlib import Path
import pandas as pd
from src import paths
from src.spikes import (SEASONAL_REGIONS, classify_seasonal, daily_counts, detect, gap_days, with_duration)
DATA = paths.UNIFIED
TYPE_RU = {"seasonal": "сезонное", "anomaly": "необычное", "без типа": "не с чем сравнить"}

def load_events(path=DATA, revision=None):
    """Дневные ряды и лента всплесков. Параметры детектора — как в src.spikes.

    Возвращает (daily, det, ev): дневные счётчики по срезам, таблицу с флагом
    всплеска по дням и ленту событий с длительностью и типом. Считается один раз
    и кладётся в кэш: детектор идёт по всем срезам всех регионов.
    """
    paths.require_source_marker(Path(path).parent / "SOURCE")
    full = pd.read_parquet(path, columns=["created_at", "region", "topic", "appeal_class"])
    full["created_at"] = pd.to_datetime(full["created_at"])
    problem = full[full.appeal_class == "problem"].copy()
    daily, bounds = daily_counts(problem)
    # Пропуски выгрузки ищутся по всем классам — см. gap_days в src/spikes.py
    det = detect(daily, blocked=gap_days(full, bounds))
    ev = classify_seasonal(with_duration(det), det, bounds, SEASONAL_REGIONS)
    # spike_type приходит как None либо NaN — оба значат «сравнивать не с чем»
    ev["тип"] = [TYPE_RU["без типа"] if (x is None or pd.isna(x)) else TYPE_RU.get(x, x)
                 for x in ev["spike_type"]]
    ev["конец"] = ev["дата"] + pd.to_timedelta(ev["дней подряд"] - 1, unit="D")
    return daily, det, ev


def region_last_day(daily):
    """Последний день данных по каждому региону — точка отсчёта «новизны».

    Отсчитывать от сегодняшней даты нельзя: выгрузка историческая и у регионов
    заканчивается в разное время (Караганда 2023-12, Павлодар 2026-07). От общей
    последней даты секция «требует внимания» показывала бы только Павлодар.
    """
    return daily.groupby("region")["день"].max()


def mark_new(ev, last_day, days):
    """Пометить события, попавшие в последние `days` дней СВОЕГО региона."""
    ev = ev.copy()
    edge = ev["регион"].map(last_day) - pd.Timedelta(days=days - 1)
    ev["новое"] = ev["конец"] >= edge
    return ev

