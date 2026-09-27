"""Forecast, capacity and recurrence services; all dates refer to data."""
import math
import pandas as pd
from src import forecast as f


def forecasts(frame, health):
    problem = frame[frame.appeal_class == 'problem']
    out = []
    for region, h in sorted(health['regions'].items()):
        if not h['forecast']['available']:
            continue
        topics = [None] + list(problem[problem.region == region].topic.value_counts().head(f.TOP_TOPICS).index)
        for topic in topics:
            s = f.weekly(problem, region, topic)
            if topic is not None and s.median() < f.MIN_LEVEL:
                continue
            res = f.evaluate(s, 13, 3)
            if res is None:
                continue
            tr_folds = f.transition_folds(s, 13)
            mae = {m: f.mean_mae_folds(tr_folds, m) for m in f.MODELS} if tr_folds else {}
            valid = [m for m in mae if not math.isnan(mae[m])]
            tr = {'модель': min(valid, key=lambda m: mae[m])} if valid else None
            choice = f.forward_choice(s, 13, res, tr)
            if choice is None:
                continue
            model, reason, _ = choice
            pred = f.forecast(s, 13, model)
            if pred is None:
                continue
            end = pd.date_range(s.index.max(), periods=14, freq='W-MON')[-1]
            out.append(dict(region=region, topic=topic, model=model, reason=reason,
                            data_as_of=str(s.index.max().date()), horizon_end=str(end.date()),
                            weeks=13, demand=float(pred.sum()),
                            transition=bool(f.crosses_transition(s,s.index.max().month,end.month)),
                            limitation=h['forecast']['reason']))
    return out


def capacity(demand, operators, hours, aht, reserve, *, weeks=13):
    """Hours per operator over the entire forecast horizon, never daily hours."""
    values = (demand, operators, hours, aht, reserve, weeks)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Все параметры должны быть конечными числами')
    if demand < 0 or operators < 0 or int(operators) != operators or hours < 0 or aht <= 0 or not 0 <= reserve < 1 or weeks != 13:
        raise ValueError('Нужны неотрицательные поток/часы/целое число операторов, AHT > 0, резерв < 100%, горизонт 13 недель')
    raw = operators * hours * 60 / aht
    usable = raw * (1-reserve)
    return [dict(scenario=label, weeks=weeks, demand=demand*factor, raw_capacity=raw,
                 available_capacity=usable, utilization=demand*factor/usable if usable else None,
                 balance=usable-demand*factor,
                 required_operator_hours=demand*factor*aht/(60*(1-reserve)))
            for label, factor in [('-20%', .8), ('base', 1.), ('+20%', 1.2)]]


def recurrence(events, health):
    """Return aggregate region×topic results; no event rows are persisted."""
    if events.empty:
        return []
    result = []
    for (region, topic), group in events.groupby(['регион', 'тема'], sort=True):
        h = health['regions'][region]
        as_of = pd.Timestamp(h['last_date'])
        gaps = [(pd.Timestamp(g['first_date']), pd.Timestamp(g['last_date'])) for g in h['gaps_ge7']]
        g = group.sort_values('дата')
        starts, ends = list(g['дата']), list(g['конец'])
        def observed(a,b):
            return b <= as_of and not any(lo <= b and hi >= a for lo,hi in gaps)
        durations = []
        row = dict(region=region, topic=topic, spikes=len(g), data_as_of=str(as_of.date()))
        for horizon in (7,30,60):
            repeats = observable = 0
            for i,end in enumerate(ends):
                a, b = end+pd.Timedelta(days=1), end+pd.Timedelta(days=horizon)
                if observed(a,b):
                    observable += 1
                    repeats += int(i+1<len(starts) and end<starts[i+1]<=b)
            row[f'observable_{horizon}d'] = observable
            row[f'repeats_{horizon}d'] = repeats
            row[f'recurrence_{horizon}d'] = repeats/observable if observable else None
            row[f'censored_{horizon}d'] = (len(g)-observable)/len(g)
        for i,end in enumerate(ends[:-1]):
            next_start = starts[i+1]
            if next_start > end and observed(end+pd.Timedelta(days=1),next_start):
                durations.append((next_start-end).days)
        row['observed_pairs'] = len(durations)
        row['median_days_to_next_observed'] = float(pd.Series(durations).median()) if durations else None
        result.append(row)
    return result
