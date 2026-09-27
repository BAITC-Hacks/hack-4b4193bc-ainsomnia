"""Deterministic aggregation of validated signals, independent of Streamlit."""
import hashlib
import json

from src.event_service import mark_new

TYPES = ('NEW_SPIKE', 'SLA_RISK', 'DATA_QUALITY', 'DATA_STALE', 'FORECAST_CHANGE')
SEVERITIES = ('BLOCKED', 'HIGH', 'WATCH')
KARAGANDA = 'Карагандинская область'


def action(kind, severity, region, date, headline, evidence, reason, check,
           source, *, topic=None, data_as_of=None):
    identity = json.dumps([kind, region, topic, str(date)], ensure_ascii=False)
    return dict(stable_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                type=kind, severity=severity, region=region, topic=topic,
                event_date=str(date) if date is not None else None,
                data_as_of=str(data_as_of) if data_as_of is not None else None,
                headline=headline, evidence=evidence, reason=reason,
                recommended_next_check=check, source_module=source)


def queue(health, *, blocked=None, events=None, last_day=None, risk=None, forecasts=()):
    if blocked or health is None:
        return [action('DATA_QUALITY', 'BLOCKED' if blocked else 'WATCH',
                       None, None, 'Аналитика недоступна', {},
                       blocked or 'Нет профиля качества этой сборки',
                       'Повторить проверенную сборку nazar-build-data', 'data_health')]
    items = []
    for region, h in sorted(health['regions'].items()):
        if h['status'] != 'OK':
            items.append(action('DATA_QUALITY', 'BLOCKED' if h['status'] == 'BLOCKED' else 'WATCH',
                                region, h['last_date'], 'Проверить качество выгрузки',
                                {'rows': h['rows']}, '; '.join(h['reasons']),
                                'Открыть «Качество и свежесть данных»', 'data_health',
                                data_as_of=h['last_date']))
    if events is not None and not events.empty:
        marked = mark_new(events, last_day, 7)
        for _, e in marked[marked['новое']].iterrows():
            items.append(action('NEW_SPIKE', 'WATCH' if e['spike_type'] == 'seasonal' else 'HIGH',
                                e['регион'], e['дата'].date(), 'Новый всплеск обращений',
                                {'count': int(e['обращений']), 'baseline': float(e['медиана окна']),
                                 'excess': float(e['прирост']), 'multiple': float(e['кратность'])},
                                'Существующий детектор; последние 7 дней данных своего региона',
                                'Проверить дневной ряд и полноту выгрузки; затем уточнить ситуацию',
                                'spikes', topic=e['тема'], data_as_of=last_day[e['регион']].date()))
    if risk and risk.get('available'):
        items.append(action('SLA_RISK', 'HIGH', KARAGANDA, risk['cutoff'],
                            'Очередь проверки риска — исторический тест Караганды',
                            {k: risk[k] for k in ('n', 'precision', 'share', 'test_n')},
                            'Верхние 20% по существующей модели; точность на отложенном тесте',
                            'Открыть «Риск просрочки»; подтвердить текущую применимость',
                            'risk_view', data_as_of=risk['data_as_of']))
    elif risk and risk.get('blocked'):
        items.append(action('SLA_RISK', 'BLOCKED', KARAGANDA, None,
                            'Модуль риска недоступен', {}, risk['reason'],
                            'Проверить согласованность сохранённых артефактов риска', 'risk_view'))
    for f in forecasts:
        if f['transition']:
            items.append(action('FORECAST_CHANGE', 'HIGH', f['region'], f['data_as_of'],
                                'Окно прогноза пересекает смену сезона',
                                {'demand': f['demand'], 'weeks': f['weeks'], 'model': f['model']},
                                f['reason'], 'Открыть планирование; проверить применимость сезонного профиля',
                                'forecast', topic=f['topic'], data_as_of=f['data_as_of']))
    return sorted(items, key=lambda x: (SEVERITIES.index(x['severity']), TYPES.index(x['type']),
                  -x['evidence'].get('excess', 0), x['event_date'] or '', x['stable_id']))


def risk_summary():
    """Only aggregates leave this service; absent model is not an empty queue."""
    from src import paths
    from src.risk_view import PRED, METRICS, WORK_SHARE, headline, load_risk, top_share
    if not PRED.exists() or not METRICS.exists():
        return {'available': False, 'reason': 'Нет готового результата модели риска'}
    try:
        h, rows = headline(), load_risk()
        selected = top_share(rows, WORK_SHARE)
        if len(rows) != h['test_n'] or rows.empty:
            raise ValueError('risk rows')
        import pandas as pd
        as_of = pd.to_datetime(rows['дата'], format='%d.%m.%Y %H:%M').max().date()
        return dict(available=True, n=selected['n'], precision=selected['precision'],
                    share=WORK_SHARE, test_n=h['test_n'], cutoff=h['cutoff'],
                    data_as_of=str(as_of), source=h['source'])
    except (paths.SourceError, ValueError, KeyError, OSError):
        return {'available': False, 'blocked': True,
                'reason': 'Источник или схема артефактов риска несовместимы'}
