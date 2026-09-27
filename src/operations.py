"""Read-only operational service; shared by UI and executive exports."""
import pandas as pd
from src import paths
from src.actions import queue, risk_summary
from src.data_health import load, PROFILE, BUILD_STATE
from src.event_service import load_events, region_last_day
from src.planning import forecasts, recurrence
from src.closure import load_closure


def revision():
    files = (paths.UNIFIED, PROFILE, BUILD_STATE, paths.DATA_DIR/'SOURCE',
             paths.PREDICTIONS, paths.METRICS, paths.REPORTS_DIR/'SOURCE', paths.MODEL)
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                 if p.exists() else (str(p), None) for p in files)


def unavailable(reason, health=None):
    return dict(available=False, source=health['source'] if health else ('fake' if paths.FAKE else 'real'),
                health=health, actions=queue(None,blocked=reason), forecasts=[], recurrence=[],
                risk={'available':False,'reason':'Аналитика недоступна'}, closure={'available':False,'rows':[]},
                new_spikes=0)


def snapshot(revision_key=None):
    before = revision()
    health, blocked = load()
    if blocked or health is None:
        result = unavailable(blocked or 'Нет профиля качества этой сборки',health)
        if not blocked:
            result['actions'] = queue(None)
        return result
    try:
        paths.require_source_marker(paths.DATA_DIR/'SOURCE')
        frame = pd.read_parquet(paths.UNIFIED,columns=['created_at','region','topic','appeal_class'])
        daily, _, events = load_events()
        f = forecasts(frame,health)
        risk = risk_summary()
        items = queue(health, events=events, last_day=region_last_day(daily),risk=risk,forecasts=f)
        result = dict(available=True, source=health['source'], health=health, actions=items,
                      forecasts=f, risk=risk, recurrence=recurrence(events,health),
                      closure=load_closure(health), new_spikes=sum(x['type']=='NEW_SPIKE' for x in items))
        if revision() != before:
            return unavailable('Сборка изменилась во время чтения; обновите страницу')
        return result
    except (OSError,ValueError,KeyError,TypeError):
        return unavailable('Не удалось прочитать согласованные артефакты; повторите проверенную сборку')
