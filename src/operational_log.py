"""Closed-vocabulary JSONL: child stdout and raw exceptions never enter logs."""
import json
import logging
import sys
from src.release_store import now

MESSAGES = {
 'build_started':'Начата отдельная сборка', 'source_validated':'Источник проверен',
 'manifest_validated':'Manifest и снимок входа проверены', 'adapter_finished':'Адаптеры завершены',
 'unified_built':'Канон собран','mapping_finished':'Темы размечены',
 'labeling_finished':'Разметка проверена','quality_finished':'Качество проверено',
 'spikes_finished':'Всплески рассчитаны','forecast_finished':'Прогноз обработан',
 'release_ready':'Все обязательные проверки пройдены','release_activated':'CURRENT переключён',
 'release_failed':'Сборка отклонена; ACTIVE не изменён', 'rollback':'Выбран прежний проверенный релиз',
}


def logger(path=None):
    log=logging.Logger('nazar.operational',level=logging.INFO)
    log.propagate=False
    stream=logging.StreamHandler(sys.stdout); stream.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(stream)
    if path:
        file=logging.FileHandler(path,encoding='utf-8'); file.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(file)
    return log


def emit(log,event,release_id,source,component,duration_ms=0,status='ok'):
    value=dict(timestamp=now(),level='ERROR' if status=='failed' else 'INFO',event=event,
               release_id=release_id,source=source,component=component,message=MESSAGES[event],
               duration_ms=round(duration_ms,3),status=status)
    log.info(json.dumps(value,ensure_ascii=False,sort_keys=True))
