"""Read a reviewed aggregate snapshot, never infer resolution quality."""
import hashlib
import json
from src import paths

STATUS_RU = {'VERIFIABLE': 'Можно проверять', 'PARTIAL': 'Частично',
             'NOT_OBSERVABLE': 'Данных недостаточно'}
NOTE = ('Ноль сигналов означает: по текущим данным правило триангуляции не нашло '
        'подтверждённого сигнала. Это не означает, что все работают хорошо. '
        'Карта показывает наблюдаемость статистических каналов, не доказывает факт решения.')


def load_closure(health, path=None):
    path = path or paths.ROOT / 'reports/closure_observability.json'
    try:
        snapshot = json.loads(path.read_text())
        keys = ('source','dataset_id','mapping_revision','unified_sha256')
        if health is None or health['source'] != 'real' or any(snapshot[k] != health[k] for k in keys):
            raise ValueError('Different release')
        report = paths.ROOT / snapshot['source_report']
        if hashlib.sha256(report.read_bytes()).hexdigest() != snapshot['source_report_sha256']:
            raise ValueError('Changed evidence')
        rows = snapshot['rows']
        if len(rows) != 21 or len({(r['region'],r['channel']) for r in rows}) != 21:
            raise ValueError('Incomplete snapshot')
        if any(r['status'] not in STATUS_RU or not r['reason'] for r in rows):
            raise ValueError('Invalid state')
        return {'available': True, 'rows': rows, 'note': NOTE, 'snapshot_note': snapshot['snapshot_note']}
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'rows': [], 'note': NOTE,
                'reason': 'Для этой версии данных нет проверенной карты 5o; историческая карта не переносится автоматически'}
