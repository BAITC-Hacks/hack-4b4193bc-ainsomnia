"""Isolated staging process. Exceptions/stdout are captured by refresh."""
import importlib
import json
import sys
from src import paths
from src.release_store import write_json

MODULES = {'adapters':'src.adapters.adapters','unified':'src.adapters.build_unified',
           'mapping':'src.topic_mapping','labeling':'src.checks.labeling','quality':'src.data_health'}


def run(step):
    paths.require_writable()
    if not paths.STAGING: raise ValueError('Staging required')
    if step in MODULES:
        module=importlib.import_module(MODULES[step])
        result=module.run_all() if step=='adapters' else module.build() if step=='quality' else module.main()
        if isinstance(result,int) and result: raise ValueError('Stage failed')
    elif step=='spikes':
        from src.event_service import load_events
        _,_,events=load_events()
        counts=[{'region':r,'topic':t,'events':int(n)} for (r,t),n in events.groupby(['регион','тема']).size().items()]
        write_json(paths.REPORTS_DIR/'spikes.json',dict(source=paths.SOURCE,count=len(events),series=counts))
    elif step=='forecast':
        from src.data_health import load
        from src.planning import forecasts
        import pandas as pd
        health,blocked=load()
        if blocked or health is None: raise ValueError('Health unavailable')
        frame=pd.read_parquet(paths.UNIFIED,columns=['created_at','region','topic','appeal_class'])
        rows=forecasts(frame,health)
        write_json(paths.REPORTS_DIR/'forecast.json',dict(source=paths.SOURCE,series=rows,
                    available=bool(rows),reason='FAKE: рабочий прогноз не применяется' if paths.FAKE else 'Правила допуска Data Health'))
        if paths.FAKE:
            (paths.REPORTS_DIR/'forecast.md').write_text('ПОДДЕЛЬНЫЕ ДАННЫЕ. Рабочий прогноз не применяется.\n')
        else:
            from src.forecast import main
            sys.argv=['forecast']; main()
    elif step=='checks':
        from src.data_health import load
        from src.checks.labeling import main
        from src.topic_mapping import mapping_revision
        import pandas as pd
        if main(): raise ValueError('Labeling failed')
        health,blocked=load()
        if blocked or health is None: raise ValueError('Invalid health')
        frame=pd.read_parquet(paths.UNIFIED,columns=['created_at','region','topic','appeal_class'])
        if frame.empty or frame.created_at.isna().any(): raise ValueError('Empty canon')
        counts={str(k):int(v) for k,v in frame.appeal_class.value_counts().items()}
        if set(counts)-{'problem','info','system'}: raise ValueError('Class mismatch')
        if sum(h['rows'] for h in health['regions'].values())!=len(frame): raise ValueError('Counts mismatch')
        if paths.PREDICTIONS.exists() or paths.METRICS.exists() or paths.MODEL.exists():
            from src.actions import risk_summary
            from src.risk_explanation import explain_rank
            risk=risk_summary()
            if not risk['available'] or not explain_rank(0)['available'] or not explain_rank(risk['test_n']-1)['available']:
                raise ValueError('Explicit risk snapshot is not usable')
        write_json(paths.REPORTS_DIR/'checks.json',dict(row_count=len(frame),class_counts=counts,
                    regions=sorted(health['regions']),mapping_version=mapping_revision(),
                    data_as_of={r:h['last_date'] for r,h in health['regions'].items()}))
    else: raise ValueError('Unknown step')


if __name__=='__main__':
    try: run(sys.argv[1])
    except BaseException:
        # Caller logs only the fixed step and failure code, not raw exceptions.
        raise SystemExit(2) from None
