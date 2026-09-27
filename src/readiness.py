"""Data readiness is separate from Streamlit process liveness."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
from src import release_store as store


def health(runtime, source):
    result=dict(status='not_ready',source=source if source in ('real','fake') else 'unknown',
                release_id=None,data_as_of={},row_count=0,required_artifacts={},warnings=[])
    try:
        if not runtime: raise store.ReleaseError('Укажите NAZAR_RUNTIME_DIR; legacy mode не является ACTIVE release')
        root=store.runtime_path(runtime,source,Path(__file__).resolve().parents[1])
        identity=store.current_id(root)
        if not identity: raise store.ReleaseError('Нет ACTIVE release; выполните nazar-refresh')
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            meta=store.verify_release(root,identity,source)
        path=store.release_path(root,identity)
        profile=json.loads((path/'data/data_health.json').read_text())
        from src.data_health import validate_profile
        validate_profile(profile)
        state=json.loads((path/'data/last_build.json').read_text())
        facts=json.loads((path/'reports/checks.json').read_text())
        manifest=json.loads((path/'manifest/raw.json').read_text())
        if (state['state']!='ok' or state['source']!=source or profile['source']!=source
            or profile['dataset_id']!=manifest['dataset_id']
            or profile['unified_sha256']!=store.digest(path/'data/unified.parquet')
            or facts['row_count']!=meta['row_count'] or sum(h['rows'] for h in profile['regions'].values())!=meta['row_count']):
            raise store.ReleaseError('Проверки release/Health не согласованы')
        import pandas as pd
        frame=pd.read_parquet(path/'data/unified.parquet',columns=['created_at','region','topic','appeal_class'])
        if len(frame)!=meta['row_count'] or frame.created_at.isna().any():
            raise store.ReleaseError('Канон повреждён или число строк не совпало')
        counts={str(k):int(v) for k,v in frame.appeal_class.value_counts().items()}
        if (counts!=meta['class_counts'] or counts!=facts['class_counts']
            or sorted(frame.region.unique())!=meta['regions']
            or any(h['status']=='BLOCKED' for h in profile['regions'].values())):
            raise store.ReleaseError('Агрегаты или качество не согласованы')
        result.update(status='ready',release_id=identity,data_as_of=facts['data_as_of'],row_count=meta['row_count'],
                      required_artifacts={name:True for name in store.REQUIRED})
        result['warnings']=['Data Health: '+r for r,h in profile['regions'].items() if h['status']!='OK']
        if not (path/'reports/metrics.json').exists(): result['warnings'].append('Риск unavailable: модель не переносилась и не обучалась')
        result['warnings'].append('Витрина закрепляет release при запуске; после переключения требуется явный перезапуск')
    except (Exception,SystemExit):
        result['warnings']=['Нет согласованного ACTIVE release: проверьте CURRENT, SOURCE, metadata, checksums и журнал refresh']
    return result


def main():
    result=health(os.environ.get('NAZAR_RUNTIME_DIR'),os.environ.get('NAZAR_SOURCE','real'))
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0 if result['status']=='ready' else 2


if __name__=='__main__': raise SystemExit(main())
