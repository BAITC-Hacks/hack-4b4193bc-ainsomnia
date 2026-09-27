"""Explicit immutable data refresh and rollback, opt-in release mode."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from src import release_store as store
from src.operational_log import logger, emit

PROJECT=Path(__file__).resolve().parents[1]
STEPS=(('adapters','adapter_finished'),('unified','unified_built'),('mapping','mapping_finished'),
       ('labeling','labeling_finished'),('quality','quality_finished'),('spikes','spikes_finished'),
       ('forecast','forecast_finished'),('checks','release_ready'))


def worker(step, env):
    result=subprocess.run([sys.executable,'-m','src.release_worker',step],cwd=PROJECT,
                          env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode: raise store.ReleaseError('Обязательный этап сборки не прошёл')


def copy_risk(donor, target, manifest):
    donor=Path(donor).resolve()
    profile=json.loads((donor/'data/data_health.json').read_text())
    if profile['source']!=manifest['source'] or profile['dataset_id']!=manifest['dataset_id']:
        raise store.ReleaseError('Исторический риск относится к другому источнику/набору данных')
    if (donor/'reports/SOURCE').read_text().strip()!=manifest['source']:
        raise store.ReleaseError('Источник риска не совпадает')
    files=('reports/SOURCE','reports/metrics.json','reports/predictions.csv','models/model.pkl')
    before={f:store.digest(donor/f) for f in files}
    for f in files:
        (target/f).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(donor/f,target/f)
    if before!={f:store.digest(donor/f) for f in files} or before!={f:store.digest(target/f) for f in files}:
        raise store.ReleaseError('Артефакты риска менялись при переносе')
    return before


def refresh(root, source, *, manifest_path=None, risk_from=None, run_worker=worker, before_publish=None):
    root=store.runtime_path(root,source,PROJECT)
    store.initialize(root,source)
    with store.writer_lock(root):
        identity=uuid.uuid4().hex
        release=store.release_path(root,identity); release.mkdir()
        for name in ('data','reports','models','manifest'): (release/name).mkdir()
        log=logger(root/'logs'/f'{identity}.jsonl')
        started=time.monotonic(); component='source'
        meta=dict(release_id=identity,created_at=store.now(),source=source,raw_manifest_hash=None,
                  row_count=0,regions=[],class_counts={},mapping_version=None,status='building',
                  build_steps=[],previous_release=store.current_id(root))
        store.write_json(release/'metadata.json',meta)
        emit(log,'build_started',identity,source,'refresh',status='building')
        try:
            # Paths and manifest readers can print legacy messages; never forward them.
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                from src import paths
                from src.dataset_manifest import generate,verify
                paths.validate_raw_source(paths.RAW_DIR,recursive=True)
            if not paths.RAW_DIR.is_dir(): raise store.ReleaseError('Нет каталога raw')
            emit(log,'source_validated',identity,source,component)
            component='manifest'
            if source=='real':
                if manifest_path is None: raise store.ReleaseError('Нужен явно принятый REAL manifest')
                manifest=verify(paths.RAW_DIR,Path(manifest_path),source=source)
                fresh=generate(paths.RAW_DIR,source=source)
                if manifest['dataset_id']!=fresh['dataset_id']: raise store.ReleaseError('Manifest schema mismatch')
            else:
                manifest=generate(paths.RAW_DIR,source=source,include_xlsx=False)
                if manifest_path:
                    verify(paths.RAW_DIR,Path(manifest_path),source=source,include_xlsx=False)
            inputs=root/'inputs'/identity; inputs.mkdir()
            for entry in manifest['files']:
                name=entry['relative_path']; out=inputs/name; out.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(paths.RAW_DIR/name,out)
            for name in ('manifest/raw.json','data/raw_manifest.json'):
                store.write_json(release/name,manifest)
            verify(inputs,release/'manifest/raw.json',source=source,include_xlsx=source=='real')
            verify(paths.RAW_DIR,release/'manifest/raw.json',source=source,include_xlsx=source=='real')
            meta['raw_manifest_hash']=store.digest(release/'manifest/raw.json')
            emit(log,'manifest_validated',identity,source,component)
            # Only the accepted previous successful profile is copied for drift.
            if meta['previous_release']:
                old=store.release_path(root,meta['previous_release'])/'data/data_health.json'
                if old.exists(): shutil.copyfile(old,release/'data/data_health.json')
            if risk_from:
                component='risk'
                meta['risk_artifacts']=copy_risk(risk_from,release,manifest)
                meta['risk_dataset_id']=manifest['dataset_id']
            env=os.environ.copy()
            env.pop('NAZAR_WORK_DIR',None)
            env.update(NAZAR_SOURCE=source,NAZAR_RUNTIME_DIR=str(root),NAZAR_RAW_DIR=str(inputs),
                       _NAZAR_BUILD_RELEASE=identity)
            for index,(component,event) in enumerate(STEPS,1):
                # Existing data_health.build requires stage 5 and matching manifest.
                state=dict(state='running' if index<=5 else 'ok',stage=min(index,5),source=source,
                           dataset_id=manifest['dataset_id'],timestamp=store.now())
                store.write_json(release/'data/last_build.json',state)
                t=time.monotonic(); run_worker(component,env)
                elapsed=(time.monotonic()-t)*1000
                meta['build_steps'].append(dict(component=component,status='ok',duration_ms=round(elapsed,3)))
                store.write_json(release/'metadata.json',meta)
                if component!='checks': emit(log,event,identity,source,component,elapsed)
            component='checks'
            verify(inputs,release/'manifest/raw.json',source=source,include_xlsx=source=='real')
            facts=json.loads((release/'reports/checks.json').read_text())
            meta.update({k:facts[k] for k in ('row_count','regions','class_counts','mapping_version')})
            meta['artifacts']=store.artifact_hashes(release)
            meta['status']='ready'
            store.write_json(release/'metadata.json',meta)
            store.sync_tree(release)
            store.sync_dir(root/'releases')
            store.verify_release(root,identity,source)
            emit(log,'release_ready',identity,source,'checks',(time.monotonic()-started)*1000,status='ready')
            component='publish'
            if before_publish: before_publish()
            store.publish(root,identity,source)
            emit(log,'release_activated',identity,source,'publish',status='active')
            return identity
        except BaseException:
            # After CURRENT commit, never relabel that valid release as failed.
            if store.current_id(root)==identity:
                return identity
            meta['status']='failed'; meta['failure_component']=component
            try: store.write_json(release/'metadata.json',meta)
            except OSError: pass
            emit(log,'release_failed',identity,source,component,(time.monotonic()-started)*1000,status='failed')
            raise store.ReleaseError('Сборка отклонена; ACTIVE release сохранён') from None
        finally:
            for h in log.handlers: h.close()


def rollback(root,source,identity):
    root=store.runtime_path(root,source,PROJECT)
    with store.writer_lock(root):
        store.publish(root,identity,source)
        log=logger(root/'logs'/'rollback.jsonl')
        try: emit(log,'rollback',identity,source,'rollback',status='active')
        finally:
            for h in log.handlers: h.close()


def arguments(rollback_command=False):
    p=argparse.ArgumentParser(description='Явное переключение проверенной версии данных; без обучения риска')
    p.add_argument('--runtime',default=os.environ.get('NAZAR_RUNTIME_DIR'))
    if rollback_command: p.add_argument('--release',required=True)
    else:
        p.add_argument('--manifest',type=Path)
        p.add_argument('--risk-from',type=Path)
    return p.parse_args()


def command(rollback_command=False):
    args=arguments(rollback_command)
    source=os.environ.get('NAZAR_SOURCE','real').strip().lower()
    log=logger()
    try:
        if not args.runtime or source not in ('fake','real'):
            raise store.ReleaseError('Нужны runtime и корректный источник')
        os.environ['NAZAR_RUNTIME_DIR']=str(Path(args.runtime).expanduser().resolve())
        os.environ.pop('_NAZAR_BUILD_RELEASE',None)
        if os.environ.get('NAZAR_WORK_DIR'):
            raise store.ReleaseError('WORK_DIR несовместим с release mode')
        if rollback_command: rollback(args.runtime,source,args.release)
        else: refresh(args.runtime,source,manifest_path=args.manifest,risk_from=args.risk_from)
        return 0
    except (Exception,SystemExit):
        emit(log,'release_failed',None,source if source in ('real','fake') else 'unknown',
             'configuration' if not rollback_command else 'rollback',status='failed')
        return 2


def main(): return command()
def rollback_main(): return command(True)

if __name__=='__main__': raise SystemExit(main())
