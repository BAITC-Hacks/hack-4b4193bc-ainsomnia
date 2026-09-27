"""Real filesystem failure injection on FAKE only; no real rows are printed."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ReleaseTests(unittest.TestCase):
    def test_fake_lifecycle_in_isolated_process(self):
        env=os.environ.copy()
        for key in ('NAZAR_RUNTIME_DIR','NAZAR_WORK_DIR','NAZAR_RAW_DIR','_NAZAR_BUILD_RELEASE'):
            env.pop(key,None)
        env['NAZAR_SOURCE']='fake'
        result=subprocess.run([sys.executable,'-m','tests.test_releases','--scenario'],env=env,
                              capture_output=True,text=True)
        # Never include subprocess stdout/stderr: it could contain broken input.
        self.assertEqual(result.returncode,0,'Release scenario failed; inspect controlled assertion name locally')
        self.assertIn('release scenarios passed',result.stdout)


def scenarios():
    import json
    import shutil
    import tempfile
    from contextlib import redirect_stdout
    import io
    from unittest.mock import patch
    from src import release_store as store
    from src.refresh import refresh, rollback, worker
    from src.readiness import health
    project=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='nazar-release-tests-') as directory:
        base=Path(directory).resolve(); raw=base/'raw'; root=base/'runtime'
        shutil.copytree(project/'tests/fixtures/fake_export',raw)
        os.environ['NAZAR_RAW_DIR']=str(raw)
        output=io.StringIO()
        with redirect_stdout(output):
            first=refresh(root,'fake')
            assert store.current_id(root)==first, 'activation'
            first_path=store.release_path(root,first)
            first_hashes=store.artifact_hashes(first_path)
            h=health(root,'fake')
            assert h['status']=='ready' and h['row_count']==12910, 'readiness'
            assert json.loads((first_path/'reports/spikes.json').read_text())['count']==1, 'spikes'
            assert json.loads((first_path/'metadata.json').read_text())['class_counts']=={'problem':8266,'info':3661,'system':983}
            assert any('Риск unavailable' in x for x in h['warnings']), 'optional risk'

            def rejected(**kwargs):
                try: refresh(root,'fake',**kwargs)
                except (store.ReleaseError,OSError): pass
                else: raise AssertionError('injected failure accepted')
                assert store.current_id(root)==first, 'failed release changed CURRENT'
                assert store.artifact_hashes(first_path)==first_hashes, 'failed release changed artifacts'
                assert health(root,'fake')['status']=='ready', 'old dashboard data stopped working'

            for stage in ('adapters','labeling'):
                def fail(step,env):
                    if step==stage: raise store.ReleaseError('Injected controlled failure')
                    worker(step,env)
                rejected(run_worker=fail)

            def corrupt(step,env):
                worker(step,env)
                p=store.release_path(root,env['_NAZAR_BUILD_RELEASE'])
                if step=='unified': (p/'data/unified.parquet').write_bytes(b'broken')
            rejected(run_worker=corrupt)
            def wrong_source(step,env):
                worker(step,env)
                if step=='unified':
                    (store.release_path(root,env['_NAZAR_BUILD_RELEASE'])/'data/SOURCE').write_text('real\n')
            rejected(run_worker=wrong_source)
            def interrupt(): raise KeyboardInterrupt()
            rejected(before_publish=interrupt)

            file=next(raw.glob('*.csv')); saved=file.read_bytes()
            moved=file.with_suffix('.missing'); file.rename(moved)
            try: rejected()
            finally: moved.rename(file)
            file.write_bytes(b'bad,header\n1,2\n')
            try: rejected()
            finally: file.write_bytes(saved)
            bad_manifest=base/'bad-manifest.json'; bad_manifest.write_text('{}')
            rejected(manifest_path=bad_manifest)

            mode=(root/'releases').stat().st_mode
            (root/'releases').chmod(0o500)
            try:
                if os.geteuid()!=0: rejected()
                else:
                    with patch('src.release_store.atomic_bytes',side_effect=PermissionError):
                        with unittest.TestCase().assertRaises(PermissionError): refresh(root,'fake')
            finally: (root/'releases').chmod(mode)

            second=refresh(root,'fake')
            assert second!=first and store.current_id(root)==second, 'second publish'
            assert store.artifact_hashes(first_path)==first_hashes, 'previous release not retained'
            assert store.metadata(root,second)['previous_release']==first, 'previous link'
            rollback(root,'fake',first)
            assert store.current_id(root)==first and health(root,'fake')['status']=='ready', 'rollback'
            assert store.release_path(root,second).exists(), 'rollback removed a release'
            with unittest.TestCase().assertRaises(store.ReleaseError): rollback(root,'real',second)
            assert store.current_id(root)==first, 'source isolation'

            # External corruption makes readiness fail; restore only this test's bytes.
            profile=first_path/'data/data_health.json'; content=profile.read_bytes()
            profile.write_text('{}')
            try: assert health(root,'fake')['status']=='not_ready','corrupt health accepted'
            finally: profile.write_bytes(content)
            assert health(root,'fake')['status']=='ready'

        records=[json.loads(line) for line in output.getvalue().splitlines()]
        expected={'timestamp','level','event','release_id','source','component','message','duration_ms','status'}
        assert records and all(set(row)==expected for row in records),'log schema'
        from src.operational_log import MESSAGES
        assert all(row['message']==MESSAGES[row['event']] for row in records),'closed log vocabulary'
        assert not any('bad,header' in row['message'] for row in records),'raw in log'
        print('release scenarios passed')


if __name__=='__main__':
    if '--scenario' in sys.argv:
        try: scenarios()
        except BaseException as exc:
            # Safe assertion identifiers only, never underlying exception values.
            print('scenario failure type:',type(exc).__name__)
            raise SystemExit(1) from None
    else: unittest.main()
