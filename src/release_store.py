"""Small filesystem release store. CURRENT is the only commit point."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ID = re.compile(r'[0-9a-f]{32}')
REQUIRED = ('data/unified.parquet','data/SOURCE','data/data_health.json',
            'data/last_build.json','data/build_counts.json','data/raw_manifest.json',
            'manifest/raw.json','reports/spikes.json','reports/forecast.json',
            'reports/forecast.md','reports/checks.json')


class ReleaseError(ValueError):
    """Only fixed operational messages, never exception payloads."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_bytes(path, content):
    path=Path(path)
    fd, name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    os.replace(name,path)
    sync_dir(path.parent)


def write_json(path, value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())


def runtime_path(value, source, project):
    root=Path(value).expanduser().resolve()
    if source not in ('real','fake'):
        raise ReleaseError('Источник должен быть real или fake')
    project=Path(project).resolve()
    if root.is_relative_to(project) and not root.is_relative_to(project/'runtime'):
        raise ReleaseError('Внутри проекта runtime разрешён только в отдельном runtime/')
    protected=[project/n for n in ('data','reports','models','tests','src','analysis',
                                  'fake_run','drive-download-20260907T161509Z-1-001')]
    if root==project or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ReleaseError('Runtime пересекается с защищёнными каталогами проекта')
    raw=os.environ.get('NAZAR_RAW_DIR')
    if raw:
        p=Path(raw).expanduser().resolve()
        internal=os.environ.get('_NAZAR_BUILD_RELEASE')
        staging_input=bool(internal and ID.fullmatch(internal) and p==root/'inputs'/internal)
        if not staging_input and (root.is_relative_to(p) or p.is_relative_to(root)):
            raise ReleaseError('Runtime пересекается с сырьём')
    owner=root/'SOURCE'
    if owner.exists() and owner.read_text().strip()!=source:
        raise ReleaseError('Runtime принадлежит другому источнику')
    return root


def initialize(root, source):
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'SOURCE').exists():
        if list(root.iterdir()):
            raise ReleaseError('Нужен пустой runtime или существующий runtime этого проекта')
        # Exclusive creation also arbitrates two initializers with different sources.
        with (root/'SOURCE').open('x') as f:
            f.write(source+'\n'); f.flush(); os.fsync(f.fileno())
    if (root/'SOURCE').read_text().strip()!=source:
        raise ReleaseError('Runtime принадлежит другому источнику')
    for name in ('releases','inputs','logs'):
        p=root/name
        if p.is_symlink(): raise ReleaseError('Symlink в runtime запрещён')
        p.mkdir(exist_ok=True)


def release_path(root, identity):
    if not isinstance(identity,str) or not ID.fullmatch(identity):
        raise ReleaseError('Некорректный ID релиза')
    path=root/'releases'/identity
    if path.is_symlink() or path.resolve()!=path:
        raise ReleaseError('Symlink релиза запрещён')
    return path


def current_id(root):
    if not (root/'CURRENT').exists(): return None
    value=(root/'CURRENT').read_text().strip()
    release_path(root,value)
    return value


def metadata(root, identity):
    path=release_path(root,identity)
    try:
        obj=json.loads((path/'metadata.json').read_text())
        if obj['release_id']!=identity or obj['status'] not in ('building','ready','failed'):
            raise ValueError()
        return obj
    except (OSError,ValueError,KeyError,TypeError):
        raise ReleaseError('Metadata релиза отсутствует или повреждён') from None


def selected(root, source, staging=None):
    identity=staging or current_id(root)
    if not identity: raise ReleaseError('Нет ACTIVE release; выполните nazar-refresh')
    obj=metadata(root,identity)
    if obj['source']!=source or (root/'SOURCE').read_text().strip()!=source:
        raise ReleaseError('SOURCE релиза не совпадает с режимом')
    if staging:
        if obj['status']!='building' or current_id(root)==identity:
            raise ReleaseError('Запись разрешена только в отдельный building release')
    elif obj['status']!='ready':
        raise ReleaseError('CURRENT не указывает на READY release')
    return release_path(root,identity),identity


@contextmanager
def writer_lock(root):
    with (root/'writer.lock').open('a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReleaseError('Другой refresh/rollback уже выполняется') from None
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


def artifact_hashes(path):
    result={}
    for p in sorted(path.rglob('*')):
        if p.is_symlink(): raise ReleaseError('Symlink в артефактах запрещён')
        if p.is_file() and p.name!='metadata.json':
            result[p.relative_to(path).as_posix()]=digest(p)
    return result


def verify_release(root, identity, source):
    path=release_path(root,identity); obj=metadata(root,identity)
    if obj['status']!='ready' or obj['source']!=source or (root/'SOURCE').read_text().strip()!=source:
        raise ReleaseError('Релиз не READY или имеет другой SOURCE')
    hashes=obj.get('artifacts',{})
    if not set(REQUIRED)<=set(hashes): raise ReleaseError('Нет обязательных артефактов')
    if artifact_hashes(path)!=hashes: raise ReleaseError('Артефакты отсутствуют или checksum не совпал')
    if (path/'data/SOURCE').read_text().strip()!=source:
        raise ReleaseError('SOURCE канона не совпадает')
    if digest(path/'manifest/raw.json')!=obj['raw_manifest_hash']:
        raise ReleaseError('Manifest hash не совпадает')
    from src.topic_mapping import mapping_revision
    if obj['mapping_version']!=mapping_revision():
        raise ReleaseError('Версия mapping не совпадает с кодом')
    return obj


def publish(root, identity, source):
    verify_release(root,identity,source)
    # This rename is the commit. No data copying or release deletion.
    atomic_bytes(root/'CURRENT',(identity+'\n').encode())


def sync_tree(path):
    for p in path.rglob('*'):
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    for p in sorted((x for x in path.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):
        sync_dir(p)
    sync_dir(path)
