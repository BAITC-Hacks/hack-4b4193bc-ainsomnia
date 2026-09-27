# Atomic refresh — реализованный P3

Постановка — CLAUDE.md, P3. Legacy nazar-build-data по-прежнему поэтапный; новый nazar-refresh включается явно через NAZAR_RUNTIME_DIR. Без этой переменной прежние команды/пути не меняются.

```text
runtime/
  SOURCE
  CURRENT
  writer.lock
  releases/<id>/
    data/
    reports/
    models/
    manifest/raw.json
    metadata.json
  inputs/<id>/
  logs/<id>.jsonl
```

CURRENT — небольшой файл ID, единственная точка commit. Новый релиз создаётся отдельно с status building; source и принятый manifest проверяются, сырьё копируется в локальный снимок inputs и проверяется до/после использования. Эти снимки содержат ПДн, не коммитятся, не попадают в образ/backup автоматически и не удаляются самовольно. Переключение не копирует данные: temp → fsync/close → os.replace → fsync каталога. До него готовые файлы, metadata и каталоги fsync; symlink релизов/артефактов запрещён.

Существующие модули выполняются отдельными процессами: adapters → unified → mapping → labeling → health → spikes → forecast → checks. Второго детектора, адаптеров или forecast logic нет. Только фиксированные JSONL события, stdout/stderr стадий не пересылаются в журнал. Flock исключает второго writer/rollback; lock освобождается процессом/ОС, файл не удаляется. На macOS/Linux используются стандартные os/fcntl, сетевые файловые системы не заявлены поддержанными.

Metadata: release_id, created_at, source, raw_manifest_hash, row_count, regions, class_counts, mapping_version, status, build_steps, previous_release, artifacts/checksums и optional risk hashes. На диске завершённый metadata остаётся ready и неизменяемым; active определяется принадлежностью CURRENT. Так нет ложной транзакции сразу над pointer и несколькими metadata. Failed stage получает failed и failure_component; SIGKILL может оставить building, который не публикуется. Kill после READY, но до switch оставляет проверенный неактивный релиз. Ошибка после commit не переименовывает валидный CURRENT в failed.

Риск не обучается. --risk-from явно копирует согласованный источник с тем же dataset_id; hashes сохраняются, чтение риска и разложение крайних позиций проверяются. Без него риск unavailable, не нулевая очередь. Не допускается писать build/train в READY release штатными командами.

Каждый dashboard-процесс закрепляет один CURRENT при запуске: нет смешения старых/новых data/reports, в UI указан ID. После успешного refresh/rollback нужен **явный restart dashboard**. Hot-swap работающих сессий не реализован. nazar-health проверяет текущий указатель, а не обещает, что старый процесс уже перезапущен. Для переключения без рестарта впоследствии нужен отдельный request-scoped release context — сейчас этого обещания нет.

nazar-rollback --release ID: под writer lock проверяет source, READY, mapping и все hashes; затем тот же atomic switch. Пересборки и удаления нет. Failed и building запрещены; предыдущий release остаётся. Повреждённый текущий набор не считается ready, восстановление — из проверенного release/backup, не из частично изменённого каталога.

Тесты tests/test_releases.py выполняют настоящий FAKE build и failure injection: adapters, labeling, повреждённый unified, неверный SOURCE, interrupt перед publish, отсутствующий raw, плохая схема/manifest, read-only release directory, повторный publish и rollback. Сверяются pointer, hashes старой версии и её readiness. Отдельно проверяется отказ readiness при повреждённом профиле. Полный power-loss/fsync hardware test и запуск Docker — не проводились; это граница локальной приёмки, а не обещание устойчивости любой FS.

Эксплуатационные команды и права — [deployment.md](deployment.md); [backup/restore](backup.md). Scheduler, уведомления и автоматическая очистка не добавлены.
