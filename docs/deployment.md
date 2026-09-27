# Deployment runbook — P3

Локальная подготовка, без выполненного deployment. Основной источник правил — [CLAUDE.md](../CLAUDE.md). Запрет push/VPS/S3/DNS сохраняется до отдельного разрешения. Полный контейнерный сценарий и PDF в Docker предстоит проверить на доступной площадке.

## 1. Requirements

macOS/Linux для разработки; Linux VPS с Docker Engine/Compose, диском под отдельные releases и raw snapshots. Python 3.12 и lock — для запуска вне образа. Контейнер использует UID 1000; host runtime должен принадлежать согласованному сервисному пользователю с правом записи для refresh. Не исправлять права `chmod 777`. Права чтения REAL/backup — по политике заказчика.

## 2. Clone

После разрешённой публикации клонировать согласованный репозиторий и checkout проверенного commit. Сейчас подготовлена локальная publication, push не выполнен. Сохранить commit рядом с эксплуатационной конфигурацией. Образ строить из этого checkout, без локального raw/runtime в контексте.

## 3. Environment

[.env.example](../.env.example) содержит только имена и пустые значения. Не копировать его как готовую конфигурацию: SOURCE должен быть явно fake или real, WORK_DIR — только legacy FAKE и несовместим с release mode. S3/OIDC не требуются текущему приложению. Реальные credentials — secret store; .env/key/credentials/secrets игнорируются.

Legacy команды остаются прежними без NAZAR_RUNTIME_DIR. Установка новых команд локально: `uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .` после установки lock. Release mode включается явно:

```bash
export NAZAR_SOURCE=fake
export NAZAR_RUNTIME_DIR="$PWD/runtime/fake"
.venv/bin/nazar-refresh
.venv/bin/nazar-health
.venv/bin/nazar-dashboard
```

## 4. Fake deployment

Следующие команды — инструкция для разрешённого этапа на площадке, здесь не выполнялись. Контекст сборки допускает только встроенную FAKE fixture, не REAL raw. Из корня checkout:

```bash
export NAZAR_SOURCE=fake
export NAZAR_RUNTIME_HOST="$PWD/runtime/fake"
export NAZAR_RAW_HOST="$PWD/tests/fixtures/fake_export"
export NAZAR_MANIFEST_DIR="$PWD/runtime/manifests"
mkdir -p "$NAZAR_RUNTIME_HOST" "$NAZAR_MANIFEST_DIR"
# Проверить/согласовать владельца UID 1000 для runtime до запуска.
docker compose -f deploy/compose.release.yml config
docker compose -f deploy/compose.release.yml build dashboard
docker compose -f deploy/compose.release.yml --profile operations run --rm refresh
docker compose -f deploy/compose.release.yml --profile operations run --rm refresh nazar-health
docker compose -f deploy/compose.release.yml up -d dashboard
```

Ожидается 12 910 строк, классы 8 266 / 3 661 / 983, один всплеск. Refresh не обучает модель; риск будет unavailable, рабочий forecast FAKE не применяется. Для проверки риска модель обучается отдельной legacy FAKE-командой, затем явно переносится через `--risk-from` согласованного read-only donor mount. На REAL её переносить нельзя.

## 5. Health/readiness

`/_stcore/health` — только liveness процесса. `nazar-health` — проверка актуального CURRENT: source, metadata, hashes обязательных файлов, канон и Data Health; JSON, exit 0 ready / 2 not ready. Отсутствие неперенесённой модели — warning; исчезновение артефакта, записанного в hashes, — not ready. При failed новой сборке прежний CURRENT остаётся ready.

Dashboard закрепляет release **на весь срок процесса** и показывает ID. Это сознательная защита от смешанных версий, не hot-swap. После успешного refresh/rollback проверить CURRENT командой health и явно перезапустить dashboard. Сверить показанный в UI ID с CURRENT; health нового CURRENT не доказывает, что старый процесс уже перезапущен.

## 6. Reverse proxy

Internet → HTTPS proxy на HOST → `127.0.0.1:8501` → контейнер. Оба compose файла публикуют только loopback; внутри контейнера Streamlit может слушать `0.0.0.0`. [Docker: host address publishing](https://docs.docker.com/get-started/docker-concepts/running-containers/publishing-ports/).

Один пример — [Caddyfile.example](../deploy/Caddyfile.example). `nazar.example.kz` — placeholder, заменить на согласованный домен. Caddy здесь запускается на host, поэтому его 127.0.0.1 относится к host. Не переносить этот пример в отдельный bridge-контейнер без изменения upstream. Стандартная директива — [reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).

## 7. HTTPS

После разрешения DNS проверить A/AAAA на нужный VPS, доступность необходимых портов для выбранного способа выдачи сертификата и `caddy validate --config <approved-file> --adapter caddyfile`. Домен в Caddyfile включает [Automatic HTTPS](https://caddyserver.com/docs/automatic-https). Пока Caddy локально не установлен, валидация его бинарником и выдача сертификата не выполнены. HTTPS не заменяет auth; REAL не открывается до IAM/access решения.

## 8. S3 future integration

[Контракт S3](s3.md): SDK не установлен, подключения нет. Не включать REAL raw и backups с ПДн автоматически.

## 9. Real data prerequisites

Не выполнены автоматически: approved data location, filesystem encryption decision, access list, OIDC decision, retention decision, правовое основание. Таблица владельцев — [deployment-external.md](deployment-external.md). REAL manifest принимается прежней явной командой после просмотра агрегатов. В контейнере принятый файл доступен read-only под `/manifests/`; raw — отдельный read-only mount. SOURCE=real задаётся явно.

## 10. Refresh

Локально: `nazar-refresh --manifest /approved/raw_manifest.json --risk-from /approved/legacy-workspace` с SOURCE=real и NAZAR_RUNTIME_DIR. `--risk-from` необязателен; требует совпадающих source/dataset_id и согласованных сохранённых артефактов. Никакого автоматического train.

В контейнере: `docker compose -f deploy/compose.release.yml --profile operations run --rm refresh nazar-refresh --manifest /manifests/raw_manifest.json`. Donor mount для риска настраивается явно; по умолчанию его нет. Сначала source/manifest, отдельный снимок входа, восемь этапов, проверки, READY, затем atomic CURRENT. Ошибка оставляет CURRENT прежним. Одновременно допускается один writer. После успеха health → `docker compose -f deploy/compose.release.yml restart dashboard` → сверка ID в UI.

## 11. Rollback

Выбрать ID предыдущего READY release из metadata текущего. `nazar-rollback --release <id>` при том же runtime/source проверяет hashes и mapping, переключает CURRENT без пересборки. В compose используется тот же operations service с командой nazar-rollback. Затем health и явный restart dashboard. Failed/building не допускаются. Релизы не удаляются.

## 12. Backup/restore

[Инструкция](backup.md). Restore в новый отдельный runtime, проверка trusted checksum и безопасная распаковка, затем explicit rollback/publish выбранного READY release. Не копировать архив поверх ACTIVE.

## 13. Logs

`runtime/logs/<release-id>.jsonl`, rollback отдельно; те же JSONL в stdout CLI. Пишутся фиксированные event/component/message и длительности. Raw stdout дочерних стадий и exception payload не пересылаются. Failed metadata указывает этап; журнал не является дампом данных. Dashboard user events отсутствуют до auth. Логи не содержат caller credentials и текстов граждан. Rotation/retention — внешняя политика, автоматического удаления нет.

## 14. Troubleshooting

| Симптом | Проверка |
|---|---|
| configuration failure | Runtime/source, права и отсутствие WORK_DIR в release mode |
| source/manifest failure | Разрешённое сырьё, все ожидаемые файлы, схема и принятый manifest |
| adapters/unified/mapping/labeling failure | Этап failed metadata; воспроизводить на копии сырья в закрытом контуре, не печатать сырые строки |
| quality/checks failure | Согласованность profile, mapping, fingerprints и явно перенесённого риска |
| CURRENT не изменился | При ошибке это ожидаемая защита; прежняя версия доступна |
| building остался после kill | Не READY; не переключать на него вручную, создать новый refresh |
| UI показывает предыдущий release | Проверить health CURRENT и явно перезапустить dashboard |
| PDF без графиков | Проверить Chromium/BROWSER_PATH; существующий экспорт сообщает о деградации, это не полный PDF success |
| Permission denied | UID 1000 и права host bind mounts; dashboard runtime RO, refresh RW |

## 15. Security checklist

Перед реальным допуском подтвердить источник/manifest, отсутствие real raw/models/predictions в образе/git, loopback publishing, firewall и proxy, identity/scope, политику диска/backup/retention, секреты вне git, restore drill и read-only dashboard mounts. Реальные параметры и разрешения ещё нужны. Dockerfile содержит Chromium `/usr/bin/chromium`, DejaVu, USER app UID 1000; kaleido использует BROWSER_PATH. YAML синтаксис проверен локально, но `docker compose config`, UID/mount/Chromium/PDF внутри контейнера пока не проверены из-за отсутствия Docker. **FAKE deployment — следующий проверочный этап, не уже выполненный запуск.**
