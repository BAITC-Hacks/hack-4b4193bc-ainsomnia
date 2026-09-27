# P3 — локальная приёмка

Постановка до кода: `c6b0565`. P2 и принятый topic mapping не менялись. Push/deployment/VPS/S3/DNS не выполнялись. Ниже разделены выполненные проверки и внешний этап, который ещё предстоит.

## Что закрыто локально

- Отдельный `nazar-refresh`: snapshot входа, существующие восемь стадий, checksums, READY и atomic CURRENT. Legacy запуск совместим, его in-place build не объявляется атомарным.
- `nazar-rollback --release ID`: явная проверка и переключение без сборки/удаления. Предыдущие релизы сохраняются. Runtime владеет одним SOURCE; записи в ACTIVE штатными build/train-командами запрещены.
- JSONL operational logging с фиксированными сообщениями; сырые stdout/stderr дочерних стадий не публикуются. Пользовательских событий dashboard до auth нет.
- `nazar-health`: checksums/source/канон/Health/метаданные; JSON и exit 0/2. Streamlit endpoint остаётся liveness. Витрина закрепляет release на процесс, после switch нужен явный restart; ID показан на экране.
- Общие row/class counters через `export.summary_counts`, общий snapshot для action/spike/risk/freshness и brief. HTML/PDF сводки используют одни готовые строки; Excel/PDF прежнего экспорта — одни export_frames. При сравнении учитывается scope: старый экспорт фильтруется, оперативная сводка — все регионы.
- Подготовлены backup/restore, OIDC roles/scopes, S3 contract, внешние владельцы решений и deployment runbook. Секретов в `.env.example` нет, только пустые переменные.

## Atomic и failure injection

`tests/test_releases.py` делает настоящую FAKE-сборку в отдельном временном runtime. Проверены успешная активация, adapters failure, labeling failure, повреждение staged unified, неверный SOURCE, interrupt перед publish, missing raw, corrupted schema, wrong manifest, read-only directory, второй publish, previous_release и явный rollback. После отказов pointer и hashes старой версии совпадают, readiness прежней версии сохраняется. Проверены CLI nonzero и отсутствие traceback, JSONL schema/закрытый словарь, not-ready при повреждённом Health и исчезновении явно перенесённых metrics.

Для риска тест отдельно обучает FAKE donor в legacy-каталоге, затем refresh переносит его без обучения: hash модели совпадает. Отсутствующая изначально optional модель — warning/unavailable; потеря файла, объявленного в metadata, — повреждение и not ready. Автоматического удаления релизов нет.

Unavailable Chromium покрывается прежним `tests/test_dashboard.py`: экспорт PDF явно сообщает деградацию без графиков, исключения UI нет. Это намеренно не полный отказ PDF и не доказательство работы Chromium в Docker. Новая сводка PDF использует ReportLab без графиков/Chromium. Kill/power-loss hardware test не проводился; SIGKILL может оставить неактивный building или READY каталог. CURRENT — единственный commit marker; metadata READY не переписывается в нескольких местах ради фиктивной атомарной транзакции.

## REAL

В отдельный runtime собран REAL release с явно перенесённым прежним риском. Сверено: 988 776 строк, классы 658 103 / 308 305 / 22 368, 7 регионов, 1 225 всплесков. Канон, forecast.md, metrics.json, predictions.csv и model.pkl совпали побайтно с принятыми legacy-артефактами. Snapshot: 16 действий. Release-mode AppTest: 5 вкладок, 0 исключений. Legacy REAL build отдельно повторён для performance baseline; переобучение REAL не выполнялось.

## Локальный performance baseline

Apple M5, 16 ГБ RAM, arm64, macOS 27.0 по platform, Python 3.12.13. Однократные измерения perf_counter на локальной машине; файловый кэш ОС не сбрасывался. Это ориентиры поиска крупных регрессий, не SLA и не прогноз производительности VPS.

| Операция | Время |
|---|---:|
| Legacy build REAL | 47.552 с |
| CLI spikes REAL | 1.861 с |
| CLI forecast REAL | 2.059 с |
| Холодный Python import + dashboard load_data | 0.398 с |
| Brief HTML из готового snapshot | 0.000095 с |
| Brief PDF из готового snapshot | 0.066 с |

Время подготовки snapshot не включено в рендер HTML/PDF. Результат полного FAKE и окончательные regression/security receipts добавляются после чистого прогона без одновременной правки git.

## Container/configuration

Docker/Caddy локально отсутствуют. Оба compose YAML распарсены штатным Ruby/Psych; это не `docker compose config` и не проверка Engine. Существующие shell scripts прошли `bash -n`. Статически проверены USER app UID 1000, `/usr/bin/chromium`, BROWSER_PATH, DejaVu, отсутствие REAL runtime/raw/models/predictions в build context, loopback `127.0.0.1:8501`. Release compose: dashboard runtime RO, refresh RW, вход RAW/manifest RO, root filesystem RO, временные каталоги через tmpfs. Legacy compose дополнен RO models mount для explainability. В образ разрешена только встроенная FAKE fixture 5p, чтобы стала возможна полная контейнерная проверка.

Caddy — один host-side пример с placeholder `nazar.example.kz`, не настроенный домен. TLS, Engine, UID/bind mounts, браузер/PNG/PDF внутри контейнера, reverse proxy и доступность площадки должны быть проверены на следующем этапе. REAL доступ дополнительно зависит от OIDC/access/encryption/retention/legal решения. [Runbook](deployment.md), [внешние блокеры](deployment-external.md), [backup](backup.md), [auth](auth.md), [S3](s3.md).

Локальные артефакты тестов/замеров находятся вне git в `/private/tmp/nazar-p3/`; реальные inputs/releases содержат защищаемые данные и не предназначены для публикации. В publication уходят только код, тесты, fixture, конфигурационные примеры и проверенные агрегатные документы.
