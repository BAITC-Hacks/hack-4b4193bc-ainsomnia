# Атомарное обновление: проект миграции

Реализация отложена. Сейчас `src.cli.build_data` пишет пять шагов в действующие `data/`, а `paths.py` фиксирует пути при импорте; forecast и риск используют отдельные каталоги отчётов. `last_build.json` блокирует витрину при неуспехе, но не возвращает старые файлы. Подмена одного symlink без закрепления release у всех читателей оставила бы смешанные версии. Поэтому минимальный wrapper не даёт обещанной атомарности.

Будущее размещение на одном файловом томе:

```text
releases/
  staging-<id>/
  <release-id>/
    data/
    reports/
    release.json
current -> releases/<release-id>
build-status/last-attempt.json
```

Только один writer под эксклюзивной блокировкой. Он получает неизменяемый снимок разрешённого сырья; исходный manifest проверяется до и после чтения. Порядок: raw schema/date validation → manifest verification → adapters → unified → labeling → Data Health → spikes → forecast → проверки источника/схем/ПДн/числовых инвариантов → publish. Текущий `nazar-build-data` пока НЕ выполняет все эти этапы в release.

`release.json` содержит source, dataset_id, mapping_revision, версию кода, hashes всех обязательных результатов, время, перечень успешных проверок, допускаемые unavailable-модули и ссылки на отдельно проверенные артефакты риска. Forecast может быть явно unavailable по правилам допуска; ошибка выполнения не эквивалентна unavailable и запрещает publish. Риск не переобучляется автоматически. Старый risk artifact допустим только как явно исторический результат с собственными периодом/manifest/hash; его нельзя объявить прогнозом новой выгрузки.

После проверок: fsync файлов/каталогов → rename staging в окончательный release → создание временной ссылки → `os.replace` указателя `current` на том же томе → fsync родителя. До replace ошибка оставляет current прежним. После replace новый release должен быть целиком завершён; сбой записи журнала не делает его незавершённым. Failure metadata пишется отдельно от текущего release и не содержит сырых значений. Незавершённые каталоги не читаются. Удаление/GC — только после отдельного разрешения и retention policy.

Миграция по шагам:

1. Ввести явный immutable ReleasePaths во всех writer/read API, сохранить нынешние source/fingerprint ограничения. Не открывать real `NAZAR_WORK_DIR` как обход защиты.
2. Научить CLI запускать каждый шаг с одним release context. Побочные записи в текущие data/reports запрещены тестом.
3. Dashboard/service/cache/export закрепляют resolved current один раз на запрос/сессию экспорта; кэш включает release_id, mapping_revision и будущий auth scope.
4. Запустить staging в shadow-режиме на FAKE, сравнить с прежним конвейером; затем на разрешённом REAL. Проверить ready files.
5. Провести failure injection, затем разрешить atomic publish. Scheduler подключать только после определения канала, частоты и SLA обновления заказчиком.

Обязательные failure tests: отказ каждого из десяти этапов; kill writer до/после rename и до/после replace; заполненный диск; изменение raw во время чтения; competing writers; повреждённый manifest/result hash; чтение во время переключения; удалённая цель ссылки; различный source; экспорт, начатый на предыдущем release. Проверка — current прежний при pre-publish ошибке, читатель видит одну полную версию, risk hash не меняется. Пока эти тесты и runner не реализованы; атомарность остаётся deployment blocker.
