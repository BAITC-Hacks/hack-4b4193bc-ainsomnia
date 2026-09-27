# Актуальный backlog перед P0/P1

**Актуализация P2:** topic fix и шесть операционных возможностей реализованы; приёмка и ограничения — [p2-results.md](p2-results.md). Ниже сохранены исходный план и история P0/P1. Feedback/auth/atomic refresh остаются отложенными.

Срез 27.09.2026, main `4fa7098`. Решения — CLAUDE.md. «Закрыто» здесь означает подтверждение существующим кодом/сохранённым прогоном; новые проверки текущей задачи будут записаны отдельно. Историческая запись не считается свежим прогоном.

| ID | Проблема | Актуальна? | Доказательство | Severity | Без заказчика? | Действие |
|---|---|---|---|---|---|---|
| B1 | Полный Linux/Windows прогон | Частично | ci.yml проверяет Linux install, build, smoke, fake вне образа; локально Docker/colima нет | EXTERNAL | Частично | Полный контейнерный прогон на доступной площадке; Windows только если нужен |
| B2a | Незакреплённые зависимости | Закрыто | requirements.lock, --require-hashes во всех 4 jobs | P2 debt | Да | Сохранять lock |
| B2b | Версия REAL raw неизвестна | Да | cli.build_data не сверяет принятый manifest | P0 blocker | Да | P0.1 |
| B2c | Недетерминизм MPS | Да, R&D | 5k/5l, измеренный разброс | P2 debt | Да | Не обещать битовую воспроизводимость ML R&D |
| B3a | CI требует реальные данные | Закрыто | tests/test_fake_export.py, fixture 5p | P2 debt | Да | Сохранить |
| B3b | UI-тесты вне git | Частично | fake тест имеет happy path; analysis/ui/check_failures.py привязан к локальному REAL | P1 important | Да | Перенести полезные сценарии на fake |
| B3c | Нет CI | Закрыто | .github/workflows/ci.yml, 4 jobs | P2 debt | Да | Новый удалённый запуск требует разрешённого push |
| B3d | Новая выгрузка падает на абсолютных суммах | Закрыто | cli.BUILD_STEPS вызывает checks.labeling | P2 debt | Да | tests.test_topic_coverage оставить для mapping changes |
| B4a | Неожиданная схема/дата даёт traceback | Да | adapters._read_csv/_parse_dates | P1 important | Да | Schema preflight, безопасный отказ; без вывода значений |
| B4b | Traceback в браузере | Закрыто для unexpected exception | .streamlit/config.toml showErrorDetails=none; старый browser test 0b | P1 important | Да | Добавить CI проверку и friendly corrupt parquet |
| B4c | Ошибки сети Hugging Face | Да, R&D | finetune/embed, вне production path | P2 debt | Да | Отдельная R&D задача |
| B5 | Контейнер не проверен вообще | Устарело | CI build/smoke проверены по 0b; полный сценарий не проверен | P1 important | Частично | Не приравнивать smoke к full acceptance |
| B6 | Журнал действий/экспортов | Да | cli stdout, Streamlit default, identity нет | EXTERNAL | Контракт | Auth раньше actor audit |
| B7a | Инвентарь старых размеров шумит на новой версии | Да | adapters.INVENTORY_RAW_ROWS / EXPECTED_TOTAL | P2 debt | Да | Считать историческим эталоном, принятый manifest — версия входа |
| B7b | RAW path зашит | Закрыто | paths.NAZAR_RAW_DIR | P2 debt | Да | Сохранить |
| B7c | Нет refresh/сигнала сбоя/атомарности | Да | cli пишет шагами в действующий data | P0 blocker | Проект решения | Atomic release design; scheduler после канала |
| B8 | Нет auth/ролей/ограничения регионов | Да | dashboard/risk_view, compose loopback | EXTERNAL | Контракт | Viewer/analyst/admin; private до подключения |
| B9a | Regex не гарантирует отсутствие ПДн в тексте | Да | checks.privacy, запрет свободного текста | P1 important | Частично | Не переносить текст, читать готовые результаты |
| B9b | Encryption/retention/legal/access policy | Да | Нет эксплуатационного контракта | EXTERNAL | Нет | Решения владельца и заказчика |
| B9c | Новые category/SLA значения могут печататься | Да | adapters.topic_dictionaries, _build_canon; labeling_review пишет category | P0 blocker | Да | Вывод только безопасных идентификаторов неизвестных категорий/чисел, ready-file тест |
| C1 | Контейнер на площадке | Да | 0c неделя 1–2 / B1 | EXTERNAL | Нет | Площадка и разрешённая выгрузка |
| C2 | Проверки на площадке | Частично | CI есть; UI coverage неполная | P1 important | Частично | P0.2 локально, customer CI после выбора системы |
| C3 | Оценка по ПДн | Да | 0c неделя 2 | EXTERNAL | Нет | Юрист |
| C4 | SSO/роли | Да | B8 | EXTERNAL | Нет | OIDC параметры/матрица ролей |
| C5 | Сборщик/срок журнала | Да | B6 | EXTERNAL | Нет | Контракт логирования |
| C6 | Канал/частота обновления | Да | B7c | EXTERNAL | Нет | Источник, формат, частота, получатель ошибки |
| C7 | Хранение/удаление/шифрование | Да | B9b | EXTERNAL | Нет | Регламент |
| C8 | Windows | Не определена необходимость | 0c | EXTERNAL | Нет | Уточнить ОС |
| C9 | Итоговое внедрение | Да | Площадки/свежей выгрузки нет | EXTERNAL | Нет | Acceptance после зависимостей |
| O1 | SLA 34.899% / 7.462%, запрет общей модели | Да | 3, Q13, разные определения/закрытые-only | EXTERNAL | Нет | Сохранить границу регионов |
| O2 | updated_date не подтверждён как closure | Да | train.build_target, Q3 | EXTERNAL | Нет | Сохранить оговорку таргета |
| O3 | 15 дней не норматив | Да | train SLA default, Q4 | EXTERNAL | Нет | Не переносить чужие нормативы |
| O4 | Дедуп Туркестана / полнота выгрузки | Да | adapters._dedup, Q7/Q12 | EXTERNAL | Нет | Объяснение поставщика |
| O5 | Объединение витрин запрещено | Закрыто как неверная трактовка | 3, build_unified работает | P2 debt | Да | Запрет относится к обучению |
| T1 | Подстроки/основы и 243 неверные строки | Да | topic_mapping._match/RULES; 9, четыре категории | P1 important | Требуется решение смысла | Отдельный mapping проект, полный category→old→new→rows diff; эталоны сохранены, сейчас не менять |
| T2 | Исторические провод/опор/газ заплатки | Закрыто точечно | 5e, текущие RULES | P2 debt | Да | Не считать общей нормализацией |
| T3 | Граничный пробел norm.strip и ^/пунктуация | Да | topic_mapping.norm/_match/classify_appeal | P1 important | После T1 | Не чинить незаметно в P1 |
| T4 | Фразовый контекст и многословные основы | Да | 9: газ, отделка, окончания | P1 important | После T1 | Контекстные правила + независимый набор |
| T5 | Симметричный обмен незаметен | Закрыто в build | checks.labeling.mapping сравнивает категорию целиком | P2 debt | Да | Добавить искусственную регрессию на swap |
| T6 | Деления на ноль operating points | Да | train._sweep/_at_threshold/compare/write_baseline_report | P1 important | Да | Явная недоступность вырожденного сравнения |
| T7 | Недостижимые operating points | Частично закрыто | write_baseline_report принимает None; _point_for проверяет prefix до ties | P1 important | Да | Тест whole-threshold достижимости, реальные числа неизменны |
| T8 | answer_type timing leakage | Не проверена | CAT_TRIM не содержит поле; только отдельный A/B | EXTERNAL | Нет | Исправить чрезмерный вывод в прозе, спросить момент заполнения |
| T9 | DEEP_DIVE_REGIONS не используется | Да | spikes.py, нет чтений константы | P2 debt | Да | Не приоритет чисел |
| T10 | audit_rules учитывает ^ как символ | Да | topic_mapping.audit_rules | P2 debt | Да | Диагностика, не разметка |
| T11 | field_shift 100 / адресные ветки | Да, оставлено решением | field_shift.almaty_shift_rules | P2 debt | Нет без пересмотра | Не менять отсев |
| T12 | Имя пакета src | Да | pyproject.toml | P2 debt | Позже | После этапа 1 отдельная миграция |
| T13 | dashboard/nlq/forecast >500 строк | Да | Файлы, текст и логика вместе | P2 debt | Да | Новые Health модули отдельно |
| T14 | NLQ morphology 6/20 heldout failures | Да | tests/test_nlq.py, reports/nlq.md | P1 important | Новый протокол | Heldout не использовать для подгонки |
| T15 | NLQ GROWTH_MIN_BASE абсолютный | Да | nlq.py | P2 debt | Решение смысла | Период должен быть частью будущего протокола |
| T16 | Пороги cross_vocab литералами | Да | field_shift.cross_vocab, диагностика | P2 debt | Да | Не влияет на отсев/текущие числа |
| F1 | 28 дней не годовая сезонность | Да | spikes.detect, 5f.1 | P2 debt | Новый протокол | Оговорка, не prediction of accidents |
| F2 | Пик и фон на разных датах | Ограничение; подпись исправлена | dashboard.event_caption | P1 important | Да | Сохранять точные даты |
| F3 | MAD=0 | Да | min-count=10, 5f.3 | P2 debt | Новый протокол | Не менять порог по результату |
| F4 | Первый год без типа, 1–2 года слабые | Да | spike_type, 5f.4/V2 | P1 important | Нет новых данных | Не сравнивать доли типов регионов |
| F5 | Окно поглощает хвост | Да, принято | 5f.5 | P2 debt | Решение смысла | Оговорка |
| F6 | Рестарт после gap даёт ложный spike | Закрыто ≥7 дней | spikes.gap_days, fake burst test, итог 1223 | P1 important | Да | Исправить старую строку 5g; <7 дней остаётся ограничением |
| F7 | Mapping меняет spike magnitude | Да | 5f.7 | P1 important | Да | T1 требует полной регрессии |
| F8 | Нет доказательства раннего предупреждения | Да | 5f отдельная проверка | EXTERNAL | Нет ground truth | Только «куда смотреть» |
| G1 | Караганда прогноз исторический | Да | forecast/report, конец 2023 | P1 important | Да | Freshness подпись |
| G2 | Seasonal копирует выбросы | Да | forecast.forecast | P2 debt | Новый протокол | Не винзоризировать молча |
| G3 | Порог перехода ×3 граничный | Да | crosses_transition | P2 debt | Новый протокол | Оговорка |
| G4 | Караганда только 2 folds | Да | forecast.max_folds | P1 important | Нет истории | Показывать |
| G5 | Объёмы регионов несопоставимы | Да | 5d, forecast | EXTERNAL | Нет | Не суммировать/ранжировать |
| G6 | Туркестан gap / ВКО composition shift | Да | forecast.longest_gap, MIX_SHIFT, Q7/8 | EXTERNAL | Нет | Health/freshness + вопрос |
| H1 | NLQ не понимает произвольный язык | Да | Детерминированный parser, T14 | P1 important | Новый eval | Явный отказ |
| H2 | Неполнота полей/периода, SLA сводное | Ограничение обработано | nlq warnings, 5h | P1 important | Данные external | Сохранять оговорки |
| H3 | Старые ошибки focus/gone/прочее/среднее | Закрыто | 5h исправления и reports/nlq.md | P2 debt | Да | Регрессия прежнего отчёта |
| I1 | Одна дата и пустой фильтр ломали UI | Закрыто кодом | dashboard.main fallback периода; stop снят | P1 important | Да | Зафиксировать CI-тестом |
| I2 | Карточки расходятся с секцией | Закрыто конструкцией | top_cards получает тот же info/work | P1 important | Да | Fake тест при смене фильтров |
| I3 | Нулевой фон/дроби/seasonal labels | Да, оговорки | 5i, max(median,1) | P1 important | Новый протокол | Не округлять/не усиливать вывод |
| I4 | Plotly подписи AppTest не видит | Да | analysis/ui/live_render.py, старый Chrome прогон | P2 debt | Да | Browser отдельно, не приписывать AppTest |
| I5 | PDF OS/font/Chromium | Частично | export.FONT_CANDIDATES/PdfUnavailable | P1 important | Частично | CI safe failure tests; full Linux PDF позже |
| I6 | Терминология demo техническая | Да | src/demo.py | P2 debt | Да | Не влияет на числа |
| I7 | Risk отсутствует в Excel/PDF | Да, граница функции | export_frames | P2 debt | Новый scope | Не добавлять скрыто |
| I8 | Risk только test Караганды, не калиброван | Да | risk_view/metrics/4b | EXTERNAL | Нет свежей разметки | Историческая очередь, не вероятность |
| I9 | Рассылка/auth/state отсутствуют | Да | dashboard | EXTERNAL | Нет адресатов/SSO | План P2/P3 |
| S1 | Нет фактического закрытия в 3 регионах | Да | statuscheck.load, 5o observability | EXTERNAL | Нет | Карта наблюдаемости |
| S2 | Закрытые-only Костанай/Туркестан | Да | Q17, 5o | EXTERNAL | Нет | Нужны открытые записи |
| S3 | Исполнителя нет/небезопасен | Да | executor_search.md, 5o | EXTERNAL | Нет | Не возвращать contractor Алматы |
| S4 | Недостаточные службы/наблюдения | Да | reports/5o/signals.md | EXTERNAL | Нет | Не считать непроверенное отрицательным |
| S5 | Noise checks каналов не пройдены | Да | channel1.json/channel3.json | P1 important | Нет нового протокола | Итоговую карту, не предварительную |
| S6 | Триангуляция 0, нет допуска в витрину рейтинга | Да | signals.md, правило 5o | P1 important | Нет | P2 только наблюдаемость, без обвинений |
| S7 | Акмола статус «Передано» неоднозначен | Да | Q20 | EXTERNAL | Нет | Не менять статус догадкой |
| S8 | Каналы 1/2 один источник; хвост не авария | Да | Постановка 5o | P1 important | Нет | Сохранять формулировку |

## Раздел 10: каждый вопрос заказчику

Для Q1–Q22 доказательство — соответствующий пункт `docs/customer_questions.md`, связанный с указанным кодом/разбором. «Ответа нет» не значит неисправность кода.

| ID | Проблема | Актуальна? | Доказательство | Severity | Без заказчика? | Действие |
|---|---|---|---|---|---|---|
| Q0 | Неверное CSV экранирование | Да | field_shift, 5c | EXTERNAL | Нет | Корректная выгрузка |
| Q1 | Атрибуция систем | Да | Q1 | EXTERNAL | Нет | Подтвердить источники |
| Q2 | Алматы/Павлодар одна система? | Да | Q2, словари | EXTERNAL | Нет | Подтвердить |
| Q3 | updated_date | Да | O2 | EXTERNAL | Нет | Смысл даты |
| Q4 | Норматив Караганды | Да | O3 | EXTERNAL | Нет | Норматив/справочник |
| Q5 | Время answer_type | Да | T8 | EXTERNAL | Нет | Момент заполнения |
| Q6 | Настоящий текст обращений | Отвечено: недоступен | Ответ 25.09 в Q6 | EXTERNAL | Нет | Не планировать текстовую production модель |
| Q7 | Провал Туркестана | Да | G6 | EXTERNAL | Нет | Полнота/система не работала? |
| Q8 | Состав ВКО | Да | G6 | EXTERNAL | Нет | Изменения по темам/датам |
| Q9 | Ключ заявителя/повтора | Да | demo, 5j | EXTERNAL | Нет | Только допустимая обезличенная форма |
| Q10 | Район отсутствует/константа | Да | Канон, 5j | EXTERNAL | Нет | Фактический район |
| Q11 | Полные совпадения с разными ID | Да | 5c | EXTERNAL | Нет | Не удалять догадкой |
| Q12 | Повтор incidentcode | Да | O4 | EXTERNAL | Нет | Смысл повторов |
| Q13 | Определение slabreach | Да | O1 | EXTERNAL | Нет | Сопоставимость |
| Q14 | Нет result Туркестана | Да, не production prerequisite | 5m | EXTERNAL | Нет | Подсказки ответа не строятся |
| Q15 | Экибастузский эпизод | Да | 5f | EXTERNAL | Нет | Ground truth |
| Q16 | Язык записи оператора | Да, R&D | 5n | EXTERNAL | Нет | Доля/каналы |
| Q17 | Выгружать открытые заявки | Да | S2 | EXTERNAL | Нет | Полная выгрузка |
| Q18 | openagain/grade константы | Да | 5o | EXTERNAL | Нет | Причина пустоты |
| Q19 | Механизм массового закрытия | Да | batches.md | EXTERNAL | Нет | Смысл finishdate |
| Q20 | Статус Акмолы | Да | S7 | EXTERNAL | Нет | Конечный/промежуточный |
| Q21 | ПДн в исполнителе | Да | checks.person_names, раздел 1 | EXTERNAL | Нет | Формат будущего справочника |
| Q22 | Устойчивые коды категорий | Да | checks.labeling, T1 | EXTERNAL | Нет | Коды всех регионов |

## Дополнительно найдено при аудите

- `.env` не исключён `.gitignore`: P1 important, закрывается без заказчика; факт отсутствия текущих секретов проверять отдельным сканированием.
- Права raw/каталогов и невозможность атомарного текущего build нельзя объявить исправленными только документацией.
- Начальный profile не является «drift=0»: предыдущей принятой сборки для сравнения пока нет.

## Результат P0/P1 после аудита

Исходная таблица выше сохранена как срез до кода; актуальные изменения статусов:

| ID | Теперь | Подтверждение |
|---|---|---|
| B2b | Закрыто | REAL manifest 12 файлов; create/accept/check, отказ до адаптеров; tests.test_dataset_manifest |
| B3b, I1, I2 | Закрыто для перечисленных fake сценариев | tests.test_dashboard: filters/no_data/no_predictions/corrupt/no_font/no_chrome/failed_build; существующий fake-export job |
| B4a | Частично закрыто | Header/date preflight и safe corrupt parquet; произвольные ошибки других CLI не объявлены покрытыми |
| B4b | Сохранено, закреплено тестом | Friendly corrupt parquet + showErrorDetails=none; новый браузерный unexpected-error тест не проводился |
| B9c | Закрыто для найденных выходов | Неизвестные category → SHA256, SLA → количество; tests.test_labeling_safety читает записанный CSV |
| T5 | Закрыто и проверено отрицательным тестом | Искусственный симметричный обмен ловится при прежних суммах |
| T6, T7 | Закрыто | Empty/one-class/tiny/zero-lift/unreachable/ties tests; исправление реальной рабочей точки принято владельцем |
| T8 | Проза исправлена; вопрос external | Основная модель без answer_type; момент заполнения по-прежнему неизвестен |
| F6 | Устаревшая запись исправлена | 5g ссылается на действующее правило 5f и 1223 |
| G1, G4, G6 | Ограничения стали видимы | data_health + health_view; причины данных остаются external |
| .env/keys | Исключения добавлены | .gitignore и .dockerignore |
| B7c | Не закрыто | Build state защищает UI, но полный pipeline ещё не atomic; проект решения в operations-plan |
| B8, B9b, C1/C3–C9, O1–O4, Q* | Без изменения внешней зависимости | Заказчик/площадка/правовое основание/семантика |

Подробная приёмка и границы новых функций — `docs/operations-results.md`.
