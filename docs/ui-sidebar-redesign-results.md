# Второй UI/UX redesign: sidebar navigation

Локальный этап 2026-09-29. Аналитические сервисы, правила Action Queue, risk,
forecast, topic mapping и REAL/FAKE boundary не изменялись.

## Интерфейс

Пять верхних вкладок заменены постоянной левой навигацией на одном Streamlit
entry point. Активная страница хранится в `st.session_state["page"]`:
`operations`, `planning`, `control`, `analytics`, `reports`. Sidebar показывает
источник, семь регионов, число записей, дату данных и переключатель режима
презентации. В FAKE вместо REAL показывается заметная маркировка тестовых данных.

«Оперативно» открывается первым: четыре KPI, пять первых действий из прежней
очереди, семь карточек Data Health, preview новых всплесков и риска. В режиме
презентации остаются три действия, sidebar скрывается, а в content area есть
кнопка его возврата. «Планирование» содержит forecast, калькулятор мощности и
повторное давление. «Контроль» начинается с Data Health и timeline с amber-gap,
затем показывает drift/details и Closure Observatory. Полная лента событий,
risk queue, темы, динамика и состав потока находятся в «Аналитике». «Отчёты»
оставлены простой страницей формирования brief, Excel и PDF.

Фильтры перенесены в content area. Регион и тема — компактные dropdown
«Все / один», период — существующий date range. Их состояние общее для
«Аналитики» и «Отчётов». Технические event filters остаются в expander.

## Регрессия данных

REAL snapshot до и после совпал побайтно, SHA256:
`ccd73dddf7837c53ad0d65ae5c606317fa8aaaf144c00ae709bce0e67a435b17`.

| Показатель | До | После |
|---|---:|---:|
| REAL rows | 988 776 | 988 776 |
| REAL events | 1 225 | 1 225 |
| REAL actions | 16 | 16 |
| REAL risk queue | 4 048 | 4 048 |
| Forecast series | 14 | 14 |
| FAKE rows | 12 910 | 12 910 |
| FAKE spikes | 1 | 1 |

## Проверки

- 33 unit tests: PASS.
- AppTest всех пяти REAL-страниц: PASS, исключений нет.
- Полный `tests/test_fake_export.py`: PASS, включая source isolation, failure
  states, dashboard navigation, brief и PII boundary.
- REAL Excel: четыре листа, 12 365 ячеек, PII violations 0.
- REAL PDF: семь страниц, графики не пропущены, четыре image objects.
- `compileall`, `git diff --check`: PASS.

## Visual review

PNG находятся вне git в `/private/tmp/nazar-ui-redesign-v2/`: пять REAL-страниц
в 1920×1080 и 1366×768, плюс presentation mode в обоих размерах. Проверены
sidebar, первый viewport, card alignment, иерархия, фильтры, таблицы и пустые
состояния. На всех снимках page-level horizontal overflow отсутствует,
traceback отсутствует. `review.json`, проверочные Excel/PDF также находятся в
этом временном каталоге.

Push, VPS, S3 и DNS не затрагивались. Скриншоты в git не добавлялись.
