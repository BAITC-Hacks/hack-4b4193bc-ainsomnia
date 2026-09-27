# Внутренний сервис перед API

Работает без Streamlit и FastAPI:

```python
from src.operations import snapshot
from src.brief import render_html

state = snapshot()
document = render_html(state, "2026-09-27T12:00:00Z")
```

`snapshot()` сначала проверяет Data Health и source. При блокировке возвращает `available=False` и действие с причиной, без чтения аналитики. При успехе содержит aggregate actions, health, forecast, recurrence, историческую очередь риска и version-bound closure map. Построчные обращения в контракт не входят. Вызов не обучает модель, не пишет feedback и не публикует данные. Даты — конец данных своего региона; строка времени выше только пример времени формирования документа.

Модули: `event_service` — общий существующий детектор/feed; `actions` — прозрачные правила приоритета; `planning` — прогноз, сценарий мощности, повторное давление; `risk_explanation` — проверенное разложение сохранённой модели; `closure` — принятая карта 5o; `brief` — два рендера одних строк. `operations_view` только связывает сервис с UI.

Согласованность файлов проверяется до/после чтения. Это защита от обнаруженной смены сборки, не атомарный storage: миграция описана в [atomic-refresh.md](atomic-refresh.md). Предстоящий HTTP API требует OIDC, scope до агрегации, единой версии release и политики ошибок/лимитов. Новая production dependency не добавлена; публичного API пока нет.
