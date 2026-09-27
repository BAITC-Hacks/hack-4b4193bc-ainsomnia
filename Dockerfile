# Витрина обращений 109 — без установки Python на машине.
# Данных в образе НЕТ: data/, reports/, сырые выгрузки подключаются томами
# (docker-compose.yml), а .dockerignore не пускает их в контекст сборки.
# Инструкция — CLAUDE.md, раздел 0b, «Контейнер».
FROM python:3.12-slim

# fonts-dejavu-core — кириллица в PDF: src/export.py ищет DejaVu по пути Debian.
# chromium — картинки графиков в PDF (kaleido ищет браузер по BROWSER_PATH).
# Без Chromium PDF соберётся без графиков и скажет об этом, остальное работает;
# строку можно убрать, если образ нужен меньше (~300 МБ экономии).
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core chromium \
 && rm -rf /var/lib/apt/lists/*
ENV BROWSER_PATH=/usr/bin/chromium \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Зависимости отдельным слоем: он пересобирается, только когда меняется lock.
# requirements.lock — все пакеты с версиями и хешами (CLAUDE.md, 0b, препятствие 3);
# requirements.txt нужен pyproject.toml для метаданных пакета, ставится проект без зависимостей.
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Код и агрегаты из git. CLAUDE.md нужен выгрузке: оговорки берутся из раздела 5d.
COPY pyproject.toml CLAUDE.md train.py ./
COPY src/ src/
# Трейсбек в браузере выключен (client.showErrorDetails = "none"): nazar-dashboard
# переходит в /app и читает конфиг оттуда.
COPY .streamlit/ .streamlit/
COPY tests/ tests/
COPY reports/ reports/
RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["nazar-dashboard", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
