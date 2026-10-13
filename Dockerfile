# syntax=docker/dockerfile:1
# ============================================================
# @UPRiTRM_task_bot — Этап 1 (исправление таймаута Docker Hub)
#
# ПРОБЛЕМА: "load metadata for docker.io/library/python:3.11-slim-bookworm
#            ... DeadlineExceeded: context deadline exceeded" —
#            docker.io недоступен из текущей сети.
# РЕШЕНИЕ: базовый образ тянется из публичного зеркала
#          dockerhub.timeweb.cloud (полный путь внутри реестра для
#          официальных образов: /library/<image>). Если это зеркало
#          окажется недоступно, замените строку FROM на альтернативу:
#          FROM mirror.gcr.io/library/python:3.11-slim-bookworm
#
# Python 3.11 + aiogram 3.x + gspread; системные ffmpeg (голосовые —
# Этап 2) и curl (healthcheck/отладка).
# ============================================================

FROM dockerhub.timeweb.cloud/library/python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="uprirm-task-bot" \
      org.opencontainers.image.description="Telegram task-controller bot (Stage 1 skeleton)" \
      org.opencontainers.image.source="@UPRiTRM_task_bot"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Moscow

WORKDIR /app

# ffmpeg — конвертация .oga/.ogg (voice) в wav на следующем этапе
# tzdata — корректная локаль времени для дедлайнов задач
# curl   — healthcheck/диагностика доступности API из контейнера
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tzdata \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости — этот слой кэшируется отдельно от кода
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (код, конфиги). В dev-режиме docker-compose
# перемонтирует ./ поверх /app — изменения применяются без пересборки.
COPY . ./

# Не запускаем процесс от root
RUN useradd --create-home --shell /usr/sbin/nologin bot \
    && chown -R bot:bot /app
USER bot

CMD ["python", "-m", "src.main"]
