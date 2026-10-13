"""Конфигурация сервиса @UPRiTRM_task_bot.

Все параметры читаются из файла ``.env`` (и/или переменных окружения)
через ``pydantic-settings``. Валидация выполняется сразу при старте —
некорректная конфигурация приводит к падению процесса, а не к тихой
деградации в рантайме.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта: <root>/src/config.py -> <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Безопасные значения по умолчанию для идентификаторов моделей Groq.
# История подбора модели chat-completions:
#   * llama-3.3-70b-versatile и llama-3.1-8b-instant на текущем аккаунте
#     отвечают 404 model_not_found (в списке GET /models их нет);
#   * qwen/qwen3.8-27b — reasoning/vision модель: при передаче
#     response_format={"type": "json_object"} рвёт соединение
#     (httpx RemoteProtocolError: Server disconnected without sending a response).
# Поэтому production-дефолт — openai/gpt-oss-20b (стабильно держит JSON mode).
DEFAULT_GROQ_CHAT_MODEL: str = "openai/gpt-oss-20b"
DEFAULT_GROQ_AUDIO_MODEL: str = "whisper-large-v3"

#: Приоритет chat-моделей для авто-подбора (initialize_models):
#: основная -> запасная -> крайний fallback.
GROQ_CHAT_MODEL_PRIORITY: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)


def _resolve_env_file() -> Path:
    """Путь к .env: переменная окружения ENV_FILE -> <корень проекта>/.env."""
    override = os.getenv("ENV_FILE")
    return Path(override).expanduser().resolve() if override else (PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из .env."""

    model_config = SettingsConfigDict(
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Telegram
    # ------------------------------------------------------------------ #
    BOT_TOKEN: str = Field(
        default="",
        description="Токен бота, выданный @BotFather.",
    )
    ADMIN_TELEGRAM_ID: int = Field(
        default=0,
        description="Telegram ID администратора (число, без знака '@').",
    )
    BOT_USERNAME: str = Field(
        default="UPRiTRM_task_bot",
        description="Username бота без '@' — используется только в текстах ответа.",
    )

    # ------------------------------------------------------------------ #
    # Google Sheets
    # ------------------------------------------------------------------ #
    GOOGLE_SHEET_ID: str = Field(
        default="",
        description="ID таблицы из URL: .../spreadsheets/d/<ID>/edit",
    )
    CREDENTIALS_FILE_PATH: Path = Field(
        default=Path("credentials.json"),
        description="Путь к JSON-ключу Service Account.",
    )

    # ------------------------------------------------------------------ #
    # Groq AI (Этап 2: транскрибация Whisper + структурирование Llama)
    # Идентификаторы моделей вынесены в конфигурацию с безопасными
    # значениями по умолчанию. ВНИМАНИЕ: llama-3.3-70b-versatile отдаёт
    # 404 model_not_found на ряде аккаунтов — дефолт переведён на
    # стабильную бесплатную модель llama-3.1-8b-instant.
    # ------------------------------------------------------------------ #
    GROQ_API_KEY: str = Field(
        default="",
        description="API-ключ Groq Cloud (console.groq.com -> API Keys).",
    )
    GROQ_AUDIO_MODEL: str = Field(
        default=DEFAULT_GROQ_AUDIO_MODEL,
        description=(
            "Модель Groq Audio API для распознавания голосовых сообщений "
            "(поддерживаемые: whisper-large-v3, whisper-large-v3-turbo)."
        ),
    )
    GROQ_CHAT_MODEL: str = Field(
        default=DEFAULT_GROQ_CHAT_MODEL,
        description=(
            "Модель Groq Chat Completions для структурирования поручений. "
            "Production-дефолт — openai/gpt-oss-20b (стабильно поддерживает "
            "response_format json_object). Переопределяется в .env; при её "
            "недоступности авто-дискавери берёт модель по приоритету "
            "openai/gpt-oss-20b -> openai/gpt-oss-120b -> qwen/qwen3.8-27b."
        ),
    )
    # Алиас старого имени поля (обратная совместимость .env и кода Этапа 2):
    # если в .env задан GROQ_TRANSCRIPTION_MODEL — он подхватывается.
    GROQ_TRANSCRIPTION_MODEL: Optional[str] = Field(
        default=None,
        description="Устаревшее имя для GROQ_AUDIO_MODEL (сохранено для совместимости).",
    )

    # ------------------------------------------------------------------ #
    # Прочее
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = Field(default="INFO")
    TZ: str = Field(default="Europe/Moscow")

    # ------------------------------------------------------------------ #
    # Валидация
    # ------------------------------------------------------------------ #
    @field_validator("BOT_TOKEN")
    @classmethod
    def _validate_token(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(
                "BOT_TOKEN пуст. Скопируйте .env.example в .env и укажите токен из @BotFather."
            )
        if ":" not in value:
            raise ValueError("BOT_TOKEN имеет неверный формат (ожидается '<id>:<secret>').")
        return value

    @field_validator("GOOGLE_SHEET_ID")
    @classmethod
    def _validate_sheet_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(
                "GOOGLE_SHEET_ID пуст. Возьмите ID из URL таблицы и пропишите его в .env."
            )
        return value

    @field_validator("ADMIN_TELEGRAM_ID")
    @classmethod
    def _validate_admin_id(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ADMIN_TELEGRAM_ID должен быть положительным числом.")
        return value

    @field_validator("GROQ_API_KEY")
    @classmethod
    def _validate_groq_key(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(
                "GROQ_API_KEY пуст. Получите ключ на https://console.groq.com/keys "
                "и пропишите его в .env (он нужен для распознавания голоса и структурирования задач)."
            )
        return value

    @field_validator("GROQ_CHAT_MODEL", "GROQ_AUDIO_MODEL")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        """Модель обязана быть непустой строкой без пробелов (id Groq API)."""
        value = (value or "").strip()
        if not value:
            raise ValueError(
                "Идентификатор модели Groq не может быть пустым — "
                "оставьте значение по умолчанию или укажите корректное имя модели."
            )
        if " " in value:
            raise ValueError(f"Идентификатор модели Groq содержит пробелы: {value!r}")
        return value

    @model_validator(mode="after")
    def _apply_legacy_transcription_alias(self) -> "Settings":
        """Совместимость: GROQ_TRANSCRIPTION_MODEL из старого .env -> GROQ_AUDIO_MODEL."""
        if self.GROQ_TRANSCRIPTION_MODEL:
            legacy = self.GROQ_TRANSCRIPTION_MODEL.strip()
            if legacy and legacy != self.GROQ_AUDIO_MODEL:
                object.__setattr__(self, "GROQ_AUDIO_MODEL", legacy)
        return self

    @model_validator(mode="after")
    def _validate_credentials_file(self) -> "Settings":
        path: Path = self.CREDENTIALS_FILE_PATH
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Файл Service Account не найден: {path}. "
                f"Положите credentials.json рядом с проектом "
                f"(монтируется в контейнер как /app/secrets/credentials.json)."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise ValueError(f"Не удалось прочитать JSON ключа {path}: {exc}") from exc
        if not isinstance(data, dict) or "client_email" not in data or "type" not in data:
            raise ValueError(
                f"{path} не похож на ключ Service Account Google (нет полей 'type'/'client_email')."
            )
        # Нормализуем путь на абсолютный
        object.__setattr__(self, "CREDENTIALS_FILE_PATH", path)
        return self

    # ------------------------------------------------------------------ #
    # Удобные производные значения
    # ------------------------------------------------------------------ #
    @property
    def sheet_title_tasks(self) -> str:
        return "Задачи"

    @property
    def sheet_title_employees(self) -> str:
        return "Сотрудники"

    @property
    def groq_chat_model(self) -> str:
        """Используемая модель Chat Completions (с фолбэком на безопасный дефолт)."""
        return (self.GROQ_CHAT_MODEL or "").strip() or DEFAULT_GROQ_CHAT_MODEL

    @property
    def groq_audio_model(self) -> str:
        """Используемая модель транскрибации (GROQ_AUDIO_MODEL, с учётом legacy-алиаса)."""
        if self.GROQ_TRANSCRIPTION_MODEL:
            return self.GROQ_TRANSCRIPTION_MODEL.strip() or DEFAULT_GROQ_AUDIO_MODEL
        return (self.GROQ_AUDIO_MODEL or "").strip() or DEFAULT_GROQ_AUDIO_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Кэш одного экземпляра настроек (читает .env один раз)."""
    return Settings()


if __name__ == "__main__":  # pragma: no cover - ручная диагностика конфигурации
    settings = get_settings()
    print("Конфигурация загружена успешно:")
    print(f"  BOT_TOKEN           = {settings.BOT_TOKEN[:10]}...")
    print(f"  ADMIN_TELEGRAM_ID   = {settings.ADMIN_TELEGRAM_ID}")
    print(f"  GOOGLE_SHEET_ID     = {settings.GOOGLE_SHEET_ID}")
    print(f"  CREDENTIALS         = {settings.CREDENTIALS_FILE_PATH}")
    print(f"  GROQ_CHAT_MODEL     = {settings.groq_chat_model}")
    print(f"  GROQ_AUDIO_MODEL    = {settings.groq_audio_model}")
