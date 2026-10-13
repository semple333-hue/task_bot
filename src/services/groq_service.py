"""Сервис искусственного интеллекта на базе Groq Cloud для @UPRiTRM_task_bot.

Три функции (Этап 2):
    1. ``initialize_models``  — AUTO-DISCOVERY моделей при старте бота:
       запрашивает ``GET /models`` (``await self._client.models.list()``),
       фильтрует доступные текстовые LLM (исключая whisper/tts/audio),
       логирует список и выбирает активную chat-модель: если модель из
       настроек (``GROQ_CHAT_MODEL``) доступна — её, иначе — первую
       доступную текстовую модель с предупреждением в лог. Для голоса
       проверяется наличие Whisper (``whisper-large-v3`` /
       ``whisper-large-v3-turbo``).
    2. ``transcribe_voice``   — распознавание голосового сообщения через
       Groq Audio API активной Whisper-модели (язык ``ru``).
    3. ``structure_task``     — извлечение структуры поручения
       (task_title / assignee_name / deadline) через Groq Chat Completions
       АКТИВНОЙ текстовой модели в strict JSON mode
       (response_format={"type": "json_object"}).

История проблемы: захардкоженные модели ``llama-3.3-70b-versatile`` и даже
``llama-3.1-8b-instant`` на разных аккаунтах отвечают
404 model_not_found; reasoning/vision модель ``qwen/qwen3.8-27b`` при
``response_format={"type": "json_object"}`` рвёт HTTP-соединение
(httpx RemoteProtocolError). Production-приоритет: openai/gpt-oss-20b ->
openai/gpt-oss-120b -> qwen/qwen3.8-27b (для последней JSON mode автоматически
отключается, ответ парсится regex'ом). Поэтому идентификаторы моделей вынесены
в конфигурацию (.env, безопасные дефолты), а сервис при старте сам определяет,
что реально доступно ключу, и переключается на рабочую модель без изменения .env.

Все сетевые сбои (таймаут, отсутствие соединения, rate limit, ошибка
авторизации) НЕ пробрасываются необработанными: они логируются и
конвертируются в доменное исключение :class:`GroqServiceError` с
человекочитаемым текстом, который хэндлер показывает руководителю.
Мягкая обработка исполнителя: имя, не найденное в листе «Сотрудники»,
не является ошибкой — пишется то, что названо в сообщении, либо
«Не назначен».
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import groq
import httpx
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import (
    DEFAULT_GROQ_AUDIO_MODEL,
    DEFAULT_GROQ_CHAT_MODEL,
    GROQ_CHAT_MODEL_PRIORITY,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

# Значения по умолчанию (используются только если настройки пустые;
# канонические константы объявлены в src/config.py).
DEFAULT_WHISPER_MODEL = DEFAULT_GROQ_AUDIO_MODEL  # "whisper-large-v3"
DEFAULT_CHAT_MODEL = DEFAULT_GROQ_CHAT_MODEL      # "openai/gpt-oss-20b"
TRANSCRIPTION_LANGUAGE = "ru"

#: Chat-модели, которые НЕ поддерживают response_format={"type": "json_object"}:
#: reasoning/vision модели рвут HTTP-соединение (RemoteProtocolError: Server
#: disconnected without sending a response). Для них JSON mode отключается,
#: а ответ парсится регулярным выражением.
JSON_MODE_INCOMPATIBLE_MARKERS: tuple[str, ...] = ("qwen",)


def _supports_json_mode(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return not any(marker in lowered for marker in JSON_MODE_INCOMPATIBLE_MARKERS)


#: Префиксы/подстроки reasoning-моделей Groq (openai/gpt-oss-*): у них часть
#: бюджета токенов уходит во внутренние рассуждения (reasoning), поэтому
#: требуется max_completion_tokens вместо max_tokens и reasoning_effort="low".
REASONING_MODEL_MARKERS: tuple[str, ...] = ("gpt-oss", "o1", "o3", "deepseek-r")

#: Параметры запроса к chat completions (ТЗ Этап 2, фикс пустого content).
# NOTE: reasoning_effort НЕ передаётся в AsyncCompletions.create() — параметра
# нет в сигнатуре groq SDK (TypeError: unexpected keyword argument).
MAX_COMPLETION_TOKENS_REASONING = 2048   # бюджет И на reasoning, И на JSON-ответ
MAX_TOKENS_DEFAULT = 600                 # обычный лимит для не-reasoning моделей


def _is_reasoning_model(model_id: str) -> bool:
    """Является ли модель reasoning-моделью (требует специальной конфигурации)."""
    lowered = (model_id or "").lower()
    return any(marker in lowered for marker in REASONING_MODEL_MARKERS)

# Whisper-модели Groq, ранжированные по предпочтительности.
PREFERRED_AUDIO_MODELS: tuple[str, ...] = ("whisper-large-v3", "whisper-large-v3-turbo")

# Подстроки, по которым модель из GET /models относится к НЕ-текстовым
# (аудио/речь/визуал/голос) — такие исключаем из списка кандидатов LLM.
NON_TEXT_MODEL_MARKERS: tuple[str, ...] = (
    "whisper",
    "tts",
    "audio",
    "voice",
    "playai",
    "speakleash",
    "canary",
    "asr",
    "embed",
    "vision",
    "image",
)

# Мягкая обработка ненайденного исполнителя (ТЗ: не падать, а писать заглушку).
UNASSIGNED_LABEL = "Не назначен"
NO_DEADLINE_LABEL = "Не указан"

#: Текст для руководителя, когда Groq недоступен (сеть/таймаут/API).
GROQ_UNAVAILABLE_MESSAGE = (
    "⚠️ Не удалось связаться с ИИ-сервисом Groq. Попробуйте повторить запрос через минуту."
)


# ---------------------------------------------------------------------- #
# Схема ответа LLM (strict JSON mode: response_format json_object)
# ---------------------------------------------------------------------- #
class TaskStructure(BaseModel):
    """Структурированное поручение, которое возвращает модель."""

    model_config = ConfigDict(extra="ignore")

    task_title: str = Field(
        default="",
        description="Суть поручения — четкая, лаконичная формулировка.",
    )
    assignee_name: str = Field(
        default=UNASSIGNED_LABEL,
        description=(
            "Имя ответственного из списка сотрудников "
            "(или ровно то имя, что названо в сообщении; 'Не указан', если не упомянут)."
        ),
    )
    deadline: str = Field(
        default=NO_DEADLINE_LABEL,
        description=(
            "Срок выполнения: дата DD.MM.YYYY или краткий срок ('до пятницы 18:00'), "
            "либо 'Не указан'."
        ),
    )


@dataclass(frozen=True)
class StructuredTask:
    """Доменный результат структурирования (уже очищен от мусора модели)."""

    task_title: str
    assignee_name: str
    deadline: str
    #: True, если исполнитель НЕ сопоставился с листом «Сотрудники»
    #: (мягкая обработка: пишем имя из сообщения либо "Не назначен", не падаем).
    assignee_matched: bool = False
    #: Модель, которая фактически обработала запрос (для логов/карточек).
    model_used: str = ""


@dataclass
class ModelsDiscoveryReport:
    """Итог авто-определения моделей Groq (выполняется один раз при старте)."""

    available_chat_models: list[str] = field(default_factory=list)
    available_audio_models: list[str] = field(default_factory=list)
    requested_chat_model: str = ""
    active_chat_model: str = ""
    requested_audio_model: str = ""
    active_audio_model: str = ""
    chat_fallback_used: bool = False
    audio_fallback_used: bool = False
    discovery_ok: bool = False
    error: str = ""

    def summary(self) -> str:
        if not self.discovery_ok:
            return f"Авто-определение моделей не выполнено ({self.error or 'причина неизвестна'}); используются значения из настроек."
        parts = [
            f"chat: {self.active_chat_model}"
            + (" (fallback)" if self.chat_fallback_used else ""),
            f"audio: {self.active_audio_model or 'нет доступных whisper-моделей'}"
            + (" (fallback)" if self.audio_fallback_used else ""),
            f"LLM в аккаунте: {len(self.available_chat_models)}",
        ]
        return "; ".join(parts)


class GroqServiceError(RuntimeError):
    """Ошибка уровня AI-сервиса с текстом, пригодным для показа пользователю."""

    def __init__(self, message: str, *, network: bool = False) -> None:
        super().__init__(message)
        #: True — сбой сети/таймаут/5xx: хэндлер покажет «повторите через минуту».
        self.network = network


class _JSONModeConnectionDrop(Exception):
    """Внутренний сигнал: сервер разорвал соединение на запросе с json_object.

    Используется только внутри :meth:`GroqService._request_completion` /
    :meth:`GroqService.structure_task` для автоматического повтора запроса
    БЕЗ параметра response_format (reasoning/vision модели, такие как
    qwen/qwen3.8-27b, рвут HTTP-сессию: RemoteProtocolError).
    """

    def __init__(self, model_id: str, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.model_id = model_id
        self.cause = cause


SYSTEM_PROMPT_TEMPLATE = """\
Ты — корпоративный ассистент руководителя компании UPRiTRM. Твоя задача — \
выделить из текста поручения 3 параметра и вернуть ИСКЛЮЧИТЕЛЬНО валидный JSON \
без лишнего текста и markdown:
{{"task_title": "суть задачи", "assignee_name": "имя ответственного сотрудника", "deadline": "срок или 'Не указан'"}}

Правила заполнения полей:
1. "task_title" — суть поручения: чёткая, лаконичная формулировка задачи \
в повелительном наклонении (без приветствий, без "поручаю", без мусорных слов).
2. "assignee_name" — имя исполнителя. Сопоставь с базой сотрудников ниже; \
если в тексте названо похожее имя (уменьшительное, опечатка, никнейм) — верни \
каноническое ИМЯ ИЗ БАЗЫ. Если исполнитель назван, но его нет в базе — верни \
имя как оно звучит в тексте. Если исполнитель не упомянут вообще — верни \
"Не указан". НИКОГДА не выдумывай исполнителя.
3. "deadline" — срок. Форматы: точная дата "DD.MM.YYYY", относительный срок \
("сегодня 18:00", "до пятницы 18:00", "3 дня"). Если срок не упомянут — "Не указан".

База сотрудников (Имя | Никнейм | Telegram ID):
{employees}

Правила вывода (ОБЯЗАТЕЛЬНО):
- Верни ТОЛЬКО валидный JSON без вступительных слов и пояснений в формате:
  {{"task_title": "...", "assignee_name": "...", "deadline": "..."}}
- Ответ — ровно ОДИН валидный JSON-объект вида:
{{
  "task_title": "Суть поручения",
  "assignee_name": "Имя ответственного (или 'Не указан')",
  "deadline": "Срок (или 'Не указан')"
}}
- Все три ключа обязательны, значения — строки.
- Никакого текста до/после JSON, никаких markdown-ограждений ```...```.
- Не выдумывай данные: чего нет в поручении — того в JSON быть не должно \
(assignee_name="Не указан", deadline="Не указан").
"""


class GroqService:
    """Обёртка над AsyncGroq: авто-подбор моделей + транскрибация + структурирование."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

        # КРИТИЧНО (фикс httpx.RemoteProtocolError "Server disconnected
        # without sending a response"): Groq-прокси закрывает idle keep-alive
        # соединения раньше, чем это замечает клиент. Переиспользование такого
        # «мёртвого» сокета рвёт запрос на уровне транспорта. Отключаем пул
        # keep-alive соединений: каждый запрос идёт по новому TCP-соединению.
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0, keepalive_expiry=5.0),
            timeout=httpx.Timeout(60.0, connect=15.0),
        )
        self._owns_http_client = True
        try:
            self._client: AsyncGroq = AsyncGroq(
                api_key=self._settings.GROQ_API_KEY,
                http_client=self._http_client,
                max_retries=2,
            )
        except Exception:
            # Клиент не создан — закрываем свой httpx-клиент, чтобы не течь.
            self._schedule_http_client_close()
            raise

        #: Активные модели. До успешного initialize_models — значения из
        #: конфигурации (production-дефолты openai/gpt-oss-20b / whisper-large-v3).
        self._active_chat_model: str = self._settings.groq_chat_model
        self._active_audio_model: str = self._settings.groq_audio_model
        self._discovery: Optional[ModelsDiscoveryReport] = None
        self._discovery_lock: Optional[asyncio.Lock] = None  # создаётся лениво (см. _get_lock)

    def _schedule_http_client_close(self) -> None:
        """Best-effort закрытие httpx.AsyncClient при ошибке инициализации."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Нет event loop (например, конструирование вне async-контекста):
            # sync-транспорт у AsyncClient не открыт — закрывать нечего.
            return
        loop.create_task(self._http_client.aclose())

    # ------------------------------------------------------------------ #
    # Свойства активных моделей
    # ------------------------------------------------------------------ #
    @property
    def active_chat_model(self) -> str:
        return self._active_chat_model

    @property
    def active_audio_model(self) -> str:
        return self._active_audio_model

    @property
    def discovery_report(self) -> Optional[ModelsDiscoveryReport]:
        return self._discovery

    @staticmethod
    def _get_lock() -> asyncio.Lock:
        """Ленивое создание asyncio.Lock (без привязки к event loop в __init__)."""
        lock = getattr(GroqService, "_shared_discovery_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            GroqService._shared_discovery_lock = lock
        return lock

    # ------------------------------------------------------------------ #
    # 0. AUTO-DISCOVERY моделей (вызывается из startup-хука aiogram)
    # ------------------------------------------------------------------ #
    async def initialize_models(self) -> ModelsDiscoveryReport:
        """Определить доступные аккаунту модели Groq и выбрать активные.

        * Вызывает ``await self._client.models.list()`` (GET /models).
        * Фильтрует ответы, исключая whisper/tts/audio и прочие не-текстовые
          модели — оставляет только LLM для текста.
        * Логирует полный список доступных текстовых моделей.
        * Если ``settings.GROQ_CHAT_MODEL`` есть в списке — выбирает её;
          иначе автоматически берёт первую доступную текстовую модель и
          пишет предупреждение.
        * Для аудио проверяет наличие ``whisper-large-v3`` /
          ``whisper-large-v3-turbo``.

        Идемпотентна (protected lock). При недоступности API НЕ бросает
        исключение наружу: фиксирует ошибку в отчёте, оставляет активными
        значения из настроек и продолжает работу — понятная ошибка
        всплывёт позже при реальном запросе.
        """
        async with self._get_lock():
            report = ModelsDiscoveryReport(
                requested_chat_model=self._settings.groq_chat_model,
                requested_audio_model=self._settings.groq_audio_model,
            )

            try:
                models_page = await self._list_models_with_retry(attempts=3, base_delay=1.0)
            except GroqServiceError as exc:
                logger.warning("Groq auto-discovery не выполнен: %s", exc)
                report.error = str(exc)
                report.active_chat_model = self._active_chat_model
                report.active_audio_model = self._active_audio_model
                self._discovery = report
                return report

            raw_ids = self._extract_model_ids(models_page)
            text_models, audio_models = self._split_models(raw_ids)

            report.available_chat_models = text_models
            report.available_audio_models = audio_models
            report.discovery_ok = True

            logger.info("Доступные модели Groq: %s", text_models)
            logger.info("Доступные аудио-модели Groq (whisper): %s", audio_models)

            # --- Выбор активной CHAT-модели ----------------------------- #
            # Приоритет (ТЗ): 1) openai/gpt-oss-20b, 2) openai/gpt-oss-120b,
            # 3) qwen/qwen3.8-27b (крайний fallback), затем — первая доступная.
            requested = report.requested_chat_model
            chosen_model = ""
            if requested in text_models:
                chosen_model = requested
            else:
                for priority_model in GROQ_CHAT_MODEL_PRIORITY:
                    if priority_model in text_models:
                        chosen_model = priority_model
                        break
                if not chosen_model and text_models:
                    chosen_model = text_models[0]

            if chosen_model:
                if chosen_model != requested:
                    report.chat_fallback_used = True
                    logger.warning(
                        "Модель %s недоступна, выбрана активная: %s",
                        requested,
                        chosen_model,
                    )
                if not _supports_json_mode(chosen_model):
                    logger.warning(
                        "Модель %s не поддерживает response_format json_object "
                        "(reasoning/vision) — JSON mode будет отключён, ответ "
                        "парсится регулярным выражением.",
                        chosen_model,
                    )
            else:
                # В аккаунте нет ни одной текстовой модели — оставляем настройку,
                # реальный запрос даст понятную ошибку.
                chosen_model = requested
                report.chat_fallback_used = False
                logger.error(
                    "В аккаунте Groq не найдено ни одной текстовой модели — "
                    "структурирование задач будет недоступно."
                )
            self._active_chat_model = chosen_model
            report.active_chat_model = chosen_model

            # --- Выбор активной AUDIO-модели ---------------------------- #
            requested_audio = report.requested_audio_model
            if requested_audio and requested_audio in audio_models:
                chosen_audio = requested_audio
            else:
                chosen_audio = next(
                    (m for m in PREFERRED_AUDIO_MODELS if m in audio_models),
                    audio_models[0] if audio_models else "",
                )
                if chosen_audio and chosen_audio != requested_audio:
                    report.audio_fallback_used = True
                    logger.warning(
                        "Аудио-модель %s недоступна, выбрана активная: %s",
                        requested_audio or "<не задана>",
                        chosen_audio,
                    )
                if not chosen_audio:
                    logger.error(
                        "В аккаунте Groq нет ни одной Whisper-модели — "
                        "распознавание голосовых сообщений будет недоступно."
                    )
            if chosen_audio:
                self._active_audio_model = chosen_audio
            report.active_audio_model = self._active_audio_model

            logger.info("Groq активные модели — %s", report.summary())
            self._discovery = report
            return report

    async def _list_models_with_retry(self, attempts: int, base_delay: float) -> Any:
        """GET /models c мягким экспоненциальным ретраем на сетевые сбои."""
        last_error: Optional[GroqServiceError] = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._client.models.list()
            except groq.APIConnectionError as exc:
                last_error = GroqServiceError(
                    "Нет связи с Groq API (таймаут/сеть).", network=True
                )
                logger.warning(
                    "Groq models.list: попытка %d/%d не удалась (%s)",
                    attempt, attempts, exc,
                )
            except groq.AuthenticationError as exc:
                logger.error("Groq: неверный API-ключ: %s", exc)
                raise GroqServiceError(
                    "Отказ авторизации Groq API — проверьте GROQ_API_KEY в .env."
                ) from exc
            except groq.RateLimitError as exc:
                last_error = GroqServiceError(
                    "Groq API ограничил частоту запросов (rate limit).", network=True
                )
                logger.warning("Groq models.list: rate limit (попытка %d/%d)", attempt, attempts)
            except groq.APIStatusError as exc:
                last_error = GroqServiceError(
                    f"Groq API вернул ошибку {exc.status_code}: {exc.message}",
                    network=exc.status_code >= 500,
                )
                logger.warning(
                    "Groq models.list: статус %s (попытка %d/%d)",
                    exc.status_code, attempt, attempts,
                )
            if attempt < attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_model_ids(models_page: Any) -> list[str]:
        """Достаёт список id моделей из ответа SDK (compat с разными версиями)."""
        data = getattr(models_page, "data", None)
        if data is None and isinstance(models_page, dict):
            data = models_page.get("data")
        ids: list[str] = []
        for item in data or []:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("model") or item.get("name")
            else:
                mid = getattr(item, "id", None) or getattr(item, "model", None) or getattr(item, "name", None)
            if mid:
                ids.append(str(mid))
        # Уникальные, порядок сохранения
        seen: set[str] = set()
        unique: list[str] = []
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                unique.append(mid)
        return unique

    @classmethod
    def _is_text_model(cls, model_id: str) -> bool:
        lowered = (model_id or "").lower()
        if not lowered:
            return False
        return not any(marker in lowered for marker in NON_TEXT_MODEL_MARKERS)

    @classmethod
    def _split_models(cls, model_ids: list[str]) -> tuple[list[str], list[str]]:
        """Разделяет модели на текстовые LLM и аудио (whisper/tts/audio)."""
        text_models: list[str] = []
        audio_models: list[str] = []
        for mid in model_ids:
            lowered = mid.lower()
            if any(marker in lowered for marker in NON_TEXT_MODEL_MARKERS):
                if "whisper" in lowered:
                    audio_models.append(mid)
                continue
            text_models.append(mid)
        return text_models, audio_models

    # ------------------------------------------------------------------ #
    # 1. Транскрибация голосового сообщения (активная Whisper-модель)
    # ------------------------------------------------------------------ #
    async def transcribe_voice(self, audio_path: str | Path) -> str:
        """Распознать аудиофайл (.ogg/.oga/.opus/.mp3/.wav) и вернуть текст.

        Использует активную аудио-модель (``initialize_models``), параметр
        ``language="ru"``. Бросает :class:`GroqServiceError` при недоступности
        файла, ошибке Groq API/сети или пустом результате распознавания.
        """
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size == 0:
            raise GroqServiceError(
                f"Аудиофайл не найден или пуст: {path}. Переотправьте голосовое сообщение."
            )

        audio_model = self._active_audio_model or self._settings.groq_audio_model
        try:
            with path.open("rb") as audio_file:
                transcription = await self._client.audio.transcriptions.create(
                    file=("voice.ogg", audio_file.read()),
                    model=audio_model,
                    language=TRANSCRIPTION_LANGUAGE,
                    response_format="json",
                )
        except groq.NotFoundError as exc:
            logger.error("Groq: аудио-модель %r недоступна аккаунту: %s", audio_model, exc)
            raise GroqServiceError(
                f"Whisper-модель '{audio_model}' недоступна вашему ключу Groq "
                "(404 model_not_found). Повторите запуск бота — модели подбираются "
                "автоматически, либо укажите GROQ_AUDIO_MODEL в .env."
            ) from exc
        except groq.APIConnectionError as exc:
            logger.exception("Groq: нет соединения (transcription)")
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.AuthenticationError as exc:
            logger.error("Groq: неверный API-ключ: %s", exc)
            raise GroqServiceError(
                "Отказ авторизации Groq API — проверьте GROQ_API_KEY в .env."
            ) from exc
        except groq.RateLimitError as exc:
            logger.warning("Groq: превышен лимит запросов: %s", exc)
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.APITimeoutError as exc:
            logger.exception("Groq: таймаут (transcription)")
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.APIStatusError as exc:
            logger.exception("Groq: ошибка API (%s)", exc.status_code)
            raise GroqServiceError(
                f"Groq Audio API вернул ошибку {exc.status_code}: {exc.message}",
                network=exc.status_code >= 500,
            ) from exc

        text = self._extract_transcription_text(transcription)
        if not text.strip():
            raise GroqServiceError(
                "Не удалось распознать речь — в аудио нет различимого текста. "
                "Продиктуйте поручение громче и отчетливее."
            )

        logger.info(
            "Whisper (%s) распознал %d симв.: %.80s...", audio_model, len(text), text
        )
        return text.strip()

    @staticmethod
    def _extract_transcription_text(transcription: Any) -> str:
        """Достаёт текст из ответа Whisper независимо от формата ответа."""
        if isinstance(transcription, str):
            return transcription
        if isinstance(transcription, dict):
            return str(transcription.get("text", "") or "")
        text_attr = getattr(transcription, "text", None)
        return str(text_attr or "")

    # ------------------------------------------------------------------ #
    # 2. Структурирование поручения (активная LLM, strict JSON mode)
    # ------------------------------------------------------------------ #
    async def _request_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        use_json_mode: bool,
    ) -> Any:
        """Один запрос к Chat Completions с параметрами из ТЗ.

        ``temperature=0.1``; лимит токенов: для reasoning-моделей семейства
        ``openai/gpt-oss`` — ``max_completion_tokens=2048`` (лимит должен
        покрывать и внутренние рассуждения, и финальный ответ — иначе весь
        бюджет уходит в reasoning и ``message.content`` приходит пустым);
        для остальных — ``max_tokens=600``. При ``use_json_mode=True``
        передаётся ``response_format={"type": "json_object"}``.

        Если запрос С json_object приводит к разрыву HTTP-соединения
        (httpx.RemoteProtocolError / APIConnectionError у reasoning-моделей),
        бросается внутренний сигнал :class:`_JSONModeConnectionDrop`, чтобы
        вызывающий код повторил запрос без response_format. Прочие ошибки
        пробрасываются как есть (обрабатываются в structure_task).
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
        }
        if _is_reasoning_model(model):
            # openai/gpt-oss-*: max_completion_tokens=2048 — бюджет и на
            # внутренние рассуждения, и на финальный JSON-ответ (см. docstring).
            # ВАЖНО: параметр reasoning_effort НЕ передаётся — он отсутствует
            # в сигнатуре groq SDK AsyncCompletions.create() (проверено через
            # inspect.signature) и вызывает TypeError на этапе вызова.
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS_REASONING
        else:
            kwargs["max_tokens"] = MAX_TOKENS_DEFAULT
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return await self._client.chat.completions.create(**kwargs)
        except (httpx.RemoteProtocolError, httpx.ProtocolError) as exc:
            if use_json_mode:
                raise _JSONModeConnectionDrop(model, exc) from exc
            raise GroqServiceError(GROQ_UNAVAILABLE_MESSAGE, network=True) from exc
        except groq.APIConnectionError as exc:
            # SDK оборачивает RemoteProtocolError в APIConnectionError —
            # при включённом json_object считаем это признаком несовместимости
            # модели с JSON mode и просим повтор без response_format.
            cause = exc.__cause__
            is_protocol_drop = isinstance(cause, httpx.TransportError) and not isinstance(
                cause, httpx.TimeoutException
            )
            if use_json_mode and is_protocol_drop:
                raise _JSONModeConnectionDrop(model, exc) from exc
            raise

    async def structure_task(
        self, raw_text: str, employees: list[dict[str, Any]]
    ) -> StructuredTask:
        """Превратить текст поручения в (task_title, assignee_name, deadline).

        ``employees`` — записи листа «Сотрудники»: [{'Имя':..., 'Никнейм':...,
        'Telegram ID':...}, ...]. Список передаётся в системный промпт, чтобы
        модель сопоставляла имена с канонической базой.

        Используется АКТИВНАЯ текстовая модель (определена initialize_models;
        до определения — значение из настроек). response_format строго
        {"type": "json_object"}.

        Сетевые сбои и ошибки API логируются и превращаются в
        :class:`GroqServiceError` (флаг ``network=True`` для сбоев сети) —
        необработанных исключений метод не пробрасывает.

        Мягкая обработка исполнителя (ТЗ): если имя не найдено в списке
        сотрудников — НЕ падаем; оставляем то, что названо в сообщении, либо
        "Не назначен" (флаг ``assignee_matched=False`` для карточки/логов).
        """
        raw_text = (raw_text or "").strip()
        if not raw_text:
            raise GroqServiceError("Пустое поручение — нечего структурировать.")

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            employees=self._format_employees(employees)
        )
        model = self._active_chat_model or self._settings.groq_chat_model

        # Формируем запрос: основной путь — со strict JSON mode
        # (response_format={"type": "json_object"}), для моделей, которые его
        # не поддерживают (qwen reasoning/vision рвут соединение при
        # RemoteProtocolError) — сразу без response_format.
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ]
        use_json_mode = _supports_json_mode(model)

        completion: Any = None
        try:
            completion = await self._request_completion(
                model=model,
                messages=request_messages,
                use_json_mode=use_json_mode,
            )
        except _JSONModeConnectionDrop as exc:
            # Разрыв соединения именно на запросе с response_format —
            # автоматический повтор БЕЗ response_format (ТЗ).
            logger.warning(
                "Модель %s разорвала соединение при json_object mode (%s) — "
                "повторяю запрос без response_format.",
                model,
                exc,
            )
            use_json_mode = False
            try:
                completion = await self._request_completion(
                    model=model,
                    messages=request_messages,
                    use_json_mode=False,
                )
            except GroqServiceError:
                raise
            except Exception as retry_exc:  # noqa: BLE001 - повтор тоже упал
                logger.exception("Groq: повтор без response_format тоже не удался")
                raise GroqServiceError(
                    GROQ_UNAVAILABLE_MESSAGE, network=True
                ) from retry_exc
        except groq.NotFoundError as exc:
            logger.error("Groq: модель %r недоступна аккаунту: %s", model, exc)
            raise GroqServiceError(
                f"Модель '{model}' недоступна вашему ключу Groq (404 model_not_found). "
                "Перезапустите бота — активная модель подберётся автоматически "
                "через GET /models, либо укажите GROQ_CHAT_MODEL в .env."
            ) from exc
        except groq.APIConnectionError as exc:
            logger.exception("Groq: нет соединения (chat completions)")
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.APITimeoutError as exc:
            logger.exception("Groq: таймаут (chat completions)")
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.AuthenticationError as exc:
            logger.error("Groq: неверный API-ключ: %s", exc)
            raise GroqServiceError(
                "Отказ авторизации Groq API — проверьте GROQ_API_KEY в .env."
            ) from exc
        except groq.RateLimitError as exc:
            logger.warning("Groq: превышен лимит запросов: %s", exc)
            raise GroqServiceError(
                GROQ_UNAVAILABLE_MESSAGE, network=True
            ) from exc
        except groq.APIStatusError as exc:
            logger.exception("Groq: ошибка API (%s)", exc.status_code)
            raise GroqServiceError(
                f"Groq Chat API вернул ошибку {exc.status_code}: {exc.message}",
                network=exc.status_code >= 500,
            ) from exc
        except httpx.TransportError as exc:
            # Прямые сетевые сбои httpx (RemoteProtocolError и т.п.), не
            # обёрнутые SDK: превращаем в доменную ошибку, не молчим.
            logger.exception("Groq: сетевой сбой транспорта (%s)", type(exc).__name__)
            raise GroqServiceError(GROQ_UNAVAILABLE_MESSAGE, network=True) from exc

        content = self._extract_completion_content(completion)
        if not use_json_mode:
            # Ответ без гарантии JSON: парсер извлечёт объект regex'ом
            # \{[\s\S]*\} (реализовано в _parse_json_loose).
            logger.debug("Модель %s отвечала без json_object mode.", model)
        logger.debug("Сырой ответ модели (%s): %s", model, content[:1000])

        payload = self._parse_json_loose(content)
        if payload is None:
            raise GroqServiceError(
                "AI не смог разобрать поручение (модель вернула не-JSON). "
                "Переформулируйте задачу и отправьте ещё раз."
            )

        try:
            parsed = TaskStructure.model_validate(payload)
        except ValidationError as exc:
            logger.error("Ответ модели не прошёл валидацию схемы: %s", exc)
            raise GroqServiceError(
                "AI вернул неожиданный формат данных. Попробуйте отправить поручение ещё раз."
            ) from exc

        # --- Мягкая нормализация полей (без падений на "мусорных" ответах) --- #
        title = parsed.task_title.strip().rstrip(".!?").strip()
        if not title:
            # Fallback для слабой модели: используем исходный текст как заголовок.
            title = (raw_text.splitlines()[0] if raw_text else "").strip()
            logger.warning("Модель вернула пустой task_title — использован исходный текст.")
        if not title:
            raise GroqServiceError(
                "В сообщении не удалось увидеть конкретную задачу. "
                "Сформулируйте поручение явно (что сделать, кому, к какому сроку)."
            )

        assignee_raw = parsed.assignee_name.strip()
        matched = self._match_assignee(assignee_raw, employees)
        if matched:
            # Найдены в базе — пишем каноническое имя сотрудника.
            assignee_final = matched
            assignee_matched = True
        elif assignee_raw and assignee_raw.lower() not in {
            "не указан",
            "не назначен",
            "нет",
            "-",
            "none",
            "null",
        }:
            # Не нашли в базе, но имя названо в поручении — сохраняем как есть.
            assignee_final = assignee_raw
            assignee_matched = False
        else:
            # Исполнитель вообще не упомянут / модель вернула заглушку.
            assignee_final = UNASSIGNED_LABEL
            assignee_matched = False

        deadline = parsed.deadline.strip() or NO_DEADLINE_LABEL

        task = StructuredTask(
            task_title=title,
            assignee_name=assignee_final,
            deadline=deadline,
            assignee_matched=assignee_matched,
            model_used=model,
        )
        logger.info(
            "Структурировано (модель=%s): title=%r assignee=%r matched=%s deadline=%r",
            model,
            task.task_title,
            task.assignee_name,
            task.assignee_matched,
            task.deadline,
        )
        return task

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Нормализация имени для сравнения: регистр, ё->е, пунктуация, лишние пробелы."""
        value = (value or "").lower().replace("ё", "е")
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    # Эвристика морфологии имён: окончание имени в косвенном падеже обычно
    # является суффиксом канонического имени ("Иван" -> "Ивану", "Петров" -> "Петрова").
    MIN_STEM_LEN = 3

    @classmethod
    def _is_declined_form(cls, word: str, name_word: str) -> bool:
        """Является ли слово косвенной формой имени (суффикс канонического слова)."""
        if not word or not name_word:
            return False
        if len(name_word) < cls.MIN_STEM_LEN or len(word) < cls.MIN_STEM_LEN:
            return False
        if word == name_word:
            return True
        # косвенная форма длиннее и каноническое имя — её префикс: "ивану" ⊃ "иван"
        if len(word) > len(name_word) and word.startswith(name_word):
            return True
        # короткая форма — префикс канонического слова ("петр" из "петров")
        if len(word) < len(name_word) and name_word.startswith(word) and len(word) >= cls.MIN_STEM_LEN:
            return True
        return False

    @classmethod
    def _match_assignee(cls, query: str, employees: list[dict[str, Any]]) -> Optional[str]:
        """Сопоставление имени исполнителя с листом «Сотрудники» (мягко).

        Возвращает каноническое имя сотрудника из базы или ``None``, если
        совпадения нет. Исключений не бросает — это вспомогательная эвристика:
            1) точное совпадение нормализованного полного имени;
            2) подстрока (полное имя/запрос друг в друге);
            3) пословное сравнение с учётом падежей (косвенные формы).
        """
        key = cls._normalize_key(query)
        if not key:
            return None

        names: list[str] = []
        for emp in employees or []:
            name = str(emp.get("Имя", "") or "").strip()
            if name:
                names.append(name)

        # 1) точное совпадение нормализованного полного имени
        for name in names:
            if cls._normalize_key(name) == key:
                return name
        # 2) полное имя содержится в запросе или наоборот
        for name in names:
            norm = cls._normalize_key(name)
            if norm in key or key in norm:
                return name
        # 3) пословное сравнение с морфологией
        query_words = key.split()
        for name in names:
            name_words = cls._normalize_key(name).split()
            if not name_words or not query_words:
                continue
            hits = sum(
                1
                for qw in query_words
                if any(cls._is_declined_form(qw, nw) or cls._is_declined_form(nw, qw) for nw in name_words)
            )
            if hits and hits == len(query_words):
                return name
            strong = [
                qw
                for qw in query_words
                if len(qw) >= cls.MIN_STEM_LEN
                and any(cls._is_declined_form(qw, nw) or cls._is_declined_form(nw, qw) for nw in name_words)
            ]
            if strong:
                return name
        return None

    @staticmethod
    def _format_employees(employees: list[dict[str, Any]]) -> str:
        """Люди из листа «Сотрудники» -> компактная таблица для промпта."""
        if not employees:
            return "(база сотрудников пуста)"
        lines: list[str] = []
        for emp in employees:
            name = str(emp.get("Имя", "") or "").strip()
            nick = str(emp.get("Никнейм", "") or "").strip()
            tg_id = str(emp.get("Telegram ID", "") or "").strip()
            if not name and not nick:
                continue
            lines.append(f"- {name or '?'} | {nick.lstrip('@') or '-'} | {tg_id or '-'}")
        return "\n".join(lines) if lines else "(база сотрудников пуста)"

    @staticmethod
    def _extract_completion_content(completion: Any) -> str:
        try:
            message = completion.choices[0].message
        except (AttributeError, IndexError) as exc:
            raise GroqServiceError("Groq вернул пустой ответ модели.") from exc

        # 1) Обычный content (json_object mode у не-reasoning моделей).
        content = getattr(message, "content", None)
        if content and str(content).strip():
            return str(content)

        # 2) reasoning-модели (openai/gpt-oss-*): при жёстком лимите токенов
        #    весь вывод уходит во внутренние рассуждения, а content приходит
        #    пустым. Извлекаем текст из reasoning/reasoning_content — там
        #    gpt-oss дублирует финальный JSON.
        for attr in ("reasoning", "reasoning_content"):
            reasoning_text = getattr(message, attr, None)
            if reasoning_text and str(reasoning_text).strip():
                logger.warning(
                    "content пуст — извлекаю ответ из поля %s (reasoning-модель).",
                    attr,
                )
                return str(reasoning_text)

        # 3) Fallback: если strict mode реализован через tool call — берём args.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            arguments = getattr(tool_calls[0].function, "arguments", None)
            if arguments:
                return str(arguments)

        raise GroqServiceError(
            "Groq вернул ответ без содержимого (пустые content и reasoning). "
            "Повторите отправку поручения."
        )

    @staticmethod
    def _parse_json_loose(content: str) -> dict[str, Any] | None:
        """Парсит JSON, терпя markdown-ограждения и текст вокруг объекта."""
        if not content:
            return None

        cleaned = content.strip()

        # 1) Прямой парсинг
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        # 2) Срезаем ```json ... ```
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                data = json.loads(fence.group(1).strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # 3) Жадный regex \{[\s\S]*\} — extreme фигурные скобки вокруг JSON.
        #    Используется, когда модель отвечала БЕЗ response_format
        #    (fallback-путь для qwen/reasoning-моделей).
        greedy = re.search(r"\{[\s\S]*\}", cleaned)
        if greedy:
            try:
                data = json.loads(greedy.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------ #
    # Диагностика / завершение работы
    # ------------------------------------------------------------------ #
    async def healthcheck(self) -> dict[str, Any]:
        """Проверка доступности Groq API минимальным запросом активной моделью."""
        if self._discovery is None:
            try:
                await self.initialize_models()
            except Exception as exc:  # noqa: BLE001 - initialize_models сама всё ловит
                logger.warning("Groq auto-discovery в healthcheck дал сбой: %s", exc)

        model = self._active_chat_model or self._settings.groq_chat_model
        hc_kwargs: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": 'Верни JSON: {"status": "ok"}',
                }
            ],
            "model": model,
            "temperature": 0.0,
        }
        if _is_reasoning_model(model):
            # reasoning-модели: маленький лимит max_tokens обрезает ответ после
            # рассуждений (пустой content) — используем max_completion_tokens
            # с запасом на reasoning И на сам JSON.
            # ВАЖНО: reasoning_effort НЕ передаётся — его нет в сигнатуре
            # groq SDK AsyncCompletions.create() (TypeError, см. _request_completion).
            hc_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS_REASONING
        else:
            hc_kwargs["max_tokens"] = 64
        try:
            completion = await self._client.chat.completions.create(**hc_kwargs)
            # _extract_completion_content проверяет наличие ответа в content,
            # а при пустом content — в полях reasoning/reasoning_content.
            answer = self._extract_completion_content(completion)
            logger.debug("Groq healthcheck ответ (%s): %.120s", model, answer)
            return {
                "ok": True,
                "chat_model": model,
                "audio_model": self._active_audio_model or self._settings.groq_audio_model,
                "available_chat_models": (
                    self._discovery.available_chat_models if self._discovery else []
                ),
                "chat_fallback_used": bool(self._discovery and self._discovery.chat_fallback_used),
            }
        except groq.NotFoundError as exc:
            logger.error("Groq healthcheck: модель %r недоступна: %s", model, exc)
            return {
                "ok": False,
                "error": (
                    f"Модель '{model}' недоступна ключу (404). Запустите auto-discovery "
                    "(перезапуск бота) или смените GROQ_CHAT_MODEL в .env."
                ),
                "chat_model": model,
            }
        except groq.GroqError as exc:
            logger.warning("Groq healthcheck failed: %s", exc)
            return {"ok": False, "error": str(exc), "chat_model": model}
        except Exception as exc:  # noqa: BLE001 - healthcheck не роняет старт
            logger.warning("Groq healthcheck: неожиданная ошибка: %s", exc)
            return {"ok": False, "error": str(exc), "chat_model": model}

    async def aclose(self) -> None:
        """Закрыть HTTP-сессию клиента Groq и собственный httpx.AsyncClient."""
        try:
            await self._client.close()
        except Exception:  # noqa: BLE001 - идемпотентный shutdown
            logger.debug("Groq client уже закрыт или ошибка при close", exc_info=True)
        if getattr(self, "_owns_http_client", False):
            try:
                await self._http_client.aclose()
            except Exception:  # noqa: BLE001 - идемпотентный shutdown
                logger.debug("httpx.AsyncClient уже закрыт", exc_info=True)


if __name__ == "__main__":  # pragma: no cover - ручная диагностика AI-сервиса
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    async def _main() -> None:
        service = GroqService()
        report = await service.initialize_models()
        print("Auto-discovery:", report.summary())
        print("Chat healthcheck:", await service.healthcheck())
        result = await service.structure_task(
            "Иван, подготовь квартальный отчёт по продажам до пятницы 18:00",
            [{"Имя": "Иван Петров", "Никнейм": "@ivan_petrov", "Telegram ID": "123456789"}],
        )
        print("Structured:", result)
        await service.aclose()

    asyncio.run(_main())
