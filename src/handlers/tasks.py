"""Роутер приёма поручений руководителя (Этап 2) — @UPRiTRM_task_bot.

Конвейер обработки:
    текст/войс -> Groq (Whisper + LLM-структурирование) -> Google Sheets ->
    карточка подтверждения в чат.

Защита от зависаний:
    * все операции с gspread идут через SheetsService (asyncio.to_thread +
      asyncio.wait_for + socket.setdefaulttimeout) — event loop не блокируется;
    * каждый шаг конвейера логируется (получение, транскрибация,
      структурирование, запись, итог);
    * любая сетевая ошибка превращается в понятное сообщение пользователю,
      а не в вечное «Обрабатываю поручение...».

Фильтры aiogram 3 перечисляются ЧЕРЕЗ ЗАПЯТУЮ (никаких '&').
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import tempfile
from typing import Any, Optional

from aiogram import Bot, F, Router
from aiogram.types import Message

from src.config import Settings
from src.services.groq_service import GroqService, GroqServiceError, StructuredTask
from src.services.sheets import Employee, SheetsService

logger = logging.getLogger(__name__)

tasks_router = Router(name="tasks")

NO_RIGHTS_TEXT = "⛔ У вас нет прав для постановки задач."
GROQ_UNAVAILABLE_TEXT = (
    "⚠️ Не удалось связаться с ИИ-сервисом Groq. Попробуйте повторить запрос через минуту."
)


# ---------------------------------------------------------------------- #
# Вспомогательные функции
# ---------------------------------------------------------------------- #
def _is_admin(message: Message, settings: Settings) -> bool:
    user = message.from_user
    return bool(user and user.id == settings.ADMIN_TELEGRAM_ID)


def _esc(text: Any) -> str:
    """Экранирование для HTML-разметки сообщений Telegram."""
    return html.escape(str(text), quote=False)


async def _safe_edit(status_msg: Optional[Message], text: str) -> None:
    """Редактирует статусное сообщение; при сбое edit — отправляет новое."""
    if status_msg is None:
        return
    try:
        await status_msg.edit_text(text, parse_mode="HTML")
    except Exception as exc:  # noqa: BLE001 - rate limit / старое сообщение и т.п.
        logger.warning("edit_text не удался (%s) — отправляю новым сообщением", exc)
        try:
            await status_msg.answer(text, parse_mode="HTML")
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось доставить сообщение даже без редактирования")


def _employees_to_dicts(employees: list[Employee]) -> list[dict[str, str]]:
    """Employee -> словари с русскими ключами (формат, который ждёт GroqService)."""
    return [
        {
            "Имя": emp.name,
            "Никнейм": emp.nickname,
            "Telegram ID": emp.tg_id_str,
        }
        for emp in employees
    ]


def _resolve_assignee_tg_id(assignee_name: str, employees: list[Employee]) -> str:
    """TG ID ответственного по имени (мягко: не нашли -> пустая строка)."""
    cleaned = (assignee_name or "").strip().lower().replace("ё", "е")
    if not cleaned or cleaned in {"не указан", "не назначен", "none", "null"}:
        return ""
    for emp in employees:
        emp_name = emp.name.lower().replace("ё", "е").strip()
        if cleaned == emp_name or cleaned in emp_name or emp_name in cleaned:
            return emp.tg_id_str
    return ""


# ---------------------------------------------------------------------- #
# Единый конвейер обработки поручения
# ---------------------------------------------------------------------- #
async def process_directive(
    message: Message,
    directive_text: str,
    sheets: SheetsService,
    groq_service: GroqService,
    status_msg: Optional[Message] = None,
) -> None:
    """text -> Groq-структурирование -> запись в Google Sheets -> карточка."""
    if status_msg is None:
        status_msg = await message.answer("⏳ Распознаю задачу через ИИ...")
    else:
        await _safe_edit(status_msg, "⏳ Распознаю задачу через ИИ...")

    directive_text = (directive_text or "").strip()
    if not directive_text:
        await _safe_edit(status_msg, "❌ Пустое сообщение — нечего распознавать.")
        return

    # ------------------------------------------------------------ #
    # Шаг 1. База сотрудников (не критична: сбой -> [])
    # ------------------------------------------------------------ #
    employees: list[Employee] = []
    try:
        employees = await sheets.get_employees()
        logger.info("Сотрудники загружены из таблицы: %d", len(employees))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось загрузить сотрудников (%s) — продолжаю с пустой базой", exc)

    # ------------------------------------------------------------ #
    # Шаг 2. Структурирование через Groq
    # ------------------------------------------------------------ #
    structured: Optional[StructuredTask] = None
    try:
        structured = await groq_service.structure_task(
            directive_text, _employees_to_dicts(employees)
        )
        logger.info(
            "Groq вернул структуру: title=%r assignee=%r deadline=%r matched=%s",
            structured.task_title,
            structured.assignee_name,
            structured.deadline,
            structured.assignee_matched,
        )
    except asyncio.CancelledError:
        raise
    except GroqServiceError as exc:
        logger.error("Groq structure_task: %s (network=%s)", exc, exc.network)
        if exc.network:
            await _safe_edit(status_msg, GROQ_UNAVAILABLE_TEXT)
        else:
            await _safe_edit(
                status_msg,
                "⚠️ ИИ не смог распознать структуру поручения. "
                "Сформулируйте явно: что сделать, кому, к какому сроку.",
            )
        return
    except Exception as exc:  # noqa: BLE001 - сетевые/HTTP сбои Groq SDK
        logger.exception("Сбой структурирования поручения через Groq")
        await _safe_edit(status_msg, GROQ_UNAVAILABLE_TEXT)
        return

    if structured is None:
        await _safe_edit(status_msg, GROQ_UNAVAILABLE_TEXT)
        return

    # TG ID ответственного (если имя сопоставилось с базой сотрудников)
    assignee_tg_id = _resolve_assignee_tg_id(structured.assignee_name, employees)
    logger.info(
        "Ответственный: %r -> tg_id=%r", structured.assignee_name, assignee_tg_id or "—"
    )

    # ------------------------------------------------------------ #
    # Шаг 3. Запись в Google Таблицу (неблокирующая, таймаут 15 с)
    # ------------------------------------------------------------ #
    await _safe_edit(status_msg, "💾 Записываю задачу в Google Таблицу...")

    try:
        task_data = await sheets.add_task(
            title=structured.task_title,
            assignee=structured.assignee_name,
            deadline=structured.deadline,
            tg_id=assignee_tg_id,
        )
        await status_msg.edit_text(
            f"✅ <b>Поручение зафиксировано!</b>\n\n"
            f"🆔 <b>Код:</b> {task_data['task_id']}\n"
            f"📌 <b>Задача:</b> {_esc(task_data['title'])}\n"
            f"👤 <b>Ответственный:</b> {_esc(task_data['assignee'])} "
            f"(TG: {_esc(task_data['tg_id'] or 'Не привязан')})\n"
            f"📅 <b>Срок:</b> {_esc(task_data['deadline'])}\n"
            f"📊 <b>Статус:</b> Новая",
            parse_mode="HTML",
        )
        logger.info("Поручение %s обработано полностью", task_data["task_id"])
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Ошибка записи в Google Sheets: {e}", exc_info=True)
        await status_msg.edit_text(
            f"⚠️ <b>Задача распознана, но не записана в таблицу</b>:\n\n"
            f"📌 {_esc(structured.task_title)}\n"
            f"👤 {_esc(structured.assignee_name)}\n"
            f"📅 {_esc(structured.deadline)}\n\n"
            f"Ошибка Google Sheets: {_esc(str(e)[:100])}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------- #
# Хэндлеры
# ---------------------------------------------------------------------- #
@tasks_router.message(F.text, ~F.text.startswith("/"))
async def handle_admin_text(
    message: Message,
    sheets: SheetsService,
    groq_service: GroqService,
    settings: Settings,
) -> None:
    """Текстовое поручение от руководителя."""
    if not _is_admin(message, settings):
        logger.info(
            "Отказ в постановке задачи: user_id=%s не является руководителем",
            message.from_user.id if message.from_user else "?",
        )
        await message.answer(NO_RIGHTS_TEXT)
        return

    text = (message.text or "").strip()
    logger.info(
        "Получено текстовое поручение от user_id=%s: %.100s",
        message.from_user.id if message.from_user else "?",
        text,
    )
    # Сразу при получении отвечаем пользователю статусным сообщением
    status_msg = await message.answer("⏳ Распознаю задачу через ИИ...")

    try:
        await process_directive(
            message, text, sheets, groq_service, status_msg=status_msg
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - последний рубеж: бот не падает на апдейте
        logger.exception("Непредвиденная ошибка при обработке текстового поручения")
        try:
            await _safe_edit(status_msg, "❌ Внутренняя ошибка. Попробуйте повторить позже.")
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось сообщить пользователю об ошибке")


@tasks_router.message(F.voice)
async def handle_admin_voice(
    message: Message,
    bot: Bot,
    sheets: SheetsService,
    groq_service: GroqService,
    settings: Settings,
) -> None:
    """Голосовое поручение: скачать .ogg -> Whisper -> общий конвейер."""
    if not _is_admin(message, settings):
        logger.info(
            "Отказ в постановке задачи (голос): user_id=%s не является руководителем",
            message.from_user.id if message.from_user else "?",
        )
        await message.answer(NO_RIGHTS_TEXT)
        return

    voice = message.voice
    if voice is None or not voice.file_id:
        await message.answer("❌ Не удалось прочитать голосовое сообщение.")
        return

    # Сразу при получении отвечаем пользователю статусным сообщением
    status_msg = await message.answer("⏳ Распознаю задачу через ИИ...")
    logger.info(
        "Получено голосовое поручение от user_id=%s (duration=%ss)",
        message.from_user.id if message.from_user else "?",
        voice.duration,
    )

    temp_fd, temp_path = tempfile.mkstemp(suffix=".ogg", prefix="uprirm_voice_")
    os.close(temp_fd)

    try:
        # 1. Скачивание голосового файла из Telegram
        try:
            await bot.download(voice.file_id, destination=temp_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось скачать голосовое сообщение")
            await _safe_edit(
                status_msg,
                f"❌ Не удалось скачать аудиофайл: {_esc(str(exc)[:100])}. Повторите попытку.",
            )
            return

        file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        logger.info("Голосовое скачано: %s байт -> %s", file_size, temp_path)

        # 2. Транскрибация через Groq Whisper
        try:
            transcript = await groq_service.transcribe_voice(temp_path)
        except asyncio.CancelledError:
            raise
        except GroqServiceError as exc:
            logger.error("Groq transcribe_voice: %s (network=%s)", exc, exc.network)
            await _safe_edit(status_msg, GROQ_UNAVAILABLE_TEXT)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Сбой транскрибации голоса")
            await _safe_edit(status_msg, GROQ_UNAVAILABLE_TEXT)
            return

        text = (transcript or "").strip()
        if not text:
            await _safe_edit(
                status_msg,
                "❌ Речь не распознана. Повторите попытку или отправьте поручение текстом.",
            )
            return
        logger.info("Расшифровано (%d символов): %.100s", len(text), text)

        # 3. Дальше — общий конвейер (структурирование -> таблица -> карточка)
        await process_directive(
            message, text, sheets, groq_service, status_msg=status_msg
        )

    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Непредвиденная ошибка при обработке голосового поручения")
        try:
            await _safe_edit(
                status_msg, "❌ Ошибка при обработке голосового сообщения. Попробуйте ещё раз."
            )
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось сообщить об ошибке обработки голоса")
    finally:
        # Гарантированное удаление временного файла
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug("Временный файл удалён: %s", temp_path)
            except OSError as exc:
                logger.warning("Не удалось удалить %s: %s", temp_path, exc)
