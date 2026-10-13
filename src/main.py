"""Точка входа @UPRiTRM_task_bot (Этап 2).

Порядок запуска:
    1. Загрузка и валидация конфигурации из ``.env`` (pydantic-settings).
    2. Инициализация aiogram (Bot / Dispatcher), подключение роутеров:
         * ``tasks_router`` — приём текстовых/голосовых поручений (Groq + Sheets);
         * хэндлер /start — только команда, без эхо-фильтров F.text (!).
    3. Startup-хук: проверка связи с Google Таблицей (листы «Задачи»/«Сотрудники»,
       заголовки) и диагностика Groq API.
    4. set_my_commands + long polling.

ВАЖНО (фикс TypeError "unsupported operand type(s) for &: 'str' and
'_InvertFilter'"): в этом файле НЕЛЬЗЯ объявлять комбинированные фильтры
через операторы & / | над magic-фильтрами. Здесь зарегистрирован ТОЛЬКО
``@dp.message(CommandStart())``; обработка простого текста живёт целиком в
src/handlers/tasks.py через запятые: ``@tasks_router.message(F.text, ~F.text.startswith("/"))``.

Ошибки конфигурации фатальны (процесс падает сразу с понятным логом),
ошибки внешних сервисов на старте НЕ фатальны: бот поднимется и честно
сообщит о статусе подключения в ответах на команды.
"""

from __future__ import annotations

import asyncio
import html
import logging
import sys
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message

from src.config import Settings, get_settings
from src.handlers.tasks import tasks_router
from src.services.groq_service import GroqService
from src.services.sheets import SheetsService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Логирование
# ---------------------------------------------------------------------- #
def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # Не дублируем хэндлеры при повторных вызовах (тесты / reload)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    # Сетевые библиотеки слишком разговорчивы на DEBUG
    for noisy in ("aiogram.dispatcher", "urllib3", "googleapiclient", "httpx"):
        logging.getLogger(noisy).setLevel(max(level, logging.INFO))


# ---------------------------------------------------------------------- #
# Тексты ответов
# ---------------------------------------------------------------------- #
def _escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def build_welcome_text(settings: Settings, sheets_status: dict[str, Any]) -> str:
    """Приветствие /start: статус коннекта к Google Таблице + инструкция Этапа 2."""
    lines: list[str] = [
        "👋 <b>Привет! Я бот-контролёр задач UPRiTRM.</b>",
        f"<code>@{settings.BOT_USERNAME}</code> · Этап 2 (приём поручений)",
        "",
    ]

    if sheets_status.get("ok"):
        lines.append("✅ <b>Google Таблица подключена успешно.</b>")
        lines.append(f"📊 Таблица: <code>{_escape(sheets_status['spreadsheet_title'])}</code>")
        lines.append(f"🆔 ID таблицы: <code>{_escape(sheets_status['spreadsheet_id'])}</code>")
        lines.append("")
        lines.append("Структура проверена и готова к работе:")
        for sheet_name in sheets_status.get("sheets", []):
            lines.append(f"   ✔️ <code>{_escape(sheet_name)}</code>")

        created = sheets_status.get("created_sheets", [])
        written = sheets_status.get("written_headers", [])
        replaced = sheets_status.get("replaced_headers", [])
        notes: list[str] = []
        if created:
            notes.append("созданы листы: " + ", ".join(_escape(x) for x in created))
        if written:
            notes.append("записаны заголовки: " + ", ".join(_escape(x) for x in written))
        if replaced:
            notes.append("исправлены заголовки: " + ", ".join(_escape(x) for x in replaced))
        if notes:
            lines.append("")
            lines.append("🔧 Изменения при старте — " + "; ".join(notes) + ".")
        else:
            lines.append("")
            lines.append("🔧 Изменений не потребовалось — схема уже корректна.")
    else:
        error = _escape(sheets_status.get("error", "неизвестная ошибка"))
        lines.append("❌ <b>Google Таблица недоступна.</b>")
        lines.append(f"Причина: <code>{error}</code>")
        lines.append("")
        lines.append("Проверьте <code>GOOGLE_SHEET_ID</code>, наличие и права "
                     "<code>credentials.json</code> (расшарьте таблицу на email сервисного аккаунта).")

    lines.extend(
        [
            "",
            "🎯 <b>Как ставить задачи (только руководителю):</b>",
            "• пришлите <b>текст</b> поручения — например:",
            "  <i>«Ивану подготовить квартальный отчёт до пятницы 18:00»</i>;",
            "• или <b>голосовое сообщение</b> — расшифрую через Whisper и",
            "  структурирую через Llama 3.3, затем запишу в лист «Задачи».",
            "",
            "Команды: <code>/start</code> — статус, <code>/ping</code> — связь с таблицей.",
            "",
            f"Владелец бота: <code>{settings.ADMIN_TELEGRAM_ID}</code>",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Хэндлеры основного диспетчера
# ---------------------------------------------------------------------- #
# Единственный message-хэндлер здесь — строго /start.
# Никаких F.text / эхо-хэндлеров / fallback'ов в main.py быть не должно:
# весь текст и голос обрабатывает tasks_router (src/handlers/tasks.py).
async def cmd_start(message: Message, sheets: SheetsService) -> None:
    """Подтверждает, что бот жив и видит Google Таблицу."""
    logger.info(
        "/start от user_id=%s chat_id=%s",
        message.from_user.id if message.from_user else "?",
        message.chat.id,
    )
    status = await sheets.healthcheck()
    await message.answer(build_welcome_text(get_settings(), status), parse_mode=ParseMode.HTML)


async def cmd_ping(message: Message, sheets: SheetsService) -> None:
    try:
        title = await sheets.ping()
        await message.answer(
            f"🏓 Коннект с Google Таблицей в порядке: <b>{_escape(title)}</b>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:  # noqa: BLE001 - показываем причину юзеру
        logger.warning("/ping: таблица недоступна: %s", exc)
        await message.answer(
            f"⚠️ Таблица недоступна: <code>{_escape(exc)}</code>",
            parse_mode=ParseMode.HTML,
        )


# ---------------------------------------------------------------------- #
# Lifecycle
# ---------------------------------------------------------------------- #
async def on_startup(bot: Bot, sheets: SheetsService, groq_service: GroqService) -> None:
    """Startup-хук: проверка связи с таблицей, диагностика Groq, регистрация команд."""
    settings = get_settings()

    # 1. Google Sheets — критичный для работы сервис
    try:
        report = await sheets.ensure_schema()
        logger.info("Google Sheets OK.\n%s", report.as_human_text())
        startup_note = "🟢 <b>Бот перезапущен (Этап 2).</b>\n\n" + \
                       f"<pre>{_escape(report.as_human_text())}</pre>"
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - старт не роняем
        logger.exception("Google Sheets недоступна на старте: %s", exc)
        startup_note = (
            "🔴 <b>Бот запущен, но Google Таблица недоступна.</b>\n"
            f"<code>{_escape(exc)}</code>"
        )

    # 2. Groq AI — AUTO-DISCOVERY активных моделей + диагностика (не фатально).
    #    initialize_models() сам гасит все сетевые ошибки и пишет отчёт;
    #    после него healthcheck() выполняется уже активной моделью.
    try:
        discovery = await groq_service.initialize_models()
        startup_note += "\n\n🧠 <b>Groq:</b> " + _escape(discovery.summary())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq auto-discovery не выполнен: %s", exc)

    try:
        groq_status = await groq_service.healthcheck()
        if groq_status.get("ok"):
            logger.info(
                "Groq API OK (активная chat-модель: %s, audio: %s)",
                groq_status.get("chat_model"),
                groq_status.get("audio_model"),
            )
        else:
            logger.warning("Groq API недоступна: %s", groq_status.get("error"))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq healthcheck не выполнен: %s", exc)

    # 3. Уведомление администратору
    try:
        await bot.send_message(chat_id=settings.ADMIN_TELEGRAM_ID, text=startup_note,
                               parse_mode=ParseMode.HTML)
    except Exception:  # noqa: BLE001 - нет доступа к API Telegram, остаётся только лог
        logger.exception("Не удалось отправить приветственный статус администратору")

    await bot.set_my_commands(
        [
            BotCommand(command="/start", description="Статус подключения и инструкция"),
            BotCommand(command="/ping", description="Проверить связь с таблицей"),
        ]
    )
    me = await bot.get_me()
    logger.info("Бот готов: @%s (id=%s). Запускаю long polling…", me.username, me.id)


async def on_shutdown(sheets: SheetsService, groq_service: GroqService) -> None:
    """Shutdown-хук: корректно закрываем внешние соединения."""
    logger.info("Останавливаюсь: закрываю соединения с Google Sheets и Groq…")
    await sheets.aclose()
    await groq_service.aclose()


# ---------------------------------------------------------------------- #
# Сборка приложения
# ---------------------------------------------------------------------- #
def create_bot(settings: Settings) -> Bot:
    return Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def create_dispatcher(sheets: SheetsService, groq_service: GroqService,
                      settings: Settings) -> Dispatcher:
    dp = Dispatcher()

    # Порядок важен: сначала роутер задач (текст/голос), затем точечные
    # команды на самом диспетчере. В диспетчере НЕТ ни одного F.text-хэндлера.
    dp.include_router(tasks_router)

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_ping, Command("ping"))

    # DI: сервисы доступны каждому хэндлеру по имени параметра
    dp["sheets"] = sheets
    dp["groq_service"] = groq_service
    dp["settings"] = settings

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    return dp


async def main() -> int:
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError / FileNotFoundError
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logger.critical("Конфигурация некорректна: %s", exc)
        return 2

    setup_logging(settings.LOG_LEVEL)
    logger.info("Запуск @%s (Этап 2). Таблица: %s", settings.BOT_USERNAME, settings.GOOGLE_SHEET_ID)

    sheets = SheetsService(settings)
    groq_service = GroqService(settings)
    bot = create_bot(settings)
    dispatcher = create_dispatcher(sheets, groq_service, settings)

    try:
        # delete_webhook безопасен, если webhook не установлен
        try:
            await bot.delete_webhook(drop_pending_updates=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_webhook пропущен: %s", exc)

        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
        return 0
    except (KeyboardInterrupt, SystemExit):
        logger.info("Остановка по сигналу.")
        return 0
    finally:
        await sheets.aclose()
        await groq_service.aclose()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем (Ctrl+C).")
        sys.exit(0)
