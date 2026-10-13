"""Сервис Google Таблиц (gspread) для @UPRiTRM_task_bot — Этап 2.

Против зависаний (сеть/VPN без таймаутов):
    * socket.setdefaulttimeout(15) — глобальный сокетный таймаут для всех
      HTTP-запросов gspread/google-auth (иначе запрос к Sheets API может
      висеть в recv() бесконечно и блокировать рабочий поток);
    * все вызовы gspread выполняются СТРОГО в asyncio.to_thread с внешним
      asyncio.wait_for (10 с чтение, 15 с запись) — event loop aiogram не
      блокируется никогда;
    * ссылки на листы self._tasks_sheet / self._employees_sheet кэшируются
      при connect()/ensure_schema() — таблица НЕ открывается заново на каждую
      добавленную задачу;
    * подробное логирование КАЖДОГО шага (начало операции, старт потока,
      успешное завершение, таймаут).
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError as GspreadAPIError
from gspread.exceptions import WorksheetNotFound

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Глобальный сокетный таймаут: любые сетевые операции (в т.ч. внутри
# requests-сессии gspread) обязаны отвалиться по таймауту, а не висеть вечно.
socket.setdefaulttimeout(15)

SCOPES: tuple[str, ...] = (
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
)

SHEET_TASKS_TITLE: str = "Задачи"
SHEET_EMPLOYEES_TITLE: str = "Сотрудники"

TASKS_HEADERS: tuple[str, ...] = (
    "ID",
    "Дата",
    "Задача",
    "Ответственный",
    "Telegram ID",
    "Дедлайн",
    "Статус",
)

EMPLOYEES_HEADERS: tuple[str, ...] = ("Имя", "Никнейм", "Telegram ID")

STATUS_NEW: str = "Новая"
TASK_ID_PREFIX: str = "T-"
UNASSIGNED_LABEL: str = "Не привязан"

#: Таймауты операций (секунды).
WRITE_TIMEOUT_SECONDS: float = 15.0
READ_TIMEOUT_SECONDS: float = 10.0


@dataclass(frozen=True)
class Employee:
    """Запись листа «Сотрудники»."""

    name: str
    nickname: str = ""
    telegram_id: Optional[int] = None

    @property
    def display(self) -> str:
        if self.nickname:
            nick = self.nickname if self.nickname.startswith("@") else f"@{self.nickname}"
            return f"{self.name} ({nick})"
        return self.name

    @property
    def tg_id_str(self) -> str:
        return str(self.telegram_id) if self.telegram_id else ""


@dataclass(frozen=True)
class SchemaReport:
    """Итог проверки структуры таблицы на старте (для /start и логов)."""

    spreadsheet_title: str
    spreadsheet_id: str
    sheets: list[str]
    created_sheets: list[str] = field(default_factory=list)
    written_headers: list[str] = field(default_factory=list)
    replaced_headers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spreadsheet_title": self.spreadsheet_title,
            "spreadsheet_id": self.spreadsheet_id,
            "sheets": list(self.sheets),
            "created_sheets": list(self.created_sheets),
            "written_headers": list(self.written_headers),
            "replaced_headers": list(self.replaced_headers),
        }

    def as_human_text(self) -> str:
        lines: list[str] = [
            f"Таблица: {self.spreadsheet_title} (id={self.spreadsheet_id})",
            "Листы: " + ", ".join(self.sheets),
        ]
        notes: list[str] = []
        if self.created_sheets:
            notes.append("созданы листы: " + ", ".join(self.created_sheets))
        if self.written_headers:
            notes.append("записаны заголовки: " + ", ".join(self.written_headers))
        if self.replaced_headers:
            notes.append("исправлены заголовки: " + ", ".join(self.replaced_headers))
        lines.append("Изменения при старте: " + ("; ".join(notes) if notes else "нет"))
        return "\n".join(lines)


class SheetsServiceError(RuntimeError):
    """Ошибка уровня сервиса таблиц (авторизация, доступ, сеть, таймаут)."""


class SheetsService:
    """Асинхронная неблокирующая обёртка над синхронным gspread.

    Публичные методы можно вызывать прямо из хэндлеров aiogram: каждый
    сетевой вызов живёт в отдельном потоке и ограничен таймаутом.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._client: Optional[gspread.Client] = None
        self._spreadsheet: Optional[gspread.Spreadsheet] = None
        # Кэшированные ссылки на листы (не переоткрываем документ на каждую задачу).
        self._tasks_sheet: Optional[gspread.Worksheet] = None
        self._employees_sheet: Optional[gspread.Worksheet] = None
        self._schema_flags: dict[str, Any] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Внутреннее исполнение: синхронный вызов -> поток -> wait_for
    # ------------------------------------------------------------------ #
    async def _run_sync(
        self,
        func: Callable[[], Any],
        *,
        timeout: float = WRITE_TIMEOUT_SECONDS,
        op_name: str = "op",
    ) -> Any:
        """Выполнить синхронную операцию gspread в потоке с явным таймаутом.

        Двойная защита от зависаний:
            1. socket.setdefaulttimeout(15) — сокеты сами отваливаются по таймауту;
            2. asyncio.wait_for — event loop гарантированно не ждёт дольше `timeout`.
        """
        logger.debug("Sheets[%s]: старт операции (timeout=%.0fs)", op_name, timeout)
        try:
            result = await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout)
            logger.debug("Sheets[%s]: операция завершена", op_name)
            return result
        except asyncio.TimeoutError as exc:
            logger.error(
                "Sheets[%s]: таймаут %.0fs — Google API не ответил "
                "(проверьте сеть/VPN/доступ к sheets.googleapis.com)",
                op_name,
                timeout,
            )
            raise SheetsServiceError(
                f"Google Sheets не ответила за {timeout:.0f} сек (операция: {op_name})."
            ) from exc
        except GspreadAPIError as exc:
            logger.error("Sheets[%s]: ошибка Google API: %s", op_name, exc)
            message = str(exc)
            lowered = message.lower()
            if "403" in message or "permission" in lowered or "accessnotfound" in lowered:
                raise SheetsServiceError(
                    "Нет доступа к таблице. Расшарьте документ для email сервисного "
                    f"аккаунта. Ответ API: {message}"
                ) from exc
            if "404" in message or "notfound" in lowered:
                raise SheetsServiceError(
                    f"Таблица с ID '{self._settings.GOOGLE_SHEET_ID}' не найдена. "
                    "Проверьте GOOGLE_SHEET_ID в .env."
                ) from exc
            raise SheetsServiceError(f"Google Sheets API вернул ошибку: {message}") from exc

    # ------------------------------------------------------------------ #
    # Авторизация / подключение / схема
    # ------------------------------------------------------------------ #
    def _authorize_sync(self) -> gspread.Client:
        logger.info(
            "Sheets[auth]: авторизация Service Account (%s)…",
            self._settings.CREDENTIALS_FILE_PATH,
        )
        credentials = Credentials.from_service_account_file(
            str(self._settings.CREDENTIALS_FILE_PATH),
            scopes=list(SCOPES),
        )
        client = gspread.authorize(credentials)
        logger.info(
            "Авторизация в Google выполнена (service account: %s)",
            getattr(credentials, "service_account_email", "<unknown>"),
        )
        return client

    def _open_and_cache_sync(self) -> dict[str, Any]:
        """Открыть таблицу, проверить схему и ЗАКЭШИРОВАТЬ ссылки на листы.

        Выполняется целиком в одном потоке (последовательные сетевые шаги),
        снаружи ограничен asyncio.wait_for.
        """
        assert self._client is not None, "_authorize_sync должен предшествовать"

        spreadsheet = self._client.open_by_key(self._settings.GOOGLE_SHEET_ID)
        logger.info(
            "Открыта таблица '%s' (id=%s)",
            spreadsheet.title,
            self._settings.GOOGLE_SHEET_ID,
        )

        created_sheets: list[str] = []
        written_headers: list[str] = []
        replaced_headers: list[str] = []

        tasks_sheet, tasks_info = self._get_or_create_worksheet_sync(
            spreadsheet, SHEET_TASKS_TITLE, TASKS_HEADERS
        )
        employees_sheet, employees_info = self._get_or_create_worksheet_sync(
            spreadsheet, SHEET_EMPLOYEES_TITLE, EMPLOYEES_HEADERS
        )

        for info in (tasks_info, employees_info):
            if info["created"]:
                created_sheets.append(info["title"])
            if info["written_headers"]:
                written_headers.append(info["title"])
            if info["replaced_headers"]:
                replaced_headers.append(info["title"])

        # Кэшируем ссылки на листы - переоткрывать документ на каждую задачу не нужно.
        self._spreadsheet = spreadsheet
        self._tasks_sheet = tasks_sheet
        self._employees_sheet = employees_sheet
        return {
            "title": spreadsheet.title,
            "sheets": [SHEET_TASKS_TITLE, SHEET_EMPLOYEES_TITLE],
            "created_sheets": created_sheets,
            "written_headers": written_headers,
            "replaced_headers": replaced_headers,
        }

    @staticmethod
    def _get_or_create_worksheet_sync(
        spreadsheet: gspread.Spreadsheet, title: str, headers: tuple[str, ...]
    ) -> tuple[gspread.Worksheet, dict[str, Any]]:
        """Возвращает (worksheet, info) где info фиксирует изменения схемы."""
        info: dict[str, Any] = {
            "title": title,
            "created": False,
            "written_headers": False,
            "replaced_headers": False,
        }
        try:
            worksheet = spreadsheet.worksheet(title)
        except WorksheetNotFound:
            logger.warning("Лист '%s' отсутствует — создаём.", title)
            worksheet = spreadsheet.add_worksheet(
                title=title, rows=1000, cols=max(10, len(headers))
            )
            info["created"] = True

        values = worksheet.get_values(range_name="1:1")
        first_row = [str(v).strip() for v in (values[0] if values else [])]
        current = [c for c in first_row if c]

        if current != list(headers):
            if not current:
                logger.info("Лист '%s' пуст — записываем заголовки: %s", title, list(headers))
                info["written_headers"] = True
            else:
                logger.warning(
                    "Лист '%s': заголовки %s не совпадают со схемой %s — приводим A1 к схеме.",
                    title,
                    current,
                    list(headers),
                )
                info["replaced_headers"] = True
            last_col = SheetsService._column_letter(len(headers))
            worksheet.update(values=[list(headers)], range_name=f"A1:{last_col}1")
        else:
            logger.debug("Лист '%s': структура в норме.", title)
        return worksheet, info

    @staticmethod
    def _column_letter(index_1_based: int) -> str:
        letters = ""
        while index_1_based > 0:
            index_1_based, remainder = divmod(index_1_based - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    async def connect(self) -> None:
        """Гарантирует соединение и кэш листов. Идемпотентно."""
        if self._tasks_sheet is not None and self._employees_sheet is not None:
            return
        async with self._lock:
            if self._tasks_sheet is not None and self._employees_sheet is not None:
                return
            if self._client is None:
                self._client = await self._run_sync(
                    self._authorize_sync, timeout=WRITE_TIMEOUT_SECONDS, op_name="authorize"
                )
            schema_info = await self._run_sync(
                self._open_and_cache_sync, timeout=WRITE_TIMEOUT_SECONDS, op_name="open+schema"
            )
            self._schema_flags = schema_info
            logger.info("Google Sheets: соединение установлено, листы закэшированы.")

    async def ensure_schema(self) -> SchemaReport:
        """Startup-проверка связи и структуры таблицы (используется main.py)."""
        await self.connect()
        return SchemaReport(
            spreadsheet_title=self._spreadsheet.title if self._spreadsheet else "",
            spreadsheet_id=self._settings.GOOGLE_SHEET_ID,
            sheets=[SHEET_TASKS_TITLE, SHEET_EMPLOYEES_TITLE],
            created_sheets=list(self._schema_flags.get("created_sheets", [])),
            written_headers=list(self._schema_flags.get("written_headers", [])),
            replaced_headers=list(self._schema_flags.get("replaced_headers", [])),
        )

    async def ping(self) -> str:
        """Лёгкая проверка связи: название таблицы."""
        await self.connect()
        return await self._run_sync(
            lambda: self._spreadsheet.title,
            timeout=READ_TIMEOUT_SECONDS,
            op_name="ping",
        )

    async def healthcheck(self) -> dict[str, Any]:
        """Статус подключения (для /start и /ping). Никогда не бросает наружу."""
        try:
            report = await self.ensure_schema()
            return {"ok": True, **report.as_dict()}
        except (SheetsServiceError, OSError, ValueError) as exc:
            logger.exception("Google Sheets недоступна")
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Сотрудники
    # ------------------------------------------------------------------ #
    async def get_employees(self) -> list[Employee]:
        """Читает лист «Сотрудники» (to_thread + таймаут 10 с).

        Пустой лист или сетевой сбой -> пустой список [] (бот не падает,
        структурирование продолжится без базы сотрудников).
        """
        try:
            await self.connect()
            rows: list[list[Any]] = await self._run_sync(
                lambda: self._employees_sheet.get_all_values(),
                timeout=READ_TIMEOUT_SECONDS,
                op_name="get_employees",
            )
        except (SheetsServiceError, GspreadAPIError, OSError) as exc:
            logger.warning("get_employees: не удалось прочитать лист (%s) -> []", exc)
            return []

        employees: list[Employee] = []
        for row in rows[1:]:  # пропускаем шапку
            if not any(str(cell).strip() for cell in row):
                continue
            name = str(row[0]).strip() if len(row) > 0 else ""
            nickname = str(row[1]).strip() if len(row) > 1 else ""
            tg_raw = str(row[2]).strip() if len(row) > 2 else ""
            if not name:
                continue
            try:
                tg_id: Optional[int] = int(tg_raw) if tg_raw else None
            except ValueError:
                tg_id = None
            employees.append(Employee(name=name, nickname=nickname, telegram_id=tg_id))

        logger.info("get_employees: получено сотрудников: %d", len(employees))
        return employees

    async def find_employee(self, query: str) -> Optional[Employee]:
        """Поиск сотрудника по имени/никнейму (нечёткое сопоставление)."""
        cleaned = (query or "").strip().lower().replace("ё", "е")
        if not cleaned:
            return None
        employees = await self.get_employees()
        return self._match_employee(cleaned, employees)

    @staticmethod
    def _match_employee(normalized_query: str, employees: list[Employee]) -> Optional[Employee]:
        for emp in employees:
            emp_name = emp.name.lower().replace("ё", "е").strip()
            if normalized_query == emp_name:
                return emp
        for emp in employees:
            emp_name = emp.name.lower().replace("ё", "е").strip()
            if normalized_query in emp_name or emp_name in normalized_query:
                return emp
        query_words = set(normalized_query.split())
        for emp in employees:
            emp_words = set(emp.name.lower().replace("ё", "е").split())
            if query_words & emp_words:
                return emp
        for emp in employees:
            nick = emp.nickname.lstrip("@").lower()
            if nick and (nick == normalized_query or nick in normalized_query):
                return emp
        return None

    # ------------------------------------------------------------------ #
    # Задачи
    # ------------------------------------------------------------------ #
    async def add_task(
        self,
        title: str,
        assignee: str,
        deadline: str,
        tg_id: str = "",
    ) -> dict[str, str]:
        """Добавляет задачу в лист «Задачи» — неблокирующе, с таймаутом.

        Вся синхронная последовательность (col_values -> append_row) выполняется
        ОДНИМ блоком в потоке; снаружи asyncio.wait_for(timeout=15с). При
        превышении — SheetsServiceError (бот НЕ зависает).

        Возвращает: {"task_id", "title", "assignee", "deadline", "tg_id"}.
        """
        await self.connect()

        logger.info(f"Начинаю запись задачи в Google Sheets: {title}")

        def _sync_append() -> str:
            # 1. Получаем номер строки (количество непустых ID + шапка)
            sheet = self._tasks_sheet
            assert sheet is not None, "Лист 'Задачи' не закэширован (connect() не выполнен)"
            ids_column = sheet.col_values(1)
            used_rows = sum(1 for cell in ids_column[1:] if str(cell).strip())
            task_id = f"{TASK_ID_PREFIX}{used_rows + 1}"

            # 2. Формируем дату
            now = datetime.now().strftime("%d.%m.%Y %H:%M")
            row = [
                task_id,
                now,
                title,
                assignee,
                str(tg_id) if tg_id else UNASSIGNED_LABEL,
                deadline,
                STATUS_NEW,
            ]

            # 3. Добавляем строку
            sheet.append_row(row, value_input_option="USER_ENTERED")
            return task_id

        task_id = await self._run_sync(
            _sync_append, timeout=WRITE_TIMEOUT_SECONDS, op_name="add_task"
        )
        logger.info(f"Задача {task_id} успешно записана в Google Sheets")

        return {
            "task_id": task_id,
            "title": title,
            "assignee": assignee,
            "deadline": deadline,
            "tg_id": str(tg_id) if tg_id else "",
        }

    async def next_task_id(self) -> str:
        """Следующий ID задачи (для превью/тестов)."""
        await self.connect()

        def _calc() -> str:
            sheet = self._tasks_sheet
            assert sheet is not None
            ids_column = sheet.col_values(1)
            used_rows = sum(1 for cell in ids_column[1:] if str(cell).strip())
            return f"{TASK_ID_PREFIX}{used_rows + 1}"

        return await self._run_sync(_calc, timeout=READ_TIMEOUT_SECONDS, op_name="next_task_id")

    async def get_tasks(self) -> list[list[Any]]:
        """Все данные листа «Задачи» (для будущих этапов)."""
        await self.connect()
        return await self._run_sync(
            lambda: self._tasks_sheet.get_all_values(),
            timeout=READ_TIMEOUT_SECONDS,
            op_name="get_tasks",
        )

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        """Освобождение HTTP-сессии gspread (идемпотентно, ошибки логируются)."""
        async with self._lock:
            if self._client is not None:
                session = getattr(self._client, "session", None)
                close = getattr(session, "close", None)
                if callable(close):
                    try:
                        await asyncio.wait_for(asyncio.to_thread(close), timeout=5.0)
                    except Exception as exc:
                        logger.warning("Sheets: закрытие сессии с замечанием: %s", exc)
            self._client = None
            self._spreadsheet = None
            self._tasks_sheet = None
            self._employees_sheet = None
            logger.info("Соединение с Google Sheets закрыто.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    async def _main() -> None:
        service = SheetsService()
        print(await service.ensure_schema())
        print(await service.get_employees())
        await service.aclose()

    asyncio.run(_main())
