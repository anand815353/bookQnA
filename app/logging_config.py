import logging
import re
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

from app.settings import LOG_DIR, LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES, LOG_LEVEL


_REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    _REQUEST_ID_CTX.set(request_id or "-")


def get_request_id() -> str:
    return _REQUEST_ID_CTX.get()


def clear_request_id() -> None:
    _REQUEST_ID_CTX.set("-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z\-_]{16,}"),
    re.compile(r"(?:api[_-]?key|token|secret)\s*[:=]\s*([^\s,;]+)", re.IGNORECASE),
]


def redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated>"


def sanitize_for_debug(value: str, max_chars: int) -> str:
    return truncate_text(redact_text((value or "").strip()), max_chars)


def setup_logging() -> None:
    level_name = LOG_LEVEL.upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = LOG_DIR
    max_bytes = LOG_FILE_MAX_BYTES
    backup_count = LOG_FILE_BACKUP_COUNT

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
    )
    request_filter = RequestIdFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)

    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
