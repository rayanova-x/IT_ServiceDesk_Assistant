"""
Structured, JSON-formatted logging for the live serving path -- replaces
the ~225 bare print() calls scattered across src/*.py with something a
real pilot could actually search/alert on after the fact. print() output
only exists in whichever terminal happened to be watching when it ran;
a JSON line in a rotating file survives a server restart, greps cleanly,
and carries the same fields every time instead of whatever string a
given print() call happened to format.

One line per log record: timestamp, level, logger name, message, plus
whatever extra fields the caller passes via `extra={"session_id": ...}`
(standard logging.Logger kwarg -- not custom plumbing). Written to BOTH
stdout (so `python ui/server.py` in a terminal still shows activity live,
matching every script's existing print()-based convention) and a
rotating file under logs/ (so it survives past that terminal's scrollback
and past a restart).
"""

import json
import logging
import logging.handlers

from config import LOG_DIR

# Fields every stdlib LogRecord carries that aren't useful noise in a
# structured line -- excluded so `extra={...}` fields stand out instead
# of being buried among internal logging-module bookkeeping.
_STANDARD_RECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_KEYS}
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Call ONCE, at process startup (ui/server.py's module import time) --
    configures the root logger so every module's `logging.getLogger(...)`
    call (gemini_retry.py, chat_service.py, this module's own callers)
    inherits the same JSON formatting and handlers without each module
    needing to configure anything itself.
    """
    LOG_DIR.mkdir(exist_ok=True)
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
