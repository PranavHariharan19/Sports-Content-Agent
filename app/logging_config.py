from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if available
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include extra attributes attached to the record
        if hasattr(record, "extra_fields"):
            log_obj["extra"] = record.extra_fields

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure root logger with structured formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
