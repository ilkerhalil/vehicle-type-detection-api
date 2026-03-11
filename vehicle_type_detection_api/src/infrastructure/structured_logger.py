"""
Structured JSON logging with correlation ID support.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any

from ..core.correlation import get_correlation_id, correlation_id_var


class StructuredLogFormatter(logging.Formatter):
    """JSON log formatter with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class TextLogFormatter(logging.Formatter):
    """Text log formatter with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", None) or get_correlation_id()
        record.correlation_id = cid
        return f"{datetime.utcnow().isoformat()} [{record.levelname}] [{cid}] {record.getMessage()}"


def setup_structured_logger(name: str, log_format: str = "json") -> logging.Logger:
    """Setup logger with structured formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Set formatter based on format
    if log_format == "json":
        formatter = StructuredLogFormatter()
    else:
        formatter = TextLogFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes correlation ID."""

    def process(self, msg: str, kwargs: Any) -> tuple:
        kwargs.setdefault("extra", {})
        kwargs["extra"]["correlation_id"] = get_correlation_id()
        return msg, kwargs
