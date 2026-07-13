"""Small, dependency-free structured logging utilities for HistoWeave.

The formatter deliberately writes one JSON object per line to stderr.  Correlation
fields live in :mod:`contextvars`, so concurrent runs and nested benchmark steps do
not overwrite one another.  Values are redacted before serialization; callers should
still avoid putting credentials in log messages in the first place.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

_CORRELATION: ContextVar[dict[str, str] | None] = ContextVar(
    "histoweave_log_correlation", default=None
)
_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|passwd|secret|token|api[_-]?key)",
    re.IGNORECASE,
)
_TEXT_REDACTIONS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(password|passwd|secret|token|api[_-]?key)"
        r"(\s*[=:]\s*)([^\s,;]+)"
    ),
    re.compile(r"(?i)(https?://[^\s:/@]+:)([^\s/@]+)(@)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


def redact_text(value: str) -> str:
    """Redact common credential forms from free-form text."""

    redacted = _TEXT_REDACTIONS[0].sub("Bearer [REDACTED]", value)
    redacted = _TEXT_REDACTIONS[1].sub(r"\1\2[REDACTED]", redacted)
    redacted = _TEXT_REDACTIONS[2].sub(r"\1[REDACTED]\3", redacted)
    return _TEXT_REDACTIONS[3].sub("[REDACTED]", redacted)


def redact(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe, recursively redacted representation of ``value``."""

    if key is not None and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact(item) for item in value]
    return redact_text(str(value))


def correlation() -> dict[str, str]:
    """Return a copy of the current run/step correlation fields."""

    return dict(_CORRELATION.get() or {})


@contextmanager
def log_context(*, run_id: str | None = None, step_id: str | None = None) -> Iterator[None]:
    """Temporarily attach run and/or step identifiers to HistoWeave log records."""

    values = correlation()
    if run_id is not None:
        values["run_id"] = str(run_id)
    if step_id is not None:
        values["step_id"] = str(step_id)
    token = _CORRELATION.set(values)
    try:
        yield
    finally:
        _CORRELATION.reset(token)


class JsonFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a redacted JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
            **correlation(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = redact(value, key=key)
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


class RedactingTextFormatter(logging.Formatter):
    """Apply the same secret policy to human-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def get_logger(name: str) -> logging.Logger:
    """Return a logger in HistoWeave's hierarchy without configuring global logging."""

    return logging.getLogger(name)


def configure_logging(
    *,
    level: str | int = "WARNING",
    log_format: str = "text",
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure HistoWeave's logger hierarchy and return its root logger.

    Only handlers installed by this function are replaced.  Application handlers and
    pytest's capture handler remain untouched.
    """

    if log_format not in {"text", "json"}:
        raise ValueError("log_format must be 'text' or 'json'")
    logger = logging.getLogger("histoweave")
    for handler in list(logger.handlers):
        if getattr(handler, "_histoweave_managed", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler._histoweave_managed = True  # type: ignore[attr-defined]
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(RedactingTextFormatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Emit a named event with structured fields through the standard logging API."""

    safe_fields = {key: redact(value, key=key) for key, value in fields.items()}
    logger.log(level, redact_text(message), extra={"event": event, **safe_fields})


__all__ = [
    "JsonFormatter",
    "configure_logging",
    "correlation",
    "get_logger",
    "log_context",
    "log_event",
    "redact",
    "redact_text",
]
