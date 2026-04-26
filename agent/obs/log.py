"""Structured JSON logging with secret redaction.

Every log line is a single JSON object on its own line, suitable for `jq`. Required
fields: ts (UTC ISO-8601), call_sid, mode, event. Sensitive fields are redacted at
format time so misuse can't leak — see CLAUDE.md observability schema.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from typing import IO

_REDACT_KEYS_FULL = frozenset({"api_key", "authorization", "bearer", "auth_token", "secret"})
_REDACT_KEYS_PHONE = frozenset({"to_phone", "from_phone", "phone_number"})

# LogRecord stdlib attributes we never want to surface in the JSON payload.
# Derived from a fresh LogRecord at module load so this stays in sync with
# whatever attrs CPython adds (e.g., taskName landed in 3.12).
_REFERENCE_RECORD = logging.LogRecord(
    name="", level=0, pathname="", lineno=0, msg="", args=None, exc_info=None,
)
_STDLIB_RECORD_ATTRS: frozenset[str] = frozenset(_REFERENCE_RECORD.__dict__) | {
    "message", "asctime", "taskName",
}


def _redact(payload: dict[str, object]) -> dict[str, object]:
    """Return a copy of `payload` with secret fields masked."""
    out: dict[str, object] = {}
    for key, value in payload.items():
        if key in _REDACT_KEYS_FULL:
            out[key] = "[REDACTED]"
        elif key in _REDACT_KEYS_PHONE and isinstance(value, str):
            out[key] = _mask_phone(value)
        else:
            out[key] = value
    return out


def _mask_phone(number: str) -> str:
    """Mask the middle digits of a phone number (positions 5-7 of the digit run)."""
    digits = [c for c in number if c.isdigit()]
    if len(digits) < 8:
        return number
    seen = 0
    out_chars: list[str] = []
    for ch in number:
        if ch.isdigit():
            seen += 1
            if 5 <= seen <= 7:
                out_chars.append("*")
                continue
        out_chars.append(ch)
    return "".join(out_chars)


class _JsonFormatter(logging.Formatter):
    """Format every record as a single-line JSON object with redacted secrets."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, object] = {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STDLIB_RECORD_ATTRS:
                continue
            payload[key] = value
        return json.dumps(_redact(payload), default=str)


def build_logger(name: str, *, stream: IO[str] | None = None) -> logging.Logger:
    """Build a logger emitting one redacted JSON object per line."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
