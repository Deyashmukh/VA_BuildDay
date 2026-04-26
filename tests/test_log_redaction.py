"""Tests for secret-redacting JSON log formatter."""

from __future__ import annotations

import io
import json

from agent.obs.log import build_logger


def test_log_line_includes_required_fields() -> None:
    buf = io.StringIO()
    logger = build_logger("test_required", stream=buf)
    logger.info("event", extra={"call_sid": "CA123", "mode": "ivr", "event": "prompt_classified"})
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["call_sid"] == "CA123"
    assert payload["mode"] == "ivr"
    assert payload["event"] == "prompt_classified"
    assert "ts" in payload


def test_redaction_masks_phone_digits() -> None:
    buf = io.StringIO()
    logger = build_logger("test_phone", stream=buf)
    logger.info("dial", extra={"to_phone": "+15125550147", "event": "dial"})
    payload = json.loads(buf.getvalue().strip())
    assert payload["to_phone"] == "+1512***0147"  # digits 5-7 of the digit run masked


def test_redaction_masks_authorization_header() -> None:
    buf = io.StringIO()
    logger = build_logger("test_auth", stream=buf)
    logger.info("req", extra={"authorization": "Bearer sk-abcd1234abcd1234", "event": "req"})
    payload = json.loads(buf.getvalue().strip())
    assert payload["authorization"] == "[REDACTED]"


def test_redaction_masks_api_key_field() -> None:
    buf = io.StringIO()
    logger = build_logger("test_apikey", stream=buf)
    logger.info("init", extra={"api_key": "sk-real", "event": "init"})
    payload = json.loads(buf.getvalue().strip())
    assert payload["api_key"] == "[REDACTED]"
