"""Tests for the destination allowlist precheck."""

from __future__ import annotations

import pytest

from agent.safety import DestinationNotAllowedError, assert_destination_allowed


def test_allowed_number_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147,+15125550148")
    assert_destination_allowed("+15125550147")


def test_blocked_number_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147")
    with pytest.raises(DestinationNotAllowedError):
        assert_destination_allowed("+19999999999")


def test_empty_env_blocks_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "")
    with pytest.raises(DestinationNotAllowedError):
        assert_destination_allowed("+15125550147")


def test_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", " +15125550147 , +15125550148 ")
    assert_destination_allowed("+15125550147")
