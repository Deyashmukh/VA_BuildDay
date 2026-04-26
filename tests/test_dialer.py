"""Tests for the outbound dialer (Twilio SDK is mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.runtime.dialer import place_call
from agent.safety import DestinationNotAllowedError


def test_place_call_blocks_unallowed_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147")
    client = MagicMock()
    with pytest.raises(DestinationNotAllowedError):
        place_call(client, to="+19999999999", from_="+15555550100", voice_url="https://x/voice")
    client.calls.create.assert_not_called()


def test_place_call_invokes_twilio_with_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147")
    client = MagicMock()
    client.calls.create.return_value = MagicMock(sid="CA123")
    sid = place_call(
        client, to="+15125550147", from_="+15555550100", voice_url="https://x/voice"
    )
    assert sid == "CA123"
    client.calls.create.assert_called_once_with(
        to="+15125550147",
        from_="+15555550100",
        url="https://x/voice",
        record=False,
    )
