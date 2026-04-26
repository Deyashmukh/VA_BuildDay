"""Twilio outbound dialer with hard ALLOWED_DESTINATIONS precheck."""

from __future__ import annotations

from twilio.rest import Client  # pyright: ignore[reportMissingTypeStubs]

from agent.safety import assert_destination_allowed


def place_call(client: Client, *, to: str, from_: str, voice_url: str) -> str:
    """Place an outbound call. Returns the Twilio call SID.

    The destination is hard-checked against ALLOWED_DESTINATIONS BEFORE any
    network call; the SDK is never invoked for a blocked number.
    """
    assert_destination_allowed(to)
    call = client.calls.create(to=to, from_=from_, url=voice_url, record=False)
    return str(call.sid)
