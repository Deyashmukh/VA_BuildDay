"""Safety prechecks that gate any outbound action."""

from __future__ import annotations

import os


class DestinationNotAllowedError(RuntimeError):
    """Raised when a dial target is not on the ALLOWED_DESTINATIONS env allowlist."""


def assert_destination_allowed(to_number: str) -> None:
    """Raise if `to_number` is not in ALLOWED_DESTINATIONS.

    Comma-separated list of E.164 numbers; whitespace is stripped. An unset
    or empty env var blocks every number — fail closed.
    """
    raw = os.environ.get("ALLOWED_DESTINATIONS", "")
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    if to_number not in allowed:
        raise DestinationNotAllowedError(
            f"{to_number!r} is not in ALLOWED_DESTINATIONS "
            f"(allowlist size: {len(allowed)})"
        )
