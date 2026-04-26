"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _env_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a known-safe ALLOWED_DESTINATIONS during tests."""
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15555550100")
