"""Tests for the FastAPI app skeleton."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.server import build_app


def test_health_endpoint() -> None:
    app = build_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_voice_endpoint_returns_twiml_with_connect_stream() -> None:
    app = build_app(public_url="https://example.ngrok.io")
    client = TestClient(app)
    resp = client.post("/voice", data={})
    assert resp.status_code == 200
    body = resp.text
    assert "<Connect>" in body
    assert "<Stream" in body
    assert 'url="wss://example.ngrok.io/stream"' in body


def test_voice_endpoint_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When build_app() called without public_url, /voice should use PUBLIC_URL env."""
    monkeypatch.setenv("PUBLIC_URL", "https://from-env.ngrok.io")
    app = build_app()
    client = TestClient(app)
    resp = client.post("/voice", data={})
    assert "wss://from-env.ngrok.io/stream" in resp.text
