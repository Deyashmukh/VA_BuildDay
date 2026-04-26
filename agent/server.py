"""FastAPI app: /start, /voice (Twilio webhook), /stream (Pipecat WS), /health.

The CLI is responsible for setting `app.state.pipeline_runner` to a callable
that runs one Pipecat pipeline per WebSocket connection BEFORE serving. This
keeps the server module decoupled from the (yet-to-exist) pipeline module
and lets pyright stay happy without lazy-import gymnastics.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

PipelineRunner = Callable[[WebSocket, FastAPI], Awaitable[None]]


def build_app(public_url: str | None = None) -> FastAPI:
    """Construct the FastAPI app.

    `public_url` is the externally reachable base (e.g. ngrok URL). If unset,
    falls back to env var `PUBLIC_URL`, then to a localhost placeholder that
    fails when Twilio fetches it (fine for unit tests; set explicitly in CLI).
    """
    app = FastAPI()
    base = public_url or os.environ.get("PUBLIC_URL", "wss://localhost:8000")

    @app.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.post("/voice")
    async def voice(_: Request) -> PlainTextResponse:  # pyright: ignore[reportUnusedFunction]
        ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/stream"
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Connect><Stream url="{ws_url}"/></Connect>'
            "</Response>"
        )
        return PlainTextResponse(twiml, media_type="application/xml")

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        runner: PipelineRunner | None = getattr(app.state, "pipeline_runner", None)
        if runner is None:
            await websocket.close(code=1011, reason="pipeline_runner not configured")
            return
        await runner(websocket, app)

    return app
