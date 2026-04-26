"""Build and run the Pipecat pipeline for one call.

Phase 1 (this commit): transport-only echo. Phases 2 and 4 add STT, TTS,
StateMachineProcessor, and DTMFProcessor between input and output.

`PipelineRunner.run(task)` returns when the WebSocket closes (Twilio's
`stop` event). We wrap in try/finally so any exception path still cancels
the task — leaves no zombie subprocesses.

`call_sid` is plumbed from Twilio's `start` event (the first WS message)
into TwilioFrameSerializer; in Phases 4+ it also becomes the LangGraph
`thread_id` for per-call state isolation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, WebSocket


async def run_pipeline_for_call(websocket: WebSocket, app: FastAPI) -> None:
    """Run the Pipecat pipeline for one call. Returns when call hangs up.

    Reads `app.state.twilio_account_sid`, `app.state.twilio_auth_token` set by
    the CLI before serving. The order is read from `app.state.order` (used by
    Phase 4's StateMachineProcessor; not needed in Phase 1's echo).
    """
    await websocket.accept()

    # Pump messages until the `start` event arrives — that's when we have
    # stream_sid + call_sid.
    stream_sid: str | None = None
    call_sid: str | None = None
    while stream_sid is None:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=app.state.twilio_account_sid,
        auth_token=app.state.twilio_auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # Phase 1: echo only — input → output. Phases 2 and 4 insert STT,
    # StateMachineProcessor, DTMFProcessor, TTS between these.
    pipeline = Pipeline([transport.input(), transport.output()])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        await task.cancel()
