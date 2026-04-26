"""Smoke test: Pipecat is installed with the right extras and primitives import."""

from __future__ import annotations


def test_pipecat_imports_pipeline_primitives() -> None:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    assert Pipeline is not None
    assert PipelineRunner is not None
    assert PipelineTask is not None
    assert FrameProcessor is not None
    assert FrameDirection is not None


def test_pipecat_twilio_serializer_imports() -> None:
    from pipecat.serializers.twilio import TwilioFrameSerializer
    assert TwilioFrameSerializer is not None


def test_pipecat_fastapi_websocket_transport_imports() -> None:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
    assert FastAPIWebsocketTransport is not None


def test_pipecat_deepgram_service_imports() -> None:
    from pipecat.services.deepgram.stt import DeepgramSTTService
    assert DeepgramSTTService is not None


def test_pipecat_cartesia_service_imports() -> None:
    from pipecat.services.cartesia.tts import CartesiaTTSService
    assert CartesiaTTSService is not None


def test_pipecat_silero_vad_imports() -> None:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    assert SileroVADAnalyzer is not None
