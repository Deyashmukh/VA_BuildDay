"""Tests for project-local Pipecat frames."""

from __future__ import annotations

from pipecat.frames.frames import SystemFrame

from agent.pipeline.frames import RestDTMFFrame


def test_rest_dtmf_frame_default_digits() -> None:
    f = RestDTMFFrame()
    assert f.digits == ""


def test_rest_dtmf_frame_carries_digits() -> None:
    f = RestDTMFFrame(digits="1234")
    assert f.digits == "1234"


def test_rest_dtmf_frame_is_a_system_frame() -> None:
    """SystemFrames are routed differently than data frames — verify subclass relationship."""
    f = RestDTMFFrame(digits="5")
    assert isinstance(f, SystemFrame)
