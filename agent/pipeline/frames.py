"""Project-local Pipecat frames."""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


@dataclass
class RestDTMFFrame(SystemFrame):
    """Send DTMF digits via Twilio REST sidechannel — not via audio synthesis.

    Consumed by `DTMFProcessor` (Phase 2) which calls Twilio's `<Play digits>`
    update API. Never confuse with `pipecat.frames.frames.OutputDTMFFrame`,
    which on FastAPIWebsocketTransport synthesizes tones into audio (see
    CLAUDE.md foot-guns).
    """

    digits: str = ""
