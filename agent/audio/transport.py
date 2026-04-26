"""Audio transport: convert between Twilio μ-law 8 kHz and arbitrary PCM16 rates.

Twilio is μ-law 8 kHz. Silero VAD wants PCM16 16 kHz. Cartesia outputs PCM16
(typically 24 kHz). Wrong rates produce chipmunk audio that reads as a bot.
Use `audioop` + `scipy.signal.resample_poly` and round-trip-test.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd
from typing import cast

import audioop
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly  # type: ignore[import-untyped]


def _resample(samples: NDArray[np.float32], up: int, down: int) -> NDArray[np.float32]:
    """Polyphase resample wrapped to give pyright a typed return."""
    return cast(NDArray[np.float32], resample_poly(samples, up=up, down=down))


def mulaw8k_to_pcm16(mulaw: bytes, *, dst_rate: int) -> bytes:
    """Convert μ-law 8 kHz mono to PCM16 little-endian mono at `dst_rate`."""
    pcm_8k = audioop.ulaw2lin(mulaw, 2)
    if dst_rate == 8_000:
        return pcm_8k
    samples = np.frombuffer(pcm_8k, dtype=np.int16).astype(np.float32)
    up, down = _ratio(dst_rate, 8_000)
    out = _resample(samples, up, down).astype(np.int16)
    return out.tobytes()


def pcm16_to_mulaw8k(pcm: bytes, *, src_rate: int) -> bytes:
    """Convert PCM16 little-endian mono at `src_rate` to μ-law 8 kHz mono."""
    if src_rate == 8_000:
        return audioop.lin2ulaw(pcm, 2)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    up, down = _ratio(8_000, src_rate)
    out = _resample(samples, up, down).astype(np.int16)
    return audioop.lin2ulaw(out.tobytes(), 2)


@lru_cache(maxsize=8)
def _ratio(target: int, source: int) -> tuple[int, int]:
    """Reduce target/source to coprime up/down for resample_poly."""
    g = gcd(target, source)
    return target // g, source // g
