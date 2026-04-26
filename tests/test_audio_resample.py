"""Tests for μ-law 8 kHz ↔ PCM16 resampling at multiple rates."""

from __future__ import annotations

import math

import numpy as np

from agent.audio.transport import (
    mulaw8k_to_pcm16,
    pcm16_to_mulaw8k,
)


def _sine(freq_hz: float, duration_s: float, sample_rate: int) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (np.sin(2 * math.pi * freq_hz * t) * 0.6 * 32767).astype(np.int16)
    return samples.tobytes()


def _rms(b: bytes) -> float:
    return float(
        np.sqrt(np.mean((np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768) ** 2))
    )


def test_round_trip_24khz_preserves_energy() -> None:
    pcm = _sine(440.0, 1.0, 24_000)
    mu = pcm16_to_mulaw8k(pcm, src_rate=24_000)
    pcm_back = mulaw8k_to_pcm16(mu, dst_rate=24_000)
    assert abs(_rms(pcm_back) - _rms(pcm)) / _rms(pcm) < 0.20


def test_round_trip_16khz_preserves_energy() -> None:
    pcm = _sine(440.0, 1.0, 16_000)
    mu = pcm16_to_mulaw8k(pcm, src_rate=16_000)
    pcm_back = mulaw8k_to_pcm16(mu, dst_rate=16_000)
    assert abs(_rms(pcm_back) - _rms(pcm)) / _rms(pcm) < 0.20


def test_8khz_to_16khz_doubles_length() -> None:
    import audioop
    pcm_8k = _sine(440.0, 0.1, 8_000)
    mu = audioop.lin2ulaw(pcm_8k, 2)
    pcm_16k = mulaw8k_to_pcm16(mu, dst_rate=16_000)
    assert len(pcm_16k) == 800 * 2 * 2  # 2x rate, 2 bytes/sample


def test_24khz_to_8khz_one_third_length() -> None:
    pcm = _sine(440.0, 0.1, 24_000)
    mu = pcm16_to_mulaw8k(pcm, src_rate=24_000)
    assert len(mu) == 800
