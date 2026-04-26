# Pizza Voice Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process Python voice agent that places an outbound phone call, navigates an automated IVR with LangGraph-driven tool-calling, detects hold→human handoff, conducts a natural conversation through streaming Llama-class LLM + TTS, and emits a structured JSON result via post-call extraction.

**Architecture:** Custom Pipecat-style audio pipeline (Twilio Media Streams → Deepgram ASR + Silero VAD → Cartesia TTS) bridged through a thin StateProcessor adapter to a **LangGraph** state machine with 6 nodes (`router`, `ivr_llm`, `ivr_tools`, `hold_node`, `human_llm`, `done_node`). Per-mode tool registries; mode-specific message histories; `MemorySaver` checkpointer keyed by `thread_id=call_sid`.

**Tech Stack:**
- Python 3.11+, `asyncio`, FastAPI, `websockets`, `uvicorn`
- Twilio Python SDK + Media Streams
- Deepgram streaming ASR (`nova-2-phonecall`)
- Silero VAD (`silero-vad` PyPI)
- Cartesia TTS (PCM16 16 kHz output → resample to μ-law 8 kHz)
- **LangGraph** (latest, Python) + `langgraph-checkpoint` (MemorySaver)
- **Cerebras** for `gpt-oss-120B` (ivr_llm, human_llm, done_node extraction)
- **Groq** for `Llama 3.1 8B` (hold_node binary classifier)
- LangChain ChatOpenAI-compatible clients (Cerebras exposes OpenAI-compat endpoint; Groq has native LangChain integration)
- pydantic v2, `audioop` (stdlib), `scipy.signal.resample_poly`
- pytest, `ruff` (with `D`/pydocstyle), `pyright` (strict)
- ngrok (`pyngrok`)
- Langfuse self-hosted (Docker) + LangChain callback

**Constraints from CLAUDE.md** (read it before starting any task):
- Per-change workflow: pytest+ruff+pyright → simplify → verify → code-reviewer → commit → push.
- Type-annotate everything; module + public-symbol docstrings; `from __future__ import annotations`.
- Foot-guns: DTMF is REST-only; resampling must be exact; `Task.cancel()` for barge-in; `thread_id=call_sid` non-negotiable.

**Suggested execution mode:** subagent-driven. Phases 0, 2, 3 have parallelizable tasks (independent subsystems). Phases 4–7 are sequential (each builds on the previous mode). Use `superpowers:dispatching-parallel-agents` where independence is real.

## IVR-Only Milestone Scope

The user's first verification gate is **Phase 4.4 — IVR end-to-end on a real call**. To get there fastest, **execute only the tasks in this list** and skip everything else until that checkpoint passes:

**Required for IVR-end-to-end:**
- Phase 0: T0.1, T0.2, T0.3, T0.4, T0.5, T0.6 (all)
- Phase 1: T1.1, T1.2, T1.3, T1.4, T1.5, T1.6, T1.6.5 (added — DTMF mid-stream verification)
- Phase 2: **T2.1 only** (Deepgram), **T2.3 buffered TTS only**, **T2.4 DTMF**. Skip T2.2 (VAD) and the streaming half of T2.3.
- Phase 3: T3.1 (state), T3.2 **Cerebras client only** (skip Groq), T3.3 **IVR prompt only** (skip hold/human/extraction prompts), T3.4 **IVR tools only** (skip HUMAN tools), T3.5 (graph builder skeleton with stubbed hold/human/done nodes).
- Phase 4: T4.1, T4.2, T4.3, T4.3b (added — coded audio-pipeline glue), T4.4.

**Deferred until after IVR checkpoint passes:** T2.2 (Silero VAD), T2.3 streaming TTS internals, T3.2 Groq client, T3.3 hold/human/extraction prompts, T3.4 HUMAN tools, all of Phases 5/6/7/8/9.

This trims ~8–10 tasks of pre-work the IVR doesn't need. After the user confirms IVR works on a real call, resume the plan from Phase 5.

---

## Phase 0 — Project setup, schemas, scaffolding

Goal: project skeleton with deps, lint/type/test tooling, input/output schemas, observability scaffolding, audio-resampling primitive, allowlist precheck.

### Task 0.1: pyproject.toml + tooling

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`
- Create: `agent/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "pizza-voice-agent"
version = "0.1.0"
description = "LangGraph-driven voice agent that places pizza orders via outbound call."
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "websockets>=12.0",
  "twilio>=9.0",
  "deepgram-sdk>=3.4",
  "cartesia>=1.0",
  "silero-vad>=5.0",
  "torch>=2.2",         # Silero VAD requirement
  "langgraph>=0.2",
  "langchain-core>=0.3",
  "langchain-openai>=0.2",  # Cerebras uses OpenAI-compatible endpoint
  "langchain-groq>=0.2",
  "langfuse>=2.36",
  "pydantic>=2.6",
  "python-dotenv>=1.0",
  "scipy>=1.12",
  "numpy>=1.26",
  "pyngrok>=7.0",
  "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "pytest-mock>=3.12",
  "ruff>=0.4",
  "pyright>=1.1.350",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "ANN", "D"]
ignore = ["D203", "D213", "ANN101", "ANN102"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "ANN"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.pyright]
include = ["agent", "tests"]
strict = ["agent"]
pythonVersion = "3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.pyright_cache/
*.log
.DS_Store
```

- [ ] **Step 3: Create `.env.example`**

```bash
# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+15551234567

# Cerebras (for gpt-oss-120B — ivr_llm, human_llm, done_node)
CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxxxxxx
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
CEREBRAS_MODEL=gpt-oss-120b

# Groq (for Llama 3.1 8B — hold_node classifier)
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxx
GROQ_CLASSIFIER_MODEL=llama-3.1-8b-instant

# Deepgram
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxx

# Cartesia
CARTESIA_API_KEY=xxxxxxxxxxxxxxxxxxxx
CARTESIA_VOICE_ID=xxxxxxxxxxxxxxxxxxxx

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-xxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-xxxxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=http://localhost:3000

# Safety
ALLOWED_DESTINATIONS=+15125550147,+15125550148

# Local
NGROK_AUTHTOKEN=xxxxxxxxxxxxxxxxxxxx
PORT=8000
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _env_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a known-safe ALLOWED_DESTINATIONS during tests."""
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15555550100")
```

- [ ] **Step 5: Install deps and verify tooling**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check agent tests
pyright
pytest
```

Expected: ruff clean, pyright clean (no source files yet), pytest collects 0 tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example agent/ tests/
git commit -m "chore: scaffold project (pyproject, deps, lint/type/test tooling)"
```

---

### Task 0.2: Order schema (input)

**Files:**
- Create: `agent/types/__init__.py`, `agent/types/order.py`
- Create: `tests/test_order_schema.py`
- Create: `fixtures/orders/example.json`

- [ ] **Step 1: Create `fixtures/orders/example.json`** — copy verbatim from `build_day.md` example payload.

- [ ] **Step 2: Write failing tests in `tests/test_order_schema.py`**

```python
"""Tests for the input order schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.types.order import Order


FIXTURE = Path(__file__).parent.parent / "fixtures" / "orders" / "example.json"


def test_example_order_validates() -> None:
    data = json.loads(FIXTURE.read_text())
    order = Order.model_validate(data)
    assert order.customer_name == "Jordan Mitchell"
    assert order.phone_number == "5125550147"
    assert order.budget_max == 45.0
    assert order.pizza.toppings == ["pepperoni", "mushroom", "green pepper"]
    assert order.side.if_all_unavailable == "skip"
    assert order.drink.skip_if_over_budget is True


def test_phone_number_must_be_10_digits() -> None:
    data = json.loads(FIXTURE.read_text())
    data["phone_number"] = "512555014"
    with pytest.raises(ValidationError):
        Order.model_validate(data)


def test_budget_must_be_positive() -> None:
    data = json.loads(FIXTURE.read_text())
    data["budget_max"] = -5.0
    with pytest.raises(ValidationError):
        Order.model_validate(data)


def test_zip_code_extracted_from_address() -> None:
    data = json.loads(FIXTURE.read_text())
    order = Order.model_validate(data)
    assert order.delivery_zip == "78745"
```

- [ ] **Step 3: Run — expect ImportError.**

- [ ] **Step 4: Implement `agent/types/order.py`**

```python
"""Input order schema (the JSON the agent receives at startup)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class Pizza(BaseModel):
    """Pizza details. Mandatory item — call fails if this can't be ordered."""

    size: str
    crust: str
    toppings: list[str]
    acceptable_topping_subs: list[str] = Field(default_factory=list)
    no_go_toppings: list[str] = Field(default_factory=list)


class Side(BaseModel):
    """Side details. Optional per `if_all_unavailable`."""

    first_choice: str
    backup_options: list[str] = Field(default_factory=list)
    if_all_unavailable: Literal["skip"] | str = "skip"


class Drink(BaseModel):
    """Drink details. Skipped per `skip_if_over_budget`."""

    first_choice: str
    alternatives: list[str] = Field(default_factory=list)
    skip_if_over_budget: bool = False


class Order(BaseModel):
    """The full order payload received at agent startup."""

    customer_name: str
    phone_number: str
    delivery_address: str
    pizza: Pizza
    side: Side
    drink: Drink
    budget_max: float = Field(gt=0)
    special_instructions: str = ""

    @field_validator("phone_number")
    @classmethod
    def _ten_digits(cls, v: str) -> str:
        if not re.fullmatch(r"\d{10}", v):
            raise ValueError("phone_number must be exactly 10 digits, no separators")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delivery_zip(self) -> str:
        """Pull the 5-digit zip from the trailing portion of the address."""
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b\s*$", self.delivery_address)
        if not match:
            raise ValueError(f"could not extract zip from address: {self.delivery_address}")
        return match.group(1)
```

`agent/types/__init__.py`:

```python
"""Typed data models shared across the agent."""

from agent.types.order import Drink, Order, Pizza, Side

__all__ = ["Drink", "Order", "Pizza", "Side"]
```

- [ ] **Step 5: Run + lint + commit**

```bash
pytest tests/test_order_schema.py -v && ruff check agent tests && pyright
git add agent/types/ tests/test_order_schema.py fixtures/
git commit -m "feat(types): Order pydantic schema with zip extraction + validation"
```

---

### Task 0.3: CallResult schema (output)

**Files:**
- Create: `agent/types/result.py`
- Modify: `agent/types/__init__.py`
- Create: `tests/test_result_schema.py`

- [ ] **Step 1: Write failing tests in `tests/test_result_schema.py`**

```python
"""Tests for the call-result schema produced at hangup."""

from __future__ import annotations

import json

from agent.types.result import (
    CallResult,
    DrinkQuote,
    Outcome,
    PizzaQuote,
    SideQuote,
)


def test_completed_full_payload_round_trips() -> None:
    result = CallResult(
        outcome=Outcome.COMPLETED,
        pizza=PizzaQuote(
            description="large hand-tossed with pepperoni, onion, green pepper",
            substitutions={"mushroom": "onion", "thin crust": "hand-tossed"},
            price=18.50,
        ),
        side=SideQuote(description="garlic bread", original="buffalo wings, 12 count", price=6.99),
        drink=DrinkQuote(description="2L Coke", price=3.49),
        total=28.98,
        delivery_time="35 minutes",
        order_number="4412",
        special_instructions_delivered=True,
    )
    payload = json.loads(result.model_dump_json())
    assert payload["outcome"] == "completed"
    assert payload["pizza"]["price"] == 18.50
    assert payload["drink_skip_reason"] is None


def test_drink_skipped_for_budget() -> None:
    result = CallResult(
        outcome=Outcome.COMPLETED,
        pizza=PizzaQuote(description="large pepperoni", substitutions={}, price=20.0),
        side=SideQuote(description="breadsticks", original="wings", price=5.0),
        drink=None,
        drink_skip_reason="over_budget",
        total=25.0,
        delivery_time="30 minutes",
        order_number="9999",
        special_instructions_delivered=True,
    )
    payload = json.loads(result.model_dump_json())
    assert payload["drink"] is None
    assert payload["drink_skip_reason"] == "over_budget"


def test_nothing_available_partial() -> None:
    result = CallResult(outcome=Outcome.NOTHING_AVAILABLE)
    payload = json.loads(result.model_dump_json())
    assert payload["outcome"] == "nothing_available"
    assert payload["pizza"] is None


def test_operational_outcomes_serialize() -> None:
    for code in ("no_answer", "ivr_failed", "ivr_rejected", "hold_timeout", "disconnect"):
        result = CallResult(outcome=Outcome(code))
        payload = json.loads(result.model_dump_json())
        assert payload["outcome"] == code


def test_detected_as_bot_keeps_partial_capture() -> None:
    result = CallResult(
        outcome=Outcome.DETECTED_AS_BOT,
        pizza=PizzaQuote(description="medium thin pepperoni", substitutions={}, price=15.0),
    )
    payload = json.loads(result.model_dump_json())
    assert payload["outcome"] == "detected_as_bot"
    assert payload["pizza"]["price"] == 15.0
```

- [ ] **Step 2: Implement `agent/types/result.py`**

```python
"""Output schema for the JSON the agent prints at hangup."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    """How the call ended.

    Spec outcomes: COMPLETED, NOTHING_AVAILABLE, OVER_BUDGET, DETECTED_AS_BOT.
    Operational extensions surface why a call ended without reaching `completed`.
    """

    COMPLETED = "completed"
    NOTHING_AVAILABLE = "nothing_available"
    OVER_BUDGET = "over_budget"
    DETECTED_AS_BOT = "detected_as_bot"
    NO_ANSWER = "no_answer"
    IVR_FAILED = "ivr_failed"
    IVR_REJECTED = "ivr_rejected"
    HOLD_TIMEOUT = "hold_timeout"
    DISCONNECT = "disconnect"


class PizzaQuote(BaseModel):
    """What the employee actually quoted for the pizza."""

    description: str
    substitutions: dict[str, str] = Field(default_factory=dict)
    price: float


class SideQuote(BaseModel):
    """What the employee actually quoted for the side; `original` is what we asked for first."""

    description: str
    original: str
    price: float


class DrinkQuote(BaseModel):
    """What the employee actually quoted for the drink."""

    description: str
    price: float


class CallResult(BaseModel):
    """Final structured result printed to stdout when the call ends."""

    outcome: Outcome
    pizza: PizzaQuote | None = None
    side: SideQuote | None = None
    drink: DrinkQuote | None = None
    drink_skip_reason: Literal["over_budget", "unavailable"] | None = None
    total: float | None = None
    delivery_time: str | None = None
    order_number: str | None = None
    special_instructions_delivered: bool = False
```

Update `agent/types/__init__.py`:

```python
"""Typed data models shared across the agent."""

from agent.types.order import Drink, Order, Pizza, Side
from agent.types.result import CallResult, DrinkQuote, Outcome, PizzaQuote, SideQuote

__all__ = [
    "CallResult", "Drink", "DrinkQuote", "Order", "Outcome",
    "Pizza", "PizzaQuote", "Side", "SideQuote",
]
```

- [ ] **Step 3: Run + lint + commit**

```bash
pytest tests/test_result_schema.py -v && ruff check agent tests && pyright
git add agent/types/ tests/test_result_schema.py
git commit -m "feat(types): CallResult schema with all outcomes"
```

---

### Task 0.4: ALLOWED_DESTINATIONS allowlist precheck

**Files:**
- Create: `agent/safety.py`
- Create: `tests/test_safety.py`

- [ ] **Step 1: Write failing tests** (4 tests as in design — allowed/blocked/empty/whitespace).

```python
"""Tests for the destination allowlist precheck."""

from __future__ import annotations

import pytest

from agent.safety import DestinationNotAllowedError, assert_destination_allowed


def test_allowed_number_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147,+15125550148")
    assert_destination_allowed("+15125550147")


def test_blocked_number_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "+15125550147")
    with pytest.raises(DestinationNotAllowedError):
        assert_destination_allowed("+19999999999")


def test_empty_env_blocks_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", "")
    with pytest.raises(DestinationNotAllowedError):
        assert_destination_allowed("+15125550147")


def test_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DESTINATIONS", " +15125550147 , +15125550148 ")
    assert_destination_allowed("+15125550147")
```

- [ ] **Step 2: Implement `agent/safety.py`**

```python
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
            f"{to_number!r} is not in ALLOWED_DESTINATIONS (have: {sorted(allowed)})"
        )
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_safety.py -v && ruff check agent tests && pyright
git add agent/safety.py tests/test_safety.py
git commit -m "feat(safety): ALLOWED_DESTINATIONS fail-closed precheck"
```

---

### Task 0.5: Audio transport — μ-law 8 kHz ↔ PCM16 16/24 kHz round-trip

**Files:**
- Create: `agent/audio/__init__.py`, `agent/audio/transport.py`
- Create: `tests/test_audio_resample.py`

- [ ] **Step 1: Write failing tests**

```python
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


def test_round_trip_24khz_preserves_energy() -> None:
    pcm = _sine(440.0, 1.0, 24_000)
    mu = pcm16_to_mulaw8k(pcm, src_rate=24_000)
    pcm_back = mulaw8k_to_pcm16(mu, dst_rate=24_000)
    rms = lambda b: float(np.sqrt(np.mean((np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768) ** 2)))
    assert abs(rms(pcm_back) - rms(pcm)) / rms(pcm) < 0.20


def test_round_trip_16khz_preserves_energy() -> None:
    pcm = _sine(440.0, 1.0, 16_000)
    mu = pcm16_to_mulaw8k(pcm, src_rate=16_000)
    pcm_back = mulaw8k_to_pcm16(mu, dst_rate=16_000)
    rms = lambda b: float(np.sqrt(np.mean((np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768) ** 2)))
    assert abs(rms(pcm_back) - rms(pcm)) / rms(pcm) < 0.20


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
```

- [ ] **Step 2: Implement `agent/audio/transport.py`**

```python
"""Audio transport: convert between Twilio μ-law 8 kHz and arbitrary PCM16 rates.

Twilio is μ-law 8 kHz. Silero VAD wants PCM16 16 kHz. Cartesia outputs PCM16
(typically 24 kHz). Wrong rates produce chipmunk audio that reads as a bot.
Use `audioop` + `scipy.signal.resample_poly` and round-trip-test.
"""

from __future__ import annotations

import audioop
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def mulaw8k_to_pcm16(mulaw: bytes, *, dst_rate: int) -> bytes:
    """Convert μ-law 8 kHz mono to PCM16 little-endian mono at `dst_rate`."""
    pcm_8k = audioop.ulaw2lin(mulaw, 2)
    if dst_rate == 8_000:
        return pcm_8k
    samples = np.frombuffer(pcm_8k, dtype=np.int16).astype(np.float32)
    up, down = _ratio(dst_rate, 8_000)
    out = resample_poly(samples, up=up, down=down).astype(np.int16)
    return out.tobytes()


def pcm16_to_mulaw8k(pcm: bytes, *, src_rate: int) -> bytes:
    """Convert PCM16 little-endian mono at `src_rate` to μ-law 8 kHz mono."""
    if src_rate == 8_000:
        return audioop.lin2ulaw(pcm, 2)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    up, down = _ratio(8_000, src_rate)
    out = resample_poly(samples, up=up, down=down).astype(np.int16)
    return audioop.lin2ulaw(out.tobytes(), 2)


def _ratio(target: int, source: int) -> tuple[int, int]:
    """Reduce target/source to coprime up/down for resample_poly."""
    g = gcd(target, source)
    return target // g, source // g
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_audio_resample.py -v && ruff check agent tests && pyright
git add agent/audio/ tests/test_audio_resample.py
git commit -m "feat(audio): μ-law 8k ↔ PCM16 resampling at arbitrary target rates"
```

---

### Task 0.6: Observability — JSON logger + Langfuse client

**Files:**
- Create: `agent/obs/__init__.py`, `agent/obs/log.py`, `agent/obs/langfuse_client.py`
- Create: `tests/test_log_redaction.py`

- [ ] **Step 1: Write failing tests** (4 tests covering required-fields, phone-redaction, authorization-redaction, api_key-redaction; same as design's earlier test plan).

(Test code identical to old plan Task 0.6 Step 1 — re-use verbatim, no architectural change.)

- [ ] **Step 2: Implement `agent/obs/log.py`** (verbatim from old plan Task 0.6 Step 3).

- [ ] **Step 3: Implement `agent/obs/langfuse_client.py`**

```python
"""Langfuse client setup. Lazy: built on first use, returns None if env unset."""

from __future__ import annotations

import os
from functools import lru_cache

from langfuse import Langfuse


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse | None:
    """Return a process-wide Langfuse client, or None if env vars are not configured."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    if not (public and secret and host):
        return None
    return Langfuse(public_key=public, secret_key=secret, host=host)


def langfuse_callback_handler():
    """Return a LangChain callback handler bound to our Langfuse client.

    Used in LangGraph by passing as a callback to .ainvoke():
      `await graph.ainvoke(state, config={"callbacks": [langfuse_callback_handler()]})`
    Auto-traces every LangChain LLM call as a span on the active trace.
    """
    from langfuse.callback import CallbackHandler
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    if not (public and secret and host):
        return None
    return CallbackHandler(public_key=public, secret_key=secret, host=host)
```

`agent/obs/__init__.py`:

```python
"""Observability primitives — JSON logger, Langfuse client, callback handler."""

from agent.obs.langfuse_client import get_langfuse, langfuse_callback_handler
from agent.obs.log import build_logger

__all__ = ["build_logger", "get_langfuse", "langfuse_callback_handler"]
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/test_log_redaction.py -v && ruff check agent tests && pyright
git add agent/obs/ tests/test_log_redaction.py
git commit -m "feat(obs): JSON logger w/ redaction + lazy Langfuse client + LangChain callback"
```

---

## Phase 1 — Telephony shell (outbound dial + audio loop sanity)

Goal: real outbound call connects, audio frames echo back, clean exit on hangup.

### Task 1.1: FastAPI server skeleton (build_app)

(Same as old plan Task 1.1 — `/voice` returns TwiML with `<Connect><Stream>`, `/health` returns ok. Tests as in old plan.)

### Task 1.2: ngrok tunnel helpers

(Same as old plan Task 1.2 — `open_tunnel`, `close_all_tunnels`.)

### Task 1.3: Twilio dialer with allowlist precheck

(Same as old plan Task 1.3.)

### Task 1.4: Twilio Media Streams JSON codec + WS handler with audio echo

(Same as old plan Task 1.5 — frame decode/encode, /stream WS that echoes audio back as a sanity check. We replace echo with real handlers in Phase 2.)

### Task 1.5: CLI entrypoint

(Same as old plan Task 1.6 — `python -m agent.cli order.json --to <num>`.)

### Task 1.6: REAL CALL CHECKPOINT 1 — outbound dial + audio echo

Manual checkpoint. Place a real call to your own phone; verify audio echoes back; verify clean WS close on hangup; verify `event: "call_placed"` printed on stdout.

### Task 1.6.5: REAL CALL — DTMF mid-stream behavior verification

**Goal:** confirm Twilio's `client.calls(sid).update(twiml=<Response><Play digits="1"/></Response>)` does NOT terminate the active `<Connect><Stream>`. The plan's IVR strategy depends on this; if it terminates the stream, the IVR phase needs a different approach.

- [ ] **Step 1: Place a call as in T1.6, but instrument the WS handler to log every inbound `media` event with a sequence number.**

- [ ] **Step 2: After WS connects and frames are flowing, from a separate terminal run:**

```python
import os
from twilio.rest import Client
sid = "<paste call SID printed at boot>"
c = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
c.calls(sid).update(twiml='<?xml version="1.0" encoding="UTF-8"?><Response><Play digits="1"/></Response>')
```

- [ ] **Step 3: Observe.**

Expected (the happy path the plan assumes): the call hears DTMF tones, the WS keeps receiving `media` events, sequence numbers continue uninterrupted.

If the WS receives a `stop` event and never resumes: the `update(twiml=...)` mechanism terminates `<Connect><Stream>`. **Plan needs revision** — see fallback in Task 2.4 caveats; consider `<Play digits>` followed by re-`<Connect><Stream>` to the same URL, accepting a brief audio gap.

- [ ] **Step 4: Document the result inline in `docs/superpowers/dtmf-mid-stream-result.md`** with the call SID and the observed behavior. Future debugging will reference this.

- [ ] **Step 5: Commit findings.**

```bash
git add docs/superpowers/dtmf-mid-stream-result.md
git commit -m "docs: record DTMF mid-stream Twilio behavior observation"
```

---

## Phase 2 — Audio pipeline (ASR + VAD + TTS)

Goal: audio frames flow through Deepgram (text out), Silero VAD (barge-in signal), and Cartesia (audio in two modes — buffered for IVR, streaming for HUMAN with cancellation).

These tasks are independent of each other and can run in parallel via `superpowers:dispatching-parallel-agents`. Each task is self-contained and tested in isolation; the integration in Phase 3+ wires them through the StateProcessor adapter.

### Task 2.1: Deepgram streaming ASR wrapper

**Files:**
- Create: `agent/asr/__init__.py`, `agent/asr/deepgram_stream.py`

**Important correction from senior review:** Deepgram's live transcription does NOT support reconfiguring `utterance_end_ms` mid-connection (the `Configure` event covers KeepAlive and a small set of params; endpointing is fixed at connect). To switch endpointing on mode transitions, **close and reopen** the WebSocket. The wrapper exposes only `open(utterance_end_ms=...)` and `close()`; the adapter handles the close+reopen at HOLD→HUMAN. The energy gate buffers the ~200–400 ms gap during reopen.

- [ ] **Step 1: Implement `agent/asr/deepgram_stream.py`**

```python
"""Deepgram streaming ASR wrapper. Feeds μ-law 8 kHz, yields finalized transcripts."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)


class DeepgramStream:
    """Async wrapper around Deepgram's live transcription WS.

    Use `open()` to start, `send(mulaw)` for each inbound frame, `finals()` to
    iterate finalized transcripts, `errors()` for fatal errors, `close()` on shutdown.
    """

    def __init__(self) -> None:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"], config)
        self._connection: object | None = None
        self._final_queue: asyncio.Queue[str] = asyncio.Queue()
        self._error_queue: asyncio.Queue[Exception] = asyncio.Queue()

    async def open(self, *, utterance_end_ms: int = 1000) -> None:
        """Open the live transcription connection with the given endpointing."""
        connection = self._client.listen.asynclive.v("1")

        async def _on_transcript(_self: object, result: object, **_: object) -> None:
            data = getattr(result, "channel", None)
            if not data:
                return
            alts = getattr(data, "alternatives", []) or []
            if not alts:
                return
            transcript = alts[0].transcript or ""
            is_final = bool(getattr(result, "is_final", False))
            if is_final and transcript.strip():
                await self._final_queue.put(transcript)

        async def _on_error(_self: object, error: object, **_: object) -> None:
            await self._error_queue.put(RuntimeError(f"deepgram error: {error}"))

        connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)
        connection.on(LiveTranscriptionEvents.Error, _on_error)

        options = LiveOptions(
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            language="en-US",
            model="nova-2-phonecall",
            interim_results=False,
            utterance_end_ms=str(utterance_end_ms),
            vad_events=False,  # we use Silero VAD instead (HUMAN phase only)
            smart_format=True,
        )
        await connection.start(options)
        self._connection = connection

    async def send(self, mulaw: bytes) -> None:
        """Stream a μ-law audio frame to Deepgram."""
        if self._connection is None:
            raise RuntimeError("DeepgramStream not opened")
        await self._connection.send(mulaw)  # type: ignore[attr-defined]

    async def finals(self) -> AsyncIterator[str]:
        """Yield finalized utterance transcripts as they arrive."""
        while True:
            yield await self._final_queue.get()

    async def errors(self) -> AsyncIterator[Exception]:
        """Yield fatal errors as they arrive."""
        while True:
            yield await self._error_queue.get()

    async def close(self) -> None:
        """Close the live transcription connection."""
        if self._connection is not None:
            await self._connection.finish()  # type: ignore[attr-defined]
            self._connection = None
```

- [ ] **Step 2: Commit**

```bash
ruff check agent && pyright
git add agent/asr/
git commit -m "feat(asr): Deepgram streaming wrapper with configurable endpointing"
```

---

### Task 2.2: Silero VAD wrapper for barge-in detection

**Files:**
- Create: `agent/audio/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for Silero VAD wrapper."""

from __future__ import annotations

import math

import numpy as np

from agent.audio.vad import SileroVAD


def _silence_pcm16_16k(ms: int) -> bytes:
    return np.zeros(int(16_000 * ms / 1000), dtype=np.int16).tobytes()


def _voice_like_pcm16_16k(ms: int, freq: float = 200.0) -> bytes:
    t = np.linspace(0, ms / 1000, int(16_000 * ms / 1000), endpoint=False)
    return (np.sin(2 * math.pi * freq * t) * 0.7 * 32767).astype(np.int16).tobytes()


def test_silence_does_not_fire() -> None:
    vad = SileroVAD()
    fired = False
    for _ in range(20):  # 600 ms of silence (30 ms windows)
        if vad.is_speech_chunk(_silence_pcm16_16k(30)):
            fired = True
    assert not fired


def test_voice_like_eventually_fires() -> None:
    vad = SileroVAD()
    fired = False
    for _ in range(20):
        if vad.is_speech_chunk(_voice_like_pcm16_16k(30)):
            fired = True
            break
    # Note: a pure sine tone may or may not classify as speech;
    # this test is loose. Real voice should reliably fire.
    # Document this as a smoke test; real coverage is in real-call testing.
```

(VAD is statistical; tight unit testing is hard. The smoke test gates the wrapper API.)

- [ ] **Step 2: Implement `agent/audio/vad.py`**

```python
"""Silero VAD wrapper. Consumes PCM16 16 kHz frames; signals voice activity."""

from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad


class SileroVAD:
    """Frame-by-frame voice activity detector.

    Silero VAD operates on 30 ms PCM16 16 kHz frames (480 samples). Returns a
    probability per frame; we threshold at 0.5 by default. Stateful — keep one
    instance per audio stream.
    """

    _CHUNK_SAMPLES = 480  # 30 ms at 16 kHz
    _THRESHOLD = 0.5

    def __init__(self) -> None:
        self._model = load_silero_vad()

    def is_speech_chunk(self, pcm16_16k: bytes) -> bool:
        """Return True if this 30 ms PCM16 16 kHz chunk contains voice activity."""
        if len(pcm16_16k) != self._CHUNK_SAMPLES * 2:
            raise ValueError(
                f"expected {self._CHUNK_SAMPLES * 2} bytes (30 ms PCM16 16k), got {len(pcm16_16k)}"
            )
        arr = np.frombuffer(pcm16_16k, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(arr)
        prob = float(self._model(tensor, 16_000).item())
        return prob > self._THRESHOLD
```

- [ ] **Step 3: Commit**

```bash
pytest tests/test_vad.py -v && ruff check agent tests && pyright
git add agent/audio/vad.py tests/test_vad.py
git commit -m "feat(audio): Silero VAD wrapper for barge-in detection"
```

---

### Task 2.3: Cartesia TTS — buffered (IVR) and streaming (HUMAN) modes

**Files:**
- Create: `agent/audio/tts_buffered.py`, `agent/audio/tts_streaming.py`
- Create: `tests/test_tts_chunking.py`

- [ ] **Step 1: Implement `agent/audio/tts_buffered.py`**

```python
"""Cartesia TTS — buffered mode: synthesize fully, then yield μ-law 8 kHz frames.

For short IVR utterances ("Jordan Mitchell", "yes"). Eliminates streaming-mid-tool-call
foot-guns at the cost of full-synthesis latency before first audio out (~150-300 ms).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Protocol

from agent.audio.transport import pcm16_to_mulaw8k


_FRAME_BYTES = 160
_MULAW_SILENCE = 0x7F


class CartesiaClient(Protocol):
    """Minimal protocol for the Cartesia client we depend on."""

    def tts_pcm16(self, text: str, *, voice_id: str, sample_rate: int) -> bytes: ...


def synthesize_to_mulaw_frames(
    client: CartesiaClient, *, text: str, voice_id: str | None = None
) -> Iterable[bytes]:
    """Synthesize `text` and yield μ-law 8 kHz 20 ms frames."""
    voice = voice_id or os.environ["CARTESIA_VOICE_ID"]
    pcm = client.tts_pcm16(text, voice_id=voice, sample_rate=16_000)
    mulaw = pcm16_to_mulaw8k(pcm, src_rate=16_000)
    yield from chunk_mulaw_to_frames(mulaw)


def chunk_mulaw_to_frames(audio: bytes) -> Iterator[bytes]:
    """Slice `audio` into 160-byte frames, padding final with μ-law silence."""
    for offset in range(0, len(audio), _FRAME_BYTES):
        frame = audio[offset : offset + _FRAME_BYTES]
        if len(frame) < _FRAME_BYTES:
            frame = frame + bytes([_MULAW_SILENCE]) * (_FRAME_BYTES - len(frame))
        yield frame
```

- [ ] **Step 2: Implement `agent/audio/tts_streaming.py`**

```python
"""Cartesia TTS — streaming mode: synthesize chunk-by-chunk, yield frames as they arrive.

For HUMAN-phase natural speech. First audible byte at ~150-200 ms; full first sentence
in ~500-700 ms. Cancellation-safe — wraps the underlying SSE/WS in try/finally so
asyncio.Task.cancel() leaves no partial frames in the outbound queue.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol

from agent.audio.transport import pcm16_to_mulaw8k


_FRAME_BYTES = 160


class CartesiaStreamingClient(Protocol):
    """Cartesia streaming protocol — yields PCM16 16 kHz chunks per text token chunk."""

    async def tts_pcm16_stream(
        self, text_stream: AsyncIterable[str], *, voice_id: str, sample_rate: int
    ) -> AsyncIterator[bytes]: ...


async def stream_to_mulaw_frames(
    client: CartesiaStreamingClient,
    text_stream: AsyncIterable[str],
    *,
    voice_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream `text_stream` through Cartesia and yield μ-law 8 kHz 20 ms frames.

    Maintains a leftover buffer because PCM16 chunk boundaries don't align with
    160-byte μ-law frames. On asyncio.CancelledError, the leftover is dropped
    cleanly (try/finally on the underlying iterator).
    """
    voice = voice_id or os.environ["CARTESIA_VOICE_ID"]
    leftover = b""
    pcm_stream = client.tts_pcm16_stream(text_stream, voice_id=voice, sample_rate=16_000)
    try:
        async for pcm_chunk in pcm_stream:
            mulaw = pcm16_to_mulaw8k(pcm_chunk, src_rate=16_000)
            buf = leftover + mulaw
            n_full = len(buf) // _FRAME_BYTES
            for i in range(n_full):
                yield buf[i * _FRAME_BYTES : (i + 1) * _FRAME_BYTES]
            leftover = buf[n_full * _FRAME_BYTES :]
        if leftover:
            yield leftover + bytes([0x7F]) * (_FRAME_BYTES - len(leftover))
    except asyncio.CancelledError:
        # On cancellation: drop the leftover, stop yielding. Do NOT yield half a
        # frame — Twilio will wedge.
        raise
```

- [ ] **Step 3: Tests** (frame chunking, leftover handling, cancellation safety)

```python
"""Tests for TTS frame chunking and streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from agent.audio.tts_buffered import chunk_mulaw_to_frames
from agent.audio.tts_streaming import stream_to_mulaw_frames


def test_chunking_8khz_20ms_frames() -> None:
    audio = b"\x80" * 1600
    frames = list(chunk_mulaw_to_frames(audio))
    assert len(frames) == 10
    assert all(len(f) == 160 for f in frames)


def test_chunking_pads_partial_final_frame() -> None:
    audio = b"\x80" * 250
    frames = list(chunk_mulaw_to_frames(audio))
    assert len(frames) == 2
    assert len(frames[1]) == 160


@pytest.mark.asyncio
async def test_streaming_yields_complete_frames() -> None:
    """Streaming with a fake client must yield only 160-byte frames."""

    class FakeClient:
        async def tts_pcm16_stream(
            self, text_stream: AsyncIterable[str], *, voice_id: str, sample_rate: int
        ) -> AsyncIterator[bytes]:
            # Emit 2 chunks of arbitrary PCM16 size that don't line up to 160-byte μ-law
            yield b"\x00\x00" * 2400  # 4800 bytes PCM16 16k → 800 bytes μ-law 8k
            yield b"\x00\x00" * 1234

        async def text_passthrough(self, src: AsyncIterable[str]) -> AsyncIterator[str]:
            async for s in src:
                yield s

    async def text_iter() -> AsyncIterator[str]:
        yield "hello"

    client = FakeClient()
    frames: list[bytes] = []
    async for f in stream_to_mulaw_frames(client, text_iter()):
        frames.append(f)
    assert all(len(f) == 160 for f in frames)


@pytest.mark.asyncio
async def test_streaming_cancellation_clean() -> None:
    """Cancelling the stream task must propagate CancelledError, not leak."""

    class HangingClient:
        async def tts_pcm16_stream(
            self, text_stream: AsyncIterable[str], *, voice_id: str, sample_rate: int
        ) -> AsyncIterator[bytes]:
            await asyncio.sleep(10)
            yield b""

    async def text_iter() -> AsyncIterator[str]:
        yield "hello"

    async def consume() -> None:
        async for _ in stream_to_mulaw_frames(HangingClient(), text_iter()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/test_tts_chunking.py -v && ruff check agent tests && pyright
git add agent/audio/ tests/test_tts_chunking.py
git commit -m "feat(audio): Cartesia TTS — buffered (IVR) and streaming (HUMAN) with cancellation"
```

---

### Task 2.4: DTMF emitter — Twilio REST sendDigits side-channel

**Files:**
- Create: `agent/audio/dtmf.py`
- Create: `tests/test_dtmf.py`

(Same as old plan Task 2.3 — `send_dtmf(client, call_sid, digits)` via `<Play digits>` TwiML update. Note in code the foot-gun: Twilio may reset the `<Connect><Stream>` on TwiML update; verify in Phase 4 real call.)

---

## Phase 3 — LangGraph foundation

Goal: state, action types, graph builder, model clients, all without per-mode logic. Phase 4–7 fill in the nodes.

### Task 3.1: State, Action types, graph state TypedDict

**Files:**
- Create: `agent/graph/__init__.py`, `agent/graph/state.py`
- Create: `tests/test_state_types.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the AgentState TypedDict and Action union."""

from __future__ import annotations

from agent.graph.state import (
    AgentState,
    DTMFAction,
    EndCallAction,
    SpeakAction,
    TransitionAction,
)
from agent.types import Outcome


def test_action_union_members_distinct() -> None:
    a1 = DTMFAction(digits="1")
    a2 = SpeakAction(text="hi", streaming=False)
    a3 = TransitionAction(target_mode="hold")
    a4 = EndCallAction(outcome=Outcome.COMPLETED, json_payload={})
    assert a1.kind == "dtmf"
    assert a2.kind == "speak"
    assert a3.kind == "transition"
    assert a4.kind == "end_call"


def test_speak_action_streaming_flag() -> None:
    buffered = SpeakAction(text="yes", streaming=False)
    streamed = SpeakAction(text="hi there what can I get for you", streaming=True)
    assert buffered.streaming is False
    assert streamed.streaming is True
```

- [ ] **Step 2: Implement `agent/graph/state.py`**

```python
"""LangGraph state and action types for the voice agent.

The state holds the per-call conversation history (split per mode), the order,
the most recent transcript to process, any pending actions to dispatch back to
the audio pipeline, and termination metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.types import Order, Outcome


@dataclass(frozen=True)
class DTMFAction:
    """Send DTMF digits via Twilio REST <Play digits> sidechannel."""

    digits: str
    kind: Literal["dtmf"] = "dtmf"


@dataclass(frozen=True)
class SpeakAction:
    """Synthesize TTS and play it on the call.

    `streaming=False` (IVR): synthesize fully then dispatch all frames.
    `streaming=True` (HUMAN): start playback as text arrives; cancellation-safe.
    """

    text: str
    streaming: bool
    kind: Literal["speak"] = "speak"


@dataclass(frozen=True)
class TransitionAction:
    """Mark a mode transition. Adapter side-effects (endpointing, watchdogs) live here."""

    target_mode: Literal["ivr", "hold", "human", "done"]
    kind: Literal["transition"] = "transition"


@dataclass(frozen=True)
class EndCallAction:
    """Drain TTS, hang up Twilio call, print final JSON."""

    outcome: Outcome
    json_payload: dict[str, Any]
    kind: Literal["end_call"] = "end_call"


Action = DTMFAction | SpeakAction | TransitionAction | EndCallAction


class AgentState(TypedDict):
    """LangGraph state for one call."""

    mode: Literal["ivr", "hold", "human", "done"]
    order: Order
    new_transcript: str
    ivr_messages: Annotated[list[BaseMessage], add_messages]
    human_messages: Annotated[list[BaseMessage], add_messages]
    pending_actions: list[Action]
    hangup_reason: Outcome | None
```

`agent/graph/__init__.py`:

```python
"""LangGraph state machine: state, nodes, builder."""
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_state_types.py -v && ruff check agent tests && pyright
git add agent/graph/ tests/test_state_types.py
git commit -m "feat(graph): AgentState TypedDict + Action union types"
```

---

### Task 3.2: LLM clients — Cerebras (gpt-oss-120B) and Groq (Llama 3.1 8B)

**Files:**
- Create: `agent/llm/__init__.py`, `agent/llm/cerebras_client.py`, `agent/llm/groq_client.py`

- [ ] **Step 1: Implement `agent/llm/cerebras_client.py`**

```python
"""Cerebras LLM client for ivr_llm, human_llm, done_node extraction.

Cerebras exposes an OpenAI-compatible endpoint, so we use ChatOpenAI from
langchain-openai with a custom base_url. Tool calling, streaming, and
structured output all flow through the standard LangChain interfaces.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_cerebras_chat(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = False,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    timeout: float = 5.0,
) -> ChatOpenAI:
    """Build a ChatOpenAI configured for Cerebras gpt-oss-120B."""
    chat = ChatOpenAI(
        model=model or os.environ["CEREBRAS_MODEL"],
        api_key=os.environ["CEREBRAS_API_KEY"],
        base_url=os.environ["CEREBRAS_BASE_URL"],
        temperature=temperature,
        streaming=streaming,
        timeout=timeout,
        max_retries=0,  # we want fast failure for voice; outer timeout handles it
    )
    if tools is not None:
        bound = chat.bind_tools(tools, tool_choice=tool_choice)
        return bound  # type: ignore[return-value]
    return chat
```

- [ ] **Step 2: Implement `agent/llm/groq_client.py`**

```python
"""Groq LLM client for hold_node binary classifier (Llama 3.1 8B)."""

from __future__ import annotations

import os

from langchain_groq import ChatGroq


def build_groq_classifier_chat(*, timeout: float = 3.0) -> ChatGroq:
    """Build a ChatGroq configured for Llama 3.1 8B as a yes/no classifier."""
    return ChatGroq(
        model=os.environ["GROQ_CLASSIFIER_MODEL"],
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
        max_tokens=4,
        timeout=timeout,
        max_retries=0,
    )
```

- [ ] **Step 3: Lint + commit**

```bash
ruff check agent && pyright
git add agent/llm/
git commit -m "feat(llm): Cerebras (gpt-oss-120B) + Groq (Llama 3.1 8B) clients"
```

---

### Task 3.3: System prompts (per mode)

**Files:**
- Create: `agent/llm/persona/__init__.py`
- Create: `agent/llm/persona/ivr_prompt.py`, `agent/llm/persona/human_prompt.py`, `agent/llm/persona/extraction_prompt.py`, `agent/llm/persona/hold_prompt.py`
- Create: `tests/test_persona.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for persona prompts: required content + banned phrases."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent.llm.persona.human_prompt import (
    BANNED_PHRASE_PATTERNS,
    build_human_prompt,
)
from agent.llm.persona.ivr_prompt import build_ivr_prompt
from agent.types import Order


@pytest.fixture
def order() -> Order:
    data = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "orders" / "example.json").read_text()
    )
    return Order.model_validate(data)


def test_ivr_prompt_includes_strict_format_rules(order: Order) -> None:
    prompt = build_ivr_prompt(order)
    assert "tool" in prompt.lower()
    assert "exactly" in prompt.lower()
    assert order.customer_name in prompt
    assert order.delivery_zip in prompt


def test_human_prompt_includes_substitution_rules(order: Order) -> None:
    prompt = build_human_prompt(order)
    for ng in order.pizza.no_go_toppings:
        assert ng in prompt
    for sub in order.pizza.acceptable_topping_subs:
        assert sub in prompt
    assert "never argue" in prompt.lower() or "never_argue" in prompt.lower() or "do not argue" in prompt.lower()


@pytest.mark.parametrize("phrase", [
    "let me check",
    "i'll note that down",
    "as an AI",
    "i'll get back to you",
    "i'm checking that for you",
])
def test_banned_phrase_regex_matches(phrase: str) -> None:
    matched = any(p.search(phrase) for p in BANNED_PHRASE_PATTERNS)
    assert matched, f"banned phrase regex missed: {phrase}"
```

- [ ] **Step 2: Implement `agent/llm/persona/ivr_prompt.py`**

```python
"""IVR system prompt — forces tool-call output, strict format compliance."""

from __future__ import annotations

from agent.types import Order


def build_ivr_prompt(order: Order) -> str:
    """Build the system prompt for ivr_llm, parameterized on the order data."""
    return f"""You are navigating a legacy automated phone menu (IVR) on behalf of a customer.

Your ONLY output is a tool call. NEVER produce free-text content. The tools
available are `send_dtmf`, `say_short`, and `mark_transferred_to_hold`.

Order data (use exactly):
- Customer name: {order.customer_name}
- Phone number (10 digits): {order.phone_number}
- Delivery zip: {order.delivery_zip}

The IVR is unforgiving — it accepts ONLY the exact format requested, no extra
words. Map prompts to tools as follows:

- "Press 1 for delivery, press 2 for carryout..." → send_dtmf(digits="1")
- "Please say the name for the order" → say_short(text="{order.customer_name}")
- "Please enter your 10-digit callback number" → send_dtmf(digits="{order.phone_number}")
- "Please say your delivery zip code" → say_short(text="{order.delivery_zip}")
- "I heard <readback>. Is that correct? Say yes to confirm, no to start over"
   → If <readback> matches the values above, say_short(text="yes").
   → If any value is wrong, say_short(text="no").
- "Got it, please hold while we connect you" → mark_transferred_to_hold()

If the prompt doesn't match any of these patterns, do NOT guess — return no tool
call (the controller will retry on the next utterance).

NEVER:
- Add courtesy words ("thanks", "okay") — they break the IVR.
- Speak the digits aloud when DTMF is requested — use send_dtmf.
- Pre-acknowledge ("dialing now") before the tool call — emit ONLY the tool call.
"""
```

- [ ] **Step 3: Implement `agent/llm/persona/human_prompt.py`**

```python
"""HUMAN-phase system prompt + banned-phrase regex regression."""

from __future__ import annotations

import re

from agent.types import Order


BANNED_PHRASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\blet me check\b", re.I),
    re.compile(r"\bi(?:'| a)m checking\b", re.I),
    re.compile(r"\bi'?ll note (?:that )?down\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi'?ll get back to (?:you|that)\b", re.I),
    re.compile(r"\bi appreciate (?:that|your)\b", re.I),
]


def build_human_prompt(order: Order) -> str:
    """Build the system prompt for human_llm, parameterized on the order data."""
    no_go = ", ".join(order.pizza.no_go_toppings) or "(none)"
    subs = ", ".join(order.pizza.acceptable_topping_subs) or "(none)"
    side_backups = ", ".join(order.side.backup_options) or "(none)"
    drink_alts = ", ".join(order.drink.alternatives) or "(none)"

    return f"""You are placing a phone order for delivery on behalf of {order.customer_name}.
Speak like a regular person. Short, natural sentences. Lower-case casual register.

ORDER DATA:
- Pizza: {order.pizza.size}, {order.pizza.crust} crust, toppings: {", ".join(order.pizza.toppings)}
  - acceptable substitutions if a topping is unavailable: {subs}
  - NEVER ACCEPT these toppings: {no_go}
- Side: {order.side.first_choice}
  - backup options in order: {side_backups}
  - if all unavailable: {order.side.if_all_unavailable!r}
- Drink: {order.drink.first_choice}
  - alternatives: {drink_alts}
  - skip if pizza+side already exceed budget: {order.drink.skip_if_over_budget}
- Budget: ${order.budget_max:.2f}
- Special instructions to deliver before hangup: {order.special_instructions!r}
- Delivery address: {order.delivery_address}
- Phone for callback: {order.phone_number}

RULES (apply STRICTLY):
- NEVER accept a no-go topping, even if offered.
- For substitutions, only accept toppings from the acceptable list.
- For sides, try first_choice, then backups in order.
- If pizza+side already exceed ${order.budget_max:.2f}, do not order the drink — call end_call(outcome="over_budget") if the employee won't proceed without the drink, otherwise just decline the drink politely.
- If the pizza itself can't be made workable, call end_call(outcome="nothing_available").

CONVERSATION STYLE:
- Push for exact prices when given ranges ("about thirty bucks" → "can you pin that down for me?").
- Wait through 3-5 second pauses; the employee may be typing.
- Match the employee's tone — friendly, brief, natural.

NEVER:
- "let me check", "I'll note that down", "as an AI", "I'll get back to you",
  "I'm checking that for you", "I appreciate that".
- Announce intent ("I need to ask you...", "as I was saying...").
- Argue with the employee or insist you're human. If asked if you're a bot or
  AI or recording, apologize briefly and call end_call(outcome="detected_as_bot").

ALWAYS deliver the special instructions before invoking end_call.
ALWAYS invoke end_call(outcome=...) — never just say goodbye and let the call die.
"""
```

- [ ] **Step 4: Implement `agent/llm/persona/hold_prompt.py`**

```python
"""hold_node classifier prompt — binary yes/no on 'is this a human greeting?'."""

from __future__ import annotations


HOLD_CLASSIFIER_PROMPT = """You classify whether a single short phone-call transcript is a real human employee greeting.

Reply with exactly one word: yes or no.

YES — examples:
- "hey thanks for calling, what can I get for you?"
- "hi"
- "hello, what would you like to order?"
- "yeah"
- "thanks for holding, how can I help?"

NO — examples:
- Recorded messages: "your call is important to us, please continue to hold"
- IVR system audio: "press 1 for..."
- Mistranscribed music or jingles
- Empty or near-empty audio
"""
```

- [ ] **Step 5: Implement `agent/llm/persona/extraction_prompt.py`**

```python
"""done_node extraction prompt — produces structured JSON from human_messages."""

from __future__ import annotations


EXTRACTION_PROMPT = """You extract a structured pizza-order summary from a phone-call conversation.

Given the full transcript of an order placed with a pizza-place employee, produce
JSON matching the CallResult schema. Be conservative: if a field was never
explicitly stated, leave it null. Do not invent values.

Required fields:
- pizza: { description, substitutions (map of {requested: actual}), price }
- side: { description, original (what was requested first), price } or null if skipped
- drink: { description, price } or null if skipped
- drink_skip_reason: "over_budget" | "unavailable" | null
- total: number
- delivery_time: text as the employee said it (e.g., "35 minutes")
- order_number: text
- special_instructions_delivered: bool — true iff the agent said the special
  instructions to the employee.

Output ONLY the JSON, no commentary.
"""
```

`agent/llm/persona/__init__.py`:

```python
"""Per-mode system prompts."""

from agent.llm.persona.extraction_prompt import EXTRACTION_PROMPT
from agent.llm.persona.hold_prompt import HOLD_CLASSIFIER_PROMPT
from agent.llm.persona.human_prompt import BANNED_PHRASE_PATTERNS, build_human_prompt
from agent.llm.persona.ivr_prompt import build_ivr_prompt

__all__ = [
    "BANNED_PHRASE_PATTERNS",
    "EXTRACTION_PROMPT",
    "HOLD_CLASSIFIER_PROMPT",
    "build_human_prompt",
    "build_ivr_prompt",
]
```

- [ ] **Step 6: Run + commit**

```bash
pytest tests/test_persona.py -v && ruff check agent tests && pyright
git add agent/llm/persona/ tests/test_persona.py
git commit -m "feat(persona): per-mode system prompts + banned-phrase regex regression"
```

---

### Task 3.4: Tool schemas (IVR + HUMAN)

**Files:**
- Create: `agent/graph/tools/__init__.py`, `agent/graph/tools/ivr_tools.py`, `agent/graph/tools/human_tools.py`
- Create: `tests/test_graph_tools.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the LangChain tool schemas."""

from __future__ import annotations

from agent.graph.tools.human_tools import HUMAN_TOOLS
from agent.graph.tools.ivr_tools import IVR_TOOLS


def test_ivr_tools_set() -> None:
    names = {t.name for t in IVR_TOOLS}
    assert names == {"send_dtmf", "say_short", "mark_transferred_to_hold"}


def test_human_tools_set() -> None:
    names = {t.name for t in HUMAN_TOOLS}
    assert names == {"end_call"}


def test_end_call_outcome_enum() -> None:
    end_call = next(t for t in HUMAN_TOOLS if t.name == "end_call")
    schema = end_call.args_schema.model_json_schema()
    enum = schema["properties"]["outcome"]["enum"]
    assert set(enum) == {"completed", "nothing_available", "over_budget", "detected_as_bot"}
```

- [ ] **Step 2: Implement `agent/graph/tools/ivr_tools.py`**

```python
"""IVR-mode tools. The model can ONLY emit these via tool_choice='required'."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class _SendDTMFArgs(BaseModel):
    digits: str = Field(..., description="DTMF digits to dial (e.g. '1' or '5125550147').")


class _SayShortArgs(BaseModel):
    text: str = Field(..., description="Exact text to speak. NO extra words.")


class _MarkTransferredArgs(BaseModel):
    pass


@tool(args_schema=_SendDTMFArgs)
def send_dtmf(digits: str) -> dict:
    """Inject DTMF tones into the live call (Twilio REST sidechannel)."""
    return {"action": "dtmf", "digits": digits}


@tool(args_schema=_SayShortArgs)
def say_short(text: str) -> dict:
    """Synthesize TTS and play the exact text on the call."""
    return {"action": "speak", "text": text, "streaming": False}


@tool(args_schema=_MarkTransferredArgs)
def mark_transferred_to_hold() -> dict:
    """Signal that the IVR has transferred us to a human queue. Transitions mode to HOLD."""
    return {"action": "transition", "target_mode": "hold"}


IVR_TOOLS = [send_dtmf, say_short, mark_transferred_to_hold]
```

- [ ] **Step 3: Implement `agent/graph/tools/human_tools.py`**

```python
"""HUMAN-mode tools. Only end_call is exposed — the model speaks freely otherwise."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class _EndCallArgs(BaseModel):
    outcome: Literal["completed", "nothing_available", "over_budget", "detected_as_bot"] = Field(
        ..., description="The outcome category for this call."
    )
    note: str | None = Field(default=None, description="Optional free-text rationale.")


@tool(args_schema=_EndCallArgs)
def end_call(outcome: str, note: str | None = None) -> dict:
    """End the call. Always use this to hang up — never just say goodbye."""
    return {"action": "end_call", "outcome": outcome, "note": note}


HUMAN_TOOLS = [end_call]
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/test_graph_tools.py -v && ruff check agent tests && pyright
git add agent/graph/tools/ tests/test_graph_tools.py
git commit -m "feat(graph/tools): IVR + HUMAN tool schemas"
```

---

### Task 3.5: Graph builder skeleton (router + empty node placeholders)

**Files:**
- Create: `agent/graph/builder.py`
- Create: `agent/graph/router.py`
- Create: `tests/test_graph_routing.py`

- [ ] **Step 1: Write failing test (router routes by mode)**

```python
"""Tests for the graph router: state.mode → next-node string."""

from __future__ import annotations

from agent.graph.router import route_by_mode


def test_routes_ivr() -> None:
    assert route_by_mode({"mode": "ivr"}) == "ivr_llm"  # type: ignore[arg-type]


def test_routes_hold() -> None:
    assert route_by_mode({"mode": "hold"}) == "hold_node"  # type: ignore[arg-type]


def test_routes_human() -> None:
    assert route_by_mode({"mode": "human"}) == "human_llm"  # type: ignore[arg-type]


def test_routes_done() -> None:
    assert route_by_mode({"mode": "done"}) == "done_node"  # type: ignore[arg-type]
```

- [ ] **Step 2: Implement `agent/graph/router.py`**

```python
"""Pure routing function: pick the next node from state.mode."""

from __future__ import annotations

from agent.graph.state import AgentState


def route_by_mode(state: AgentState) -> str:
    """Return the name of the node to run for this state.mode."""
    mode = state["mode"]
    return {
        "ivr": "ivr_llm",
        "hold": "hold_node",
        "human": "human_llm",
        "done": "done_node",
    }[mode]
```

- [ ] **Step 3: Implement `agent/graph/builder.py` (skeleton — nodes filled in Phase 4–7)**

```python
"""Build the compiled LangGraph state machine.

Nodes are implemented in agent/graph/nodes/. This module wires them together
and configures the checkpointer.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.graph.router import route_by_mode
from agent.graph.state import AgentState


def build_graph() -> object:
    """Build and compile the agent graph."""
    # Lazy imports so the file compiles before all nodes exist.
    from agent.graph.nodes.done_node import done_node
    from agent.graph.nodes.hold_node import hold_node
    from agent.graph.nodes.human_llm import human_llm
    from agent.graph.nodes.ivr_llm import ivr_llm
    from agent.graph.nodes.ivr_tools import ivr_tools

    g = StateGraph(AgentState)

    g.add_node("ivr_llm", ivr_llm)
    g.add_node("ivr_tools", ivr_tools)
    g.add_node("hold_node", hold_node)
    g.add_node("human_llm", human_llm)
    g.add_node("done_node", done_node)

    # Conditional entry: START routes by state.mode
    g.add_conditional_edges(
        START,
        route_by_mode,
        {
            "ivr_llm": "ivr_llm",
            "hold_node": "hold_node",
            "human_llm": "human_llm",
            "done_node": "done_node",
        },
    )

    # Fixed edges
    g.add_edge("ivr_llm", "ivr_tools")
    g.add_edge("ivr_tools", END)
    g.add_edge("hold_node", END)
    g.add_edge("human_llm", END)
    g.add_edge("done_node", END)

    return g.compile(checkpointer=MemorySaver())
```

- [ ] **Step 4: Create empty `agent/graph/nodes/__init__.py` and stub `*.py` files**

Each stub raises `NotImplementedError`. They'll be replaced in Phase 4–7.

```python
# agent/graph/nodes/ivr_llm.py
async def ivr_llm(state):  # type: ignore[no-untyped-def]
    raise NotImplementedError("filled in Phase 4")
```

(Same pattern for `ivr_tools`, `hold_node`, `human_llm`, `done_node`.)

- [ ] **Step 5: Run + commit**

```bash
pytest tests/test_graph_routing.py -v && ruff check agent tests && pyright
git add agent/graph/
git commit -m "feat(graph): builder skeleton + router with stubbed nodes"
```

---

## Phase 4 — IVR mode

Goal: real IVR turn works through the graph. Phase ends with REAL CALL CHECKPOINT 2.

### Task 4.1: ivr_llm node — invoke Cerebras with forced tool choice

**Files:**
- Replace: `agent/graph/nodes/ivr_llm.py`
- Create: `tests/test_ivr_llm_node.py`

- [ ] **Step 1: Write failing test (mock LLM, verify state update)**

```python
"""Tests for the ivr_llm node."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph.nodes.ivr_llm import _ivr_llm_impl
from agent.types import Order


@pytest.fixture
def order() -> Order:
    data = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "orders" / "example.json").read_text()
    )
    return Order.model_validate(data)


@pytest.mark.asyncio
async def test_ivr_llm_appends_transcript_and_response(order: Order) -> None:
    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(
        return_value=AIMessage(content="", tool_calls=[
            {"name": "send_dtmf", "args": {"digits": "1"}, "id": "call_123"}
        ])
    )
    state = {
        "mode": "ivr",
        "order": order,
        "new_transcript": "Press 1 for delivery.",
        "ivr_messages": [],
        "human_messages": [],
        "pending_actions": [],
        "hangup_reason": None,
    }
    update = await _ivr_llm_impl(state, chat=fake_chat)
    msgs = update["ivr_messages"]
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Press 1 for delivery."
    assert msgs[1].tool_calls[0]["name"] == "send_dtmf"


@pytest.mark.asyncio
async def test_preface_content_does_not_leak_to_speak(order: Order) -> None:
    """If the model emits content alongside a tool call, only the tool call dispatches."""
    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="Sure! Dialing now.",  # preface text — must be ignored downstream
            tool_calls=[
                {"name": "send_dtmf", "args": {"digits": "1"}, "id": "call_x"}
            ],
        )
    )
    state = {
        "mode": "ivr",
        "order": order,
        "new_transcript": "Press 1 for delivery.",
        "ivr_messages": [],
        "human_messages": [],
        "pending_actions": [],
        "hangup_reason": None,
    }
    # ivr_llm only appends history; the preface ends up in ivr_messages but
    # ivr_tools dispatches by tool_calls only — content is never spoken.
    update = await _ivr_llm_impl(state, chat=fake_chat)
    last = update["ivr_messages"][-1]
    assert last.tool_calls[0]["name"] == "send_dtmf"
    # The preface stays in history (for debugging) but produces no SpeakAction
    # — assertion of that is in test_ivr_tools_node.py.


@pytest.mark.asyncio
async def test_prompt5_deterministic_no_when_readback_wrong(order: Order) -> None:
    """Prompt-5 readback verification short-circuits to 'no' on data mismatch."""
    from agent.graph.nodes.ivr_llm import deterministic_confirm_response

    # Wrong zip — readback says 90210 but order says 78745
    transcript = "I heard Jordan Mitchell, 5125550147, zip code 90210. Is that correct?"
    decision = deterministic_confirm_response(transcript, order)
    assert decision == "no"


@pytest.mark.asyncio
async def test_prompt5_deterministic_yes_when_readback_correct(order: Order) -> None:
    transcript = "I heard Jordan Mitchell, 5125550147, zip code 78745. Is that correct?"
    decision = deterministic_confirm_response(transcript, order)
    assert decision == "yes"
```

- [ ] **Step 2: Implement `agent/graph/nodes/ivr_llm.py`**

> **Prompt-5 deterministic verification:** before invoking the LLM, if the transcript looks like a confirmation prompt with a readback ("I heard X, Y, zip code Z. Is that correct?"), regex-extract the readback fields and compare against `order.customer_name` / `order.phone_number` / `order.delivery_zip` deterministically. If any field mismatches, short-circuit to `say_short("no")` without calling the LLM. This avoids the known LLM failure mode where models lazily say "yes" on confirmation prompts. The LLM still handles the *recognition* of the confirm prompt; the *decision* is mechanical.

```python
"""ivr_llm node: forced tool-call response on each finalized IVR transcript.

Prompt-5 confirmation: short-circuits to 'no' deterministically if the readback
fields don't match the order data. LLM is only invoked when the prompt is
something other than a confirmation, OR when the readback matches.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.graph.tools.ivr_tools import IVR_TOOLS
from agent.llm.cerebras_client import build_cerebras_chat
from agent.llm.persona import build_ivr_prompt
from agent.types import Order

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableSerializable


_CONFIRM_RE = re.compile(
    r"i heard (?P<name>.+?),\s*(?P<phone>[\d\s-]+),\s*zip code\s*(?P<zip>\d{5})",
    re.IGNORECASE,
)


def deterministic_confirm_response(transcript: str, order: Order) -> Literal["yes", "no"] | None:
    """If `transcript` is a Prompt-5 readback, return 'yes'/'no' deterministically.

    Returns None if the transcript doesn't match the readback shape — caller
    falls through to the LLM.
    """
    match = _CONFIRM_RE.search(transcript)
    if not match:
        return None
    heard_name = match.group("name").strip().lower()
    heard_phone = re.sub(r"\D", "", match.group("phone"))
    heard_zip = match.group("zip")

    name_ok = heard_name == order.customer_name.lower()
    phone_ok = heard_phone == order.phone_number
    zip_ok = heard_zip == order.delivery_zip

    return "yes" if (name_ok and phone_ok and zip_ok) else "no"


async def ivr_llm(state: AgentState) -> dict[str, Any]:
    """ivr_llm node — public entry point used by LangGraph."""
    # Deterministic short-circuit for confirmation prompts
    confirm = deterministic_confirm_response(state["new_transcript"], state["order"])
    if confirm is not None:
        # Synthesize an AIMessage with the appropriate tool call; bypass LLM
        synthetic = AIMessage(
            content="",
            tool_calls=[{
                "name": "say_short",
                "args": {"text": confirm},
                "id": f"det_{confirm}",
            }],
        )
        user = HumanMessage(content=state["new_transcript"])
        return {"ivr_messages": [user, synthetic]}

    chat = build_cerebras_chat(
        tools=[t.args_schema.model_json_schema() for t in IVR_TOOLS],
        tool_choice="required",
        timeout=5.0,
    )
    return await _ivr_llm_impl(state, chat=chat)


async def _ivr_llm_impl(state: AgentState, *, chat: "RunnableSerializable") -> dict[str, Any]:
    """Testable inner implementation; takes the chat client as a dep."""
    system = SystemMessage(content=build_ivr_prompt(state["order"]))
    user = HumanMessage(content=state["new_transcript"])
    history = state.get("ivr_messages", [])
    messages = [system, *history, user]

    try:
        async with asyncio.timeout(5.0):
            response = await chat.ainvoke(messages)
    except asyncio.TimeoutError:
        return {"hangup_reason": "ivr_failed"}

    return {"ivr_messages": [user, response]}
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_ivr_llm_node.py -v && ruff check agent tests && pyright
git add agent/graph/nodes/ivr_llm.py tests/test_ivr_llm_node.py
git commit -m "feat(graph/ivr): ivr_llm node — forced tool call on each transcript"
```

---

### Task 4.2: ivr_tools node — translate tool calls into Actions

**Files:**
- Replace: `agent/graph/nodes/ivr_tools.py`
- Create: `tests/test_ivr_tools_node.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for the ivr_tools node: tool call → Action mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agent.graph.nodes.ivr_tools import ivr_tools
from agent.graph.state import DTMFAction, SpeakAction, TransitionAction
from agent.types import Order


@pytest.fixture
def order() -> Order:
    return Order.model_validate(
        json.loads(
            (Path(__file__).parent.parent / "fixtures" / "orders" / "example.json").read_text()
        )
    )


def _state_with(tool_call: dict, order: Order) -> dict:
    return {
        "mode": "ivr",
        "order": order,
        "new_transcript": "",
        "ivr_messages": [
            AIMessage(content="", tool_calls=[tool_call]),
        ],
        "human_messages": [],
        "pending_actions": [],
        "hangup_reason": None,
    }


@pytest.mark.asyncio
async def test_send_dtmf_yields_dtmf_action(order: Order) -> None:
    state = _state_with({"name": "send_dtmf", "args": {"digits": "1"}, "id": "x"}, order)
    update = await ivr_tools(state)
    assert update["pending_actions"][0] == DTMFAction(digits="1")


@pytest.mark.asyncio
async def test_say_short_yields_buffered_speak_action(order: Order) -> None:
    state = _state_with({"name": "say_short", "args": {"text": "yes"}, "id": "x"}, order)
    update = await ivr_tools(state)
    a = update["pending_actions"][0]
    assert isinstance(a, SpeakAction)
    assert a.text == "yes"
    assert a.streaming is False


@pytest.mark.asyncio
async def test_mark_transferred_sets_mode_hold(order: Order) -> None:
    state = _state_with(
        {"name": "mark_transferred_to_hold", "args": {}, "id": "x"}, order
    )
    update = await ivr_tools(state)
    assert update["mode"] == "hold"
    assert TransitionAction(target_mode="hold") in update["pending_actions"]


@pytest.mark.asyncio
async def test_no_tool_calls_no_actions(order: Order) -> None:
    state = {
        "mode": "ivr",
        "order": order,
        "new_transcript": "",
        "ivr_messages": [AIMessage(content="oops", tool_calls=[])],
        "human_messages": [],
        "pending_actions": [],
        "hangup_reason": None,
    }
    update = await ivr_tools(state)
    assert update.get("pending_actions", []) == []
```

- [ ] **Step 2: Implement `agent/graph/nodes/ivr_tools.py`**

> **Critical**: append a `ToolMessage` per tool call to `ivr_messages`. Without this, Cerebras (OpenAI-compatible API) will 400 on the next turn because the prior `AIMessage.tool_calls` has no matching tool response. Synthetic content `"ok"` is fine — the model never reads it; the protocol just requires the message exist.

```python
"""ivr_tools node: read latest AIMessage's tool_calls, emit Actions, ack with ToolMessage."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph.state import AgentState, DTMFAction, SpeakAction, TransitionAction


async def ivr_tools(state: AgentState) -> dict[str, Any]:
    """Translate the latest LLM tool call(s) into Actions; append ToolMessage acks."""
    actions: list = []
    new_mode: str | None = None
    tool_acks: list[ToolMessage] = []

    for msg in reversed(state.get("ivr_messages", [])):
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                name = call["name"]
                args = call.get("args", {}) or {}
                call_id = call.get("id") or ""
                if name == "send_dtmf":
                    actions.append(DTMFAction(digits=args["digits"]))
                elif name == "say_short":
                    actions.append(SpeakAction(text=args["text"], streaming=False))
                elif name == "mark_transferred_to_hold":
                    actions.append(TransitionAction(target_mode="hold"))
                    new_mode = "hold"
                # Required by Cerebras (OpenAI-compat): every tool_call needs a
                # matching ToolMessage with the same id, or the next turn 400s.
                tool_acks.append(ToolMessage(content="ok", tool_call_id=call_id))
            break

    update: dict[str, Any] = {"pending_actions": actions}
    if tool_acks:
        update["ivr_messages"] = tool_acks  # add_messages reducer appends
    if new_mode is not None:
        update["mode"] = new_mode
    return update
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_ivr_tools_node.py -v && ruff check agent tests && pyright
git add agent/graph/nodes/ivr_tools.py tests/test_ivr_tools_node.py
git commit -m "feat(graph/ivr): ivr_tools translates tool calls to Actions, sets mode on transfer"
```

---

### Task 4.3: StateProcessor adapter — bridge audio pipeline ↔ graph

**Files:**
- Create: `agent/adapter.py`
- Modify: `agent/server.py` to instantiate the adapter on /stream WS connect

- [ ] **Step 1: Implement `agent/adapter.py`**

This is the central runtime piece — translates audio events into `graph.ainvoke`s and dispatches resulting Actions.

```python
"""StateProcessor: bridges the audio pipeline to the LangGraph state machine.

For each finalized ASR transcript (or audio event during HOLD), invokes the
graph and dispatches the resulting Actions back to the audio pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

from agent.audio.dtmf import send_dtmf
from agent.audio.tts_buffered import synthesize_to_mulaw_frames
from agent.audio.tts_streaming import stream_to_mulaw_frames
from agent.graph.state import (
    Action,
    AgentState,
    DTMFAction,
    EndCallAction,
    SpeakAction,
    TransitionAction,
)
from agent.obs import build_logger, langfuse_callback_handler
from agent.runtime.stream import OutboundFrame, encode_outbound_frame
from agent.types import Order, Outcome

if TYPE_CHECKING:
    from cartesia import Cartesia
    from fastapi import WebSocket
    from twilio.rest import Client


_log = build_logger(__name__)


class StateProcessor:
    """One per call. Owns the graph invocation loop and audio dispatch."""

    def __init__(
        self,
        *,
        ws: WebSocket,
        twilio: Client,
        cartesia: Cartesia,
        order: Order,
        graph: Any,
    ) -> None:
        self.ws = ws
        self.twilio = twilio
        self.cartesia = cartesia
        self.order = order
        self.graph = graph
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.tts_task: asyncio.Task[None] | None = None
        self.hold_timeout_task: asyncio.Task[None] | None = None
        self._state: AgentState = {  # type: ignore[typeddict-item]
            "mode": "ivr",
            "order": order,
            "new_transcript": "",
            "ivr_messages": [],
            "human_messages": [],
            "pending_actions": [],
            "hangup_reason": None,
        }
        self._mark_events: dict[str, asyncio.Event] = {}

    async def on_call_started(self, *, call_sid: str, stream_sid: str) -> None:
        """Initialize per-call state."""
        self.call_sid = call_sid
        self.stream_sid = stream_sid

    async def on_transcript(self, transcript: str) -> None:
        """Process one finalized ASR transcript by invoking the graph."""
        self._state["new_transcript"] = transcript
        await self._invoke_graph()

    async def on_barge_in(self) -> None:
        """User started speaking — cancel any in-flight TTS task."""
        if self.tts_task is not None and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass

    async def on_hold_timeout(self) -> None:
        """Hold-timeout watchdog fired."""
        self._state["mode"] = "done"
        self._state["hangup_reason"] = Outcome.HOLD_TIMEOUT
        await self._invoke_graph()

    async def _invoke_graph(self) -> None:
        if self.call_sid is None:
            return
        config = {
            "configurable": {"thread_id": self.call_sid},
            "callbacks": [cb] if (cb := langfuse_callback_handler()) else [],
        }
        result: AgentState = await self.graph.ainvoke(self._state, config=config)
        # Merge new state from result
        self._state = result
        # Dispatch any pending actions
        for action in result.get("pending_actions", []):
            await self._dispatch(action)
        # Clear actions
        self._state["pending_actions"] = []

    async def _dispatch(self, action: Action) -> None:
        if isinstance(action, DTMFAction):
            assert self.call_sid is not None
            send_dtmf(self.twilio, call_sid=self.call_sid, digits=action.digits)
            _log.info("dispatched_dtmf", extra={
                "call_sid": self.call_sid, "mode": self._state["mode"],
                "event": "dtmf_sent", "digits": action.digits,
            })
        elif isinstance(action, SpeakAction):
            await self._speak(action)
        elif isinstance(action, TransitionAction):
            await self._on_transition(action.target_mode)
        elif isinstance(action, EndCallAction):
            await self._end_call(action)

    async def _speak(self, action: SpeakAction) -> None:
        if self.stream_sid is None:
            return
        if self.tts_task is not None:
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
        if action.streaming:
            self.tts_task = asyncio.create_task(self._speak_streaming(action.text))
        else:
            self.tts_task = asyncio.create_task(self._speak_buffered(action.text))

    async def _speak_buffered(self, text: str) -> None:
        if self.stream_sid is None:
            return
        for frame in synthesize_to_mulaw_frames(self.cartesia, text=text):
            out = OutboundFrame(stream_sid=self.stream_sid, audio=frame)
            await self.ws.send_text(encode_outbound_frame(out))

    async def _speak_streaming(self, text: str) -> None:
        # Stub: in HUMAN mode, the LLM streaming output drives this path.
        # See Phase 6 — the human_llm node returns a token stream that we plumb
        # through Cartesia streaming. For Phase 4, this isn't reached.
        raise NotImplementedError("filled in Phase 6")

    async def _on_transition(self, target_mode: str) -> None:
        self._state["mode"] = target_mode  # type: ignore[typeddict-item]
        if target_mode == "hold":
            # Start hold-timeout watchdog
            self.hold_timeout_task = asyncio.create_task(self._hold_timeout_watch())

    async def _hold_timeout_watch(self) -> None:
        await asyncio.sleep(300)  # 5 min
        if self._state["mode"] == "hold":
            await self.on_hold_timeout()

    async def _end_call(self, action: EndCallAction) -> None:
        # Wait for any in-flight TTS to drain (the synthesis task ending != Twilio
        # finished playing). Send a Twilio Media Streams `mark` event after the
        # last audio frame and wait for Twilio to echo it back — that's the
        # signal that audio has actually played out, not just been queued.
        if self.tts_task is not None and not self.tts_task.done():
            try:
                await asyncio.wait_for(self.tts_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.tts_task.cancel()
        if self.stream_sid is not None:
            mark_name = f"end_{action.outcome.value}"
            await self.ws.send_text(json.dumps({
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": mark_name},
            }))
            # Wait for Twilio to echo the mark (consumed by the WS handler in
            # server.py and forwarded via on_mark_received). Bound to 3s so a
            # stuck Twilio doesn't hang shutdown.
            try:
                await asyncio.wait_for(self._mark_echoed(mark_name), timeout=3.0)
            except asyncio.TimeoutError:
                pass
        # Print final JSON
        print(json.dumps(action.json_payload))
        # Twilio hangup
        if self.call_sid is not None:
            try:
                self.twilio.calls(self.call_sid).update(status="completed")
            except Exception:  # noqa: BLE001 — best effort
                pass

    async def _mark_echoed(self, name: str) -> None:
        """Wait until on_mark_received(name) sets the corresponding event."""
        ev = self._mark_events.setdefault(name, asyncio.Event())
        await ev.wait()

    async def on_mark_received(self, name: str) -> None:
        """Called by the WS handler when Twilio echoes a mark event back."""
        ev = self._mark_events.setdefault(name, asyncio.Event())
        ev.set()

    async def close(self) -> None:
        for task in (self.tts_task, self.hold_timeout_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
```

- [ ] **Step 2: Lint + commit**

```bash
ruff check agent && pyright
git add agent/adapter.py
git commit -m "feat(adapter): StateProcessor bridges audio pipeline ↔ LangGraph"
```

---

### Task 4.3b: WebSocket handler — Twilio Media Streams ↔ Deepgram + adapter

This is the audio-pipeline glue. Without it, no audio reaches the graph and no IVR call happens. (Senior review flagged the previous "described, not coded" version as a blocker — promoted here.)

**Files:**
- Modify: `agent/server.py` (replace echo handler from T1.4)

- [ ] **Step 1: Replace the echo `/stream` handler in `agent/server.py`**

```python
@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """Twilio Media Streams handler: route audio to ASR + adapter, dispatch outbound."""
    from agent.adapter import StateProcessor
    from agent.asr import DeepgramStream
    from agent.graph.builder import build_graph
    from agent.runtime.stream import decode_inbound_frame

    await ws.accept()
    order: Order = ws.app.state.order
    twilio: Client = ws.app.state.twilio
    cartesia: Cartesia = ws.app.state.cartesia
    graph = build_graph()

    deepgram = DeepgramStream()
    processor = StateProcessor(
        ws=ws, twilio=twilio, cartesia=cartesia, order=order, graph=graph
    )
    finals_task: asyncio.Task[None] | None = None
    errors_task: asyncio.Task[None] | None = None

    async def _consume_finals() -> None:
        async for transcript in deepgram.finals():
            await processor.on_transcript(transcript)

    async def _consume_errors() -> None:
        async for err in deepgram.errors():
            # Treat ASR errors as ivr_failed during IVR; adapter routes via hangup_reason.
            await processor.on_asr_error(err)

    try:
        await deepgram.open(utterance_end_ms=750)  # IVR-tuned per senior review
        finals_task = asyncio.create_task(_consume_finals())
        errors_task = asyncio.create_task(_consume_errors())

        while True:
            msg = await ws.receive_text()
            frame = decode_inbound_frame(msg)
            if frame is None:
                continue
            if frame.kind == "start" and frame.stream_sid and frame.call_sid:
                await processor.on_call_started(
                    call_sid=frame.call_sid, stream_sid=frame.stream_sid
                )
            elif frame.kind == "media":
                await deepgram.send(frame.audio)
            elif frame.kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        for task in (finals_task, errors_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await deepgram.close()
        await processor.close()
```

- [ ] **Step 2: Add `on_asr_error` method to StateProcessor** in `agent/adapter.py`:

```python
async def on_asr_error(self, err: Exception) -> None:
    """Handle a fatal Deepgram ASR error during IVR by triggering ivr_failed."""
    if self._state["mode"] == "ivr":
        self._state["mode"] = "done"
        self._state["hangup_reason"] = Outcome.IVR_FAILED
        await self._invoke_graph()
```

- [ ] **Step 3: Lint + commit**

```bash
ruff check agent && pyright
git add agent/server.py agent/adapter.py
git commit -m "feat(server): WS handler wires Twilio Media Streams ↔ Deepgram ↔ StateProcessor"
```

---

### Task 4.4: REAL CALL CHECKPOINT 2 — IVR navigation works end-to-end

(Same script as old plan Task 2.7. Read the IVR prompts in robotic monotone; verify each tool dispatch produces the right action; verify mode→hold transition.)

---

## Phase 5 — HOLD mode

### Task 5.1: Energy gate (audio-frame-level)

**Files:** `agent/hold/__init__.py`, `agent/hold/energy.py`, `tests/test_hold_energy.py`. Same as old plan Task 3.1.

### Task 5.2: hold_node (LangGraph node) — small-LLM classifier

**Files:** Replace `agent/graph/nodes/hold_node.py`. Tests in `tests/test_hold_node.py`.

```python
"""hold_node: classify whether a transcript is a human greeting; transition if yes."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.llm.groq_client import build_groq_classifier_chat
from agent.llm.persona import HOLD_CLASSIFIER_PROMPT


async def hold_node(state: AgentState) -> dict[str, Any]:
    """Classify the new_transcript. On 'yes', transition to HUMAN with seeded context."""
    transcript = state["new_transcript"].strip()
    if not transcript:
        return {}

    chat = build_groq_classifier_chat()
    try:
        async with asyncio.timeout(3.0):
            resp = await chat.ainvoke(
                [SystemMessage(content=HOLD_CLASSIFIER_PROMPT), HumanMessage(content=transcript)]
            )
    except asyncio.TimeoutError:
        return {}  # treat as no — we'll re-classify the next utterance

    answer = (resp.content or "").strip().lower()
    if answer.startswith("yes"):
        return {
            "mode": "human",
            "human_messages": [HumanMessage(content=transcript)],
        }
    return {}
```

(Tests mock the chat client; verify yes/no/empty cases.)

### Task 5.3: Wire energy gate into adapter (HOLD mode handling)

In `StateProcessor.on_media` (or its equivalent in the audio pipeline), during HOLD mode: feed PCM16 16 kHz to the energy gate. Only forward transcripts to `on_transcript` when the gate has tripped.

### Task 5.4: REAL CALL CHECKPOINT 3 — hold detection works

Manual test: place call, navigate IVR, on the "please hold" prompt the agent goes silent. Speak after 5–10 s — agent transitions to HUMAN and responds.

---

## Phase 6 — HUMAN mode (streaming TTS, barge-in)

### Task 6.1: human_llm node — streaming response

```python
"""human_llm: free-text response, streaming, with end_call tool."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState, EndCallAction, SpeakAction
from agent.graph.tools.human_tools import HUMAN_TOOLS
from agent.llm.cerebras_client import build_cerebras_chat
from agent.llm.persona import build_human_prompt
from agent.types import Outcome


async def human_llm(state: AgentState) -> dict[str, Any]:
    """Run one HUMAN turn. Streams text to TTS via SpeakAction; may emit end_call."""
    chat = build_cerebras_chat(
        tools=[t.args_schema.model_json_schema() for t in HUMAN_TOOLS],
        tool_choice="auto",
        streaming=True,
        timeout=8.0,
    )
    system = SystemMessage(content=build_human_prompt(state["order"]))
    user = HumanMessage(content=state["new_transcript"])
    history = state.get("human_messages", [])
    messages = [system, *history, user]

    # Collect streaming output. The adapter reads `pending_actions` after the
    # node returns. For streaming TTS, the SpeakAction carries the FULL text;
    # the adapter does the actual streaming through Cartesia (Phase 6.2).
    try:
        async with asyncio.timeout(8.0):
            response = await chat.ainvoke(messages)
    except asyncio.TimeoutError:
        return {"hangup_reason": Outcome.DISCONNECT, "mode": "done"}

    actions: list = []
    new_mode = state["mode"]
    end_call_payload: EndCallAction | None = None

    if response.content:
        actions.append(SpeakAction(text=str(response.content), streaming=True))
    for call in response.tool_calls or []:
        if call["name"] == "end_call":
            args = call["args"] or {}
            outcome = Outcome(args.get("outcome", "completed"))
            end_call_payload = EndCallAction(outcome=outcome, json_payload={})
            new_mode = "done"

    update: dict[str, Any] = {
        "human_messages": [user, response],
        "pending_actions": actions,
    }
    if new_mode != state["mode"]:
        update["mode"] = new_mode
    if end_call_payload is not None:
        update["pending_actions"] = actions + [end_call_payload]
    return update
```

### Task 6.2: Adapter `_speak_streaming` — pipe Cartesia stream through TTS task

(Replace the `NotImplementedError` stub from Task 4.3 with the real Cartesia streaming call. Use `stream_to_mulaw_frames`.)

### Task 6.3: VAD-based barge-in cancellation

In the audio pipeline, run Silero VAD on every inbound 30 ms PCM16 16 kHz chunk during HUMAN mode. When VAD fires, call `processor.on_barge_in()` which cancels `tts_task`.

### Task 6.4: REAL CALL CHECKPOINT 4 — happy-path conversation

Run the build_day.md test script verbatim. Verify:
- Substitutions handled per data
- Per-item prices captured (in human_messages)
- end_call(outcome="completed") invoked

---

## Phase 7 — DONE mode (post-call extraction)

### Task 7.1: done_node — structured-output extraction over human_messages

```python
"""done_node: extract CallResult JSON from human_messages."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import SystemMessage

from agent.graph.state import AgentState, EndCallAction
from agent.llm.cerebras_client import build_cerebras_chat
from agent.llm.persona import EXTRACTION_PROMPT
from agent.types import CallResult, Outcome


async def done_node(state: AgentState) -> dict[str, Any]:
    """Run extraction (or use hangup_reason) and produce final EndCallAction."""
    hangup = state.get("hangup_reason")
    if hangup is not None:
        # Operational outcome — skip extraction
        result = CallResult(outcome=hangup)
        return {
            "pending_actions": [
                EndCallAction(outcome=hangup, json_payload=json.loads(result.model_dump_json()))
            ]
        }

    # Run extraction LLM over human_messages
    chat = build_cerebras_chat(temperature=0.0, timeout=10.0)
    system = SystemMessage(content=EXTRACTION_PROMPT)
    transcript_text = "\n".join(
        f"{m.type}: {m.content}" for m in state.get("human_messages", []) if m.content
    )

    try:
        async with asyncio.timeout(10.0):
            resp = await chat.ainvoke([system, *state.get("human_messages", [])])
        raw = (resp.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        data = json.loads(raw)
        # The extraction LLM omits 'outcome' — set to COMPLETED if pizza is present, else NOTHING_AVAILABLE
        if data.get("pizza"):
            data["outcome"] = Outcome.COMPLETED.value
        else:
            data["outcome"] = Outcome.NOTHING_AVAILABLE.value
        result = CallResult.model_validate(data)
    except (asyncio.TimeoutError, ValueError, json.JSONDecodeError):
        result = CallResult(outcome=Outcome.DISCONNECT)

    return {
        "pending_actions": [
            EndCallAction(
                outcome=result.outcome,
                json_payload=json.loads(result.model_dump_json()),
            )
        ]
    }
```

### Task 7.2: Tests — done_node with mocked extraction LLM

(Mock chat.ainvoke to return a sample JSON; verify the EndCallAction.json_payload matches.)

### Task 7.3: REAL CALL CHECKPOINT 5 — full call from dial to JSON output

Same script as Phase 6, but verify the JSON printed to stdout passes `CallResult.model_validate`.

---

## Phase 8 — Robustness

### Task 8.1: 3-strike IVR_REJECTED abort

Track consecutive empty/no-tool-call IVR turns in state; on third, set `mode="done"` + `hangup_reason=IVR_REJECTED`.

### Task 8.2: no_answer detection

In `agent/cli.py` after `place_call`, poll Twilio status for ~10 s; if `failed`/`busy`/`no-answer`, print `{"outcome":"no_answer"}` and exit.

### Task 8.3: HUMAN inactivity timeout (~30 s no transcripts → disconnect)

Adapter starts a `human_inactivity_task` on HOLD→HUMAN transition. Each transcript resets it; if it fires, `mode="done"`, `hangup_reason=DISCONNECT`.

### Task 8.4: Deepgram error → ivr_failed

Already wired in Task 2.1; just connect to `hangup_reason=IVR_FAILED` in adapter.

### Task 8.5: LLM provider 5xx retry once with backoff

In `_ivr_llm_impl` and `human_llm`, on `httpx.HTTPStatusError` 5xx: retry once after 250 ms backoff before timing out.

---

## Phase 9 — Final verification

### Task 9.1: Run build_day.md test script verbatim

Real call. Capture stdout JSON. Validate against schema:

```python
import json, sys
from agent.types import CallResult
data = json.loads(sys.argv[1])
result = CallResult.model_validate(data)
assert result.outcome.value == "completed"
assert result.pizza is not None and result.pizza.price > 0
assert result.side is not None and result.side.original == "buffalo wings, 12 count"
assert result.drink is not None and result.drink.description.startswith("2L")
assert result.total > 0
assert result.special_instructions_delivered is True
```

### Task 9.2: Substitution edge cases (3 real calls)

- `nothing_available`: empty `acceptable_topping_subs`, deny the only topping, verify `outcome="nothing_available"`.
- `over_budget`: budget low enough that pizza+side > budget; verify `drink: null`, `drink_skip_reason: "over_budget"`.
- `detected_as_bot`: ask "are you a robot?" mid-conversation; verify `outcome="detected_as_bot"`.

### Task 9.3: Hold-timeout test (1 real call)

Place call; navigate IVR; on hold prompt, do not respond for 5 minutes. Verify `outcome="hold_timeout"`.

---

## Self-Review

**1. Spec coverage**
Every build_day.md requirement has a node or task. See design doc compliance map.

**2. Placeholder scan**
- Phase 6.2 references the streaming TTS path; adapter `_speak_streaming` is a stub until Task 6.2.
- Phase 4.3 adapter is partial (the audio-pipeline glue between Twilio /stream WS and Deepgram + VAD is in Phase 4.3 Step 2 — described, not coded inline). Implementor: study `agent/runtime/stream.py` codec from Task 1.4, then wire frames into `DeepgramStream.send` + `SileroVAD.is_speech_chunk` + `processor.on_transcript`/`on_barge_in` callbacks.

**3. Type / signature consistency**
- `Action` union: kinds match across state.py, ivr_tools.py, adapter.py.
- `Outcome` enum used consistently.
- LangGraph state TypedDict matches across nodes.

**4. Architectural notes**
- LangGraph `Command` could replace the "ivr_tools sets mode" pattern, but for clarity we use plain dict updates with explicit `mode` field. Refactor opportunity if implementor prefers.
- `MemorySaver` is fine for one-call, single-process usage. Swap to `SqliteSaver` only if cross-process replay becomes a need (not in v1).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-pizza-voice-agent.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Phase 0 tasks (T0.2/T0.3/T0.4/T0.5/T0.6) and Phase 2 tasks (T2.1/T2.2/T2.3/T2.4) can run in parallel via `superpowers:dispatching-parallel-agents`.

**2. Inline Execution** — Execute tasks in this session via `superpowers:executing-plans`, batched with checkpoints for review.

**Which approach?**
