# Pizza Voice Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process Python voice agent that places an outbound phone call, navigates an automated IVR with LangGraph-driven tool-calling, detects hold→human handoff, conducts a natural conversation through streaming Llama-class LLM + TTS, and emits a structured JSON result via post-call extraction.

**Architecture:** **Pipecat** (the library) owns the audio pipeline (Twilio transport, Deepgram STT, Silero VAD, Cartesia TTS, frame routing). A custom `StateMachineProcessor` (Pipecat `FrameProcessor` subclass) sits in the pipeline and bridges audio events to a **LangGraph** state machine with 6 nodes (`router`, `ivr_llm`, `ivr_tools`, `hold_node`, `human_llm`, `done_node`). LangGraph runs OUTSIDE the Pipecat pipeline; the processor calls `graph.ainvoke()` per finalized transcript. Per-mode tool registries; mode-specific message histories; `MemorySaver` checkpointer keyed by `thread_id=call_sid`.

**Tech Stack:**
- Python 3.13+, `asyncio`, FastAPI, `websockets`, `uvicorn`
- **Pipecat** (`pipecat-ai`, exact version pinned) — provides: `FastAPIWebsocketTransport`, `TwilioFrameSerializer`, `DeepgramSTTService`, `CartesiaTTSService`, `SileroVADAnalyzer`, `Pipeline`, `PipelineRunner`, `PipelineTask`, `FrameProcessor` base class
- Twilio Python SDK (for outbound dial + DTMF REST sidechannel)
- **LangGraph** + `langgraph-checkpoint` (MemorySaver)
- **Cerebras** for `gpt-oss-120B` (ivr_llm, human_llm, done_node extraction)
- **Groq** for `Llama 3.1 8B` (hold_node binary classifier)
- LangChain ChatOpenAI-compatible clients (Cerebras exposes OpenAI-compat endpoint; Groq has native LangChain integration)
- pydantic v2, `audioop-lts` (Py 3.13 stdlib drop), `scipy.signal.resample_poly` (used at boundaries — Pipecat handles most resampling internally)
- pytest, `ruff` (with `D`/pydocstyle), `pyright` (strict)
- ngrok (`pyngrok`)
- Langfuse self-hosted (Docker) + LangChain callback

**Constraints from CLAUDE.md** (read it before starting any task):
- Per-change workflow: pytest+ruff+pyright → simplify → verify → code-reviewer → commit → push.
- Type-annotate everything; module + public-symbol docstrings; `from __future__ import annotations`.
- Foot-guns: DTMF is REST-only; resampling must be exact; `Task.cancel()` for barge-in; `thread_id=call_sid` non-negotiable.

**Suggested execution mode:** subagent-driven. Phases 0, 2, 3 have parallelizable tasks (independent subsystems). Phases 4–7 are sequential (each builds on the previous mode). Use `superpowers:dispatching-parallel-agents` where independence is real.

## IVR-Only Milestone Scope (Pipecat-based)

The user's first verification gate is **Phase 4.4 — IVR end-to-end on a real call**. With Pipecat owning the audio pipeline, the scope is now:

**Required for IVR-end-to-end:**
- Phase 0: ✅ DONE (T0.1–T0.6 + simplify fixes already on `feat/ivr-e2e`)
- Phase 1: P1.1 add `pipecat-ai` dep + pin, P1.2 ngrok helper (already-coded conceptually), P1.3 Twilio dialer (allowlist precheck — already-coded conceptually), P1.4 FastAPI server with `/voice` returning Pipecat-compatible TwiML, P1.5 build_pipecat_pipeline skeleton (transport-only — echo via Pipecat's pass-through), P1.6 CLI entrypoint, P1.7 REAL CALL CHECKPOINT 1 (audio echo through Pipecat), P1.8 DTMF mid-stream verify.
- Phase 2: **T2.1** wire `DeepgramSTTService` into pipeline, **T2.2** wire `CartesiaTTSService` into pipeline, **T2.3** custom `DTMFProcessor` (FrameProcessor that emits Twilio REST DTMF when it sees a `DTMFFrame`). Skip Silero VAD (HUMAN-phase only) and streaming-TTS specifics (already covered by Cartesia service).
- Phase 3: T3.1 (state + Action types), T3.2 **Cerebras client only**, T3.3 **IVR prompt only**, T3.4 **IVR tools only**, T3.5 (graph builder skeleton with stubbed hold/human/done nodes).
- Phase 4: T4.1 ivr_llm node (with Prompt-5 deterministic verification), T4.2 ivr_tools node (with `ToolMessage` acks), T4.3 `StateMachineProcessor` (FrameProcessor subclass that bridges to LangGraph), T4.4 wire it into the Pipecat pipeline at the right position, T4.5 REAL CALL CHECKPOINT 2.

**Deferred until after IVR checkpoint passes:** Silero VAD wiring (Phase 6), Groq client (Phase 5), hold/human/extraction prompts (Phases 5–7), HUMAN tools (Phase 6), all of Phases 5/6/7/8/9.

After the user confirms IVR works on a real call, resume the plan from Phase 5.

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

## Phase 1 — Pipecat telephony shell (Pipecat-revised)

Goal: real outbound call connects, audio frames flow through Pipecat's pipeline runner echoing back to the caller, clean exit on hangup, DTMF mid-stream behavior verified.

### Task 1.1: pipecat-ai dependency — DONE

Already on `feat/ivr-e2e` (commit `669cd42`). Pinned at `pipecat-ai==0.0.108`. Smoke tests cover key import paths.

**One follow-up needed (Task 1.1b):** the senior review flagged the install spec should be `pipecat-ai[cartesia,deepgram,silero,websocket]==0.0.108`. The implementer dropped `silero` thinking it wasn't a declared extra; it actually is (no-op deps in 0.0.108) and including it is future-proofing. Add `[silero,websocket]` via:

```bash
source .venv/bin/activate
uv add 'pipecat-ai[cartesia,deepgram,silero,websocket]==0.0.108'
git add pyproject.toml uv.lock
git commit -m "chore(deps): pipecat-ai add silero+websocket extras for future-proofing"
```

### Task 1.2: Project-local Pipecat frames

**Files:** `agent/pipeline/__init__.py`, `agent/pipeline/frames.py`, `tests/test_frames.py`

We define a `RestDTMFFrame(SystemFrame)` that signals "send these DTMF digits via Twilio REST." We **do not** use Pipecat's `OutputDTMFFrame` because on `FastAPIWebsocketTransport` it falls back to synthesizing μ-law tones into the audio stream — exactly the foot-gun CLAUDE.md forbids.

```python
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
```

Test: import and instantiate; assert digits round-trip.

Commit: `feat(pipeline): RestDTMFFrame for out-of-band DTMF via Twilio REST`.

### Task 1.3: ngrok tunnel helpers

**Files:** `agent/runtime/__init__.py`, `agent/runtime/tunnel.py`

```python
"""ngrok tunnel boot/teardown helpers."""

from __future__ import annotations

import os

from pyngrok import conf, ngrok


def open_tunnel(port: int) -> str:
    """Open an ngrok HTTPS tunnel to localhost:port; return the public URL."""
    auth = os.environ.get("NGROK_AUTHTOKEN")
    if auth:
        conf.get_default().auth_token = auth
    tunnel = ngrok.connect(port, "http", bind_tls=True)
    return str(tunnel.public_url)


def close_all_tunnels() -> None:
    """Close every active ngrok tunnel."""
    ngrok.kill()
```

No unit tests (network side-effect; covered by Task 1.8 real call).

Commit: `feat(runtime): ngrok tunnel open/close helpers`.

### Task 1.4: Twilio dialer with allowlist precheck

**Files:** `agent/runtime/dialer.py`, `tests/test_dialer.py`

```python
"""Twilio outbound dialer with hard ALLOWED_DESTINATIONS precheck."""

from __future__ import annotations

from twilio.rest import Client

from agent.safety import assert_destination_allowed


def place_call(client: Client, *, to: str, from_: str, voice_url: str) -> str:
    """Place an outbound call. Returns the Twilio call SID.

    The destination is hard-checked against ALLOWED_DESTINATIONS BEFORE any
    network call; the SDK is never invoked for a blocked number.
    """
    assert_destination_allowed(to)
    call = client.calls.create(to=to, from_=from_, url=voice_url, record=False)
    return str(call.sid)
```

Tests (mock the SDK): allowlist blocks → no SDK call; allowed → SDK called with right args; SID returned.

Commit: `feat(runtime): Twilio outbound dialer with hard allowlist precheck`.

### Task 1.5: FastAPI server with /voice + /start + /stream

**Files:** `agent/server.py`, `tests/test_server.py`

```python
"""FastAPI app: /start (place call), /voice (Twilio webhook), /stream (Pipecat WS), /health."""

from __future__ import annotations

import os

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse


def build_app(public_url: str | None = None) -> FastAPI:
    """Construct the FastAPI app. /stream WS handler is wired in Task 1.6."""
    app = FastAPI()
    base = public_url or os.environ.get("PUBLIC_URL", "wss://localhost:8000")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/voice")
    async def voice(_: Request) -> PlainTextResponse:
        ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/stream"
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Connect><Stream url="{ws_url}"/></Connect>'
            "</Response>"
        )
        return PlainTextResponse(twiml, media_type="application/xml")

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        # Wired in Task 1.6 — runs build_pipecat_pipeline.
        from agent.pipeline.builder import run_pipeline_for_call
        await run_pipeline_for_call(websocket, app)

    return app
```

Test: `/voice` returns TwiML with `<Connect><Stream>`, `/health` returns ok.

Commit: `feat(server): FastAPI with /voice TwiML, /health, /stream WS wired to Pipecat`.

### Task 1.6: build_pipecat_pipeline skeleton (transport-only echo)

**Files:** `agent/pipeline/builder.py`

This is the core of the Pipecat integration. Build a `Pipeline` with just the transport (input + output) so audio echoes back. We expand to STT + TTS + StateMachineProcessor in Phase 2 + Phase 4.

Use the canonical 0.0.108 paths verified in T1.1:
- `pipecat.transports.websocket.fastapi.FastAPIWebsocketTransport`
- `pipecat.serializers.twilio.TwilioFrameSerializer`
- `pipecat.pipeline.pipeline.Pipeline`
- `pipecat.pipeline.runner.PipelineRunner`
- `pipecat.pipeline.task.PipelineTask`

```python
"""Build and run the Pipecat pipeline for one call."""

from __future__ import annotations

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

    Reads `app.state.twilio_account_sid`, `app.state.twilio_auth_token`,
    `app.state.order` set by the CLI before serving.
    """
    # 1. Wait for Twilio's `start` event so we have stream_sid + call_sid
    start_data = await websocket.receive_json()
    while start_data.get("event") != "start":
        start_data = await websocket.receive_json()
    stream_sid = start_data["start"]["streamSid"]
    call_sid = start_data["start"]["callSid"]

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
            vad_analyzer=SileroVADAnalyzer(),  # used for barge-in in Phase 6; harmless here
            serializer=serializer,
        ),
    )

    # Phase 1: echo only — input → output.
    # Phase 2 will insert: transport.input() → STT → ... → TTS → transport.output()
    pipeline = Pipeline([transport.input(), transport.output()])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        # Pipecat's PipelineTask handles its own teardown on hangup; this is
        # belt-and-suspenders in case of exceptions.
        await task.cancel()
```

> **Lifecycle note (per senior review gap):** `PipelineRunner.run(task)` returns when the pipeline ends (call hangup signaled by transport). The `try/finally` ensures `task.cancel()` runs on any exception; Pipecat's runner handles SIGINT separately (`handle_sigint=False` because we manage our own).

> **`call_sid` plumbing (per senior review gap):** the `start` event carries `call_sid`. We extract it before pipeline creation and use it as both `TwilioFrameSerializer`'s `call_sid` AND (in Phase 4) the LangGraph `thread_id`. The `StateMachineProcessor` will read it from app.state set by the pipeline builder.

Commit: `feat(pipeline): build_pipecat_pipeline skeleton with transport-only echo`.

### Task 1.7: CLI entrypoint

**Files:** `agent/cli.py`, `agent/__main__.py`

```python
"""CLI: load order, boot server, open ngrok, place call, exit when call ends."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from twilio.rest import Client

from agent.runtime.dialer import place_call
from agent.runtime.tunnel import close_all_tunnels, open_tunnel
from agent.safety import assert_destination_allowed
from agent.server import build_app
from agent.types import Order


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent")
    p.add_argument("order_file", type=Path, help="Path to order JSON")
    p.add_argument("--to", required=True, help="Destination phone number, E.164 (+1...)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code."""
    load_dotenv()
    args = _parse_args(argv or sys.argv[1:])

    order = Order.model_validate(json.loads(args.order_file.read_text()))
    assert_destination_allowed(args.to)

    port = int(os.environ.get("PORT", 8000))
    public_url = open_tunnel(port)

    app = build_app(public_url=public_url)
    app.state.twilio_account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    app.state.twilio_auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    app.state.order = order

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    server_thread.start()

    try:
        client = Client(app.state.twilio_account_sid, app.state.twilio_auth_token)
        sid = place_call(
            client, to=args.to, from_=os.environ["TWILIO_FROM_NUMBER"],
            voice_url=f"{public_url}/voice",
        )
        print(json.dumps({"event": "call_placed", "call_sid": sid, "order": order.customer_name}))

        while server_thread.is_alive():
            server_thread.join(timeout=1)
    finally:
        close_all_tunnels()

    return 0
```

`agent/__main__.py`:
```python
"""Module entrypoint: `python -m agent.cli ...`."""

from agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Commit: `feat(cli): order.json + --to entry point that boots server, ngrok, places call`.

### Task 1.8: REAL CALL CHECKPOINT 1 — Pipecat echo

Manual checkpoint:

1. Configure `.env` with real Twilio + ngrok auth.
2. Set `ALLOWED_DESTINATIONS` to your own phone in E.164.
3. Run: `python -m agent.cli fixtures/orders/example.json --to +1YOURPHONE`
4. Pick up your phone; speak; verify you hear yourself echoed back.
5. Hang up; verify the Python process exits cleanly with no exceptions.

If audio is chipmunked, the `TwilioFrameSerializer` is misconfigured — re-check `stream_sid`/`call_sid` are pulled from the `start` event.

If WS connects but no audio: the pipeline isn't routing transport.input() → transport.output(); check `audio_in_enabled` and `audio_out_enabled` in `FastAPIWebsocketParams`.

Commit any fixes before Task 1.9.

### Task 1.9: REAL CALL — DTMF mid-stream behavior verification

Confirms Twilio's `update(twiml=<Play digits>)` REST call does NOT terminate the active `<Connect><Stream>`. Our IVR strategy depends on this; if it terminates the stream, we need a fallback.

1. Place a call as in Task 1.8.
2. While audio is flowing, from a separate terminal:
   ```python
   import os
   from twilio.rest import Client
   sid = "<paste call_placed SID>"
   c = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
   c.calls(sid).update(twiml='<?xml version="1.0"?><Response><Play digits="1"/></Response>')
   ```
3. Verify: the call hears DTMF tones; the WS keeps receiving `media` events.

If the WS disconnects (`stop` event fires and never resumes): document the failure in `docs/superpowers/dtmf-mid-stream-result.md` and consider the fallback (re-`<Connect><Stream>` after `<Play digits>`).

Commit findings either way:
```bash
git add docs/superpowers/dtmf-mid-stream-result.md
git commit -m "docs: record DTMF mid-stream Twilio behavior observation"
```

---

## Phase 2 — Pipecat services + custom DTMFProcessor

Goal: extend the transport-only echo from Phase 1 with Pipecat's prebuilt `DeepgramSTTService` and `CartesiaTTSService`, plus a custom `DTMFProcessor` that consumes our project-local `RestDTMFFrame`.

These three tasks are largely independent (each adds one processor to the pipeline). Run sequentially — they all modify `agent/pipeline/builder.py` so worktrees would conflict on the merge.

### Task 2.1: Wire `DeepgramSTTService` into the pipeline

**Files:**
- Modify: `agent/pipeline/builder.py`

Pipecat owns the WS lifecycle and audio frame routing — we just instantiate the service and put it in the `Pipeline` list.

```python
# In agent/pipeline/builder.py
import os
from pipecat.services.deepgram.stt import DeepgramSTTService

# Inside run_pipeline_for_call, after `transport = ...`:
stt = DeepgramSTTService(
    api_key=os.environ["DEEPGRAM_API_KEY"],
    model="nova-2-phonecall",
    # Note: utterance_end_ms is fixed at connect (per senior review).
    # Mode-specific endpointing (IVR=750ms vs HUMAN=1500ms) requires
    # closing+reopening the service in Phase 5/6, not configurable here.
    sample_rate=8000,  # μ-law from Twilio
)

pipeline = Pipeline([
    transport.input(),
    stt,
    transport.output(),  # echo TTS removed — STT is silent on the output side
])
```

> **Deepgram reopen note (per senior review gap):** Pipecat's `DeepgramSTTService` is constructed once per pipeline. To swap endpointing at IVR→HOLD→HUMAN transitions, we'll need to call its internal reconnect logic OR rebuild the pipeline mid-call. For now (Phase 2 + IVR-only milestone), pick `utterance_end_ms=750` (IVR-tuned per CLAUDE.md latency math); HUMAN-phase tuning lands in Phase 6. Document the reopen mechanism choice when Phase 5 ships.

Test: smoke import only (real verification is Phase 4 real call). Commit: `feat(pipeline): wire DeepgramSTTService for IVR transcripts`.

### Task 2.2: Wire `CartesiaTTSService` into the pipeline

**Files:**
- Modify: `agent/pipeline/builder.py`

Similar shape. Cartesia consumes `TTSSpeakFrame`/`TextFrame` upstream and emits audio frames downstream. Pipecat handles the rate conversion automatically (Cartesia → μ-law 8 kHz for Twilio).

```python
# Add to run_pipeline_for_call:
from pipecat.services.cartesia.tts import CartesiaTTSService

tts = CartesiaTTSService(
    api_key=os.environ["CARTESIA_API_KEY"],
    voice_id=os.environ["CARTESIA_VOICE_ID"],
    sample_rate=8000,  # output to μ-law 8 kHz for Twilio
)

pipeline = Pipeline([
    transport.input(),
    stt,
    # StateMachineProcessor lands here in Phase 4
    tts,
    transport.output(),
])
```

For now, no upstream produces `TTSSpeakFrame`s — this is a stub until Phase 4. The pipeline still runs; STT just transcribes silently. We're proving the wiring.

Commit: `feat(pipeline): wire CartesiaTTSService into the pipeline`.

### Task 2.3: Custom `DTMFProcessor` (consumes `RestDTMFFrame` → Twilio REST)

**Files:**
- Create: `agent/pipeline/dtmf_processor.py`
- Create: `tests/test_dtmf_processor.py`

This is the foot-gun avoidance: instead of synthesizing DTMF audio, this processor receives `RestDTMFFrame` (defined in Phase 1.2) and calls Twilio's `<Play digits>` REST sidechannel.

```python
"""DTMFProcessor: consumes RestDTMFFrame, sends DTMF via Twilio REST sidechannel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.pipeline.frames import RestDTMFFrame

if TYPE_CHECKING:
    from twilio.rest import Client


class DTMFProcessor(FrameProcessor):
    """Send DTMF digits via Twilio REST when a RestDTMFFrame flows through.

    The frame is consumed (not pushed downstream) since DTMF doesn't traverse
    the audio path. Other frame types pass through unchanged.
    """

    def __init__(self, twilio_client: Client, call_sid: str) -> None:
        super().__init__()
        self._twilio = twilio_client
        self._call_sid = call_sid

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Intercept RestDTMFFrame; pass everything else through."""
        await super().process_frame(frame, direction)
        if isinstance(frame, RestDTMFFrame):
            twiml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Response><Play digits="{frame.digits}"/></Response>'
            )
            self._twilio.calls(self._call_sid).update(twiml=twiml)
            # consumed — do not push downstream
            return
        await self.push_frame(frame, direction)
```

Tests (mock Twilio client + mock parent.push_frame): `RestDTMFFrame` → REST update called, NOT pushed downstream. Other frames → pushed unchanged, REST not called.

Wire into pipeline (`agent/pipeline/builder.py`):
```python
from twilio.rest import Client
from agent.pipeline.dtmf_processor import DTMFProcessor

twilio_client = Client(app.state.twilio_account_sid, app.state.twilio_auth_token)
dtmf = DTMFProcessor(twilio_client, call_sid)

pipeline = Pipeline([
    transport.input(),
    stt,
    # StateMachineProcessor lands here in Phase 4
    dtmf,        # consumes RestDTMFFrame from upstream
    tts,
    transport.output(),
])
```

Commit: `feat(pipeline): DTMFProcessor for out-of-band DTMF via Twilio REST`.

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

## Phase 4 — IVR mode (StateMachineProcessor + LangGraph IVR nodes)

Goal: real IVR turn works end-to-end through the Pipecat pipeline + StateMachineProcessor + LangGraph. Phase ends with REAL CALL CHECKPOINT 2.

### Task 4.1: `ivr_llm` node (with deterministic Prompt-5 verification)

**Files:**
- Replace stub: `agent/graph/nodes/ivr_llm.py`
- Create: `tests/test_ivr_llm_node.py`

> **Prompt-5 deterministic verification:** before invoking the LLM, if the transcript looks like a confirmation prompt with a readback ("I heard X, Y, zip code Z. Is that correct?"), regex-extract the readback fields and compare against `order.customer_name` / `order.phone_number` / `order.delivery_zip` deterministically. If any field mismatches, short-circuit to `say_short("no")` without calling the LLM. Avoids the known LLM failure mode where models lazily say "yes" on confirmation prompts.

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
    """If `transcript` is a Prompt-5 readback, return 'yes'/'no' deterministically."""
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
    confirm = deterministic_confirm_response(state["new_transcript"], state["order"])
    if confirm is not None:
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

Tests (mock chat client): forced-tool, deterministic confirm yes/no, preface-content drop. See plan section "test_ivr_llm_node.py" below or use the implementer's discretion to mirror the patterns documented in the spec.

Commit: `feat(graph/ivr): ivr_llm node — forced tool call + deterministic Prompt-5`.

### Task 4.2: `ivr_tools` node (with `ToolMessage` acks)

**Files:**
- Replace stub: `agent/graph/nodes/ivr_tools.py`
- Create: `tests/test_ivr_tools_node.py`

> **Critical:** append a `ToolMessage(content="ok", tool_call_id=call_id)` per tool call to `ivr_messages`. Without this, Cerebras (OpenAI-compatible API) will 400 on the next turn because the prior `AIMessage.tool_calls` has no matching tool response.

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
                tool_acks.append(ToolMessage(content="ok", tool_call_id=call_id))
            break

    update: dict[str, Any] = {"pending_actions": actions}
    if tool_acks:
        update["ivr_messages"] = tool_acks
    if new_mode is not None:
        update["mode"] = new_mode
    return update
```

Commit: `feat(graph/ivr): ivr_tools translates tool calls to Actions with ToolMessage acks`.

### Task 4.3: `StateMachineProcessor` (enqueue-and-drain FrameProcessor)

**Files:**
- Create: `agent/pipeline/state_machine_processor.py`
- Create: `tests/test_state_machine_processor.py`

This is the bridge between Pipecat (audio frames) and LangGraph (state machine). It lives INSIDE the Pipecat pipeline as a `FrameProcessor` subclass; the compiled graph runs OUTSIDE Pipecat.

**Design (per senior review):**
- `process_frame()` returns immediately after enqueueing a `TranscriptionFrame` — does NOT await the LLM. This avoids backpressuring upstream STT.
- A separate drain task consumes the inbound queue serially, calls `graph.ainvoke()`, and pushes outbound `Action`s as Pipecat frames.
- An `InterruptionFrame` from upstream cancels the in-flight LLM task (barge-in).
- The processor holds the `call_sid` for `thread_id` plumbing (set on the `start` event in Phase 1.6 builder).

```python
"""StateMachineProcessor: bridges Pipecat audio events to a LangGraph state machine.

The processor lives INSIDE the Pipecat pipeline as a FrameProcessor. The compiled
LangGraph graph runs OUTSIDE Pipecat — the processor calls `graph.ainvoke()` from
a drain task that consumes an inbound queue. This non-blocking pattern avoids
backpressuring upstream STT while still serializing graph invocations per call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.graph.state import (
    Action,
    AgentState,
    DTMFAction,
    EndCallAction,
    SpeakAction,
    TransitionAction,
)
from agent.pipeline.frames import RestDTMFFrame
from agent.types import Order


class StateMachineProcessor(FrameProcessor):
    """Bridge Pipecat ↔ LangGraph. Enqueue-and-drain pattern.

    - `process_frame` is non-blocking: enqueue and return so upstream isn't
      backpressured.
    - A drain task consumes the inbound queue serially, awaits
      `graph.ainvoke()`, and pushes resulting Action frames downstream.
    - On `InterruptionFrame` (barge-in), the in-flight LLM task is cancelled.
    """

    def __init__(self, *, graph: Any, order: Order, call_sid: str) -> None:
        super().__init__()
        self._graph = graph
        self._call_sid = call_sid
        self._inbound_q: asyncio.Queue[TranscriptionFrame] = asyncio.Queue()
        self._drain_task: asyncio.Task[None] | None = None
        self._llm_task: asyncio.Task[None] | None = None
        self._state: AgentState = {  # type: ignore[typeddict-item]
            "mode": "ivr",
            "order": order,
            "new_transcript": "",
            "ivr_messages": [],
            "human_messages": [],
            "pending_actions": [],
            "hangup_reason": None,
        }

    async def setup(self, *args: Any, **kwargs: Any) -> None:
        """Start the drain task when the pipeline boots."""
        await super().setup(*args, **kwargs)
        self._drain_task = asyncio.create_task(self._drain())

    async def cleanup(self) -> None:
        """Cancel drain + in-flight LLM tasks on pipeline teardown."""
        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Non-blocking dispatch. Enqueue transcripts; cancel LLM on interruption."""
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self._inbound_q.put(frame)
            return  # consumed — drain task drives the response
        if isinstance(frame, InterruptionFrame) and self._llm_task is not None:
            if not self._llm_task.done():
                self._llm_task.cancel()
        await self.push_frame(frame, direction)

    async def _drain(self) -> None:
        """Consume the inbound queue serially; run one graph turn per transcript."""
        while True:
            tframe = await self._inbound_q.get()
            self._state["new_transcript"] = tframe.text
            self._llm_task = asyncio.create_task(self._run_graph_turn())
            try:
                await self._llm_task
            except asyncio.CancelledError:
                continue  # barge-in cancelled this turn; wait for next utterance

    async def _run_graph_turn(self) -> None:
        """One graph invocation; dispatch resulting Actions as Pipecat frames."""
        config = {"configurable": {"thread_id": self._call_sid}}
        result: AgentState = await self._graph.ainvoke(self._state, config=config)
        self._state = result
        for action in result.get("pending_actions", []):
            await self._dispatch(action)
        self._state["pending_actions"] = []

    async def _dispatch(self, action: Action) -> None:
        """Translate an Action to a Pipecat frame and push downstream."""
        if isinstance(action, DTMFAction):
            await self.push_frame(RestDTMFFrame(digits=action.digits), FrameDirection.DOWNSTREAM)
        elif isinstance(action, SpeakAction):
            await self.push_frame(TTSSpeakFrame(text=action.text), FrameDirection.DOWNSTREAM)
        elif isinstance(action, TransitionAction):
            # Mode transition recorded in self._state; no outbound frame.
            pass
        elif isinstance(action, EndCallAction):
            # Outbound JSON + Twilio hangup wired in Phase 7 done_node.
            pass
```

Tests (mock the graph; assert process_frame returns fast, drain serializes, LLM cancellation on InterruptionFrame, RestDTMFFrame pushed for DTMFAction).

Commit: `feat(pipeline): StateMachineProcessor — non-blocking bridge to LangGraph`.

### Task 4.4: Wire `StateMachineProcessor` into the Pipecat pipeline

**Files:**
- Modify: `agent/pipeline/builder.py`

```python
# In agent/pipeline/builder.py (extending Phase 2):
from agent.graph.builder import build_graph
from agent.pipeline.state_machine_processor import StateMachineProcessor

graph = build_graph()  # compiled with MemorySaver checkpointer
state_machine = StateMachineProcessor(
    graph=graph,
    order=app.state.order,
    call_sid=call_sid,
)

pipeline = Pipeline([
    transport.input(),
    stt,
    state_machine,   # bridge to LangGraph
    dtmf,            # DTMFProcessor consumes RestDTMFFrame
    tts,             # CartesiaTTSService consumes TTSSpeakFrame
    transport.output(),
])
```

Commit: `feat(pipeline): wire StateMachineProcessor between STT and DTMF/TTS`.

### Task 4.5: REAL CALL CHECKPOINT 2 — IVR navigation works end-to-end

Manual checkpoint:

1. Place a call: `python -m agent.cli fixtures/orders/example.json --to +1YOURPHONE`
2. Read the IVR script in robotic monotone:
   - "Thank you for calling. Press 1 for delivery..." → expect DTMF `1`
   - "Please say the name for the order." → expect agent says "Jordan Mitchell"
   - "Please enter your 10-digit callback number." → expect DTMF `5125550147`
   - "Please say your delivery zip code." → expect agent says "78745"
   - "I heard Jordan Mitchell, 5125550147, zip code 78745. Is that correct?" → expect "yes"
   - "Got it. Please hold..." → expect agent goes silent (mode → hold)
3. Verify:
   - Each IVR response is exactly one tool dispatch (no extra speech)
   - DTMF tones audibly reach the call (confirm mid-stream behavior from T1.9)
   - Mode transition logged when "please hold" matched
   - No exceptions on hangup
4. If Prompt-5 readback fails (synthetic_no fires when readback mismatches): verify the readback was actually wrong; the deterministic verification is working.

Commit any fixes before moving to Phase 5.

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

In `StateMachineProcessor` (the Pipecat `FrameProcessor` subclass), during HOLD mode: feed audio frames to the energy gate (likely as a separate processor upstream of STT, OR by intercepting `AudioRawFrame` in `process_frame` before enqueueing). Only forward transcripts to the inbound queue when the gate has tripped.

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
