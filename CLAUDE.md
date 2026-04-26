# CLAUDE.md — Pizza Voice Agent

Guidance for Claude Code when working in this repository.

## Project

A voice agent that places an outbound phone call and orders pizza for delivery. Spec: `build_day.md`. Design: `docs/superpowers/specs/2026-04-26-pizza-voice-agent-design.md`.

**Architecture:** hybrid — deterministic Python IVR + OpenAI Realtime API for the human conversation phase. Hand-rolled FastAPI + websockets + SDKs (no Pipecat, no LangGraph). Single Python process per call. Langfuse for observability.

**Modes:** `IVR → HOLD → HUMAN → DONE`. Unidirectional transitions only.

**Speed posture:** this is a one-day build. Optimize for time-to-first-real-call. Parallelize independent work. Use git worktrees for any independent feature stream so multiple chunks can progress simultaneously without branch-switching overhead. Heavyweight review skills are session-end gates, not per-commit gates — see workflow below.

## Software best practices

- **Single responsibility per module.** Each file owns one purpose. If a file grows past ~250 lines or you can't describe its job in one sentence, split it.
- **Clear interface boundaries.** Modules communicate through typed dataclasses / pydantic models, never shared mutable globals. Audio frames are bytes; everything else is a typed object.
- **No silent failures.** Every `except` either re-raises, transitions to a hangup outcome, or logs to Langfuse with full context. Bare `except: pass` is forbidden.
- **No retries inside the audio loop.** Retries belong at the controller level only. The audio path stays low-latency.
- **`try/finally` around every long-lived resource** (Realtime session, ngrok tunnel, Twilio call, Deepgram WS). Cancellation must always close the call and emit partial result JSON.
- **Use real cancellation (`asyncio.Task.cancel()`), not boolean flags.** See "Cancellation discipline" below for what makes a handler cancellation-safe.
- **Structured outputs over free-text parsing.** Realtime tool calls record state. Never parse prices/items/totals out of free-form model speech.
- **Trust framework guarantees at internal boundaries.** Don't add input validation between our own modules. Validate at system boundaries: CLI input, Twilio webhooks, external API responses.
- **Type-annotate everything.** Every function parameter, every return type, every class attribute. Use `from __future__ import annotations` so forward references are free. Run `pyright` (or at minimum `python -m mypy --strict` on each module) before commit; type errors fail the commit.
- **Docstring every module and every public symbol.** See "Documentation style" below for the required format.
- **Inline comments only when the *why* is non-obvious** — a constraint, an invariant, a workaround for a specific bug. Names + types + docstrings cover the *what*; inline comments are reserved for the *why*.
- **Don't add features beyond what the task requires.** No speculative abstractions, no future-proofing, no half-finished implementations.

## Documentation style

Every Python module gets a docstring; every public symbol gets a docstring. Keep them short — readability is the goal, not word count.

### Module docstring (top of every `.py` file)

```python
"""<one-line purpose>.

<2-4 line description: what this module owns, what it depends on,
what consumes it. No usage examples here — those go in the design doc.>
"""
```

### Class docstring

```python
class CallController:
    """Owns the per-call lifecycle and mode transitions.

    State: current Mode, the active sub-handler (IVR/Hold/Human), and a
    reference to the result builder. Receives audio frames from the
    Twilio WebSocket; routes them to the active sub-handler; emits
    DTMF intents and outbound audio.
    """
```

### Function docstring

Use a compact form. Args/Returns only when the type alone doesn't tell the story.

```python
def classify(asr_text: str) -> PromptType:
    """Map a finalized IVR ASR transcript to a PromptType enum.

    Returns PromptType.UNKNOWN if no rule matches; the controller then
    waits for the next utterance rather than guessing.
    """
```

For dataclasses / pydantic models, document the *role* of the type, not each field — fields explain themselves through names and types.

### What docstrings should NOT contain

- The current task or PR number ("added for the order-flow feature").
- Caller lists ("called from server.py:45").
- WHAT the function does in prose when names already say it.
- Multi-paragraph essays. If you need that, write a section in the design doc and link it.

### Tooling

- `ruff` with `D` (pydocstyle) enabled — fails CI/local check if a public symbol is missing a docstring.
- `pyright` strict mode — fails on missing annotations or `Any` leaks.

## Cancellation discipline

A handler is cancellation-safe iff:
1. Every `await` site is acceptable to raise `CancelledError` from. No mutable mid-flight state that would be left inconsistent if the next `await` raised.
2. Network sends are atomic per frame — a partially-sent JSON envelope corrupts the WebSocket. Wrap multi-step sends so cancellation either completes or sends nothing.
3. `try/finally` releases the resource. `asyncio.shield()` is used **only** on the close/finalize path so cleanup itself can't be interrupted.
4. `CancelledError` is re-raised after cleanup. Never swallowed.
5. Outer task awaits are bounded by `asyncio.timeout()` — Deepgram and Realtime calls must not hang forever.

## Foot-guns specific to this project

These are the bugs that bite hardest in real-time voice work. Every one is in the design spec; the rules below are non-negotiable.

- **DTMF is out-of-band.** Twilio Media Streams will not deliver DTMF inside the WS audio. Use `twilio.calls(sid).update(send_digits=...)` REST side-channel. **Never** synthesize DTMF tones into the audio stream.
- **μ-law 8 kHz ↔ PCM16 24 kHz resampling must be exact.** Off-by-one rate mismatch produces chipmunk audio that sounds like a bot — straight to `detected_as_bot`. Use `audioop` or `scipy.signal.resample_poly` and round-trip-test it.
- **Realtime reconnect is one attempt only.** On WS close mid-call: try once with last ~3 s of transcript prepended as `conversation.item.create role=user`. If reconnect fails, `outcome="disconnect"` and hang up. Don't retry-loop — that masks real failures.
- **ngrok URL changes per restart.** Emit it on boot, log it, validate Twilio webhook reaches it (`/voice` returns 200) **before** placing the call.
- **`ALLOWED_DESTINATIONS` precheck.** Every `calls.create()` passes through a hard env-var allowlist. Five lines of code; prevents the worst possible bug (calling a real number while developing).
- **Secret hygiene in logs.** Redact API keys, bearer tokens, and customer phone numbers in the JSON log formatter. Langfuse spans must not capture raw `Authorization` headers. Test the redactor.
- **Deepgram scope.** Deepgram is used for IVR + hold only. The HUMAN phase uses OpenAI Realtime, which handles its own ASR/TTS — do not double-up.

## Testing & verification

### Component unit tests (write inline as modules are built)

| Test file | Concrete assertions |
|---|---|
| `test_ivr_classifier.py` | ≥3 phrasings per `PromptType`, including transcription noise (lowercase, missing punctuation, partial words) |
| `test_responder.py` | Each `PromptType` yields the correct *intent type* (`DTMFIntent` vs `SpeechIntent`) and value, given a sample `order.json` |
| `test_result_builder.py` | One test per outcome (`completed`, `nothing_available`, `over_budget`, `detected_as_bot`, plus the 5 operational outcomes); each round-trips through pydantic output schema |
| `test_persona_banned_phrases.py` | Greps agent output for "let me check", "I'll note that down", "as an AI", "I'll get back to", "I'm checking that" — fail if any appear |
| `test_order_schema.py` | pydantic validation: spec example passes; 3 malformed variants fail with informative errors |
| `test_audio_resample.py` | Round-trip μ-law 8 kHz ↔ PCM16 24 kHz, assert energy preserved within tolerance, no clipping |

### Verification rules

- **`pytest` runs locally before every real call.** No commits, no PRs, no claims of "it works" without `pytest` passing.
- **Real call testing** is reserved for end-to-end behavior: barge-in feel, latency, IVR navigation, hold detection, full happy-path script. Twilio paid account required.
- **Verification before any completion claim.** Never say "this works" / "this is fixed" / "tests pass" without running the verification command and confirming output. Evidence before assertions, always.

## Observability

Concrete schema. Don't wing it.

### Langfuse

- **One trace per call**, named `call_<call_sid>`. Tags: `outcome`, `mode_durations` summary at trace end.
- **One span per mode** (`ivr`, `hold`, `human`). Span starts on mode entry, ends on transition.
- **Span events** (within each mode):
  - IVR: `prompt_classified` (`{prompt_type, asr_text, confidence}`), `intent_emitted` (`{kind: dtmf|speech, value, latency_ms}`), `transition_to` (`{next_mode, reason}`)
  - HOLD: `energy_gate_trip`, `classifier_call` (`{utterance, is_human, latency_ms}`), `transition_to`
  - HUMAN: `realtime_session_open`, `tool_call` (`{name, args, latency_ms}`), `realtime_session_close`, `transition_to`

### JSON logs

- One event per line. No multi-line records.
- Required fields on every line: `{ts, call_sid, mode, event, ...event-specific fields...}`.
- Stream to stdout in dev; tail with `jq` during real calls.
- Redact: `api_key`, `authorization`, `bearer`, `to_phone`, `from_phone` (replace digits 4–7 with `*`).

## Per-change workflow

For every meaningful chunk of work (a feature, a fix, a refactor), follow this sequence in order:

1. **Implement** the change. TDD where it fits (write the test first, watch it fail, make it pass).
2. **Run `pytest`, `ruff check`, and `pyright` locally.** All tests pass, no lint errors, no type errors. Annotations and docstrings are required to land.
3. **Run the `simplify` skill** on the working changes. Review for reuse, quality, and efficiency; clean up before the commit lands.
4. **Run the `superpowers:verification-before-completion` skill.** Confirm via verification commands that the change actually works end-to-end. No success claim until verification passes.
5. **Run the `pr-review-toolkit:code-reviewer` skill** on the staged diff *if this chunk is significant* (a full feature, a major refactor, or anything touching the audio path). For small fixes inside an existing module, skip this in-loop and let the session-end review catch it.
6. **Commit** the cleaned-up, verified, reviewed state to git. Small, focused commits with descriptive messages. Co-Authored-By Claude when applicable.
7. **Push.** Push to GitHub. Survives laptop death and gives rollback.

### Session-end / pre-merge workflow (run once per PR, not per commit)

8. **Create the PR** with `gh pr create`. Title under 70 chars; body has Summary + Test plan.
9. **Run the `pr-review-toolkit:review-pr` skill** on the open PR for comprehensive multi-agent review. Address findings before requesting human review or merging.

If a step surfaces a problem, fix it before moving on. Don't batch up "I'll address review later" — the workflow exists so problems get caught at the cheapest stage.

### Why verification + code-reviewer run BEFORE commit

A senior-engineer review of an earlier draft of this guide flagged that running them after commit guarantees fix-up commits, contradicting "small focused commits" and "no WIP merged to main." Pre-commit is the right gate — the cost of catching issues there is minutes; post-merge is hours.

## Parallelization & worktrees

This is a single-process voice agent at runtime, but development work splits cleanly across modules. Use these to keep multiple things moving at once:

- **Use the `superpowers:using-git-worktrees` skill** before starting any feature stream that's independent of the current workspace. Examples: building `ivr/` while another stream wires up `human/realtime_bridge.py`; iterating on the persona prompt while another stream improves the result builder.
- **Use the `superpowers:dispatching-parallel-agents` skill** for 2+ independent build tasks. Independent = no shared mutable state, no sequential dependency. Examples: writing the 6 component test files in parallel; drafting `audio/transport.py` and `audio/dtmf.py` in parallel.
- **Default to parallel tool calls** within a single Claude turn when reads/searches don't depend on each other.
- **Treat speed as a first-class constraint.** A correct change that ships in 2 hours beats a perfect change that ships in 6.

## Commit & PR conventions

- **Commit messages:** focus on *why*, not *what*. The diff shows what.
- **Commit boundaries:** one logical change per commit. No "WIP" or "fixup" merged to main.
- **PR titles:** under 70 characters. Use the body for detail.
- **PR body:** Summary (1–3 bullets) + Test plan (markdown checklist).
- **Never `--no-verify`** to bypass hooks. Never `--force` to main without explicit user authorization.

## What not to do

- Don't reach for a framework (LangGraph, Pipecat, LiveKit Agents) — design rejected these for this project. Hand-rolled.
- Don't put LLM calls inside the audio frame loop or VAD callback. Latency tanks.
- Don't conflate IVR DTMF entry with speech (different transports — REST API vs audio).
- Don't model the whole call as one big LLM prompt. The IVR phase is deterministic; the conversation is open-ended; one model can't serve both.
- Don't skip observability "for now." Every IVR classification, every state transition, every Realtime tool call goes to JSON logs + Langfuse per the schema above.
- Don't try to handle every edge case before one end-to-end real-call test. Happy path first, escape hatches second.
- Don't commit secrets. Twilio/OpenAI/Deepgram/Cartesia/Langfuse keys live in `.env`, which is gitignored.
- Don't run heavyweight review skills on every commit. They are gates at session boundaries, not per-line linters.
