# Pizza Voice Agent — Design

**Date:** 2026-04-26
**Status:** Design approved, pending plan
**Scope:** One-day solo build (~6–10 hours). Goal: happy-path call works end-to-end on a real phone.

## Problem

Build a voice agent that places an outbound phone call, navigates an automated phone menu (IVR), waits on hold, then converses with a human employee to place a pizza order per a JSON spec. The agent prints a structured JSON result when the call ends. Full task description: `build_day.md`.

## Decisions Recap

| Decision | Choice | Rationale |
|---|---|---|
| Timeline | One day solo, ~6–10h | Build-day event; happy path is success |
| API access | Full kit (OpenAI Realtime + small classifier model, Twilio, Deepgram, Cartesia, Langfuse) | Eshan provisions on request |
| Architecture | Hybrid — DIY Python IVR + OpenAI Realtime API for human phase | IVR is deterministic and unforgiving; conversation is open-ended. Mixing them in one model is wrong on cost, latency, and reliability. |
| Pipeline framework | Hand-rolled (FastAPI + websockets + SDKs) | ~5 modes, one Realtime session/call. Framework setup time competes with shipping. Rejected: Pipecat (pre-1.0 churn), LiveKit Agents (Twilio bridge overhead), LangGraph (audio is not its primitive; cancellation paradigm mismatch). |
| Hold detection | Energy + duration gate, then on trip transcribe via Deepgram (reused) and classify via small LLM (gpt-4o-mini or claude-haiku) | Robust to silence/music/human-voice without expensive LLM polling. Reuses the Deepgram session already open for IVR — no third transcription provider. |
| Invocation | `python agent.py order.json --to <num>` (FastAPI + ngrok in same process) | One command per test iteration |
| Observability | Structured JSON logs + Langfuse self-hosted | Voice calls can't be debugged after the fact without traces |
| Testing | Component pytest + real-call iteration loop on paid Twilio. **No** local audio fixture replay (skipped — real calls are cheap on paid account). | Skip fixture-replay infrastructure; commit constantly to git. Skip GitHub Actions CI. |

## Architecture

The agent is a single Python process that wears three hats during one outbound call. Mode transitions are unidirectional and explicit: `IVR → HOLD → HUMAN → DONE`. No backwards transitions for v1.

```
                   ┌───────────────────────────────────────────────────────┐
                   │                   agent.py (FastAPI)                  │
                   │                                                       │
  CLI ──order.json─►  /start  ──places call──┐                             │
                   │                          │                             │
                   │            ┌──── Twilio ◄┘                             │
                   │            │             ▲                             │
                   │            │  /voice ────┘ (TwiML → Media Streams URL) │
                   │            ▼                                           │
                   │      WS /stream  (μ-law 8kHz audio frames)             │
                   │            │                                           │
                   │            ▼                                           │
                   │   ┌──────────────────┐                                 │
                   │   │  Call Controller │                                 │
                   │   │  (state machine) │                                 │
                   │   └──────┬───────────┘                                 │
                   │          │                                             │
                   │          ▼                                             │
                   │  ┌────────────────┬─────────────┬──────────────┐       │
                   │  │ IVR Navigator  │  Hold Watch │ Human Phase  │       │
                   │  │ (deterministic)│ (gate+LLM)  │ (Realtime API│       │
                   │  │                │             │  bridge)     │       │
                   │  └────────────────┴─────────────┴──────────────┘       │
                   │          │             │              │                │
                   │          ▼             ▼              ▼                │
                   │       Twilio REST  Deepgram +    OpenAI Realtime       │
                   │       (DTMF +TTS)  small LLM     (full duplex audio)   │
                   │                    classifier                          │
                   └───────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                       Langfuse + JSON logs
```

**Why three sub-handlers and not one big LLM:** the IVR phase is unforgiving and benefits from deterministic Python with regex/keyword classification. The hold phase is mostly waiting. The human phase is open-ended dialogue. One LLM call cannot serve all three well. Forcing the IVR through Realtime is wrong on cost (~$0.30/min for what should be free), reliability (the model wants to converse, not say "yes"), and latency.

## Components & Interfaces

```
agent/
├── cli.py                  # entry point: parses args, boots server, places call
├── server.py               # FastAPI: /start, /voice, ws /stream
├── controller.py           # CallController: state machine + per-call lifecycle
├── ivr/
│   ├── classifier.py       # regex/keyword → IVR prompt type
│   └── responder.py        # emits DTMF intent or short TTS phrase
├── hold/
│   └── watcher.py          # energy gate + Whisper-tiny classifier
├── human/
│   ├── realtime_bridge.py  # OpenAI Realtime WS bridge
│   ├── persona.py          # system prompt + banned-phrase regex
│   └── tools.py            # Realtime function-calling tool schemas
├── audio/
│   ├── transport.py        # μ-law 8kHz ↔ PCM 24kHz resampling
│   └── dtmf.py             # Twilio REST sendDigits side-channel
├── result/
│   └── builder.py          # accumulator → final structured JSON output
└── obs/
    └── trace.py            # Langfuse + JSON-line logger
```

### Interface contracts

| Module | Inputs | Outputs |
|---|---|---|
| `CallController` | mode, audio frames in, ASR text | mode transitions, audio frames out, DTMF intents |
| `ivr.classifier` | finalized ASR text | `PromptType` enum: `MENU`, `NAME`, `PHONE`, `ZIP`, `CONFIRM`, `TRANSFER`, `UNKNOWN` |
| `ivr.responder` | `PromptType`, order data | `DTMFIntent("1")` or `SpeechIntent("Jordan Mitchell")` |
| `hold.watcher` | audio frames | `HoldEvent.HUMAN_DETECTED` (one-shot) |
| `human.realtime_bridge` | audio frames in, tool registry | audio frames out, structured tool calls |
| `result.builder` | tool calls + state transitions | final JSON dict matching spec output schema |

### Two non-obvious choices

1. **`ivr.responder` emits intent objects, not actions.** It returns `DTMFIntent("1")`; the controller decides whether to send via Twilio REST or speak via TTS. The brief's "DTMF is a separate channel from TTS" principle. Keeps the IVR layer pure and testable.
2. **`result.builder` is its own module, populated incrementally.** Every Realtime tool call appends here, not to scattered state. When any hangup condition fires, `builder.finalize()` produces the final JSON regardless of outcome. Avoids the brief's "easy to miss per-item price tracking" foot-gun by making per-item capture a tool-call obligation rather than a free-text parse.

### Realtime tools (initial set)

The Realtime model's only way to record state or end the call is through these tools. No free-text "I'll hang up now" — tool-call only.

```python
record_pizza_quote(description: str, price: float, substitutions: dict[str, str])
record_side_quote(description: str, price: float, original: str)
record_drink_quote(description: str, price: float)
skip_drink(reason: Literal["over_budget", "unavailable"])
record_total(amount: float)
record_delivery_time(text: str)
record_order_number(text: str)
mark_special_instructions_delivered()
end_call(outcome: Literal["completed", "nothing_available", "over_budget", "detected_as_bot"], note: str | None = None)
```

## Data Flow (one successful call)

```
T+0.0s   CLI loads order.json, validates against pydantic schema
T+0.1s   FastAPI starts on port 8000; ngrok tunnel opens
T+0.5s   POST /start → CallController instantiated, Twilio call placed
            twilio.calls.create(to=+1512..., from_=TWILIO_NUM, url=NGROK/voice)
T+2-4s   Twilio dials user's phone; user picks up
T+~4s    Twilio fetches /voice → returns TwiML <Connect><Stream url=NGROK/stream>
T+~5s    WebSocket /stream connects, Media Streams audio frames begin flowing
            (μ-law 8kHz, 20ms frames, base64 in JSON, both directions)

         ─── MODE: IVR ─────────────────────────────────────────────────────

T+~5s    Controller emits Mode.IVR
         Inbound audio → Deepgram streaming ASR (utterance_end_ms=400)
         IVR responder maps PromptType → action:
            MENU       → DTMFIntent("1")    via Twilio REST sendDigits
            NAME       → SpeechIntent("Jordan Mitchell")    via Cartesia TTS
            PHONE      → DTMFIntent("5125550147")
            ZIP        → SpeechIntent("78745")
            CONFIRM    → SpeechIntent("yes") (or "no" → reset to NAME)
            TRANSFER   → mode transition

T+~25s   "Got it. Please hold..." matched → Mode.HOLD

         ─── MODE: HOLD ────────────────────────────────────────────────────

         Energy gate (RMS over 300ms windows) suppresses music/silence
         On utterance candidate: Deepgram-transcribed text → small LLM
         classifier ("is this a human greeting?")
         If yes → Mode.HUMAN  [classifier latency ~300-500ms]

         ─── MODE: HUMAN ───────────────────────────────────────────────────

T+~30s   Open OpenAI Realtime WebSocket session
            session.update with system prompt + tool schemas + voice config
         Bridge audio:
            Twilio inbound (μ-law 8kHz) → resample to PCM16 24kHz → Realtime
            Realtime output (PCM16 24kHz) → resample to μ-law 8kHz → Twilio
         Barge-in: Realtime's server-side VAD cancels in-flight TTS within ~150ms

         Per turn: Realtime emits transcript + agent_speech + tool_call events
         Tool calls → result.builder
         Persona enforces no robotic phrases (banned-phrase regex test)

T+~3min  Tool: end_call(outcome="completed")
         → Mode.DONE; Realtime session closed; TwiML <Hangup>; call ends

T+~3min  result.builder.finalize() → JSON to stdout
         Server shuts down, ngrok closes, process exits
```

### Three places this can interact badly (mitigations baked in)

1. **IVR Prompt 5 misheard.** If the readback contains a wrong digit/word vs. the order data, classifier emits "no" and we reset to Mode.IVR Prompt 2 (NAME). Bounded retries — three "no"s and we abort.
2. **Hold ends with employee mid-sentence.** When transitioning to HUMAN, the most recent ~3 seconds of transcribed audio gets prepended to the Realtime session as `conversation.item.create role=user` so the model knows it just heard a greeting.
3. **Realtime tool call without preceding info.** If `record_total` fires but `record_pizza_quote` was never called, builder finalize switches outcome to `nothing_available`. Tools are advisory; builder is the source of truth.

## Error Handling & Hangup Conditions

### Spec outcomes

| Outcome | Trigger | Builder emits |
|---|---|---|
| `completed` | `end_call(outcome="completed")` AND builder validates: pizza + side (or skipped) + total + delivery_time + order_number + special_instructions_delivered=True | Full JSON per spec |
| `nothing_available` | `end_call(outcome="nothing_available")`, OR builder finalize finds pizza missing | Whatever was collected, `pizza: null` if applicable |
| `over_budget` | `end_call(outcome="over_budget")` after pizza+side already exceed `budget_max` | Captured items, `drink: null`, `drink_skip_reason: "over_budget"` |
| `detected_as_bot` | `end_call(outcome="detected_as_bot")` (persona prompt: "if asked if you're a bot, apologize, hang up — never argue") | Whatever's captured before suspicion |

### Operational outcomes (extensions beyond spec)

| Outcome | Trigger |
|---|---|
| `no_answer` | Twilio CallStatus webhook = `no-answer`/`busy`/`failed` |
| `ivr_failed` | Deepgram WS error after one retry |
| `ivr_rejected` | Same prompt classified 3× consecutively (legacy IVR's hangup response) |
| `hold_timeout` | No human detected after 5 minutes on hold |
| `disconnect` | Realtime WS closes mid-call and reconnect fails |

These appear in the output JSON's `outcome` field. Diagnostic over spec-strict — debugging a failed grader call is much easier with specific outcomes.

### Operational failures (recovery paths)

| Failure | Detection | Response |
|---|---|---|
| Twilio outbound dial fails | `calls.create()` raises | Print actionable error, exit 1, no JSON |
| Caller doesn't pick up | CallStatus webhook | `outcome: "no_answer"` |
| ASR provider timeout/error during IVR | Deepgram WS errors | One retry on the same prompt, then `outcome: "ivr_failed"` |
| Realtime session disconnects mid-call | WS close event | One reconnect attempt with prepended transcript context; if fails, `outcome: "disconnect"` |
| Cartesia TTS fails during IVR | API error during SpeechIntent | Fallback to Twilio's built-in `<Say>`; log warning |
| `order.json` validation fails on startup | pydantic raises | Exit 1 before any call placed |
| Outbound destination not on `ALLOWED_DESTINATIONS` env var | hard precheck | Exit 1 with clear message — brief flagged this as the worst-possible bug |

### General error-handling principles

- **No silent failures.** Every `except` either re-raises, transitions to a hangup outcome, or logs to Langfuse with full context.
- **No retries inside the audio loop.** Retries are at the controller level only. Audio path stays low-latency.
- **`try/finally`** around any in-flight task that holds a Realtime session, ngrok tunnel, or Twilio call. Cancellation must always close the call and dump partial result JSON.

## Testing

### Component unit tests (write inline with implementation)

| Test file | Covers |
|---|---|
| `test_ivr_classifier.py` | 6 PromptType cases × 3 phrasings each |
| `test_responder.py` | Each PromptType maps to correct DTMF/Speech intent given a sample order.json |
| `test_result_builder.py` | Construct builder, append tool calls, assert finalize() output matches spec; one test per outcome |
| `test_persona_banned_phrases.py` | Regex over corpus of "bot-sounding" phrases ("let me check", "I'll note that", etc.) — fail if any appear in agent output |
| `test_order_schema.py` | pydantic validation over spec example + 3 malformed variants |

### Real call testing

Reserve real calls for what only they can verify:
1. End-to-end first call (dialing, IVR navigation, hold detection, human pickup)
2. Barge-in feel and latency
3. Spec test script (build_day.md happy path)

**Twilio paid account required** — trial-account quirks (verified destinations only, prepended voicemail, DTMF timing) are blockers per the brief.

### Explicitly out of scope

- Local audio fixture replay (decided to skip — paid Twilio makes real-call iteration cheap enough)
- Different telephony providers
- Different LLM providers
- Real hold music against the energy gate (test setup uses 5–10s silence)
- Concurrent calls
- International numbers / non-US zip codes
- GitHub Actions / CI (commit constantly, run pytest locally before each real call iteration)

## Output Schema (per spec)

```json
{
  "outcome": "completed | nothing_available | over_budget | detected_as_bot | no_answer | ivr_failed | ivr_rejected | hold_timeout | disconnect",
  "pizza": { "description": "...", "substitutions": {...}, "price": 18.50 },
  "side": { "description": "...", "original": "...", "price": 6.99 } | null,
  "drink": { "description": "...", "price": 3.49 } | null,
  "drink_skip_reason": "over_budget" | "unavailable" | null,
  "total": 28.98,
  "delivery_time": "35 minutes",
  "order_number": "4412",
  "special_instructions_delivered": true
}
```

For non-`completed` outcomes, fields populate with whatever was captured before the call ended; missing fields are `null`.

## Open Questions / Out-of-Scope for v1

None gating the build. Items below are explicit non-goals for the day:
- Multi-call orchestration (one call at a time)
- Persisting call history to disk for replay (Langfuse traces cover post-mortem)
- Configurable persona / voice per order
- Non-English IVR or non-US phone formatting
