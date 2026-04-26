# Pizza Voice Agent — Design

**Date:** 2026-04-26 (architecture revised)
**Status:** Design approved, implementation plan pending
**Scope:** Build a voice agent that places an outbound phone call, navigates an automated phone menu (IVR), waits on hold, then orders a pizza per a JSON spec by speaking with a human employee. Emits a structured JSON result. Full task: `build_day.md`. The solution must match or surpass build_day.md's acceptable-output schema.

## Problem and constraints

Real-time voice work has hard latency floors. Each turn must feel natural (~600–900 ms from user speech-end to agent speech-start). The IVR phase is unforgiving — wrong word, wrong format, repeats or hangs up. The human phase is open-ended dialogue with substitutions, prices, and budget enforcement. We need full observability and per-LLM-call inspection — black-box realtime APIs don't satisfy this.

## Decisions Recap

| Decision | Choice | Rationale |
|---|---|---|
| Approach | Custom Pipecat-style audio pipeline + LangGraph state machine | Full control over latency, cost, observability. No black-box realtime API. |
| State machine | LangGraph (`StateGraph` with checkpointer) | Architecture is now node-shaped (per-utterance LLM calls, conditional routing, tool dispatch). Earlier "no LangGraph" decision was tied to the realtime-API design — that constraint is gone. |
| ASR (IVR + HOLD) | Deepgram streaming, μ-law 8 kHz | `nova-2-phonecall` model; `utterance_end_ms` configurable per phase |
| TTS (IVR + HUMAN) | Cartesia, streaming for HUMAN, buffered for IVR | Streaming for natural HUMAN speech; buffered for tiny IVR utterances |
| VAD | Silero VAD on inbound audio | Barge-in detection for HUMAN-phase TTS cancellation |
| LLMs | **gpt-oss-120B on Cerebras** for ivr_llm + human_llm + done_node extraction; **Llama 3.1 8B on Groq** for hold_node classifier | Researched 2026-04-26 against current open-source landscape. gpt-oss-120B leads TauBench (matches o4-mini), runs at ~3000 t/s on Cerebras. Llama 3.1 8B at 1000+ t/s on Groq is right-sized for binary classification. |
| Telephony | Twilio Media Streams over WebSocket | Same as before; paid account required for serious testing |
| Invocation | `python -m agent.cli order.json --to <num>` (FastAPI + ngrok in same process) | One command per test iteration |
| Observability | Structured JSON logs + Langfuse self-hosted | Per-LLM-call traces; one trace per call_sid |
| Output state capture | **Post-call extraction** in `done_node` | Tool calls only in IVR; HUMAN-phase LLM speaks freely; structured extraction happens after the call |

## Architecture

### High-level

```
   Twilio Media Streams (μ-law 8 kHz, 20 ms frames)
                  │
                  ▼
   ┌──────────────────────────────────────────┐
   │  Audio Pipeline (Pipecat-style, hand-rolled)
   │
   │   InboundFrame ──┬──► Deepgram ASR ─────► finalized_transcript
   │                  │
   │                  └──► Silero VAD ─────► barge_in event
   │
   │   OutboundFrame ◄── Cartesia TTS ◄──── tts_text_stream
   └──────────────────┬───────────────────────┘
                      │ events: TRANSCRIPT, BARGE_IN, START, STOP
                      ▼
   ┌──────────────────────────────────────────┐
   │  Adapter (StateProcessor)
   │
   │   - Receives audio events
   │   - Owns task interruption: cancel TTS task on barge_in
   │   - Calls graph.ainvoke({...new_transcript}) per event
   │   - Routes graph outputs to the audio pipeline:
   │     • DTMFAction        → Twilio REST <Play digits>
   │     • SpeakAction (buf) → Cartesia synth → frame queue
   │     • SpeakAction (str) → Cartesia stream → frame queue
   │     • EndCallAction     → drain TTS, then Twilio hangup
   │   - Hold-timeout watchdog (5 min) → mode="done", outcome=HOLD_TIMEOUT
   └──────────────────┬───────────────────────┘
                      │ graph.ainvoke(thread_id=call_sid, ...)
                      ▼
   ┌──────────────────────────────────────────┐
   │  LangGraph compiled graph + checkpointer (MemorySaver)
   │
   │      ┌─────── router ───────┐
   │      │                      │
   │      ▼                      ▼
   │   ivr_llm              hold_node      ──┐
   │      │                                  │
   │      ▼                                  │ (per-mode subgraph)
   │   ivr_tools                             │
   │      │                                  │
   │      ▼                                  │
   │     END                                ─┘
   │
   │   human_llm  ─────────────► END
   │   done_node  ─────────────► END
   └──────────────────────────────────────────┘
```

### State

```python
class AgentState(TypedDict):
    mode: Literal["ivr", "hold", "human", "done"]
    order: Order                            # input, never mutated
    new_transcript: str                     # the utterance to process this turn
    ivr_messages: Annotated[list[BaseMessage], add_messages]
    human_messages: Annotated[list[BaseMessage], add_messages]
    pending_actions: list[Action]           # graph output → adapter dispatches
    result: ResultBuilderState              # accumulator (currently empty in IVR/HOLD/HUMAN; populated by done_node extraction)
    hangup_reason: Outcome | None           # set by adapter on timeouts/errors
```

`Action` is a tagged union: `DTMFAction(digits)`, `SpeakAction(text, streaming: bool)`, `TransitionAction(target_mode)`, `EndCallAction(outcome, json)`.

### Nodes

| Node | LLM? | Provider | Job |
|---|---|---|---|
| `router` | – | – | Reads `state.mode`; picks next node |
| `ivr_llm` | gpt-oss-120B | Cerebras | Forced tool calls only (`tool_choice="required"`); IVR system prompt; IVR-scoped messages |
| `ivr_tools` | – | – | Translates tool call → Action; may set `mode="hold"` |
| `hold_node` | Llama 3.1 8B | Groq | Binary "is this transcript a human greeting?" classifier; on yes, transitions to HUMAN and seeds `human_messages` |
| `human_llm` | gpt-oss-120B | Cerebras | Natural-text response (streamed to TTS); only `end_call` tool available |
| `done_node` | gpt-oss-120B | Cerebras | Structured-output extraction over `human_messages` → `CallResult` JSON |

### Edges

```
START                   ──conditional── (mode="ivr")    ──► ivr_llm
                                       (mode="hold")   ──► hold_node
                                       (mode="human")  ──► human_llm
                                       (mode="done")   ──► done_node

ivr_llm                  ──fixed────────────────────────► ivr_tools
ivr_tools                ──fixed────────────────────────► END

hold_node                ──fixed────────────────────────► END

human_llm                ──fixed────────────────────────► END

done_node                ──fixed────────────────────────► END
```

Each `ainvoke()` runs through one node + (for IVR) the tool node, then exits. State persists across invokes via the checkpointer (keyed by `thread_id=call_sid`).

### Tool registries

**IVR tools** (loaded only on `ivr_llm`, `tool_choice="required"`):
- `send_dtmf(digits: str)`
- `say_short(text: str)`
- `mark_transferred_to_hold()`

**HUMAN tools** (loaded only on `human_llm`, `tool_choice="auto"`):
- `end_call(outcome: "completed" | "nothing_available" | "over_budget" | "detected_as_bot")`

Mode-specific registries eliminate cross-mode hallucination (no `send_dtmf` mid-conversation; no `end_call` mid-IVR).

### Build_day.md compliance map

| build_day.md requirement | Where it's handled |
|---|---|
| Press 1 for delivery (Prompt 1) | ivr_llm decides → ivr_tools → DTMFAction("1") |
| Say customer name (Prompt 2) | ivr_llm → say_short(order.customer_name) |
| Enter 10-digit callback (Prompt 3) | ivr_llm → send_dtmf(order.phone_number) |
| Say delivery zip (Prompt 4) | ivr_llm → say_short(order.delivery_zip) |
| Confirmation Prompt 5 | ivr_llm cross-checks readback against order data; says "yes" or "no" |
| Hold prompt (Prompt 6) | ivr_llm → mark_transferred_to_hold → mode=hold |
| Stay silent on hold | mode=hold; adapter suppresses outbound TTS |
| Detect human pickup | energy gate (adapter) → hold_node classifier |
| Order pizza per data | human_llm system prompt encodes substitution rules + budget |
| Per-item prices | human_llm pushes for explicit prices; done_node extracts |
| Total / delivery time / order # | done_node extracts from `human_messages` |
| Special instructions before hangup | human_llm system prompt requires delivery before end_call |
| Four spec outcomes | end_call enum |
| Operational outcomes | adapter sets hangup_reason on timeouts/errors; routes to done_node |
| Structured JSON output | done_node finalize → adapter prints |

## Data flow (one successful call)

```
T+0.0s   CLI loads order.json, validates schema
T+0.1s   FastAPI starts; ngrok tunnel opens
T+0.5s   POST /start → Twilio outbound call placed (after ALLOWED_DESTINATIONS check)
T+2-4s   Twilio dials user's phone; user picks up
T+~5s    Twilio /voice → TwiML <Connect><Stream> → WS /stream connects
         Audio frames flow

         ─── MODE: IVR ─────────────────────────────────────────────
T+~5s    Audio pipeline online: Deepgram streaming, Silero VAD running
         (VAD signals barge-in but is no-op during IVR — agent does
         not preface or speak unprompted)
         Adapter sets state.mode = "ivr"
         Per Deepgram-finalized utterance:
           adapter.ainvoke(graph, state | {new_transcript: <text>})
           router → ivr_llm (gpt-oss-120B, tool_choice="required")
           ivr_llm emits one tool call → ivr_tools dispatches:
             - DTMFAction("1") → Twilio REST <Play digits>
             - SpeakAction("Jordan Mitchell", streaming=False) →
                 Cartesia synth → frame queue → Twilio
             - mark_transferred_to_hold → mode="hold"
T+~25s   Prompt 6 ("Got it. Please hold...") matched →
         ivr_llm → mark_transferred_to_hold
         ivr_tools sets mode="hold"
         Adapter relaxes Deepgram endpointing (longer utterance_end_ms),
         starts hold-timeout watchdog (5 min), starts energy gate

         ─── MODE: HOLD ───────────────────────────────────────────
         Energy gate observes audio frames continuously
         Most frames suppressed (silence/music); gate trips on
         speech-shaped utterance candidate
         When Deepgram finalizes a transcript AND gate has tripped:
           adapter.ainvoke(graph, ... new_transcript=<text>)
           router → hold_node (Llama 3.1 8B, binary classify)
           if "yes (human greeting)":
              state.mode = "human"
              state.human_messages.append(HumanMessage(<text>))
           if "no": no state changes
         Loop continues; on next gate-trip + transcript, re-classify

         ─── MODE: HUMAN ──────────────────────────────────────────
         Adapter cancels hold-timeout, switches Deepgram to
         human-conversation endpointing (~1500-2000ms utterance_end_ms
         to tolerate employee typing pauses)
         Per Deepgram-finalized utterance:
           adapter.ainvoke(graph, ... new_transcript=<text>)
           router → human_llm (gpt-oss-120B, tool_choice="auto")
           Streaming response:
             - text deltas → SpeakAction(streaming=True) →
                 Cartesia stream → μ-law frames → Twilio
             - if end_call tool emitted: state.mode = "done",
                 EndCallAction queued
         Barge-in: VAD detects user speech during TTS playback →
                   adapter cancels in-flight TTS task

         ─── MODE: DONE ───────────────────────────────────────────
         Adapter waits for TTS queue to drain (don't cut goodbye)
         adapter.ainvoke(graph, ... mode="done")
         router → done_node:
           - Runs structured-output extraction LLM over human_messages
           - Builds CallResult (pizza, side, drink, total, delivery
             time, order number, special_instructions_delivered)
           - Validates against pydantic schema
           - Adds EndCallAction(outcome, json) to pending_actions
         Adapter prints JSON to stdout, calls Twilio hangup,
         closes pipelines, exits
```

### Three places this can interact badly (mitigations baked in)

1. **IVR Prompt 5 misheard.** ivr_llm sees the readback transcript, cross-references the order data. If the readback contains the wrong digit/word, ivr_llm calls `say_short("no")` and the cycle resets. Bounded by 3-strike abort: 3 consecutive `UNKNOWN` IVR transcripts → adapter sets `mode="done"`, `hangup_reason=IVR_REJECTED`.

2. **Hold ends with employee mid-sentence.** When hold_node detects human, it appends the gate-tripping transcript to `human_messages` as a `HumanMessage`. The first turn of human_llm sees that greeting in context, so it can respond appropriately rather than starting cold. If the employee said something like "thanks for holding, what's your order?" — the agent picks up on it.

3. **HUMAN-phase model speaks before tool call.** Even with `tool_choice="auto"`, the model may say "thanks, ordering that now" then call `end_call`. We tolerate this in HUMAN — text streams to TTS as natural speech. The tool call still routes to done_node correctly.

## Error handling & hangup conditions

### Spec outcomes

| Outcome | Trigger | done_node emits |
|---|---|---|
| `completed` | end_call(outcome="completed") AND extraction yields non-null pizza | Full JSON per spec |
| `nothing_available` | end_call(outcome="nothing_available") OR extraction finds pizza missing | Whatever was extracted; `pizza: null` |
| `over_budget` | end_call(outcome="over_budget") after pizza+side already exceeded `budget_max` | Captured items, `drink: null`, `drink_skip_reason: "over_budget"` |
| `detected_as_bot` | end_call(outcome="detected_as_bot") (HUMAN system prompt: "if asked if you're a bot, apologize and call end_call") | Whatever was captured before suspicion |

### Operational outcomes (extensions, surfaced in JSON)

| Outcome | Trigger |
|---|---|
| `no_answer` | Twilio CallStatus = `no-answer`/`busy`/`failed` within 10 s of dial |
| `ivr_failed` | Deepgram WS error during IVR after one retry |
| `ivr_rejected` | 3 consecutive UNKNOWN IVR transcripts |
| `hold_timeout` | Hold-timeout watchdog fires (>5 min) without human pickup |
| `disconnect` | Twilio Media Streams disconnects mid-call OR HUMAN inactivity timeout (~30 s no transcripts) |

These appear in the JSON's `outcome` field. The schema accepts all 9 string values.

### General error principles

- **No silent failures.** Every `except` either re-raises, sets `hangup_reason`, or logs to Langfuse.
- **No retries inside the audio loop.** Retries belong at the adapter level only.
- **`try/finally`** around every long-lived resource (Deepgram WS, Cartesia stream, Twilio call, ngrok tunnel).
- **Real cancellation:** `asyncio.Task.cancel()` for barge-in TTS interruption. Flag-checks don't survive a 1-second blocking LLM call.

## Testing

### Component unit tests (write inline)

| Test file | Covers |
|---|---|
| `test_order_schema.py` | pydantic validation: spec example + 3 malformed variants |
| `test_result_schema.py` | Output schema across all outcomes |
| `test_safety.py` | ALLOWED_DESTINATIONS hard precheck (fail-closed allowlist) |
| `test_audio_resample.py` | μ-law 8 kHz ↔ PCM16 16 kHz round-trip (Silero VAD wants 16 kHz) |
| `test_log_redaction.py` | Secret-redacting JSON formatter |
| `test_ivr_tools.py` | LangGraph tool schemas validate; `ivr_tools` node correctly maps tool call → Action |
| `test_human_tools.py` | `end_call` schema; outcome enum |
| `test_persona_banned_phrases.py` | Banned-phrase regex over corpus ("let me check", "as an AI", etc.) — fails if any leak into prompts |
| `test_hold_classifier.py` | Mock LLM client; "yes" / "no" / "" cases |
| `test_hold_energy_gate.py` | Silence does not trip; flat music does not trip; speech-with-gaps trips |
| `test_router.py` | Each `state.mode` value routes to correct next node |
| `test_done_node_extraction.py` | Mock LLM; given a sample `human_messages`, extracts correct pizza/side/drink/total/etc. |
| `test_action_dispatch.py` | Adapter dispatches each Action type correctly (mocks Twilio, Cartesia clients) |

### Real call testing

Reserved for end-to-end behavior:
1. End-to-end first call (dial, IVR navigation, hold detection, human pickup)
2. Barge-in feel and HUMAN-phase latency
3. Spec test script (build_day.md) full happy-path
4. Substitution edge cases (no-go topping offered, all sides unavailable, over-budget, bot detection)

**Twilio paid account required** — trial-account quirks are blockers.

### Out of scope

- Local audio fixture replay (skipped — paid Twilio makes real-call iteration cheap)
- Multiple telephony providers
- Multiple LLM providers per node beyond what's specified
- Concurrent calls (one call at a time)
- International numbers / non-US zip codes
- GitHub Actions / CI (commit constantly, run pytest locally before each real call)

## Output schema

```json
{
  "outcome": "completed | nothing_available | over_budget | detected_as_bot | no_answer | ivr_failed | ivr_rejected | hold_timeout | disconnect",
  "pizza": { "description": "...", "substitutions": {...}, "price": 18.50 } | null,
  "side": { "description": "...", "original": "...", "price": 6.99 } | null,
  "drink": { "description": "...", "price": 3.49 } | null,
  "drink_skip_reason": "over_budget" | "unavailable" | null,
  "total": 28.98 | null,
  "delivery_time": "35 minutes" | null,
  "order_number": "4412" | null,
  "special_instructions_delivered": true | false
}
```

For non-`completed` outcomes, fields populate with whatever extraction recovered; missing fields are `null`.

## Module layout (preview — full mapping in implementation plan)

```
agent/
├── cli.py                     # entry point
├── server.py                  # FastAPI: /start, /voice, ws /stream, /health
├── adapter.py                 # StateProcessor: bridges audio pipeline ↔ LangGraph
├── audio/
│   ├── pipeline.py            # frame router, ASR + VAD subscriptions
│   ├── transport.py           # μ-law 8k ↔ PCM16 16k/24k resampling
│   ├── vad.py                 # Silero VAD wrapper
│   ├── tts_buffered.py        # Cartesia synth-then-emit (IVR)
│   ├── tts_streaming.py       # Cartesia streaming (HUMAN), with cancellation
│   └── dtmf.py                # Twilio REST sendDigits side-channel
├── asr/
│   └── deepgram_stream.py     # Deepgram streaming wrapper
├── graph/
│   ├── state.py               # AgentState TypedDict + Action union
│   ├── builder.py             # build_graph() returns compiled graph
│   ├── router.py              # router node
│   ├── nodes/
│   │   ├── ivr_llm.py
│   │   ├── ivr_tools.py
│   │   ├── hold_node.py
│   │   ├── human_llm.py
│   │   └── done_node.py
│   └── tools/
│       ├── ivr_tools.py       # send_dtmf, say_short, mark_transferred_to_hold schemas
│       └── human_tools.py     # end_call schema
├── llm/
│   ├── cerebras_client.py     # gpt-oss-120B client (LangChain ChatOpenAI-compatible against Cerebras endpoint)
│   ├── groq_client.py         # Llama 3.1 8B client (LangChain ChatGroq)
│   └── persona/
│       ├── ivr_prompt.py
│       ├── human_prompt.py
│       └── extraction_prompt.py
├── result/
│   └── builder.py             # CallResult helpers
├── runtime/
│   ├── tunnel.py              # ngrok
│   ├── dialer.py              # Twilio dialer with allowlist
│   └── stream.py              # Twilio Media Streams JSON codec
├── obs/
│   ├── log.py                 # JSON formatter with redaction
│   └── langfuse_client.py     # lazy Langfuse setup
├── safety.py                  # ALLOWED_DESTINATIONS precheck
└── types/
    ├── order.py               # Order schema (input)
    └── result.py              # CallResult schema (output)
```

## Open Questions / Out-of-Scope for v1

- LangGraph docs review (in progress) may surface primitives that adjust the node implementation but not the architecture. The implementation plan will integrate findings before code is written.
- Multi-call orchestration, persistent call history, configurable persona/voice, non-English IVR — explicit non-goals for v1.
