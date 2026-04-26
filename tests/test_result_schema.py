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
