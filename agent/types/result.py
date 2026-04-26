"""Output schema for the JSON the agent prints at hangup."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Outcome(StrEnum):
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
