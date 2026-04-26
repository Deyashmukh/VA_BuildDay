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
