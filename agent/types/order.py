"""Input order schema (the JSON the agent receives at startup)."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, computed_field

PhoneNumber = Annotated[str, StringConstraints(pattern=r"^\d{10}$")]


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
    phone_number: PhoneNumber
    delivery_address: str
    pizza: Pizza
    side: Side
    drink: Drink
    budget_max: float = Field(gt=0)
    special_instructions: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delivery_zip(self) -> str:
        """Pull the 5-digit zip from the trailing portion of the address."""
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b\s*$", self.delivery_address)
        if not match:
            raise ValueError(f"could not extract zip from address: {self.delivery_address}")
        return match.group(1)
