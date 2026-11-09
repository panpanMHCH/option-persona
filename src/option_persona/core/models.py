"""Validated data models for contracts, quotes and personal positions.

Design rule: a market observation always carries its provenance (source,
timestamp, whether it is synthetic, whether it is delayed). Theoretical values
are never stored in these models; they are computed on demand by the pricing
functions so that observed and modelled numbers cannot be confused.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from option_persona.core.time import require_aware
from option_persona.core.types import ExerciseStyle, OptionType, SettlementType


class OptionContract(BaseModel):
    """Static contract specification (no prices)."""

    model_config = ConfigDict(frozen=True)

    underlying: str = Field(min_length=1, description="Ticker of the underlying, e.g. NVDA")
    option_type: OptionType
    strike: float = Field(gt=0)
    expiry: date
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN
    settlement: SettlementType = SettlementType.PHYSICAL
    multiplier: float = Field(default=100.0, gt=0, description="Shares (or index units) per contract")
    currency: str = Field(default="USD", min_length=3, max_length=3)

    @field_validator("underlying")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("option_type", mode="before")
    @classmethod
    def _parse_type(cls, v: object) -> OptionType:
        return OptionType.parse(v)  # type: ignore[arg-type]


class Provenance(BaseModel):
    """Where a number came from and when it was observed."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1, description="e.g. 'IBKR snapshot', 'manual', 'synthetic:bsm'")
    observed_at: datetime
    is_synthetic: bool = False
    is_delayed: bool = False

    @field_validator("observed_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return require_aware(v, "observed_at")


class OptionQuote(BaseModel):
    """A two-sided option quotation. Executable prices are bid (sell) and ask (buy);
    the mid is an indicative reference, never an execution price."""

    model_config = ConfigDict(frozen=True)

    contract: OptionContract
    bid: float = Field(ge=0)
    ask: float = Field(gt=0)
    underlying_price: float = Field(gt=0, description="Underlying reference at observation time")
    provenance: Provenance
    bid_size: int | None = Field(default=None, ge=0)
    ask_size: int | None = Field(default=None, ge=0)
    underlying_bid: float | None = Field(
        default=None, gt=0, description="Underlying bid, if quoted (indices usually have none)"
    )
    underlying_ask: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _crossed(self) -> Self:
        if self.bid > self.ask:
            raise ValueError(f"Crossed quote: bid {self.bid} exceeds ask {self.ask}.")
        if (self.underlying_bid is None) != (self.underlying_ask is None):
            raise ValueError("Supply both underlying_bid and underlying_ask, or neither.")
        if (self.underlying_bid is not None and self.underlying_ask is not None
                and self.underlying_bid > self.underlying_ask):
            raise ValueError(
                f"Crossed underlying quote: bid {self.underlying_bid} exceeds ask {self.underlying_ask}."
            )
        return self

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        return self.spread / self.mid if self.mid > 0 else float("inf")


class OptionPosition(BaseModel):
    """A personal long option position (MVP scope: long calls and long puts only).

    Prices are per share (per unit of underlying); dollar amounts are obtained
    by multiplying by ``contract.multiplier * quantity``. Fees are totals in
    the contract currency.
    """

    contract: OptionContract
    quantity: int = Field(gt=0, description="Number of contracts held long")
    entry_premium: float = Field(gt=0, description="Premium paid per share, e.g. 2.38")
    entry_underlying_price: float = Field(gt=0)
    entry_time: datetime | None = Field(
        default=None, description="Timezone-aware entry timestamp; None if not recorded"
    )
    entry_fees: float = Field(default=0.0, ge=0)
    observed_entry_iv: float | None = Field(default=None, gt=0, lt=20)
    observed_entry_delta: float | None = Field(default=None, ge=-1, le=1)
    exit_premium: float | None = Field(default=None, ge=0)
    exit_time: datetime | None = None
    exit_fees: float = Field(default=0.0, ge=0)
    notes: str = ""

    @field_validator("entry_time", "exit_time")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        return None if v is None else require_aware(v, "position timestamp")

    @model_validator(mode="after")
    def _checks(self) -> Self:
        if self.observed_entry_delta is not None:
            d = self.observed_entry_delta
            if self.contract.option_type is OptionType.CALL and d < 0:
                raise ValueError(f"A long call cannot have negative delta ({d}).")
            if self.contract.option_type is OptionType.PUT and d > 0:
                raise ValueError(f"A long put cannot have positive delta ({d}).")
        if self.exit_time and self.entry_time and self.exit_time < self.entry_time:
            raise ValueError("exit_time precedes entry_time.")
        return self

    @property
    def units(self) -> float:
        """Underlying units controlled at expiry (multiplier x contracts)."""
        return self.contract.multiplier * self.quantity

    @property
    def premium_paid(self) -> float:
        """Total premium outlay excluding fees."""
        return self.entry_premium * self.units

    @property
    def is_closed(self) -> bool:
        return self.exit_premium is not None

    def realised_pnl(self) -> float | None:
        """Realised P&L after all fees, or None if the position is open."""
        if self.exit_premium is None:
            return None
        return (self.exit_premium - self.entry_premium) * self.units - self.entry_fees - self.exit_fees
