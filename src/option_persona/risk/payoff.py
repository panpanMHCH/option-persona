"""Expiration payoff analysis for long option positions (model-free).

Everything in this module depends only on the contract terms and the price
paid; no volatility or pricing model is involved. Values at expiry are exact
under the stated assumptions:

* the option is held to expiry and settles at intrinsic value
  max(S_T - K, 0) for a call, max(K - S_T, 0) for a put;
* the settlement price S_T is the official settlement reference;
* exercise/assignment fees and any taxes are excluded unless recorded as
  ``exit_fees`` on the position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from option_persona.core.models import OptionPosition
from option_persona.core.types import OptionType
from option_persona.core.validation import as_float_array, require_non_negative
from option_persona.pricing.black_scholes import FloatOrArray


def intrinsic_value(option_type: OptionType | str, S_T: ArrayLike, K: float) -> FloatOrArray:
    """Payoff per unit of underlying at expiry."""
    s = as_float_array(S_T, "Underlying price at expiry")
    require_non_negative(s, "Underlying price at expiry")
    otype = OptionType.parse(option_type)
    v = np.maximum(s - K, 0.0) if otype is OptionType.CALL else np.maximum(K - s, 0.0)
    return float(v) if v.ndim == 0 else v


def _total_fees(position: OptionPosition, include_fees: bool) -> float:
    return (position.entry_fees + position.exit_fees) if include_fees else 0.0


def expiration_pnl(position: OptionPosition, S_T: ArrayLike,
                   include_fees: bool = True) -> FloatOrArray:
    """Position P&L in currency if held to expiry, for terminal price(s) ``S_T``."""
    c = position.contract
    payoff = np.asarray(intrinsic_value(c.option_type, S_T, c.strike))
    pnl = (payoff - position.entry_premium) * position.units - _total_fees(position, include_fees)
    return float(pnl) if pnl.ndim == 0 else pnl


def breakeven_at_expiry(position: OptionPosition, include_fees: bool = True) -> float:
    """Terminal underlying price at which expiry P&L is exactly zero."""
    c = position.contract
    cost_per_unit = position.entry_premium + _total_fees(position, include_fees) / position.units
    if c.option_type is OptionType.CALL:
        return c.strike + cost_per_unit
    return c.strike - cost_per_unit  # may be <= 0: then the put cannot break even


@dataclass(frozen=True)
class ExpirationProfile:
    premium_paid: float
    total_fees: float
    max_loss: float  # positive number: the most that can be lost
    max_gain: float  # math.inf for long calls
    breakeven: float
    breakeven_reachable: bool
    max_loss_condition: str
    max_gain_condition: str


def expiration_profile(position: OptionPosition, include_fees: bool = True) -> ExpirationProfile:
    """Closed-form risk/reward at expiry for a long call or long put."""
    c = position.contract
    fees = _total_fees(position, include_fees)
    loss = position.premium_paid + fees
    be = breakeven_at_expiry(position, include_fees)
    if c.option_type is OptionType.CALL:
        gain = math.inf
        loss_cond = f"{c.underlying} settles at or below {c.strike:g}"
        gain_cond = "Unbounded as the underlying rises"
        reachable = True
    else:
        gain = (c.strike - position.entry_premium) * position.units - fees
        loss_cond = f"{c.underlying} settles at or above {c.strike:g}"
        gain_cond = f"{c.underlying} settles at 0 (theoretical floor)"
        reachable = be > 0
    return ExpirationProfile(
        premium_paid=position.premium_paid, total_fees=fees, max_loss=loss, max_gain=gain,
        breakeven=be, breakeven_reachable=reachable, max_loss_condition=loss_cond,
        max_gain_condition=gain_cond,
    )
