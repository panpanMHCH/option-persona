"""Pre-expiration scenario analysis using an explicitly selected model.

Every output produced here is a *model estimate*, conditional on the
assumptions in :class:`ModelAssumptions` (constant volatility, rate and
dividend yield over the horizon, lognormal dynamics). It is not a forecast of
what the market will quote: in practice implied volatility moves with the
underlying (it typically rises as equities fall), bid-ask spreads widen, and
the market may price jump risk the model ignores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionPosition
from option_persona.core.time import DAYS_PER_YEAR, require_aware, time_to_expiry
from option_persona.core.types import ExerciseStyle, OptionType
from option_persona.pricing.black_scholes import bsm_price


@dataclass(frozen=True)
class ModelAssumptions:
    """Inputs the scenario engine holds constant. Shown alongside every result."""

    volatility: float  # annualised decimal, e.g. 0.46
    rate: float = 0.04
    dividend_yield: float = 0.0
    model: str = "Black-Scholes-Merton (European exercise)"

    def __post_init__(self) -> None:
        if not 0 < self.volatility < 20:
            raise InvalidInputError(
                f"Volatility must be a positive decimal (0.46 = 46%), got {self.volatility}."
            )

    def describe(self) -> str:
        return (f"{self.model}; sigma = {self.volatility:.2%} constant; r = {self.rate:.2%} "
                f"(continuous); q = {self.dividend_yield:.2%} (continuous)")


def model_caveats(position: OptionPosition) -> list[str]:
    """Warnings about where the selected model is known to be wrong for this position."""
    notes = [
        "Model values assume constant volatility; real implied volatility changes with "
        "price, time and events (earnings, macro releases).",
        "Values are theoretical mid-market estimates, not executable bid or ask prices.",
    ]
    c = position.contract
    if c.exercise_style is ExerciseStyle.AMERICAN:
        if c.option_type is OptionType.PUT:
            notes.append(
                "American put valued with a European model: BSM omits the early-exercise "
                "premium, so values are a lower bound (the gap grows deep in the money)."
            )
        else:
            notes.append(
                "American call valued with a European model: exact only if no dividend is "
                "paid before expiry; otherwise early exercise before ex-dividend may matter."
            )
    return notes


def model_value_per_unit(position: OptionPosition, spot: float | np.ndarray,
                         valuation_time: datetime, assumptions: ModelAssumptions,
                         vol_shift: float = 0.0) -> float | np.ndarray:
    """BSM value per unit of underlying at ``valuation_time`` (expiry at 16:00 ET)."""
    c = position.contract
    T = time_to_expiry(require_aware(valuation_time, "valuation_time"), c.expiry)
    sigma = assumptions.volatility + vol_shift
    if sigma <= 0:
        raise InvalidInputError(f"Volatility shift {vol_shift:+.2%} makes volatility non-positive.")
    return bsm_price(c.option_type, spot, c.strike, T, assumptions.rate, sigma,
                     assumptions.dividend_yield)


def scenario_grid(position: OptionPosition, spots: Sequence[float], days_forward: Sequence[float],
                  valuation_time: datetime, assumptions: ModelAssumptions,
                  include_fees: bool = True) -> pd.DataFrame:
    """Model P&L across underlying prices and future dates (long/tidy format).

    Horizons beyond expiry are clipped to expiry, where the model collapses to
    intrinsic value (identical to the model-free expiration payoff).
    """
    require_aware(valuation_time, "valuation_time")
    spots_arr = np.asarray(spots, dtype=float)
    fees = (position.entry_fees + position.exit_fees) if include_fees else 0.0
    cost = position.premium_paid + fees
    rows = []
    for d in days_forward:
        t = valuation_time + timedelta(days=float(d))
        T = time_to_expiry(t, position.contract.expiry)
        per_unit = np.asarray(model_value_per_unit(position, spots_arr, t, assumptions))
        value = per_unit * position.units
        rows.append(pd.DataFrame({
            "days_forward": float(d),
            "days_to_expiry": T * DAYS_PER_YEAR,
            "spot": spots_arr,
            "model_value_per_unit": per_unit,
            "position_value": value,
            "pnl": value - cost,
            "pnl_pct_of_cost": (value - cost) / cost,
        }))
    return pd.concat(rows, ignore_index=True)


def spot_vol_stress(position: OptionPosition, spot: float, valuation_time: datetime,
                    assumptions: ModelAssumptions, spot_moves: Sequence[float],
                    vol_shifts: Sequence[float], horizon_days: float = 0.0,
                    include_fees: bool = True) -> pd.DataFrame:
    """P&L matrix (rows: % spot move, columns: absolute vol shift) at a horizon."""
    t = require_aware(valuation_time, "valuation_time") + timedelta(days=horizon_days)
    fees = (position.entry_fees + position.exit_fees) if include_fees else 0.0
    cost = position.premium_paid + fees
    moves = np.asarray(spot_moves, dtype=float)
    shocked = spot * (1.0 + moves)
    if np.any(shocked <= 0):
        raise InvalidInputError("Spot moves of -100% or worse are not valid stress scenarios.")
    out = {}
    for dv in vol_shifts:
        per_unit = np.asarray(model_value_per_unit(position, shocked, t, assumptions, dv))
        out[f"vol {dv * 100:+.0f} pts"] = per_unit * position.units - cost
    return pd.DataFrame(out, index=pd.Index([f"{m:+.0%}" for m in moves], name="spot move"))
