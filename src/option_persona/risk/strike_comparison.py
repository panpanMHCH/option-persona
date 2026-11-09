r"""Side-by-side comparison of candidate strikes for a long call or put.

Purpose
-------
Choosing a strike is choosing a trade-off between cost, the move needed to
profit, time decay and leverage. This module lays those quantities out for
several strikes at once so the trade-off is visible. It does not rank strikes
or recommend one: which trade-off is acceptable depends on the user's view,
horizon and risk budget.

Price basis per strike
----------------------
* **Observed** (bid and ask supplied): the entry price is the **ask**, because a
  buyer pays the offer. The volatility for that row is the BSM implied
  volatility *of the ask*, so the row reflects the market's skew at that strike.
* **Model** (no quote): the entry price is the BSM value at the single
  reference volatility in :class:`ModelAssumptions` (a flat smile). Real chains
  usually price lower-strike puts at higher IVs, so model rows understate their cost.

Quantities
----------
* Breakeven at expiry: K + p + f/units for calls, K − p − f/units for puts,
  where f is the *entry* commission only (exit costs are unknown at this stage).
* Expected move (one standard deviation, reference σ): S·σ·√T. This is a
  quoting convention for scale, not a forecast.
* Probabilities are risk-neutral (Q-measure) at the row's volatility: N(±d2)
  for finishing in the money, and the same evaluated at the breakeven price for
  finishing beyond breakeven. They are not real-world probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from option_persona.core.errors import InvalidInputError, OptionPersonaError
from option_persona.core.time import DAYS_PER_YEAR, require_aware, time_to_expiry
from option_persona.core.types import OptionType
from option_persona.pricing.black_scholes import bsm_greeks, risk_neutral_prob_itm
from option_persona.pricing.implied_vol import implied_volatility
from option_persona.risk.scenarios import ModelAssumptions


@dataclass(frozen=True)
class StrikeCandidate:
    """A strike to compare, optionally with an observed bid/ask (per unit)."""

    strike: float
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise InvalidInputError(f"Strike must be positive, got {self.strike}.")
        if (self.bid is None) != (self.ask is None):
            raise InvalidInputError(f"Strike {self.strike}: give both bid and ask, or neither.")
        if self.ask is not None and self.bid is not None:
            if self.ask <= 0 or self.bid < 0:
                raise InvalidInputError(f"Strike {self.strike}: need ask > 0 and bid >= 0.")
            if self.bid > self.ask:
                raise InvalidInputError(f"Strike {self.strike}: crossed quote (bid {self.bid} > ask "
                                        f"{self.ask}).")

    @property
    def is_observed(self) -> bool:
        return self.ask is not None


def expected_move(spot: float, sigma: float, T: float) -> float:
    """One-standard-deviation move S·σ·√T: a scale reference, not a forecast."""
    return spot * sigma * math.sqrt(max(T, 0.0))


def compare_strikes(
    option_type: OptionType | str,
    spot: float,
    expiry: date,
    valuation_time: datetime,
    candidates: Sequence[StrikeCandidate | float],
    assumptions: ModelAssumptions,
    contracts: int = 1,
    multiplier: float = 100.0,
    fee_per_contract: float = 0.0,
) -> pd.DataFrame:
    """One row per candidate strike, ordered by strike.

    Dollar columns are for ``contracts`` contracts. ``theta_per_day`` and
    ``vega_per_vol_point`` are model sensitivities at the row's volatility.
    """
    otype = OptionType.parse(option_type)
    if spot <= 0:
        raise InvalidInputError(f"Underlying price must be positive, got {spot}.")
    if contracts < 1:
        raise InvalidInputError(f"Contracts must be at least 1, got {contracts}.")
    if multiplier <= 0 or fee_per_contract < 0:
        raise InvalidInputError("Multiplier must be > 0 and fees >= 0.")
    T = time_to_expiry(require_aware(valuation_time, "valuation_time"), expiry)
    if T <= 0:
        raise InvalidInputError("Valuation time is at or after expiry: there is nothing left to compare.")
    cands = [c if isinstance(c, StrikeCandidate) else StrikeCandidate(float(c)) for c in candidates]
    if not cands:
        raise InvalidInputError("Give at least one strike to compare.")
    strikes = [c.strike for c in cands]
    if len(set(strikes)) != len(strikes):
        raise InvalidInputError("Each strike may appear only once.")

    r, q, ref_sigma = assumptions.rate, assumptions.dividend_yield, assumptions.volatility
    units = multiplier * contracts
    fees = fee_per_contract * contracts
    move_1sd = expected_move(spot, ref_sigma, T)
    sign = 1.0 if otype is OptionType.CALL else -1.0

    rows = []
    for c in sorted(cands, key=lambda c: c.strike):
        note = ""
        if c.is_observed:
            entry = float(c.ask)  # type: ignore[arg-type]
            basis = "ask (observed)"
            try:
                sigma = implied_volatility(entry, otype, spot, c.strike, T, r, q)
            except OptionPersonaError as exc:
                sigma = float("nan")
                note = f"No implied volatility for the ask: {exc}"
            spread = (c.ask - c.bid) / (0.5 * (c.ask + c.bid))  # type: ignore[operator]
        else:
            sigma = ref_sigma
            spread = float("nan")
            basis = "model (flat σ)"
            entry = float(bsm_greeks(otype, spot, c.strike, T, r, sigma, q).price)

        breakeven = c.strike + sign * (entry + fees / units)
        row = {
            "strike": c.strike, "price_basis": basis, "entry_price": entry,
            "cost": entry * units + fees, "breakeven": breakeven,
            "move_to_breakeven": breakeven / spot - 1.0,
            "breakeven_in_expected_moves": (breakeven - spot) / move_1sd if move_1sd > 0 else float("nan"),
            "volatility": sigma, "relative_spread": spread,
        }
        if math.isfinite(sigma) and sigma > 0:
            g = bsm_greeks(otype, spot, c.strike, T, r, sigma, q)
            value = float(g.price)
            row |= {
                "delta": float(g.delta),
                "q_prob_itm": float(risk_neutral_prob_itm(otype, spot, c.strike, T, r, sigma, q)),
                "q_prob_beyond_breakeven": (
                    float(risk_neutral_prob_itm(otype, spot, breakeven, T, r, sigma, q))
                    if breakeven > 0 else 0.0),
                "theta_per_day": float(g.theta_per_day) * units,
                "theta_pct_of_value_per_day": float(g.theta_per_day) / value if value > 0 else float("nan"),
                "vega_per_vol_point": float(g.vega_per_vol_point) * units,
                "elasticity": float(g.delta) * spot / value if value > 0 else float("nan"),
            }
        else:
            row |= dict.fromkeys(("delta", "q_prob_itm", "q_prob_beyond_breakeven", "theta_per_day",
                                  "theta_pct_of_value_per_day", "vega_per_vol_point", "elasticity"),
                                 float("nan"))
        row["note"] = note
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs |= {
        "days_to_expiry": T * DAYS_PER_YEAR, "expected_move": move_1sd, "spot": spot,
        "reference_volatility": ref_sigma, "assumptions": assumptions.describe(),
    }
    return out
