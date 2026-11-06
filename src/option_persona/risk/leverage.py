r"""Leverage, exposure and capital-efficiency analytics for long option positions.

Key quantities (per position, currency units unless stated)
-----------------------------------------------------------
* Share-equivalent exposure   = delta x multiplier x contracts
  (the number of shares whose P&L matches the option's for a *small* move).
* Delta-adjusted notional     = share-equivalent exposure x S.
* Elasticity (lambda, Omega)  = (dV/V) / (dS/S) = delta x S / V.
  A put with lambda = -12 loses roughly 12% of its value for a +1% move in
  the underlying — before gamma, theta and vega effects.
* Premium at risk             = the most a long option holder can lose:
  the current value (or, from inception, premium paid plus fees).

Why leverage does not raise expected returns by itself
------------------------------------------------------
Under the risk-neutral measure every traded asset, options included, earns the
risk-free rate in expectation; leverage scales risk, not risk-adjusted reward.
Under the real-world measure an option's expected excess return is roughly
lambda times the underlying's (and is negative for puts on assets with a
positive risk premium), *before* subtracting any volatility risk premium and
transaction costs. High elasticity magnifies forecasting errors exactly as much
as it magnifies correct forecasts, and delta and lambda themselves change as
the underlying moves (gamma), so they are local, not global, measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionPosition
from option_persona.core.time import DAYS_PER_YEAR, require_aware, time_to_expiry
from option_persona.pricing.black_scholes import bsm_greeks
from option_persona.risk.scenarios import ModelAssumptions, model_value_per_unit


@dataclass(frozen=True)
class RiskLimits:
    """Thresholds for warnings. Personal risk preferences, not universal rules."""

    max_premium_pct_equity: float = 0.02
    max_delta_notional_pct_equity: float = 0.25
    high_elasticity: float = 10.0
    near_expiry_days: float = 7.0
    wide_relative_spread: float = 0.10


@dataclass(frozen=True)
class ExposureReport:
    spot: float
    days_to_expiry: float
    model_value_per_unit: float
    position_value: float
    delta_per_unit: float
    gamma_per_unit: float
    share_equivalent: float
    delta_notional: float
    gross_underlying_notional: float  # |units| x S, ignoring delta
    elasticity: float
    premium_at_risk: float  # current model value: the most that can still be lost
    cost_basis: float  # premium paid + entry fees
    theta_per_day: float  # position, currency per calendar day
    vega_per_vol_point: float  # position, currency per +1 vol point
    gamma_dollar_per_1pct: float  # change in delta-notional for a 1% move
    pct_equity_at_risk: float | None
    pct_equity_delta_notional: float | None
    assumptions: str
    warnings: list[str] = field(default_factory=list)


def exposure_report(position: OptionPosition, spot: float, valuation_time: datetime,
                    assumptions: ModelAssumptions, portfolio_equity: float | None = None,
                    relative_spread: float | None = None,
                    limits: RiskLimits | None = None) -> ExposureReport:
    """Compute delta-adjusted exposure, elasticity and risk warnings at ``valuation_time``."""
    if spot <= 0:
        raise InvalidInputError(f"Underlying price must be positive, got {spot}.")
    if portfolio_equity is not None and portfolio_equity <= 0:
        raise InvalidInputError(f"Portfolio equity must be positive, got {portfolio_equity}.")
    limits = limits or RiskLimits()
    c = position.contract
    T = time_to_expiry(require_aware(valuation_time, "valuation_time"), c.expiry)
    g = bsm_greeks(c.option_type, spot, c.strike, T, assumptions.rate, assumptions.volatility,
                   assumptions.dividend_yield)
    units = position.units
    value_pu = float(g.price)
    delta = float(g.delta)
    share_eq = delta * units
    delta_notional = share_eq * spot
    position_value = value_pu * units
    elasticity = delta * spot / value_pu if value_pu > 0 else float("nan")
    cost_basis = position.premium_paid + position.entry_fees

    pct_risk = position_value / portfolio_equity if portfolio_equity else None
    pct_notional = abs(delta_notional) / portfolio_equity if portfolio_equity else None
    days = T * DAYS_PER_YEAR

    warnings: list[str] = []
    if pct_risk is not None and pct_risk > limits.max_premium_pct_equity:
        warnings.append(f"Concentration: premium at risk is {pct_risk:.1%} of equity "
                        f"(limit {limits.max_premium_pct_equity:.1%}).")
    if pct_notional is not None and pct_notional > limits.max_delta_notional_pct_equity:
        warnings.append(f"Exposure: delta-adjusted notional is {pct_notional:.1%} of equity "
                        f"(limit {limits.max_delta_notional_pct_equity:.1%}).")
    if np.isfinite(elasticity) and abs(elasticity) > limits.high_elasticity:
        warnings.append(f"High leverage: elasticity {elasticity:.1f} — a 1% underlying move "
                        f"changes position value by roughly {abs(elasticity):.0f}%.")
    if days < limits.near_expiry_days:
        warnings.append(f"Near expiry ({days:.1f} days): gamma and theta are large; value can "
                        "change abruptly and decay quickly.")
    if relative_spread is not None and relative_spread > limits.wide_relative_spread:
        warnings.append(f"Liquidity: bid-ask spread is {relative_spread:.0%} of mid; exiting at "
                        "the bid would realise a material loss versus the model value.")

    return ExposureReport(
        spot=spot, days_to_expiry=days, model_value_per_unit=value_pu,
        position_value=position_value, delta_per_unit=delta, gamma_per_unit=float(g.gamma),
        share_equivalent=share_eq, delta_notional=delta_notional,
        gross_underlying_notional=units * spot, elasticity=elasticity,
        premium_at_risk=position_value, cost_basis=cost_basis,
        theta_per_day=float(g.theta_per_day) * units,
        vega_per_vol_point=float(g.vega_per_vol_point) * units,
        gamma_dollar_per_1pct=float(g.gamma) * units * spot * spot * 0.01,
        pct_equity_at_risk=pct_risk, pct_equity_delta_notional=pct_notional,
        assumptions=assumptions.describe(), warnings=warnings,
    )


def compare_with_shares(position: OptionPosition, spot: float, valuation_time: datetime,
                        assumptions: ModelAssumptions, spot_moves: Sequence[float],
                        horizon_days: float = 0.0) -> pd.DataFrame:
    """Compare the option with two share-based alternatives across spot moves.

    * Delta-equivalent shares: same initial *exposure* (share-equivalent units).
      For a put this is a short stock position — shown only as a hypothetical
      benchmark (it would require margin and stock borrow).
    * Capital-equivalent shares: the same *capital* as the option's current model
      value, deployed in shares in the option's direction (long for calls; short
      for puts, again hypothetical).

    Option P&L is model-based at the horizon; share P&L is exact (ignoring
    dividends and financing).
    """
    c = position.contract
    t0 = require_aware(valuation_time, "valuation_time")
    t1 = t0 + timedelta(days=horizon_days)
    v0 = float(model_value_per_unit(position, spot, t0, assumptions)) * position.units
    g = bsm_greeks(c.option_type, spot, c.strike, time_to_expiry(t0, c.expiry),
                   assumptions.rate, assumptions.volatility, assumptions.dividend_yield)
    delta_shares = float(g.delta) * position.units
    direction = np.sign(delta_shares) or 1.0
    capital_shares = direction * v0 / spot

    moves = np.asarray(spot_moves, dtype=float)
    s1 = spot * (1 + moves)
    if np.any(s1 <= 0):
        raise InvalidInputError("Spot moves of -100% or worse are not valid scenarios.")
    v1 = np.asarray(model_value_per_unit(position, s1, t1, assumptions)) * position.units
    return pd.DataFrame({
        "spot_move": moves,
        "spot": s1,
        "option_pnl": v1 - v0,
        "option_return_on_capital": (v1 - v0) / v0 if v0 > 0 else np.nan,
        "delta_equiv_shares_pnl": delta_shares * (s1 - spot),
        "delta_equiv_capital": abs(delta_shares) * spot,
        "capital_equiv_shares_pnl": capital_shares * (s1 - spot),
        "capital_equiv_return": direction * moves,
    }).assign(option_capital=v0, delta_equiv_shares=delta_shares,
              capital_equiv_shares=capital_shares)
