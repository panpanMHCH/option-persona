"""Synthetic European option chains with deliberately injected parity violations.

Purpose: provide ground-truth test data for the arbitrage scanner. Every row is
flagged ``is_synthetic = True`` and carries ``source = "synthetic:bsm"``; these
prices must never be presented as market observations.

Construction
------------
1. Each strike gets a volatility from a simple quadratic smile in log-moneyness,
   sigma(K) = sigma_atm + skew * k + curvature * k^2 with k = ln(K / F).
2. Call and put fair values come from BSM using the *same* sigma(K), r and q,
   so their mids satisfy put-call parity exactly regardless of the smile shape
   (parity is model-free; the smile does not affect it).
3. Bid/ask quotes are built around fair value with a half-spread that widens
   for cheaper options (a crude but realistic pattern), rounded to the tick.
4. Violations are injected by shifting both the call bid and ask by a chosen
   amount per share at chosen strikes; the shift is recorded in
   ``injected_call_shift`` so tests can verify detection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from option_persona.arbitrage.parity import (
    ExecutionCosts,
    ParityQuoteSet,
    assess_parity,
)
from option_persona.core.time import UTC
from option_persona.core.types import ExerciseStyle, OptionType, SettlementType
from option_persona.pricing.black_scholes import bsm_price

SYNTHETIC_SOURCE = "synthetic:bsm"


def _quote_side(fair: np.ndarray, rel_half_spread: float, min_half_spread: float,
                tick: float) -> tuple[np.ndarray, np.ndarray]:
    half = np.maximum(rel_half_spread * fair, min_half_spread)
    bid = np.maximum(np.floor((fair - half) / tick) * tick, 0.0)
    ask = np.ceil((fair + half) / tick) * tick
    return bid, np.maximum(ask, tick)


def generate_synthetic_chain(
    spot: float = 100.0,
    strikes: Sequence[float] | None = None,
    time_to_expiry: float = 30 / 365,
    rate: float = 0.04,
    dividend_yield: float = 0.01,
    sigma_atm: float = 0.20,
    skew: float = -0.25,
    curvature: float = 0.6,
    rel_half_spread: float = 0.01,
    min_half_spread: float = 0.025,
    tick: float = 0.01,
    underlying_half_spread: float = 0.01,
    observed_at: datetime | None = None,
    violations: Mapping[float, float] | None = None,
) -> pd.DataFrame:
    """Return one row per strike with call/put bid/ask, fair values and provenance.

    ``violations`` maps strike -> per-share shift applied to the call bid and ask.
    """
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 5.0)
    K = np.asarray(strikes, dtype=float)
    T, r, q = time_to_expiry, rate, dividend_yield
    forward = spot * np.exp((r - q) * T)
    k = np.log(K / forward)
    sigma = np.clip(sigma_atm + skew * k + curvature * k * k, 0.05, None)

    call_fair = np.asarray(bsm_price(OptionType.CALL, spot, K, T, r, sigma, q))
    put_fair = np.asarray(bsm_price(OptionType.PUT, spot, K, T, r, sigma, q))
    call_bid, call_ask = _quote_side(call_fair, rel_half_spread, min_half_spread, tick)
    put_bid, put_ask = _quote_side(put_fair, rel_half_spread, min_half_spread, tick)

    shifts = np.zeros_like(K)
    for strike, shift in (violations or {}).items():
        mask = np.isclose(K, strike)
        if not mask.any():
            raise ValueError(f"Violation strike {strike} not in chain.")
        shifts[mask] = shift
    call_bid = np.maximum(call_bid + shifts, 0.0)
    call_ask = call_ask + shifts

    stamp = observed_at or datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
    return pd.DataFrame({
        "strike": K,
        "time_to_expiry": T,
        "sigma_used": sigma,
        "call_fair": call_fair,
        "put_fair": put_fair,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "underlying_bid": spot - underlying_half_spread,
        "underlying_ask": spot + underlying_half_spread,
        "injected_call_shift": shifts,
        "observed_at": stamp,
        "source": SYNTHETIC_SOURCE,
        "is_synthetic": True,
    })


def scan_chain_for_parity(chain: pd.DataFrame, rate: float, dividend_yield: float,
                          costs: ExecutionCosts | None = None,
                          exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
                          settlement: SettlementType = SettlementType.CASH,
                          multiplier: float = 100.0) -> pd.DataFrame:
    """Run :func:`assess_parity` on every row of a chain DataFrame."""
    rows = []
    for rec in chain.to_dict("records"):
        stamp = rec.get("observed_at")
        qs = ParityQuoteSet(
            strike=rec["strike"], time_to_expiry=rec["time_to_expiry"],
            call_bid=rec["call_bid"], call_ask=rec["call_ask"],
            put_bid=rec["put_bid"], put_ask=rec["put_ask"],
            underlying_bid=rec["underlying_bid"], underlying_ask=rec["underlying_ask"],
            exercise_style=exercise_style, settlement=settlement, multiplier=multiplier,
            call_observed_at=stamp, put_observed_at=stamp, underlying_observed_at=stamp,
            is_synthetic=bool(rec.get("is_synthetic", False)),
        )
        a = assess_parity(qs, rate, dividend_yield, costs)
        rows.append({
            "strike": a.strike,
            "mid_residual": a.mid_residual,
            "relative_mid_residual": a.relative_mid_residual,
            "half_spread_cost": a.half_spread_cost,
            "conversion_edge": a.conversion_edge,
            "reversal_edge": a.reversal_edge,
            "best_strategy": a.best_strategy,
            "classification": a.classification.value,
            "risk_notes": a.risk_notes,
        })
    return pd.DataFrame(rows)
