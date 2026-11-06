"""Shared fixtures and an optional QuantLib oracle for cross-validation."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from option_persona.core.models import OptionContract, OptionPosition
from option_persona.core.time import NEW_YORK

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover
    ql = None

requires_quantlib = pytest.mark.skipif(ql is None, reason="QuantLib not installed")


def quantlib_european(option_type: str, S: float, K: float, days: int, r: float, sigma: float,
                      q: float) -> dict[str, float]:
    """Price and Greeks from QuantLib's AnalyticEuropeanEngine (independent implementation).

    Uses Actual/365 Fixed and continuously compounded flat curves so conventions
    match ours exactly; T = days / 365.
    """
    assert ql is not None
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc, ql.Continuous))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc, ql.Continuous))
    v_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, dc))
    process = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, v_ts)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(today + days))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "theta": option.theta(),
        "rho": option.rho(),
    }


@pytest.fixture
def nvda_put_position() -> OptionPosition:
    """The user's NVDA case-study position. Entry timestamp is not recorded (None)."""
    contract = OptionContract(underlying="NVDA", option_type="put", strike=195.0,
                              expiry=date(2026, 11, 6))
    return OptionPosition(contract=contract, quantity=1, entry_premium=2.38,
                          entry_underlying_price=201.5, observed_entry_iv=0.46,
                          observed_entry_delta=-0.29)


@pytest.fixture
def long_call_position() -> OptionPosition:
    contract = OptionContract(underlying="TEST", option_type="call", strike=100.0,
                              expiry=date(2026, 12, 18))
    return OptionPosition(contract=contract, quantity=2, entry_premium=4.00,
                          entry_underlying_price=100.0, entry_fees=1.30, exit_fees=1.30)


@pytest.fixture
def valuation_time() -> datetime:
    return datetime(2026, 10, 30, 10, 0, tzinfo=NEW_YORK)
