"""Put-call parity: identity, executable-edge arithmetic and synthetic-chain detection."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from option_persona.arbitrage import (
    ExecutionCosts,
    ParityClassification,
    ParityQuoteSet,
    assess_parity,
    generate_synthetic_chain,
    implied_forward,
    put_call_parity_residual,
    scan_chain_for_parity,
)
from option_persona.core.errors import InvalidInputError, NotApplicableError
from option_persona.core.time import UTC
from option_persona.core.types import ExerciseStyle, SettlementType
from option_persona.pricing import bsm_call, bsm_put

ZERO_COSTS = ExecutionCosts(option_fee_per_contract=0.0, underlying_fee_per_unit=0.0,
                            borrow_rate_spread=0.0, lend_rate_spread=0.0, stock_borrow_fee=0.0)
NOW = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)


def _quotes(c, p, s, half=0.0, **kw) -> ParityQuoteSet:
    base = dict(strike=100.0, time_to_expiry=0.5, call_bid=c - half, call_ask=c + half,
                put_bid=p - half, put_ask=p + half, underlying_bid=s - half,
                underlying_ask=s + half, call_observed_at=NOW, put_observed_at=NOW,
                underlying_observed_at=NOW)
    return ParityQuoteSet(**(base | kw))


def test_residual_is_zero_for_bsm_prices():
    S, K, T, r, q, v = 100.0, 95.0, 0.5, 0.04, 0.01, 0.3
    c, p = bsm_call(S, K, T, r, v, q), bsm_put(S, K, T, r, v, q)
    assert put_call_parity_residual(c, p, S, K, T, r, q) == pytest.approx(0.0, abs=1e-12)


def test_implied_forward_matches_carry_forward():
    S, K, T, r, q, v = 100.0, 95.0, 0.5, 0.04, 0.01, 0.3
    c, p = bsm_call(S, K, T, r, v, q), bsm_put(S, K, T, r, v, q)
    assert implied_forward(c, p, K, T, r) == pytest.approx(S * math.exp((r - q) * T))


def test_zero_spread_zero_cost_edges_are_zero_at_fair_value():
    S, K, T, r, v = 100.0, 100.0, 0.5, 0.04, 0.25
    qs = _quotes(bsm_call(S, K, T, r, v), bsm_put(S, K, T, r, v), S)
    a = assess_parity(qs, r, 0.0, ZERO_COSTS)
    assert a.conversion_edge == pytest.approx(0.0, abs=1e-9)
    assert a.reversal_edge == pytest.approx(0.0, abs=1e-9)
    assert a.classification is ParityClassification.NO_VIOLATION


def test_conversion_edge_hand_computed():
    # Call overpriced by 0.50/share. Conversion: sell call @ bid, buy put @ ask, buy stock @ ask.
    S, K, T, r = 100.0, 100.0, 0.5, 0.04
    c_fair, p_fair = bsm_call(S, K, T, r, 0.25), bsm_put(S, K, T, r, 0.25)
    qs = _quotes(c_fair + 0.50, p_fair, S, half=0.05)
    costs = ExecutionCosts(option_fee_per_contract=0.65, underlying_fee_per_unit=0.0,
                           borrow_rate_spread=0.0, lend_rate_spread=0.0, stock_borrow_fee=0.0)
    a = assess_parity(qs, r, 0.0, costs)
    # Per share: +0.50 mispricing minus three half-spreads of 0.05 = 0.35; x100 minus 2 x 0.65 fees.
    assert a.conversion_edge == pytest.approx(0.35 * 100 - 1.30, abs=1e-9)
    assert a.best_strategy == "conversion"
    assert a.classification is ParityClassification.APPARENT_EXECUTABLE_EDGE
    assert a.half_spread_cost == pytest.approx(0.15)


def test_financing_spread_reduces_edges():
    S, K, T, r = 100.0, 100.0, 1.0, 0.04
    qs = _quotes(bsm_call(S, K, T, r, 0.25), bsm_put(S, K, T, r, 0.25), S, time_to_expiry=T)
    costs = ExecutionCosts(0.0, 0.0, borrow_rate_spread=0.01, lend_rate_spread=0.01,
                           stock_borrow_fee=0.0)
    a = assess_parity(qs, r, 0.0, costs)
    assert a.conversion_edge < 0 and a.reversal_edge < 0


def test_american_options_are_rejected():
    qs = _quotes(5.0, 4.0, 100.0, half=0.05, exercise_style=ExerciseStyle.AMERICAN)
    with pytest.raises(NotApplicableError, match="only for European"):
        assess_parity(qs, 0.04)


def test_crossed_quote_rejected():
    with pytest.raises(InvalidInputError, match="Invalid call quote"):
        _quotes(5.0, 4.0, 100.0, call_bid=5.2, call_ask=5.1)


def test_stale_leg_flagged():
    qs = _quotes(5.0, 4.0, 100.0, half=0.05, put_observed_at=NOW - timedelta(seconds=30))
    a = assess_parity(qs, 0.04)
    assert any("30.0s apart" in n for n in a.risk_notes)


def test_settlement_specific_risk_notes():
    cash = assess_parity(_quotes(5.0, 4.0, 100.0, half=0.05), 0.04)
    phys = assess_parity(_quotes(5.0, 4.0, 100.0, half=0.05, settlement=SettlementType.PHYSICAL),
                         0.04)
    assert any("index itself is not tradeable" in n for n in cash.risk_notes)
    assert any("pin risk" in n for n in phys.risk_notes)


# ---------------------------------------------------------------- synthetic chain


def test_clean_synthetic_chain_has_no_executable_edges():
    chain = generate_synthetic_chain()
    assert chain["is_synthetic"].all() and (chain["source"] == "synthetic:bsm").all()
    scan = scan_chain_for_parity(chain, 0.04, 0.01)
    assert (scan["classification"] != ParityClassification.APPARENT_EXECUTABLE_EDGE).all()
    # Fair values satisfy parity exactly despite the volatility smile.
    res = put_call_parity_residual(chain["call_fair"], chain["put_fair"], 100.0,
                                   chain["strike"], chain["time_to_expiry"], 0.04, 0.01)
    assert abs(res).max() < 1e-10


def test_injected_violations_are_detected_with_correct_direction():
    injected = {90.0: 0.60, 105.0: -0.50, 110.0: 0.03}
    scan = scan_chain_for_parity(generate_synthetic_chain(violations=injected), 0.04, 0.01)
    by_k = scan.set_index("strike")
    assert by_k.loc[90.0, "classification"] == ParityClassification.APPARENT_EXECUTABLE_EDGE
    assert by_k.loc[90.0, "best_strategy"] == "conversion"
    assert by_k.loc[105.0, "classification"] == ParityClassification.APPARENT_EXECUTABLE_EDGE
    assert by_k.loc[105.0, "best_strategy"] == "reversal"
    # A 3-cent mispricing is visible at the mid but is not executable through the spread.
    assert by_k.loc[110.0, "classification"] == ParityClassification.THEORETICAL_ONLY
    # Mid residual recovers the injected shift up to tick-rounding noise.
    for k, shift in injected.items():
        assert by_k.loc[k, "mid_residual"] == pytest.approx(shift, abs=0.03)
    assert all(any("SYNTHETIC DATA" in n for n in notes) for notes in scan["risk_notes"])
