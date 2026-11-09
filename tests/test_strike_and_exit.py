"""Strike comparison and hold-or-sell review."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pytest

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionContract, OptionPosition
from option_persona.core.time import BANGKOK, NEW_YORK, time_to_expiry
from option_persona.core.types import ExerciseStyle, SettlementType
from option_persona.pricing import bsm_greeks, bsm_price, implied_volatility, risk_neutral_prob_itm
from option_persona.risk import (
    ModelAssumptions,
    ReviewStatus,
    StrikeCandidate,
    compare_strikes,
    exit_review,
    expected_move,
)

EXPIRY = date(2026, 11, 6)
A = ModelAssumptions(volatility=0.46, rate=0.04)
NOW = datetime(2026, 10, 30, 10, 0, tzinfo=NEW_YORK)


# ====================================================================== strike comparison
def test_model_rows_match_direct_pricing():
    table = compare_strikes("put", 201.5, EXPIRY, NOW, [185, 200, 195, 190], A)
    assert list(table.strike) == [185, 190, 195, 200]  # sorted
    T = time_to_expiry(NOW, EXPIRY)
    for row in table.itertuples():
        g = bsm_greeks("put", 201.5, row.strike, T, 0.04, 0.46)
        assert row.price_basis == "model (flat σ)"
        assert row.entry_price == pytest.approx(g.price, rel=1e-12)
        assert row.cost == pytest.approx(g.price * 100, rel=1e-12)
        assert row.breakeven == pytest.approx(row.strike - g.price, rel=1e-12)
        assert row.delta == pytest.approx(g.delta, rel=1e-12)
        assert row.theta_per_day == pytest.approx(g.theta_per_day * 100, rel=1e-12)
        assert row.elasticity == pytest.approx(g.delta * 201.5 / g.price, rel=1e-12)
        assert row.q_prob_itm == pytest.approx(risk_neutral_prob_itm("put", 201.5, row.strike, T, 0.04, 0.46))
        assert math.isnan(row.relative_spread)


def test_strike_tradeoffs_are_monotone_for_otm_puts():
    t = compare_strikes("put", 201.5, EXPIRY, NOW, [185, 190, 195, 200], A)
    assert np.all(np.diff(t.cost) > 0)  # higher strike costs more
    assert np.all(np.diff(t.q_prob_itm) > 0)  # and is more likely to finish ITM
    assert np.all(np.diff(t.move_to_breakeven) > 0)  # and needs a smaller fall
    assert np.all(np.diff(np.abs(t.elasticity)) < 0)  # but is less leveraged
    assert (t.q_prob_beyond_breakeven < t.q_prob_itm).all()  # breakeven is beyond the strike


def test_calls_mirror_puts():
    t = compare_strikes("call", 100, date(2026, 12, 18), NOW, [95, 100, 110], ModelAssumptions(0.25))
    assert (t.breakeven > t.strike).all() and (t.delta > 0).all()
    assert np.all(np.diff(t.cost) < 0)


def test_observed_rows_enter_at_the_ask_with_its_own_iv():
    T = time_to_expiry(NOW, EXPIRY)
    ask_iv = 0.52  # deliberate skew relative to the 46% reference
    ask = float(bsm_price("put", 201.5, 190, T, 0.04, ask_iv))
    t = compare_strikes("put", 201.5, EXPIRY, NOW,
                        [StrikeCandidate(190, bid=ask - 0.10, ask=ask), 195], A)
    obs = t[t.strike == 190].iloc[0]
    assert obs.price_basis == "ask (observed)" and obs.entry_price == ask
    assert obs.volatility == pytest.approx(ask_iv, abs=1e-9)
    assert obs.relative_spread == pytest.approx(0.10 / (ask - 0.05))
    assert t[t.strike == 195].iloc[0].volatility == 0.46


def test_fees_shift_breakeven_by_fee_per_unit():
    base = compare_strikes("put", 201.5, EXPIRY, NOW, [195], A).iloc[0]
    fee = compare_strikes("put", 201.5, EXPIRY, NOW, [195], A, contracts=2,
                          fee_per_contract=0.65).iloc[0]
    assert fee.breakeven == pytest.approx(base.breakeven - 0.65 / 100)
    assert fee.cost == pytest.approx(2 * base.cost + 1.30)


def test_ask_below_intrinsic_has_no_iv_but_keeps_row():
    t = compare_strikes("put", 150, EXPIRY, NOW, [StrikeCandidate(195, bid=40, ask=41)], A)
    row = t.iloc[0]
    assert math.isnan(row.volatility) and math.isnan(row.delta)
    assert "No implied volatility" in row.note


def test_expected_move_and_attrs():
    T = time_to_expiry(NOW, EXPIRY)
    assert expected_move(201.5, 0.46, T) == pytest.approx(201.5 * 0.46 * math.sqrt(T))
    t = compare_strikes("put", 201.5, EXPIRY, NOW, [195], A)
    assert t.attrs["expected_move"] == pytest.approx(expected_move(201.5, 0.46, T))
    row = t.iloc[0]
    em = t.attrs["expected_move"]
    assert row.breakeven_in_expected_moves == pytest.approx((row.breakeven - 201.5) / em)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"candidates": []}, "at least one"),
    ({"candidates": [195, 195]}, "only once"),
    ({"spot": -1}, "positive"),
    ({"valuation_time": datetime(2026, 11, 6, 16, 0, tzinfo=NEW_YORK)}, "nothing left"),
])
def test_strike_comparison_input_validation(kwargs, match):
    args = {"option_type": "put", "spot": 201.5, "expiry": EXPIRY, "valuation_time": NOW,
            "candidates": [195], "assumptions": A} | kwargs
    with pytest.raises(InvalidInputError, match=match):
        compare_strikes(**args)


def test_candidate_quote_validation():
    with pytest.raises(InvalidInputError, match="both bid and ask"):
        StrikeCandidate(195, bid=1.0)
    with pytest.raises(InvalidInputError, match="crossed"):
        StrikeCandidate(195, bid=1.0, ask=0.9)


# ====================================================================== exit review
def _position(strike=195.0, otype="put", settlement=SettlementType.PHYSICAL, **kw):
    contract = OptionContract(underlying="NVDA", option_type=otype, strike=strike, expiry=EXPIRY,
                              settlement=settlement)
    return OptionPosition(**({"contract": contract, "quantity": 1, "entry_premium": 2.38,
                              "entry_underlying_price": 201.5} | kw))


def test_review_never_recommends():
    rev = exit_review(_position(), 201.5, NOW, A)
    text = " ".join(i.finding + i.why for i in rev.items).lower()
    assert "you should" not in text and "recommend" not in rev.disclaimer.replace("not a recommendation", "")
    assert {i.factor for i in rev.items} == {
        "Reason for the trade", "Re-buy test", "Time left", "Time decay", "Intrinsic vs time value",
        "Expiry settlement", "Liquidity", "Unrealised P&L"}


def test_thesis_answer_drives_status():
    assert exit_review(_position(), 201.5, NOW, A).item("Reason for the trade").status is ReviewStatus.WATCH
    assert exit_review(_position(), 201.5, NOW, A, thesis_intact=True).item(
        "Reason for the trade").status is ReviewStatus.OK
    assert exit_review(_position(), 201.5, NOW, A, thesis_intact=False).item(
        "Reason for the trade").status is ReviewStatus.ALERT


def test_time_left_reports_bangkok_instant_and_escalates():
    rev = exit_review(_position(), 201.5, NOW, A)
    assert rev.expiry_bangkok == datetime(2026, 11, 7, 4, 0, tzinfo=BANGKOK)  # 16:00 EST = 04:00 BKK
    assert rev.item("Time left").status is ReviewStatus.OK  # 7.3 days
    late = datetime(2026, 11, 5, 10, 0, tzinfo=NEW_YORK)
    assert exit_review(_position(), 201.5, late, A).item("Time left").status is ReviewStatus.ALERT


def test_values_use_bid_when_quoted_and_model_otherwise():
    model = exit_review(_position(), 201.5, NOW, A)
    T = time_to_expiry(NOW, EXPIRY)
    assert model.value_basis == "model"
    assert model.value_per_unit == pytest.approx(float(bsm_price("put", 201.5, 195, T, 0.04, 0.46)))
    quoted = exit_review(_position(entry_fees=0.65), 201.5, NOW, A, bid=2.10, ask=2.20)
    assert quoted.value_basis == "bid (observed)" and quoted.value_per_unit == 2.10
    assert quoted.unrealised_pnl == pytest.approx((2.10 - 2.38) * 100 - 0.65)
    assert quoted.intrinsic_per_unit == 0 and quoted.time_value_per_unit == 2.10


def test_rebuy_breakeven_uses_the_ask():
    rev = exit_review(_position(), 201.5, NOW, A, bid=2.10, ask=2.20)
    assert "192.80" in rev.item("Re-buy test").finding  # 195 - 2.20


def test_itm_physical_put_near_expiry_flags_short_shares():
    late = datetime(2026, 11, 5, 10, 0, tzinfo=NEW_YORK)
    item = exit_review(_position(), 190.0, late, A).item("Expiry settlement")
    assert item.status is ReviewStatus.ALERT
    assert "sell 100 shares at 195.00" in item.finding and "short stock" in item.finding
    call = exit_review(_position(strike=180, otype="call", entry_premium=10), 190.0, late, A)
    assert "buy 100 shares for 18,000" in call.item("Expiry settlement").finding


def test_near_strike_otm_is_watch_and_far_otm_is_ok():
    late = datetime(2026, 11, 5, 10, 0, tzinfo=NEW_YORK)
    assert exit_review(_position(), 197.0, late, A).item("Expiry settlement").status is ReviewStatus.WATCH
    assert exit_review(_position(), 240.0, late, A).item("Expiry settlement").status is ReviewStatus.OK


def test_cash_settled_never_mentions_share_delivery():
    late = datetime(2026, 11, 5, 10, 0, tzinfo=NEW_YORK)
    item = exit_review(_position(settlement=SettlementType.CASH), 190.0, late, A).item("Expiry settlement")
    assert item.status is ReviewStatus.INFO and "no shares" in item.finding


def test_bid_below_intrinsic_on_american_itm_option():
    pos = _position()
    assert pos.contract.exercise_style is ExerciseStyle.AMERICAN
    item = exit_review(pos, 180.0, NOW, A, bid=14.50, ask=15.40).item("Intrinsic vs time value")
    assert item.status is ReviewStatus.WATCH and "below intrinsic" in item.finding


def test_liquidity_thresholds():
    def status(bid, ask):
        return exit_review(_position(), 201.5, NOW, A, bid=bid, ask=ask).item("Liquidity").status
    assert status(2.00, 2.04) is ReviewStatus.OK  # ~2%
    assert status(2.00, 2.14) is ReviewStatus.WATCH  # ~7%
    assert status(2.00, 2.40) is ReviewStatus.ALERT  # ~18%
    assert status(0.00, 0.05) is ReviewStatus.ALERT  # no buyer
    assert exit_review(_position(), 201.5, NOW, A).item("Liquidity").status is ReviewStatus.INFO


def test_time_decay_escalates_into_expiry():
    early = exit_review(_position(), 201.5, datetime(2026, 9, 1, 10, tzinfo=NEW_YORK), A)
    late = exit_review(_position(), 201.5, datetime(2026, 11, 5, 10, tzinfo=NEW_YORK), A)
    assert early.item("Time decay").status is ReviewStatus.OK
    assert late.item("Time decay").status is ReviewStatus.ALERT


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"bid": 1.0}, "both"),
    ({"bid": 2.0, "ask": 1.0}, "Invalid quote"),
    ({"spot": 0}, "positive"),
    ({"valuation_time": datetime(2026, 11, 6, 17, tzinfo=NEW_YORK)}, "expired"),
])
def test_exit_review_validation(kwargs, match):
    args = {"position": _position(), "spot": 201.5, "valuation_time": NOW, "assumptions": A} | kwargs
    with pytest.raises(InvalidInputError, match=match):
        exit_review(**args)


def test_closed_position_rejected():
    with pytest.raises(InvalidInputError, match="already closed"):
        exit_review(_position(exit_premium=3.0), 201.5, NOW, A)


def test_iv_helper_consistency():
    """Sanity: the observed-row IV equals a direct solve on the ask."""
    T = time_to_expiry(NOW, EXPIRY)
    t = compare_strikes("put", 201.5, EXPIRY, NOW, [StrikeCandidate(195, bid=2.30, ask=2.46)], A)
    assert t.iloc[0].volatility == pytest.approx(implied_volatility(2.46, "put", 201.5, 195, T, 0.04))
