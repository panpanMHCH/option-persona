"""Expiration payoff, breakeven and maximum loss (model-free)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from option_persona.risk import (
    breakeven_at_expiry,
    expiration_pnl,
    expiration_profile,
    intrinsic_value,
)


def test_intrinsic_value():
    assert intrinsic_value("call", [90, 100, 110], 100) .tolist() == [0, 0, 10]
    assert intrinsic_value("put", [90, 100, 110], 100).tolist() == [10, 0, 0]


def test_nvda_put_expiry_profile(nvda_put_position):
    prof = expiration_profile(nvda_put_position)
    assert prof.premium_paid == pytest.approx(238.0)
    assert prof.max_loss == pytest.approx(238.0)
    assert prof.breakeven == pytest.approx(192.62)
    assert prof.max_gain == pytest.approx((195.0 - 2.38) * 100)


def test_nvda_put_pnl_points(nvda_put_position):
    pnl = expiration_pnl(nvda_put_position, np.array([210.0, 195.0, 192.62, 185.0]))
    assert pnl == pytest.approx([-238.0, -238.0, 0.0, 762.0])


def test_breakeven_is_a_zero_of_pnl(long_call_position):
    be = breakeven_at_expiry(long_call_position)
    # 2 contracts x 100 = 200 units; fees 2.60 add 0.013/unit.
    assert be == pytest.approx(100.0 + 4.0 + 2.60 / 200)
    assert expiration_pnl(long_call_position, be) == pytest.approx(0.0, abs=1e-9)


def test_call_max_gain_unbounded_and_fees_in_loss(long_call_position):
    prof = expiration_profile(long_call_position)
    assert math.isinf(prof.max_gain)
    assert prof.max_loss == pytest.approx(4.0 * 200 + 2.60)
    assert expiration_profile(long_call_position, include_fees=False).max_loss == pytest.approx(800)


def test_pnl_never_below_max_loss(long_call_position, nvda_put_position):
    grid = np.linspace(0.0, 400.0, 2001)
    for pos in (long_call_position, nvda_put_position):
        assert expiration_pnl(pos, grid).min() >= -expiration_profile(pos).max_loss - 1e-9
