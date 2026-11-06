"""Scenario engine and leverage analytics."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from option_persona.core.errors import InvalidInputError
from option_persona.core.time import time_to_expiry
from option_persona.pricing import bsm_delta, bsm_price
from option_persona.risk import (
    ModelAssumptions,
    RiskLimits,
    compare_with_shares,
    expiration_pnl,
    exposure_report,
    model_caveats,
    scenario_grid,
    spot_vol_stress,
)

ASSUME = ModelAssumptions(volatility=0.46, rate=0.04)


def test_grid_at_expiry_equals_payoff(nvda_put_position, valuation_time):
    spots = np.linspace(150, 250, 21)
    days_to_exp = time_to_expiry(valuation_time, nvda_put_position.contract.expiry) * 365
    grid = scenario_grid(nvda_put_position, spots, [days_to_exp + 1], valuation_time, ASSUME)
    assert grid["days_to_expiry"].iloc[0] == 0.0
    assert grid["pnl"].to_numpy() == pytest.approx(expiration_pnl(nvda_put_position, spots))


def test_grid_today_matches_bsm(nvda_put_position, valuation_time):
    grid = scenario_grid(nvda_put_position, [201.5], [0], valuation_time, ASSUME)
    T = time_to_expiry(valuation_time, nvda_put_position.contract.expiry)
    expected = bsm_price("put", 201.5, 195, T, 0.04, 0.46) * 100
    assert grid["position_value"].iloc[0] == pytest.approx(expected)


def test_time_decay_reduces_otm_value(nvda_put_position, valuation_time):
    grid = scenario_grid(nvda_put_position, [205.0], [0, 2, 4, 6], valuation_time, ASSUME)
    assert np.all(np.diff(grid["position_value"].to_numpy()) < 0)


def test_stress_monotone_in_spot_and_vol(nvda_put_position, valuation_time):
    m = spot_vol_stress(nvda_put_position, 201.5, valuation_time, ASSUME,
                        spot_moves=[-0.2, -0.1, 0, 0.1], vol_shifts=[-0.1, 0, 0.1])
    assert np.all(np.diff(m.to_numpy(), axis=0) < 0)  # put loses as spot rises
    assert np.all(np.diff(m.to_numpy(), axis=1) > 0)  # long option gains with vol


def test_invalid_vol_shift(nvda_put_position, valuation_time):
    with pytest.raises(InvalidInputError, match="non-positive"):
        spot_vol_stress(nvda_put_position, 200, valuation_time, ASSUME, [0.0], [-0.5])


def test_american_put_caveat(nvda_put_position):
    assert any("lower bound" in n for n in model_caveats(nvda_put_position))


def test_exposure_identities(nvda_put_position, valuation_time):
    S = 201.5
    rep = exposure_report(nvda_put_position, S, valuation_time, ASSUME, portfolio_equity=50_000)
    T = time_to_expiry(valuation_time, nvda_put_position.contract.expiry)
    delta = bsm_delta("put", S, 195, T, 0.04, 0.46)
    value = bsm_price("put", S, 195, T, 0.04, 0.46)
    assert rep.share_equivalent == pytest.approx(delta * 100)
    assert rep.delta_notional == pytest.approx(delta * 100 * S)
    assert rep.elasticity == pytest.approx(delta * S / value)
    assert rep.pct_equity_delta_notional == pytest.approx(abs(delta) * 100 * S / 50_000)
    # Elasticity equals d ln V / d ln S numerically.
    h = 1e-5
    fd = (np.log(bsm_price("put", S * (1 + h), 195, T, 0.04, 0.46))
          - np.log(bsm_price("put", S * (1 - h), 195, T, 0.04, 0.46))) / (2 * h)
    assert rep.elasticity == pytest.approx(fd, rel=1e-6)


def test_exposure_warnings(nvda_put_position, valuation_time):
    rep = exposure_report(nvda_put_position, 201.5, valuation_time, ASSUME,
                          portfolio_equity=5_000, relative_spread=0.2,
                          limits=RiskLimits(near_expiry_days=10))
    text = " ".join(rep.warnings)
    for key in ("Concentration", "Exposure", "High leverage", "Near expiry", "Liquidity"):
        assert key in text


def test_share_comparison_small_moves_agree(nvda_put_position, valuation_time):
    df = compare_with_shares(nvda_put_position, 201.5, valuation_time, ASSUME,
                             spot_moves=[-1e-4, 1e-4])
    # For tiny instantaneous moves the option behaves like its delta-equivalent shares.
    assert df["option_pnl"].to_numpy() == pytest.approx(df["delta_equiv_shares_pnl"].to_numpy(),
                                                        rel=1e-2)
    assert (df["delta_equiv_shares"] < 0).all()


def test_share_comparison_shows_convexity(nvda_put_position, valuation_time):
    df = compare_with_shares(nvda_put_position, 201.5, valuation_time, ASSUME,
                             spot_moves=[-0.10, 0.10])
    # Long gamma: option beats delta-equivalent shares on large moves in both directions.
    assert (df["option_pnl"] > df["delta_equiv_shares_pnl"]).all()
    # ...but the option can never lose more than its value.
    assert (df["option_pnl"] >= -df["option_capital"]).all()


def test_horizon_after_valuation(nvda_put_position, valuation_time):
    later = compare_with_shares(nvda_put_position, 201.5, valuation_time, ASSUME, [0.0],
                                horizon_days=3)
    assert later["option_pnl"].iloc[0] < 0  # pure theta loss at unchanged spot
    _ = timedelta  # keeps import explicit for readers
