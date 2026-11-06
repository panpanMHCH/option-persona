"""Implied volatility: round trips, conditioning, bounds and error handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from option_persona.core.errors import ArbitrageBoundError, InvalidInputError
from option_persona.pricing import (
    bsm_price,
    bsm_vega,
    european_price_bounds,
    implied_volatility,
    implied_volatility_array,
)

ROUND_TRIP = [
    (kind, K, T, sigma)
    for kind in ("call", "put")
    for K in (50.0, 80.0, 95.0, 100.0, 105.0, 125.0, 200.0)
    for T in (1 / 365, 7 / 365, 0.25, 1.0, 5.0)
    for sigma in (0.03, 0.15, 0.46, 1.0, 3.0)
]


@pytest.mark.parametrize(("kind", "K", "T", "sigma"), ROUND_TRIP)
def test_round_trip(kind, K, T, sigma):
    S, r, q = 100.0, 0.04, 0.01
    price = bsm_price(kind, S, K, T, r, sigma, q)
    lb = european_price_bounds(kind, S, K, T, r, q).lower
    time_value = price - lb
    if time_value < 1e-12 * S:
        pytest.skip("No time value at double precision: volatility is not identifiable.")
    iv = implied_volatility(price, kind, S, K, T, r, q)
    # Price-space round trip is always tight.
    assert bsm_price(kind, S, K, T, r, iv, q) == pytest.approx(price, abs=1e-11 * S)
    # Vol-space accuracy is limited by conditioning: d(sigma) ~ d(price) / vega.
    vega = bsm_vega(S, K, T, r, sigma, q)
    vol_tol = max(1e-8, 1e-11 * S / vega) if vega > 0 else 1.0
    assert iv == pytest.approx(sigma, abs=vol_tol)


def test_deep_itm_call_uses_otm_conditioning():
    # Deep ITM call: price ~ 49.99, time value tiny; parity conversion keeps accuracy.
    S, K, T, r, sigma = 150.0, 100.0, 0.1, 0.03, 0.25
    price = bsm_price("call", S, K, T, r, sigma)
    assert implied_volatility(price, "call", S, K, T, r) == pytest.approx(sigma, abs=1e-6)


def test_call_and_put_give_same_iv_under_parity():
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.05, 0.02, 0.33
    c = bsm_price("call", S, K, T, r, sigma, q)
    p = bsm_price("put", S, K, T, r, sigma, q)
    assert implied_volatility(c, "call", S, K, T, r, q) == pytest.approx(
        implied_volatility(p, "put", S, K, T, r, q), abs=1e-10)


def test_price_below_intrinsic_raises():
    S, K, T, r = 120.0, 100.0, 0.5, 0.05
    lb = european_price_bounds("call", S, K, T, r).lower
    with pytest.raises(ArbitrageBoundError, match="below its no-arbitrage lower bound"):
        implied_volatility(lb - 0.10, "call", S, K, T, r)


def test_price_above_upper_bound_raises():
    with pytest.raises(ArbitrageBoundError, match="upper bound"):
        implied_volatility(101.0, "call", 100.0, 100.0, 1.0, 0.05)


def test_price_at_lower_bound_returns_zero_vol():
    S, K, T, r = 120.0, 100.0, 0.5, 0.05
    lb = european_price_bounds("call", S, K, T, r).lower
    assert implied_volatility(lb, "call", S, K, T, r) == 0.0


def test_expired_option_has_no_iv():
    with pytest.raises(InvalidInputError, match="undefined at or after expiry"):
        implied_volatility(1.0, "call", 100, 100, 0.0, 0.05)


def test_array_version_returns_nan_for_unsolvable():
    prices = np.array([bsm_price("put", 100, 100, 0.5, 0.03, 0.2), -1.0, 500.0])
    out = implied_volatility_array(prices, "put", 100, 100, 0.5, 0.03)
    assert out[0] == pytest.approx(0.2, abs=1e-9)
    assert math.isnan(out[1]) and math.isnan(out[2])


def test_nvda_case_study_iv_is_recoverable():
    """With ~7 days to expiry (the model-consistent horizon), a $2.38 put at S=201.5 implies ~46%."""
    T = 6.96 / 365
    iv = implied_volatility(2.38, "put", 201.5, 195.0, T, 0.04)
    assert iv == pytest.approx(0.46, abs=0.002)
