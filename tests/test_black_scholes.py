"""Pricing correctness: textbook values, an independent oracle, identities and limits."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from option_persona.core.errors import InvalidInputError
from option_persona.pricing import (
    bsm_call,
    bsm_delta,
    bsm_gamma,
    bsm_greeks,
    bsm_price,
    bsm_put,
    bsm_vega,
    risk_neutral_prob_itm,
)
from tests.conftest import quantlib_european, requires_quantlib

# ---------------------------------------------------------------- published values


@pytest.mark.parametrize(
    ("kind", "args", "expected", "tol", "source"),
    [
        ("call", (42, 40, 0.5, 0.10, 0.20, 0.0), 4.76, 0.005, "Hull, OFOD, Example 15.6"),
        ("put", (42, 40, 0.5, 0.10, 0.20, 0.0), 0.81, 0.005, "Hull, OFOD, Example 15.6"),
        ("call", (60, 65, 0.25, 0.08, 0.30, 0.0), 2.1334, 5e-5, "Haug (2007), BSM call"),
        ("put", (100, 95, 0.5, 0.10, 0.20, 0.05), 2.4648, 5e-5, "Haug (2007), Merton 1973 put"),
    ],
)
def test_published_reference_values(kind, args, expected, tol, source):
    assert bsm_price(kind, *args) == pytest.approx(expected, abs=tol), source


# ---------------------------------------------------------------- QuantLib oracle

GRID = [
    (kind, S, K, days, r, sigma, q)
    for kind in ("call", "put")
    for S in (50.0, 100.0, 180.0)
    for K in (80.0, 100.0, 125.0)
    for days in (1, 30, 365, 1825)
    for r, q in ((0.0, 0.0), (0.05, 0.02), (-0.005, 0.0))
    for sigma in (0.05, 0.25, 1.2)
]


@requires_quantlib
@pytest.mark.parametrize(("kind", "S", "K", "days", "r", "sigma", "q"), GRID)
def test_price_and_greeks_match_quantlib(kind, S, K, days, r, sigma, q):
    T = days / 365.0
    ours = bsm_greeks(kind, S, K, T, r, sigma, q)
    ref = quantlib_european(kind, S, K, days, r, sigma, q)
    for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
        val = float(getattr(ours, name))
        assert val == pytest.approx(ref[name], rel=1e-9, abs=1e-10), name


# ---------------------------------------------------------------- model-free identities

pos = st.floats(min_value=1.0, max_value=1000.0)
times = st.floats(min_value=1e-4, max_value=10.0)
rates = st.floats(min_value=-0.02, max_value=0.15)
vols = st.floats(min_value=0.01, max_value=3.0)
yields = st.floats(min_value=0.0, max_value=0.10)


@settings(max_examples=500, deadline=None)
@given(S=pos, K=pos, T=times, r=rates, sigma=vols, q=yields)
def test_put_call_parity_holds_for_bsm_prices(S, K, T, r, sigma, q):
    lhs = bsm_call(S, K, T, r, sigma, q) - bsm_put(S, K, T, r, sigma, q)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10 * max(S, K))


@settings(max_examples=500, deadline=None)
@given(S=pos, K=pos, T=times, r=rates, sigma=vols, q=yields)
def test_prices_respect_no_arbitrage_bounds(S, K, T, r, sigma, q):
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    c = bsm_call(S, K, T, r, sigma, q)
    p = bsm_put(S, K, T, r, sigma, q)
    eps = 1e-10 * max(S, K)
    assert max(S * df_q - K * df_r, 0.0) - eps <= c <= S * df_q + eps
    assert max(K * df_r - S * df_q, 0.0) - eps <= p <= K * df_r + eps


def test_call_prices_decreasing_and_convex_in_strike():
    K = np.linspace(50, 150, 101)
    c = bsm_call(100.0, K, 0.5, 0.03, 0.3, 0.01)
    p = bsm_put(100.0, K, 0.5, 0.03, 0.3, 0.01)
    assert np.all(np.diff(c) < 0) and np.all(np.diff(p) > 0)
    assert np.all(np.diff(c, 2) > -1e-12) and np.all(np.diff(p, 2) > -1e-12)


def test_delta_relation_and_gamma_vega_shared():
    args = (100.0, 105.0, 0.75, 0.04, 0.35, 0.015)
    T, q = args[2], args[5]
    assert bsm_delta("call", *args) - bsm_delta("put", *args) == pytest.approx(math.exp(-q * T))
    assert bsm_greeks("call", *args).gamma == pytest.approx(bsm_greeks("put", *args).gamma)
    assert bsm_greeks("call", *args).vega == pytest.approx(bsm_greeks("put", *args).vega)


def test_risk_neutral_probabilities_sum_to_one_and_match_digital():
    args = (100.0, 95.0, 0.5, 0.05, 0.25, 0.0)
    pc = risk_neutral_prob_itm("call", *args)
    pp = risk_neutral_prob_itm("put", *args)
    assert pc + pp == pytest.approx(1.0)
    # N(d2) = -e^{rT} dC/dK (digital call value, undiscounted)
    h = 1e-4
    dcdk = (bsm_call(100, 95 + h, 0.5, 0.05, 0.25) - bsm_call(100, 95 - h, 0.5, 0.05, 0.25)) / (2 * h)
    assert pc == pytest.approx(-math.exp(0.05 * 0.5) * dcdk, abs=1e-7)


# ---------------------------------------------------------------- limits and edge cases


@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_expiry_value_is_intrinsic(S):
    assert bsm_call(S, 100.0, 0.0, 0.05, 0.3) == pytest.approx(max(S - 100, 0))
    assert bsm_put(S, 100.0, 0.0, 0.05, 0.3) == pytest.approx(max(100 - S, 0))


def test_zero_vol_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.02
    fwd_intrinsic = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert bsm_call(S, K, T, r, 0.0, q) == pytest.approx(fwd_intrinsic)
    assert bsm_put(S, K, T, r, 0.0, q) == pytest.approx(0.0)
    assert bsm_delta("call", S, K, T, r, 0.0, q) == pytest.approx(math.exp(-q * T))
    assert bsm_gamma(S, K, T, r, 0.0, q) == 0.0
    assert bsm_vega(S, K, T, r, 0.0, q) == 0.0


def test_continuity_into_degenerate_limit():
    # Approaching expiry, prices converge to intrinsic value smoothly.
    for T in (1e-4, 1e-6, 1e-8):
        assert bsm_call(105.0, 100.0, T, 0.05, 0.3) == pytest.approx(5.0, abs=1e-2)
        assert bsm_put(95.0, 100.0, T, 0.05, 0.3) == pytest.approx(5.0, abs=1e-2)


def test_deep_otm_put_is_tiny_but_non_negative():
    p = bsm_put(1000.0, 10.0, 0.1, 0.05, 0.2)
    assert 0.0 <= p < 1e-100


def test_extreme_volatility_approaches_upper_bound():
    S, q, T = 100.0, 0.01, 1.0
    assert bsm_call(S, 100.0, T, 0.03, 15.0, q) == pytest.approx(S * math.exp(-q * T), rel=1e-6)


def test_vectorised_broadcasting_matches_scalar():
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([[0.1], [1.0]])
    grid = bsm_call(100.0, K, T, 0.03, 0.2)
    assert grid.shape == (2, 3)
    assert grid[1, 2] == pytest.approx(bsm_call(100.0, 110.0, 1.0, 0.03, 0.2))


def test_scalar_inputs_return_python_float():
    assert isinstance(bsm_call(100, 100, 1, 0.05, 0.2), float)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"S": -1.0}, "Underlying price S must be strictly positive"),
        ({"K": 0.0}, "Strike K must be strictly positive"),
        ({"T": -0.1}, "non-negative"),
        ({"sigma": -0.2}, "Volatility sigma must be non-negative"),
        ({"sigma": 46.0}, "likely unit error"),
        ({"S": float("nan")}, "finite"),
    ],
)
def test_invalid_inputs_raise_clear_errors(kwargs, match):
    params = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0} | kwargs
    with pytest.raises(InvalidInputError, match=match):
        bsm_price("call", **params)


def test_invalid_option_type():
    with pytest.raises(ValueError, match="'call' or 'put'"):
        bsm_price("straddle", 100, 100, 1, 0.05, 0.2)
