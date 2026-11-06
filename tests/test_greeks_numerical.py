"""Analytical Greeks versus central finite differences, plus the BSM PDE.

Tolerances: central differences have truncation error O(h^2) and round-off
O(eps / h); with relative bump h = 1e-4 the combined error is ~1e-7 relative
for first derivatives and ~1e-5 for gamma (second derivative). We assert
rel 1e-5 (abs 1e-6) for first-order Greeks and rel 1e-4 (abs 1e-5) for gamma.
"""

from __future__ import annotations

import pytest

from option_persona.pricing import bsm_greeks, bsm_price

CASES = [
    (kind, S, K, T, r, sigma, q)
    for kind in ("call", "put")
    for S in (60.0, 100.0, 140.0)
    for K in (100.0,)
    for T in (7 / 365, 0.25, 2.0)
    for r in (0.0, 0.05)
    for sigma in (0.15, 0.6)
    for q in (0.0, 0.03)
]


@pytest.mark.parametrize(("kind", "S", "K", "T", "r", "sigma", "q"), CASES)
def test_greeks_match_finite_differences(kind, S, K, T, r, sigma, q):
    g = bsm_greeks(kind, S, K, T, r, sigma, q)

    def V(S=S, T=T, r=r, sigma=sigma):
        return bsm_price(kind, S, K, T, r, sigma, q)

    hS, hT, hr, hv = 1e-4 * S, 1e-4 * T, 1e-5, 1e-5
    fd_delta = (V(S=S + hS) - V(S=S - hS)) / (2 * hS)
    fd_gamma = (V(S=S + hS) - 2 * V() + V(S=S - hS)) / hS**2
    fd_vega = (V(sigma=sigma + hv) - V(sigma=sigma - hv)) / (2 * hv)
    fd_theta = -(V(T=T + hT) - V(T=T - hT)) / (2 * hT)  # theta = -dV/dT
    fd_rho = (V(r=r + hr) - V(r=r - hr)) / (2 * hr)

    assert g.delta == pytest.approx(fd_delta, rel=1e-5, abs=1e-6)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-4, abs=1e-5)
    assert g.vega == pytest.approx(fd_vega, rel=1e-5, abs=1e-6)
    assert g.theta == pytest.approx(fd_theta, rel=1e-5, abs=1e-6)
    assert g.rho == pytest.approx(fd_rho, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize(("kind", "S", "K", "T", "r", "sigma", "q"), CASES)
def test_black_scholes_pde_is_satisfied(kind, S, K, T, r, sigma, q):
    """theta + 1/2 sigma^2 S^2 gamma + (r - q) S delta - r V = 0."""
    g = bsm_greeks(kind, S, K, T, r, sigma, q)
    residual = g.theta + 0.5 * sigma**2 * S**2 * g.gamma + (r - q) * S * g.delta - r * g.price
    assert residual == pytest.approx(0.0, abs=1e-9 * max(S, K))


def test_unit_conversions():
    g = bsm_greeks("call", 100, 100, 1.0, 0.05, 0.2)
    assert g.theta_per_day == pytest.approx(g.theta / 365)
    assert g.vega_per_vol_point == pytest.approx(g.vega / 100)
    assert g.rho_per_rate_point == pytest.approx(g.rho / 100)
