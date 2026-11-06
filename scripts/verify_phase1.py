"""Reproducible Phase 1 verification report.

Run:  uv run python scripts/verify_phase1.py
Prints the maximum errors of our implementation against independent checks.
The output is recorded in docs/verification.md.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from option_persona.arbitrage import generate_synthetic_chain, scan_chain_for_parity
from option_persona.pricing import bsm_greeks, bsm_price, bsm_vega, implied_volatility
from option_persona.pricing.implied_vol import european_price_bounds


def textbook() -> None:
    print("1. Published reference values")
    refs = [
        ("Hull Ex. 15.6 call", "call", (42, 40, 0.5, 0.10, 0.20, 0.0), 4.76),
        ("Hull Ex. 15.6 put", "put", (42, 40, 0.5, 0.10, 0.20, 0.0), 0.81),
        ("Haug BSM call", "call", (60, 65, 0.25, 0.08, 0.30, 0.0), 2.1334),
        ("Haug Merton-73 put", "put", (100, 95, 0.5, 0.10, 0.20, 0.05), 2.4648),
    ]
    for name, kind, args, ref in refs:
        print(f"   {name:<20} OP = {bsm_price(kind, *args):.6f}   published = {ref}")


def quantlib_grid() -> None:
    try:
        import QuantLib as ql  # noqa: F401
    except ImportError:
        print("2. QuantLib not installed — skipped")
        return
    from tests.conftest import quantlib_european

    worst = dict.fromkeys(("price", "delta", "gamma", "vega", "theta", "rho"), 0.0)
    n = 0
    for kind, S, K, days, (r, q), sigma in itertools.product(
        ("call", "put"), (50.0, 100.0, 180.0), (80.0, 100.0, 125.0), (1, 30, 365, 1825),
        ((0.0, 0.0), (0.05, 0.02), (-0.005, 0.0)), (0.05, 0.25, 1.2),
    ):
        g = bsm_greeks(kind, S, K, days / 365, r, sigma, q)
        ref = quantlib_european(kind, S, K, days, r, sigma, q)
        for k in worst:
            worst[k] = max(worst[k], abs(float(getattr(g, k)) - ref[k]))
        n += 1
    print(f"2. QuantLib AnalyticEuropeanEngine, {n} cases — max absolute difference")
    for k, v in worst.items():
        print(f"   {k:<6} {v:.3e}")


def finite_differences() -> None:
    worst_abs = dict.fromkeys(("delta", "gamma", "vega", "theta", "rho"), 0.0)
    worst_rel = dict.fromkeys(worst_abs, 0.0)
    worst_pde = 0.0
    n = 0
    for kind, S, T, r, sigma, q in itertools.product(
        ("call", "put"), (60.0, 100.0, 140.0), (7 / 365, 0.25, 2.0), (0.0, 0.05), (0.15, 0.6),
        (0.0, 0.03),
    ):
        K = 100.0

        def V(S=S, T=T, r=r, sigma=sigma, kind=kind, q=q, K=K):
            return bsm_price(kind, S, K, T, r, sigma, q)

        g = bsm_greeks(kind, S, K, T, r, sigma, q)
        hS, hT, h = 1e-4 * S, 1e-4 * T, 1e-5
        fd = {
            "delta": (V(S=S + hS) - V(S=S - hS)) / (2 * hS),
            "gamma": (V(S=S + hS) - 2 * V() + V(S=S - hS)) / hS**2,
            "vega": (V(sigma=sigma + h) - V(sigma=sigma - h)) / (2 * h),
            "theta": -(V(T=T + hT) - V(T=T - hT)) / (2 * hT),
            "rho": (V(r=r + h) - V(r=r - h)) / (2 * h),
        }
        for k, v in fd.items():
            err = abs(float(getattr(g, k)) - v)
            worst_abs[k] = max(worst_abs[k], err)
            if abs(v) > 1e-4:  # relative error is only meaningful for material Greeks
                worst_rel[k] = max(worst_rel[k], err / abs(v))
        pde = g.theta + 0.5 * sigma**2 * S**2 * g.gamma + (r - q) * S * g.delta - r * g.price
        worst_pde = max(worst_pde, abs(pde))
        n += 1
    print(f"3. Central finite differences, {n} cases")
    print(f"   {'greek':<6} {'max abs err':>12} {'max rel err (|greek| > 1e-4)':>30}")
    for k in worst_abs:
        print(f"   {k:<6} {worst_abs[k]:>12.3e} {worst_rel[k]:>30.3e}")
    print(f"   BSM PDE residual theta + 1/2 s^2 S^2 gamma + (r-q) S delta - rV: max abs {worst_pde:.3e}")


def parity_and_iv() -> None:
    rng = np.random.default_rng(20261106)
    n = 100_000
    S, K = rng.uniform(1, 1000, n), rng.uniform(1, 1000, n)
    T, r = rng.uniform(1e-4, 10, n), rng.uniform(-0.02, 0.15, n)
    sig, q = rng.uniform(0.01, 3, n), rng.uniform(0, 0.1, n)
    lhs = np.asarray(bsm_price("call", S, K, T, r, sig, q)) - np.asarray(
        bsm_price("put", S, K, T, r, sig, q))
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    rel = np.abs(lhs - rhs) / np.maximum(S, K)
    print(f"4. Put-call parity on {n:,} random BSM prices — max |residual| / max(S,K) = {rel.max():.3e}")

    worst_price, worst_vol, solved, skipped = 0.0, 0.0, 0, 0
    for kind, Kk, Tt, s in itertools.product(
        ("call", "put"), (50, 80, 95, 100, 105, 125, 200), (1 / 365, 7 / 365, 0.25, 1, 5),
        (0.03, 0.15, 0.46, 1.0, 3.0),
    ):
        p = bsm_price(kind, 100.0, Kk, Tt, 0.04, s, 0.01)
        if p - european_price_bounds(kind, 100.0, Kk, Tt, 0.04, 0.01).lower < 1e-10:
            skipped += 1
            continue
        iv = implied_volatility(p, kind, 100.0, Kk, Tt, 0.04, 0.01)
        worst_price = max(worst_price, abs(bsm_price(kind, 100.0, Kk, Tt, 0.04, iv, 0.01) - p))
        if bsm_vega(100.0, Kk, Tt, 0.04, s, 0.01) > 1e-3:
            worst_vol = max(worst_vol, abs(iv - s))
        solved += 1
    print(f"5. Implied volatility round trip: {solved} solved, {skipped} with no identifiable time value")
    print(f"   max price error {worst_price:.3e};  max vol error (vega > 1e-3) {worst_vol:.3e}")


def synthetic_detection() -> None:
    injected = {90.0: 0.60, 105.0: -0.50, 110.0: 0.03}
    scan = scan_chain_for_parity(generate_synthetic_chain(violations=injected), 0.04, 0.01)
    print("6. Synthetic chain (SYNTHETIC DATA) — injected call shifts", injected)
    for rec in scan.to_dict("records"):
        print(f"   K={rec['strike']:>5.0f}  mid residual {rec['mid_residual']:+.4f}  "
              f"conv {rec['conversion_edge']:+8.2f}  rev {rec['reversal_edge']:+8.2f}  "
              f"{rec['classification']}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    textbook()
    quantlib_grid()
    finite_differences()
    parity_and_iv()
    synthetic_detection()
    _ = math
