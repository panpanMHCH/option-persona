r"""Implied volatility: invert the BSM price for sigma.

Method
------
BSM value is strictly increasing in sigma whenever T > 0 (vega > 0), so for any
price strictly inside the model-free no-arbitrage bounds

    call: max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}
    put:  max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}

a unique implied volatility exists. We bracket the root and solve with Brent's
method (``scipy.optimize.brentq``), which is guaranteed to converge for a
continuous monotone function and does not rely on vega, so it is robust where
Newton-Raphson fails (deep out-of-the-money options and very short expiries,
where vega is tiny and Newton steps overshoot).

Numerical conditioning
----------------------
An in-the-money option's price is mostly intrinsic value; its time value — the
only part that carries volatility information — may be a tiny difference of
large numbers. We therefore convert ITM inputs to the price of the
out-of-the-money option at the same strike via put-call parity (an exact
algebraic identity under the model) and solve on that. The implied volatility
is identical by construction.

Reference: P. Jäckel, "Let's Be Rational" (2015), for a faster
machine-precision method; Brent is adequate and transparent for research use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from option_persona.core.errors import ArbitrageBoundError, ConvergenceError, InvalidInputError
from option_persona.core.types import OptionType
from option_persona.core.validation import validate_bsm_inputs
from option_persona.pricing.black_scholes import bsm_price

SIGMA_MIN = 1e-8
SIGMA_MAX = 20.0  # 2000% — anything higher is treated as a data error


@dataclass(frozen=True)
class PriceBounds:
    """Model-free no-arbitrage bounds on a European option price (per unit)."""

    lower: float
    upper: float


def european_price_bounds(option_type: OptionType | str, S: float, K: float, T: float,
                          r: float, q: float = 0.0) -> PriceBounds:
    """Bounds that hold for *any* arbitrage-free model, not only BSM."""
    otype = OptionType.parse(option_type)
    fwd_spot = S * math.exp(-q * T)  # PV of the underlying delivered at T
    pv_strike = K * math.exp(-r * T)
    if otype is OptionType.CALL:
        return PriceBounds(lower=max(fwd_spot - pv_strike, 0.0), upper=fwd_spot)
    return PriceBounds(lower=max(pv_strike - fwd_spot, 0.0), upper=pv_strike)


def implied_volatility(price: float, option_type: OptionType | str, S: float, K: float,
                       T: float, r: float, q: float = 0.0, *, tol: float = 1e-12) -> float:
    """Implied volatility (annualised decimal) of a European option price.

    Raises
    ------
    ArbitrageBoundError
        If the price lies outside the no-arbitrage bounds, so no volatility can
        reproduce it. This is diagnostic information, not a trading signal:
        stale quotes, mids of wide markets, early-exercise value (American
        options) and wrong dividend/rate inputs are the usual causes.
    InvalidInputError
        For non-positive T or otherwise invalid inputs.
    ConvergenceError
        If the solver fails (should not occur for valid, bracketed inputs).
    """
    otype = OptionType.parse(option_type)
    validate_bsm_inputs(S, K, T, r, 0.0, q)
    if not math.isfinite(price):
        raise InvalidInputError(f"Option price must be finite, got {price!r}.")
    if T <= 0.0:
        raise InvalidInputError(
            "Implied volatility is undefined at or after expiry (T must be > 0): "
            "the option value is pure intrinsic value."
        )

    bounds = european_price_bounds(otype, S, K, T, r, q)
    scale = max(S, K)
    bound_tol = 1e-12 * scale
    if price < bounds.lower - bound_tol:
        raise ArbitrageBoundError(
            f"{otype.value.capitalize()} price {price:.6g} is below its no-arbitrage lower bound "
            f"{bounds.lower:.6g} (discounted forward intrinsic value). No implied volatility exists."
        )
    if price >= bounds.upper - bound_tol:
        raise ArbitrageBoundError(
            f"{otype.value.capitalize()} price {price:.6g} is at or above its upper bound "
            f"{bounds.upper:.6g}. Implied volatility would be infinite."
        )
    if price <= bounds.lower + bound_tol:
        return 0.0  # no time value: zero volatility reproduces the price

    # Solve on the out-of-the-money side for numerical conditioning.
    forward_intrinsic = S * math.exp(-q * T) - K * math.exp(-r * T)  # C - P under parity
    solve_type, target = otype, price
    if otype is OptionType.CALL and forward_intrinsic > 0:
        solve_type, target = OptionType.PUT, price - forward_intrinsic
    elif otype is OptionType.PUT and forward_intrinsic < 0:
        solve_type, target = OptionType.CALL, price + forward_intrinsic
    target = max(target, 0.0)
    if target <= bound_tol:
        return 0.0

    def objective(sigma: float) -> float:
        return float(bsm_price(solve_type, S, K, T, r, sigma, q)) - target

    lo, hi = SIGMA_MIN, 1.0
    while objective(hi) < 0.0:
        hi *= 2.0
        if hi > SIGMA_MAX:
            raise ArbitrageBoundError(
                f"Price {price:.6g} implies volatility above {SIGMA_MAX:.0%}; "
                "treating as a data or unit error."
            )
    if objective(lo) > 0.0:
        return lo
    try:
        root, info = brentq(objective, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps,
                            maxiter=500, full_output=True)
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
        raise ConvergenceError(f"Implied volatility solver failed: {exc}") from exc
    if not info.converged:  # pragma: no cover - defensive
        raise ConvergenceError(f"Implied volatility did not converge: {info.flag}")
    return float(root)


def implied_volatility_array(prices: ArrayLike, option_type: OptionType | str, S: ArrayLike,
                             K: ArrayLike, T: ArrayLike, r: ArrayLike,
                             q: ArrayLike = 0.0) -> NDArray[np.float64]:
    """Element-wise implied volatility; NaN where no implied volatility exists.

    Intended for scanning option chains, where a few unsolvable quotes should not
    abort the whole computation. Use :func:`implied_volatility` to obtain the
    reason for any individual failure.
    """
    p, S_, K_, T_, r_, q_ = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                   for x in (prices, S, K, T, r, q)))
    out = np.full(p.shape, np.nan)
    for idx in np.ndindex(p.shape):
        try:
            out[idx] = implied_volatility(float(p[idx]), option_type, float(S_[idx]),
                                          float(K_[idx]), float(T_[idx]), float(r_[idx]),
                                          float(q_[idx]))
        except (ArbitrageBoundError, InvalidInputError):
            continue
    return out
