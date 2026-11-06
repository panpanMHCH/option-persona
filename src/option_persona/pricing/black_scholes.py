r"""Black-Scholes-Merton (BSM) pricing and analytical Greeks for European options.

Model
-----
Under the risk-neutral measure Q the underlying follows

    dS_t = (r - q) S_t dt + sigma S_t dW_t^Q,

with constant continuously compounded risk-free rate ``r``, continuous dividend
yield ``q`` and volatility ``sigma``. Writing the forward F = S e^{(r-q)T},
discount factor D = e^{-rT} and total volatility v = sigma sqrt(T):

    d1 = [ln(F/K) + v^2/2] / v,     d2 = d1 - v
    C  = D [F N(d1) - K N(d2)]
    P  = D [K N(-d2) - F N(-d1)]

Puts are evaluated with N(-d) directly (not via parity) so that deep
out-of-the-money prices do not suffer catastrophic cancellation.

Degenerate limits
-----------------
When v = 0 (T = 0 or sigma = 0) the terminal distribution is a point mass at F,
so the price is the discounted forward intrinsic value D * max(F - K, 0) for a
call. We implement this by sending d1, d2 to +/-inf (or 0 at F = K) and setting
the density terms phi(d1) to zero. Consequently gamma and vega are reported as
0 in the limit; exactly at-the-money at expiry gamma is formally a Dirac delta,
which no finite number represents — callers should treat that point with care.

Greek conventions (all per unit of underlying, i.e. per share)
--------------------------------------------------------------
* delta  = dV/dS
* gamma  = d2V/dS2
* vega   = dV/dsigma  per 1.00 change in sigma (divide by 100 for per vol point)
* theta  = dV/dt = -dV/dT  per year of calendar time (divide by 365 for per day)
* rho    = dV/dr  per 1.00 change in r (divide by 100 for per 1%)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from option_persona.core.time import DAYS_PER_YEAR
from option_persona.core.types import OptionType
from option_persona.core.validation import validate_bsm_inputs

FloatOrArray = float | NDArray[np.float64]

_SQRT_2PI = np.sqrt(2.0 * np.pi)
# Below this total volatility the lognormal is numerically a point mass.
_DEGENERATE_TOTAL_VOL = 1e-12


def _pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-0.5 * x * x) / _SQRT_2PI


def _out(x: NDArray[np.float64]) -> FloatOrArray:
    return float(x) if x.ndim == 0 else x


@dataclass(frozen=True)
class _State:
    """Intermediate quantities shared by price and Greeks."""

    S: NDArray[np.float64]
    K: NDArray[np.float64]
    T: NDArray[np.float64]
    r: NDArray[np.float64]
    sigma: NDArray[np.float64]
    q: NDArray[np.float64]
    df_r: NDArray[np.float64]  # e^{-rT}
    df_q: NDArray[np.float64]  # e^{-qT}
    sqrt_T: NDArray[np.float64]
    d1: NDArray[np.float64]
    d2: NDArray[np.float64]
    pdf_d1: NDArray[np.float64]  # phi(d1), zero in the degenerate limit
    degenerate: NDArray[np.bool_]


def _state(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
           q: ArrayLike) -> _State:
    S_, K_, T_, r_, v_, q_ = validate_bsm_inputs(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T_)
    total_vol = v_ * sqrt_T
    df_r = np.exp(-r_ * T_)
    df_q = np.exp(-q_ * T_)
    # ln(F/K) computed additively to avoid forming F explicitly.
    log_fk = np.log(S_ / K_) + (r_ - q_) * T_
    degenerate = total_vol <= _DEGENERATE_TOTAL_VOL

    safe_vol = np.where(degenerate, 1.0, total_vol)
    d1_reg = log_fk / safe_vol + 0.5 * safe_vol
    d1_lim = np.where(log_fk > 0, np.inf, np.where(log_fk < 0, -np.inf, 0.0))
    d1 = np.where(degenerate, d1_lim, d1_reg)
    d2 = np.where(degenerate, d1_lim, d1_reg - safe_vol)
    pdf_d1 = np.where(degenerate, 0.0, _pdf(np.where(degenerate, 0.0, d1)))
    return _State(S_, K_, T_, r_, v_, q_, df_r, df_q, sqrt_T, d1, d2, pdf_d1, degenerate)


def _sign(option_type: OptionType | str) -> float:
    return 1.0 if OptionType.parse(option_type) is OptionType.CALL else -1.0


# --------------------------------------------------------------------------- prices


def bsm_price(option_type: OptionType | str, S: ArrayLike, K: ArrayLike, T: ArrayLike,
              r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> FloatOrArray:
    """European option value under Black-Scholes-Merton (per unit of underlying).

    Parameters are broadcast with NumPy rules. ``T`` in years, ``r``, ``q``,
    ``sigma`` as annualised decimals (0.05 = 5%).
    """
    st = _state(S, K, T, r, sigma, q)
    w = _sign(option_type)
    # w = +1: S e^{-qT} N(d1) - K e^{-rT} N(d2);  w = -1: K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    price = w * (st.S * st.df_q * ndtr(w * st.d1) - st.K * st.df_r * ndtr(w * st.d2))
    return _out(np.maximum(price, 0.0))


def bsm_call(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
             q: ArrayLike = 0.0) -> FloatOrArray:
    return bsm_price(OptionType.CALL, S, K, T, r, sigma, q)


def bsm_put(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
            q: ArrayLike = 0.0) -> FloatOrArray:
    return bsm_price(OptionType.PUT, S, K, T, r, sigma, q)


# --------------------------------------------------------------------------- Greeks


def bsm_delta(option_type: OptionType | str, S: ArrayLike, K: ArrayLike, T: ArrayLike,
              r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> FloatOrArray:
    """Call: e^{-qT} N(d1).  Put: -e^{-qT} N(-d1)."""
    st = _state(S, K, T, r, sigma, q)
    w = _sign(option_type)
    return _out(w * st.df_q * ndtr(w * st.d1))


def bsm_gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
              q: ArrayLike = 0.0) -> FloatOrArray:
    """e^{-qT} phi(d1) / (S sigma sqrt(T)); identical for calls and puts."""
    st = _state(S, K, T, r, sigma, q)
    denom = np.where(st.degenerate, 1.0, st.S * st.sigma * st.sqrt_T)
    return _out(st.df_q * st.pdf_d1 / denom)


def bsm_vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
             q: ArrayLike = 0.0) -> FloatOrArray:
    """S e^{-qT} phi(d1) sqrt(T), per 1.00 change in sigma; identical for calls and puts."""
    st = _state(S, K, T, r, sigma, q)
    return _out(st.S * st.df_q * st.pdf_d1 * st.sqrt_T)


def bsm_theta(option_type: OptionType | str, S: ArrayLike, K: ArrayLike, T: ArrayLike,
              r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> FloatOrArray:
    """dV/dt per year of calendar time (negative = value lost as time passes).

    theta = -S e^{-qT} phi(d1) sigma / (2 sqrt T) - w r K e^{-rT} N(w d2) + w q S e^{-qT} N(w d1)
    """
    st = _state(S, K, T, r, sigma, q)
    w = _sign(option_type)
    safe_sqrt_t = np.where(st.degenerate, 1.0, st.sqrt_T)
    decay = -st.S * st.df_q * st.pdf_d1 * st.sigma / (2.0 * safe_sqrt_t)
    carry = (-w * st.r * st.K * st.df_r * ndtr(w * st.d2)
             + w * st.q * st.S * st.df_q * ndtr(w * st.d1))
    return _out(decay + carry)


def bsm_rho(option_type: OptionType | str, S: ArrayLike, K: ArrayLike, T: ArrayLike,
            r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> FloatOrArray:
    """w K T e^{-rT} N(w d2), per 1.00 change in r."""
    st = _state(S, K, T, r, sigma, q)
    w = _sign(option_type)
    return _out(w * st.K * st.T * st.df_r * ndtr(w * st.d2))


@dataclass(frozen=True)
class Greeks:
    """Analytical BSM sensitivities, per unit of underlying (see module docstring)."""

    price: FloatOrArray
    delta: FloatOrArray
    gamma: FloatOrArray
    vega: FloatOrArray
    theta: FloatOrArray
    rho: FloatOrArray

    @property
    def theta_per_day(self) -> FloatOrArray:
        """Theta per calendar day (theta / 365)."""
        return self.theta / DAYS_PER_YEAR

    @property
    def vega_per_vol_point(self) -> FloatOrArray:
        """Value change for a 1 percentage-point rise in volatility."""
        return self.vega / 100.0

    @property
    def rho_per_rate_point(self) -> FloatOrArray:
        """Value change for a 1 percentage-point rise in the risk-free rate."""
        return self.rho / 100.0


def bsm_greeks(option_type: OptionType | str, S: ArrayLike, K: ArrayLike, T: ArrayLike,
               r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> Greeks:
    """Price and all first-order Greeks plus gamma in one call."""
    args = (S, K, T, r, sigma, q)
    return Greeks(
        price=bsm_price(option_type, *args),
        delta=bsm_delta(option_type, *args),
        gamma=bsm_gamma(*args),
        vega=bsm_vega(*args),
        theta=bsm_theta(option_type, *args),
        rho=bsm_rho(option_type, *args),
    )


def risk_neutral_prob_itm(option_type: OptionType | str, S: ArrayLike, K: ArrayLike,
                          T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
                          q: ArrayLike = 0.0) -> FloatOrArray:
    """Q-probability that the option finishes in the money: N(d2) for calls, N(-d2) for puts.

    This is a *risk-neutral* probability implied by the model and its volatility
    input. It is not a forecast of the real-world (P-measure) probability, which
    also depends on the underlying's expected return and on any volatility risk
    premium embedded in implied volatility.
    """
    st = _state(S, K, T, r, sigma, q)
    w = _sign(option_type)
    return _out(ndtr(w * st.d2))
