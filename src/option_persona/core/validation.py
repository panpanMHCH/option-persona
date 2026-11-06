"""Parameter validation for pricing functions.

Validation is explicit and produces messages a trader can act on, rather than
letting NaNs propagate silently through NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from option_persona.core.errors import InvalidInputError


def as_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be numeric, got {value!r}.") from exc
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError(f"{name} must be finite (no NaN or infinity), got {value!r}.")
    return arr


def require_positive(arr: NDArray[np.float64], name: str) -> None:
    if np.any(arr <= 0.0):
        raise InvalidInputError(f"{name} must be strictly positive, got {_show(arr)}.")


def require_non_negative(arr: NDArray[np.float64], name: str) -> None:
    if np.any(arr < 0.0):
        raise InvalidInputError(f"{name} must be non-negative, got {_show(arr)}.")


def validate_bsm_inputs(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    """Validate and broadcast Black-Scholes-Merton inputs.

    S, K > 0; T >= 0 (years); sigma >= 0 (annualised); r, q any finite real
    (negative rates and yields are permitted).
    """
    S_ = as_float_array(S, "Underlying price S")
    K_ = as_float_array(K, "Strike K")
    T_ = as_float_array(T, "Time to expiry T (years)")
    r_ = as_float_array(r, "Risk-free rate r")
    v_ = as_float_array(sigma, "Volatility sigma")
    q_ = as_float_array(q, "Dividend yield q")
    require_positive(S_, "Underlying price S")
    require_positive(K_, "Strike K")
    require_non_negative(T_, "Time to expiry T (years)")
    require_non_negative(v_, "Volatility sigma")
    if np.any(v_ > 20.0):
        raise InvalidInputError(
            f"Volatility sigma is {_show(v_)} — expressed as a decimal (0.46 = 46%). "
            "Values above 20 (2000%) are rejected as a likely unit error."
        )
    try:
        return tuple(np.broadcast_arrays(S_, K_, T_, r_, v_, q_))
    except ValueError as exc:
        raise InvalidInputError(f"Input shapes cannot be broadcast together: {exc}") from exc


def _show(arr: NDArray[np.float64]) -> str:
    return repr(arr.item()) if arr.ndim == 0 else np.array2string(arr, threshold=6)
