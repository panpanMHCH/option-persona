"""Synthetic example chains in the canonical CSV schema.

These exist so the importer can be exercised end to end (and the template
downloaded from the dashboard) without any market data. Every row is written
with ``is_synthetic = true`` and the underlying ``SYNTHETIC``, so a file
produced here can never be imported as a real observation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

import pandas as pd

from option_persona.arbitrage.synthetic import generate_synthetic_chain
from option_persona.core.time import require_aware, time_to_expiry
from option_persona.data.chain_csv import CANONICAL_COLUMNS

SYNTHETIC_UNDERLYING = "SYNTHETIC"


def synthetic_canonical_chain(
    observed_at: datetime,
    expiry: date,
    *,
    spot: float = 100.0,
    strikes: Sequence[float] | None = None,
    rate: float = 0.04,
    dividend_yield: float = 0.01,
    sigma_atm: float = 0.20,
    violations: Mapping[float, float] | None = None,
) -> pd.DataFrame:
    """A European, cash-settled synthetic chain as canonical long-format rows (two per strike)."""
    require_aware(observed_at, "observed_at")
    T = time_to_expiry(observed_at, expiry)
    wide = generate_synthetic_chain(spot=spot, strikes=strikes, time_to_expiry=T, rate=rate,
                                    dividend_yield=dividend_yield, sigma_atm=sigma_atm,
                                    observed_at=observed_at, violations=violations)
    rows = []
    for rec in wide.to_dict("records"):
        for side in ("call", "put"):
            rows.append({
                "underlying": SYNTHETIC_UNDERLYING, "option_type": side, "strike": rec["strike"],
                "expiry": expiry.isoformat(), "bid": round(rec[f"{side}_bid"], 2),
                "ask": round(rec[f"{side}_ask"], 2), "observed_at": observed_at.isoformat(),
                "underlying_price": "", "underlying_bid": rec["underlying_bid"],
                "underlying_ask": rec["underlying_ask"], "bid_size": "", "ask_size": "",
                "exercise_style": "european", "settlement": "cash", "multiplier": 100,
                "currency": "USD", "is_delayed": "false", "is_synthetic": "true",
            })
    return pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
