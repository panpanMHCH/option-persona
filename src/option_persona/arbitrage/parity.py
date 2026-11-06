r"""European put-call parity: theoretical residuals and a basic executable-edge assessment.

Identity
--------
For European options on the same underlying, strike K and expiry T, with a
continuous dividend yield q and a single risk-free rate r,

    C - P = S e^{-qT} - K e^{-rT}.

It follows from a static replication argument (a long call plus a short put
has the same payoff as a forward contract, S_T - K) and holds for *any*
arbitrage-free model, not only Black-Scholes. It does **not** hold as an
equality for American options, which satisfy only the bounds
S - D - K <= C - P <= S - K e^{-rT}, where D is the present value of
dividends paid before expiry (D = 0 for a non-dividend payer).

Three tiers of evidence
-----------------------
Every assessment reports three distinct things, which must never be conflated:

1. **Theoretical residual** — parity evaluated at quote mids. A non-zero mid
   residual is a *pricing observation*, not an opportunity: mids are not
   executable.
2. **Apparent executable edge** — the cash-flow edge of the conversion or
   reversal using the side of each quote you would actually trade (buy at the
   ask, sell at the bid), minus modelled fees, with asymmetric borrowing and
   lending rates. Positive values are labelled "apparent" because quotes may
   be stale, sizes may be small and legs may not fill simultaneously.
3. **Execution and operational risks** — listed explicitly; they are why a
   positive apparent edge is never described as risk-free or locked-in.

Strategies
----------
* Conversion (exploits C - P too *high*): sell call at bid, buy put at ask,
  buy e^{-qT} units of the underlying at ask (dividends reinvested). At T the
  position is worth exactly K. The initial outlay is financed at the
  *borrowing* rate:
      edge_conv = C_bid - P_ask - S_ask e^{-qT} + K e^{-r_borrow T}
* Reversal (exploits C - P too *low*): buy call at ask, sell put at bid, short
  e^{-qT} units at bid. The proceeds are invested at the *lending* rate, and
  you owe K at T. Stock-borrow fees reduce the edge:
      edge_rev = P_bid - C_ask + S_bid e^{-qT} - K e^{-r_lend T} - borrow_fee
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from option_persona.core.errors import InvalidInputError, NotApplicableError
from option_persona.core.types import ExerciseStyle, SettlementType
from option_persona.core.validation import as_float_array
from option_persona.pricing.black_scholes import FloatOrArray


def put_call_parity_residual(call_price: ArrayLike, put_price: ArrayLike, S: ArrayLike,
                             K: ArrayLike, T: ArrayLike, r: ArrayLike,
                             q: ArrayLike = 0.0) -> FloatOrArray:
    """(C - P) - (S e^{-qT} - K e^{-rT}). Zero under exact European parity."""
    C, P, S_, K_, T_, r_, q_ = (as_float_array(x, n) for x, n in (
        (call_price, "call price"), (put_price, "put price"), (S, "S"), (K, "K"),
        (T, "T"), (r, "r"), (q, "q")))
    res = (C - P) - (S_ * np.exp(-q_ * T_) - K_ * np.exp(-r_ * T_))
    return float(res) if res.ndim == 0 else res


def implied_forward(call_price: float, put_price: float, K: float, T: float, r: float) -> float:
    """Forward price implied by parity: F = K + (C - P) e^{rT}.

    Useful when the underlying is not directly tradeable (e.g. SPX), because it
    sidesteps the need for a dividend assumption.
    """
    return K + (call_price - put_price) * math.exp(r * T)


class ParityClassification(StrEnum):
    NO_VIOLATION = "no_violation"
    THEORETICAL_ONLY = "theoretical_only"  # mid residual beyond tolerance, not executable
    APPARENT_EXECUTABLE_EDGE = "apparent_executable_edge"  # survives quoted spreads & fees


@dataclass(frozen=True)
class ParityQuoteSet:
    """Synchronised two-sided quotes for one strike/expiry (per unit of underlying)."""

    strike: float
    time_to_expiry: float  # years
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    underlying_bid: float
    underlying_ask: float
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    settlement: SettlementType = SettlementType.CASH
    multiplier: float = 100.0
    call_observed_at: datetime | None = None
    put_observed_at: datetime | None = None
    underlying_observed_at: datetime | None = None
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        for name in ("call", "put", "underlying"):
            bid, ask = getattr(self, f"{name}_bid"), getattr(self, f"{name}_ask")
            if bid < 0 or ask <= 0 or bid > ask:
                raise InvalidInputError(f"Invalid {name} quote: bid={bid}, ask={ask}.")
        if self.strike <= 0 or self.time_to_expiry <= 0 or self.multiplier <= 0:
            raise InvalidInputError("Strike, time to expiry and multiplier must be positive.")


@dataclass(frozen=True)
class ExecutionCosts:
    """Basic cost model (Phase 1). Phase 2 extends this with tiered fees, slippage,
    exchange/regulatory fees and margin financing."""

    option_fee_per_contract: float = 0.65
    underlying_fee_per_unit: float = 0.0
    borrow_rate_spread: float = 0.005  # added to r when financing a conversion
    lend_rate_spread: float = 0.005  # subtracted from r when investing reversal proceeds
    stock_borrow_fee: float = 0.0025  # annualised hard-to-borrow fee for the short leg
    max_quote_age_gap_seconds: float = 1.0


@dataclass(frozen=True)
class ParityAssessment:
    strike: float
    time_to_expiry: float
    # Tier 1: theoretical (mid-based)
    mid_residual: float  # per unit of underlying
    relative_mid_residual: float  # residual / underlying mid
    half_spread_cost: float  # per unit: sum of half-spreads of the three legs
    # Tier 2: apparent executable edge (per conversion/reversal package, in currency)
    conversion_edge: float
    reversal_edge: float
    classification: ParityClassification
    # Tier 3: risks preventing any claim of a locked-in profit
    risk_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def best_edge(self) -> float:
        return max(self.conversion_edge, self.reversal_edge)

    @property
    def best_strategy(self) -> str:
        return "conversion" if self.conversion_edge >= self.reversal_edge else "reversal"


def assess_parity(quotes: ParityQuoteSet, rate: float, dividend_yield: float = 0.0,
                  costs: ExecutionCosts | None = None,
                  mid_tolerance: float = 0.02, min_edge: float = 1.0) -> ParityAssessment:
    """Assess a single strike for put-call parity violations under three tiers of evidence.

    ``mid_tolerance`` is the absolute mid residual per unit of underlying (default
    two $0.01 ticks) below which the mid-market is treated as consistent with
    parity. Tick rounding alone moves each mid by up to half a tick, and mids of
    options with a zero bid are biased upwards, so a residual smaller than a
    couple of ticks carries no information.

    The mid residual must exceed ``half_spread_cost`` (plus fees and financing
    spreads) before any executable edge can appear. ``min_edge`` (currency per
    package, default 1.00) is the materiality threshold below which a positive
    edge is treated as noise rather than an apparent opportunity.
    """
    if quotes.exercise_style is not ExerciseStyle.EUROPEAN:
        raise NotApplicableError(
            "Put-call parity holds as an equality only for European options. American options "
            "satisfy bounds (no dividends: S - K <= C - P <= S - K e^{-rT}) and carry early "
            "exercise and assignment risk; they require a separate model (Phase 3)."
        )
    costs = costs or ExecutionCosts()
    q, T, K, m = dividend_yield, quotes.time_to_expiry, quotes.strike, quotes.multiplier

    call_mid = 0.5 * (quotes.call_bid + quotes.call_ask)
    put_mid = 0.5 * (quotes.put_bid + quotes.put_ask)
    s_mid = 0.5 * (quotes.underlying_bid + quotes.underlying_ask)
    residual = float(put_call_parity_residual(call_mid, put_mid, s_mid, K, T, rate, q))
    relative = residual / s_mid
    half_spread_cost = 0.5 * ((quotes.call_ask - quotes.call_bid) + (quotes.put_ask - quotes.put_bid)
                              + (quotes.underlying_ask - quotes.underlying_bid))

    units = math.exp(-q * T)  # underlying units held per unit delivered at T
    r_borrow = rate + costs.borrow_rate_spread
    r_lend = rate - costs.lend_rate_spread
    fees = 2 * costs.option_fee_per_contract + costs.underlying_fee_per_unit * m * units

    conv_ps = (quotes.call_bid - quotes.put_ask - quotes.underlying_ask * units
               + K * math.exp(-r_borrow * T))
    borrow_cost_ps = quotes.underlying_bid * units * (math.exp(costs.stock_borrow_fee * T) - 1.0)
    rev_ps = (quotes.put_bid - quotes.call_ask + quotes.underlying_bid * units
              - K * math.exp(-r_lend * T) - borrow_cost_ps)
    conversion_edge = conv_ps * m - fees
    reversal_edge = rev_ps * m - fees

    if max(conversion_edge, reversal_edge) > min_edge:
        cls = ParityClassification.APPARENT_EXECUTABLE_EDGE
    elif abs(residual) > mid_tolerance:
        cls = ParityClassification.THEORETICAL_ONLY
    else:
        cls = ParityClassification.NO_VIOLATION

    return ParityAssessment(
        strike=K, time_to_expiry=T, mid_residual=residual, relative_mid_residual=relative,
        half_spread_cost=half_spread_cost,
        conversion_edge=conversion_edge, reversal_edge=reversal_edge, classification=cls,
        risk_notes=_risk_notes(quotes, costs, cls),
    )


def _risk_notes(quotes: ParityQuoteSet, costs: ExecutionCosts,
                cls: ParityClassification) -> tuple[str, ...]:
    notes: list[str] = []
    if quotes.is_synthetic:
        notes.append("SYNTHETIC DATA: quotes were generated for testing, not observed in a market.")
    stamps = [t for t in (quotes.call_observed_at, quotes.put_observed_at,
                          quotes.underlying_observed_at) if t is not None]
    if len(stamps) < 3:
        notes.append("Quote timestamps missing: cannot confirm the legs were observed simultaneously.")
    else:
        gap = (max(stamps) - min(stamps)).total_seconds()
        if gap > costs.max_quote_age_gap_seconds:
            notes.append(f"Legs observed {gap:.1f}s apart: the apparent edge may be a stale-quote artefact.")
    if cls is ParityClassification.APPARENT_EXECUTABLE_EDGE:
        notes.extend([
            "Legging risk: three legs must fill at quoted prices simultaneously; "
            "displayed size may be small.",
            "Quotes may be indicative or delayed; executable prices can differ by the time an order arrives.",
            "Dividend uncertainty: a continuous yield approximates discrete dividends; a dividend "
            "change alters parity.",
        ])
    if quotes.settlement is SettlementType.CASH:
        notes.append(
            "Cash-settled index options: the index itself is not tradeable. Replication requires "
            "futures or an ETF proxy (basis and tracking risk); AM-settled contracts settle on the "
            "opening print, not the last trade."
        )
    else:
        notes.append(
            "Physically settled: pin risk near the strike at expiry (uncertain assignment of the short leg)."
        )
    notes.append("Financing and stock-borrow rates are assumptions; recall of borrowed shares is possible.")
    return tuple(notes)
