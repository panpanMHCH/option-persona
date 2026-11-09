"""Hold-or-sell review for an open long option position.

This produces a **checklist of risk factors**, each with a status and the
reasoning behind it. It never outputs "sell" or "hold": the decision depends on
the user's view and circumstances, which the library cannot know (rule 1).

Statuses
--------
* ``ok``     nothing in this factor currently argues for attention.
* ``watch``  the factor is becoming material; worth checking regularly.
* ``alert``  the factor is material now (e.g. physical delivery within days).
* ``info``   context with no threshold (e.g. unrealised P&L).

Factors
-------
1. **Reason for the trade.** Whether the user says the original reason still holds.
2. **Re-buy test.** Holding an option is economically the same as buying it now at
   the bid. The review restates the trade as if it were opened today.
3. **Time left.** Days to the 16:00 ET expiry, plus the instant in Bangkok.
4. **Time decay.** Theta per day in dollars and as a share of current value.
5. **Intrinsic versus time value.** What selling captures that exercising gives up.
6. **Expiry settlement.** Automatic exercise of physically settled options that
   finish in the money, with the resulting share position.
7. **Liquidity.** The bid-ask spread as the cost of getting out.
8. **Unrealised P&L.** Shown for context. Entry price is a sunk cost; it does not
   change what the option is worth now.

Thresholds live in :class:`ExitReviewThresholds`; they are personal settings,
not universal rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionPosition
from option_persona.core.time import (
    DAYS_PER_YEAR,
    expiry_datetime,
    require_aware,
    time_to_expiry,
    to_bangkok,
)
from option_persona.core.types import ExerciseStyle, OptionType, SettlementType
from option_persona.pricing.black_scholes import bsm_greeks
from option_persona.risk.payoff import intrinsic_value
from option_persona.risk.scenarios import ModelAssumptions
from option_persona.risk.strike_comparison import expected_move

DISCLAIMER = ("A checklist of risk factors, not a recommendation. Model figures assume constant "
              "volatility and European exercise; whether to hold or sell is your decision.")


class ReviewStatus(StrEnum):
    OK = "ok"
    WATCH = "watch"
    ALERT = "alert"
    INFO = "info"


@dataclass(frozen=True)
class ReviewItem:
    factor: str
    status: ReviewStatus
    finding: str  # what is true now, with numbers
    why: str  # why this factor matters


@dataclass(frozen=True)
class ExitReviewThresholds:
    alert_days: float = 2.0
    watch_days: float = 7.0
    theta_alert_share: float = 0.10  # theta per day as a share of current value
    theta_watch_share: float = 0.05
    spread_alert: float = 0.10  # (ask - bid) / mid
    spread_watch: float = 0.05


@dataclass(frozen=True)
class ExitReview:
    days_to_expiry: float
    expiry_new_york: datetime
    expiry_bangkok: datetime
    value_basis: str  # "bid (observed)" or "model"
    value_per_unit: float
    intrinsic_per_unit: float
    time_value_per_unit: float
    unrealised_pnl: float
    items: tuple[ReviewItem, ...]
    assumptions: str
    disclaimer: str = DISCLAIMER

    def item(self, factor: str) -> ReviewItem:
        return next(i for i in self.items if i.factor == factor)


def _days_status(days: float, t: ExitReviewThresholds) -> ReviewStatus:
    if days <= t.alert_days:
        return ReviewStatus.ALERT
    return ReviewStatus.WATCH if days <= t.watch_days else ReviewStatus.OK


def exit_review(
    position: OptionPosition,
    spot: float,
    valuation_time: datetime,
    assumptions: ModelAssumptions,
    bid: float | None = None,
    ask: float | None = None,
    thesis_intact: bool | None = None,
    thresholds: ExitReviewThresholds | None = None,
) -> ExitReview:
    """Review an open long position at ``valuation_time``.

    ``bid``/``ask`` are the current option quote per unit, if the user has it.
    Without them, values fall back to the model and say so.
    """
    t = thresholds or ExitReviewThresholds()
    if spot <= 0:
        raise InvalidInputError(f"Underlying price must be positive, got {spot}.")
    if (bid is None) != (ask is None):
        raise InvalidInputError("Give both the current bid and ask, or neither.")
    if bid is not None and ask is not None and (bid < 0 or ask <= 0 or bid > ask):
        raise InvalidInputError(f"Invalid quote: bid {bid}, ask {ask} (need 0 <= bid <= ask, ask > 0).")
    if position.is_closed:
        raise InvalidInputError("The position is already closed; a hold-or-sell review applies to open "
                                "positions.")

    c = position.contract
    now = require_aware(valuation_time, "valuation_time")
    expiry_ny = expiry_datetime(c.expiry)
    T = time_to_expiry(now, c.expiry)
    days = T * DAYS_PER_YEAR
    if T <= 0:
        raise InvalidInputError("The valuation time is at or after expiry; the option has expired.")
    units = position.units
    sigma = assumptions.volatility
    g = bsm_greeks(c.option_type, spot, c.strike, T, assumptions.rate, sigma, assumptions.dividend_yield)
    model_value = float(g.price)
    has_quote = bid is not None and ask is not None
    value = float(bid) if has_quote else model_value  # type: ignore[arg-type]
    basis = "bid (observed)" if has_quote else "model"
    intrinsic = float(intrinsic_value(c.option_type, spot, c.strike))
    time_value = value - intrinsic
    pnl = (value - position.entry_premium) * units - position.entry_fees
    is_put = c.option_type is OptionType.PUT
    itm = intrinsic > 0
    move = expected_move(spot, sigma, T)
    shares = f"{units:,.0f} shares"
    items: list[ReviewItem] = []

    # 1. reason for the trade ---------------------------------------------------
    if thesis_intact is True:
        items.append(ReviewItem("Reason for the trade", ReviewStatus.OK,
                                "You say the reason you opened the trade still holds.",
                                "Paying time decay is only worthwhile while you still want the exposure."))
    elif thesis_intact is False:
        items.append(ReviewItem("Reason for the trade", ReviewStatus.ALERT,
                                "You say the reason you opened the trade no longer holds.",
                                "Without the original reason, holding means paying theta for exposure "
                                "you no longer intend to have."))
    else:
        items.append(ReviewItem("Reason for the trade", ReviewStatus.WATCH,
                                "Not answered: does the reason you opened the trade still hold?",
                                "This is the one factor only you can judge, and usually the most important."))

    # 2. re-buy test ------------------------------------------------------------
    rebuy_price = float(ask) if has_quote else model_value  # type: ignore[arg-type]
    rebuy_be = c.strike - rebuy_price if is_put else c.strike + rebuy_price
    needed = rebuy_be / spot - 1.0
    in_moves = (rebuy_be - spot) / move if move > 0 else float("nan")
    items.append(ReviewItem(
        "Re-buy test", ReviewStatus.INFO,
        f"Bought today at {rebuy_price:.2f} ({'ask' if has_quote else 'model value'}), this option "
        f"would break even at {rebuy_be:,.2f}: a {needed:+.1%} move from {spot:,.2f} in {days:.1f} days, "
        f"or {in_moves:+.2f} expected moves (±{move:,.2f} at σ = {sigma:.0%}).",
        "Holding is economically the same as buying the option now at the bid. If you would not open "
        "this trade today, ask what keeps you in it."))

    # 3. time left --------------------------------------------------------------
    items.append(ReviewItem(
        "Time left", _days_status(days, t),
        f"{days:.2f} calendar days to expiry: {expiry_ny:%a %d %b %H:%M} ET = "
        f"{to_bangkok(expiry_ny):%a %d %b %H:%M} Bangkok.",
        "In the last days gamma and theta grow fastest: value swings sharply with small moves and "
        "decays quickly when the underlying is still."))

    # 4. time decay -------------------------------------------------------------
    theta_day = float(g.theta_per_day)
    share = -theta_day / model_value if model_value > 0 else float("inf")
    status = (ReviewStatus.ALERT if share >= t.theta_alert_share else
              ReviewStatus.WATCH if share >= t.theta_watch_share else ReviewStatus.OK)
    items.append(ReviewItem(
        "Time decay", status,
        f"Theta ≈ {theta_day * units:+,.2f} $/day, {share:.1%} of the model value per day if nothing "
        "else changes.",
        "Theta is the daily price of keeping the option's remaining chance alive."))

    # 5. intrinsic versus time value -------------------------------------------
    if itm and c.exercise_style is ExerciseStyle.AMERICAN and has_quote and value < intrinsic:
        items.append(ReviewItem(
            "Intrinsic vs time value", ReviewStatus.WATCH,
            f"The bid {value:.2f} is below intrinsic value {intrinsic:.2f}.",
            "Selling at this bid realises less than exercising would, but exercise delivers "
            f"{'a short' if is_put else 'a long'} position of {shares} that needs margin or cash."))
    else:
        items.append(ReviewItem(
            "Intrinsic vs time value", ReviewStatus.INFO,
            f"Value {value:.2f} ({basis}) = intrinsic {intrinsic:.2f} + time value {time_value:.2f} per "
            "share.",
            "Selling captures the time value; exercising early gives it up. Time value falls to zero at "
            "expiry."))

    # 6. expiry settlement ------------------------------------------------------
    distance = abs(spot - c.strike)
    if c.settlement is SettlementType.CASH:
        items.append(ReviewItem(
            "Expiry settlement", ReviewStatus.INFO,
            "Cash-settled: no shares are delivered at expiry.",
            "Settlement uses the official settlement value, which can differ from the last price seen."))
    elif days > t.watch_days:
        items.append(ReviewItem(
            "Expiry settlement", ReviewStatus.OK,
            "Physically settled, but expiry is more than a week away.",
            "Becomes relevant in the final days."))
    else:
        delivered = (f"you would sell {shares} at {c.strike:,.2f} — a short stock position if you "
                     "do not own them" if is_put else
                     f"you would buy {shares} for {c.strike * units:,.0f}")
        if itm:
            status = ReviewStatus.ALERT if days <= t.alert_days else ReviewStatus.WATCH
            finding = (f"In the money by {intrinsic:.2f}. If still in the money at expiry it is normally "
                       f"exercised automatically: {delivered}.")
        elif distance <= move:
            status = ReviewStatus.WATCH
            finding = (f"Out of the money by {distance:.2f}, within one expected move (±{move:.2f}): it "
                       f"could finish either side. If it finishes in the money, {delivered}.")
        else:
            status = ReviewStatus.OK
            finding = (f"Out of the money by {distance:.2f}, more than one expected move (±{move:.2f}) "
                       "away.")
        items.append(ReviewItem(
            "Expiry settlement", status, finding,
            "US equity options in the money by $0.01 or more are normally exercised automatically "
            "(broker policies vary). The share position then appears after the close, overnight in "
            f"Bangkok ({to_bangkok(expiry_ny):%a %H:%M}), and can be held over a weekend."))

    # 7. liquidity -------------------------------------------------------------
    if not has_quote:
        items.append(ReviewItem("Liquidity", ReviewStatus.INFO,
                                "No current bid/ask entered.",
                                "Enter the live quote: the bid is what you can actually sell for."))
    elif value == 0:
        items.append(ReviewItem("Liquidity", ReviewStatus.ALERT,
                                "The bid is 0: there is currently no buyer at any positive price.",
                                "The option may only be worth something if it moves into the money."))
    else:
        mid = 0.5 * (value + float(ask))  # type: ignore[arg-type]
        spread = (float(ask) - value) / mid  # type: ignore[arg-type]
        status = (ReviewStatus.ALERT if spread >= t.spread_alert else
                  ReviewStatus.WATCH if spread >= t.spread_watch else ReviewStatus.OK)
        items.append(ReviewItem(
            "Liquidity", status,
            f"Spread {float(ask) - value:.2f} = {spread:.1%} of mid; selling at the bid costs "
            f"{(mid - value) * units:,.2f} versus mid.",
            "Spreads usually widen near expiry and after large moves, which raises the cost of exiting."))

    # 8. unrealised P&L --------------------------------------------------------
    items.append(ReviewItem(
        "Unrealised P&L", ReviewStatus.INFO,
        f"{pnl:+,.2f} at the {basis} value (after entry fees, before exit fees).",
        "Your entry price is sunk: it does not change what the option is worth now or what it can do "
        "from here."))

    return ExitReview(
        days_to_expiry=days, expiry_new_york=expiry_ny, expiry_bangkok=to_bangkok(expiry_ny),
        value_basis=basis, value_per_unit=value, intrinsic_per_unit=intrinsic,
        time_value_per_unit=time_value, unrealised_pnl=pnl, items=tuple(items),
        assumptions=assumptions.describe(),
    )
