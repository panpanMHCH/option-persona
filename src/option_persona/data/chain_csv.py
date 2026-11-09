"""Option-chain CSV ingestion into validated :class:`OptionQuote` records.

Principles
----------
* **Nothing is invented.** A blank bid is not read as zero, a missing
  timestamp is not replaced by "now", and a naive timestamp is not silently
  given a time zone. Each of these either rejects the row or requires the
  caller to state the assumption explicitly; stated assumptions are listed in
  :attr:`ChainImport.assumptions`.
* **Row-level, explainable rejection.** One bad row does not discard a file.
  Every rejected row gets an ``error`` issue with its line number, column and
  a reason; quality concerns on accepted rows are ``warning`` issues.
* **Observations only.** Vendor-computed IV, Greeks or "mark"/"theo" columns
  are deliberately ignored: they are someone else's model under undisclosed
  assumptions. The last-trade price is ignored because it is neither current
  nor executable.

Canonical schema (one row per option quote; header names are case-insensitive,
spaces and hyphens become underscores, and ``column_map`` renames others):

====================  ========  ================================================
column                required  meaning
====================  ========  ================================================
underlying            yes       ticker, e.g. ``NVDA``, ``SPX``
option_type           yes       ``call``/``put`` (``C``/``P`` accepted)
strike                yes       > 0
expiry                yes       ISO date, e.g. ``2026-11-20``
bid, ask              yes       per-unit prices; blanks are rejected, not zeroed
underlying_price      one of    underlying reference at observation time
underlying_bid/_ask   these     if both present and no ``underlying_price``,
                                the reference is their mid (recorded as info)
observed_at           *         ISO 8601 timestamp with offset; required unless
                                a file-level ``observed_at`` is supplied
bid_size, ask_size    no        displayed size in contracts (non-negative ints)
exercise_style        no        ``european``/``american``
settlement            no        ``cash``/``physical``
multiplier            no        > 0, default from the contract-spec resolution
currency              no        three-letter code, default ``USD`` (assumption)
is_delayed            *         true/false; required unless the caller declares it
is_synthetic          no        true marks the row synthetic (never cleared)
====================  ========  ================================================

Line numbers in issues assume a single header line (header = line 1).
Thousands separators (``1,234.50``) and ``$`` are stripped; decimal commas are
not supported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, tzinfo
from enum import StrEnum
from pathlib import Path
from typing import IO, Any
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import ValidationError

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionContract, OptionQuote, Provenance
from option_persona.core.time import NEW_YORK, expiry_datetime, require_aware
from option_persona.core.types import ExerciseStyle, OptionType, SettlementType
from option_persona.data.contract_specs import US_EQUITY_DEFAULT, known_spec

REQUIRED_COLUMNS = ("underlying", "option_type", "strike", "expiry", "bid", "ask")
OPTIONAL_COLUMNS = (
    "observed_at", "underlying_price", "underlying_bid", "underlying_ask", "bid_size", "ask_size",
    "exercise_style", "settlement", "multiplier", "currency", "is_delayed", "is_synthetic",
)
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

VENDOR_MODEL_COLUMNS = frozenset({
    "iv", "implied_volatility", "impl_vol", "volatility", "delta", "gamma", "theta", "vega", "rho",
    "theo", "theoretical", "theoretical_price", "mark", "model_price",
})
LAST_TRADE_COLUMNS = frozenset({"last", "last_price", "last_trade", "last_sale"})

_MISSING_TOKENS = frozenset({"", "-", "--", "n/a", "na", "nan", "null", "none"})
_TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0"})
_OPTION_TYPE_ALIASES = {"c": "call", "call": "call", "calls": "call",
                        "p": "put", "put": "put", "puts": "put"}
_EXERCISE_ALIASES = {"european": ExerciseStyle.EUROPEAN, "e": ExerciseStyle.EUROPEAN,
                     "american": ExerciseStyle.AMERICAN, "a": ExerciseStyle.AMERICAN}
_SETTLEMENT_ALIASES = {"cash": SettlementType.CASH, "physical": SettlementType.PHYSICAL}


class IssueSeverity(StrEnum):
    ERROR = "error"  # row rejected
    WARNING = "warning"  # row kept; data-quality concern the user should see
    INFO = "info"  # how the file was interpreted


@dataclass(frozen=True)
class ImportIssue:
    line: int | None  # file line number (header = 1); None for file-level issues
    column: str | None
    severity: IssueSeverity
    code: str
    message: str


@dataclass(frozen=True)
class ChainImport:
    """Result of an import: accepted quotes plus everything the user must know about them."""

    source: str
    rows_read: int
    quotes: tuple[OptionQuote, ...]
    issues: tuple[ImportIssue, ...]
    assumptions: tuple[str, ...]

    @property
    def rows_accepted(self) -> int:
        return len(self.quotes)

    @property
    def rows_rejected(self) -> int:
        return self.rows_read - self.rows_accepted

    @property
    def errors(self) -> tuple[ImportIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ImportIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.WARNING)

    def to_frame(self) -> pd.DataFrame:
        """Accepted quotes, one row each, with provenance columns kept alongside prices."""
        rows = []
        for q in self.quotes:
            c, p = q.contract, q.provenance
            rows.append({
                "underlying": c.underlying, "option_type": c.option_type.value, "strike": c.strike,
                "expiry": c.expiry, "exercise_style": c.exercise_style.value,
                "settlement": c.settlement.value, "multiplier": c.multiplier, "currency": c.currency,
                "bid": q.bid, "ask": q.ask, "mid": q.mid, "spread": q.spread,
                "bid_size": q.bid_size, "ask_size": q.ask_size,
                "underlying_price": q.underlying_price, "underlying_bid": q.underlying_bid,
                "underlying_ask": q.underlying_ask, "observed_at": p.observed_at, "source": p.source,
                "is_synthetic": p.is_synthetic, "is_delayed": p.is_delayed,
            })
        return pd.DataFrame(rows, columns=[
            "underlying", "option_type", "strike", "expiry", "exercise_style", "settlement",
            "multiplier", "currency", "bid", "ask", "mid", "spread", "bid_size", "ask_size",
            "underlying_price", "underlying_bid", "underlying_ask", "observed_at", "source",
            "is_synthetic", "is_delayed",
        ])

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [vars(i) | {"severity": i.severity.value} for i in self.issues],
            columns=["line", "column", "severity", "code", "message"],
        )


# ---------------------------------------------------------------------- parsing helpers
class _RowError(Exception):
    def __init__(self, column: str | None, code: str, message: str) -> None:
        super().__init__(message)
        self.column, self.code, self.message = column, code, message


def normalise_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _is_missing(text: str) -> bool:
    return text.strip().lower() in _MISSING_TOKENS


def _number(raw: Mapping[str, str], column: str) -> float | None:
    text = raw.get(column, "")
    if _is_missing(text):
        return None
    cleaned = text.strip().replace(",", "").replace("$", "")
    try:
        value = float(cleaned)
    except ValueError:
        raise _RowError(column, "unparseable", f"{column} {text!r} is not a number.") from None
    if not math.isfinite(value):
        raise _RowError(column, "unparseable", f"{column} {text!r} is not finite.")
    return value


def _size(raw: Mapping[str, str], column: str) -> int | None:
    value = _number(raw, column)
    if value is None:
        return None
    if value < 0 or value != int(value):
        raise _RowError(column, "invalid_value", f"{column} must be a non-negative integer, got {value}.")
    return int(value)


def _boolean(raw: Mapping[str, str], column: str) -> bool | None:
    text = raw.get(column, "")
    if _is_missing(text):
        return None
    token = text.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise _RowError(column, "unparseable", f"{column} {text!r} is not true/false.")


def _expiry(text: str) -> date:
    s = text.strip()
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise _RowError("expiry", "unparseable",
                        f"expiry {text!r} is not an ISO date (expected e.g. 2026-11-20).") from None
    if dt.time() != time(0) or dt.tzinfo is not None:
        raise _RowError("expiry", "unparseable", f"expiry {text!r} carries a time of day; give a date.")
    return dt.date()


def _timestamp(text: str, assume_timezone: tzinfo | None) -> tuple[datetime, bool]:
    """Parse an ISO 8601 timestamp; returns (aware datetime, whether a zone was assumed)."""
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        raise _RowError("observed_at", "unparseable",
                        f"observed_at {text!r} is not ISO 8601 (e.g. 2026-11-05T15:59:58-05:00).") from None
    if dt.tzinfo is None or dt.utcoffset() is None:
        if assume_timezone is None:
            raise _RowError(
                "observed_at", "naive_timestamp",
                f"observed_at {text!r} has no time zone. Add an offset, or pass assume_timezone "
                "if you know the zone the export was written in.",
            )
        return dt.replace(tzinfo=assume_timezone), True
    return dt, False


# ---------------------------------------------------------------------- import
@dataclass
class _Context:
    source: str
    is_delayed: bool | None
    observed_at: datetime | None
    assume_timezone: tzinfo | None
    exercise_style: ExerciseStyle | None
    settlement: SettlementType | None
    multiplier: float | None
    currency: str | None
    is_synthetic: bool
    assumptions: dict[str, None] = field(default_factory=dict)  # ordered set, accepted rows only
    pending: list[str] = field(default_factory=list)  # assumptions made by the row being parsed

    def assume(self, text: str) -> None:
        self.pending.append(text)

    def commit(self) -> None:
        for text in self.pending:
            self.assumptions.setdefault(text, None)
        self.pending.clear()


def _resolve_spec(raw: Mapping[str, str], underlying: str, ctx: _Context
                  ) -> tuple[ExerciseStyle, SettlementType, float]:
    spec = known_spec(underlying)

    text = raw.get("exercise_style", "")
    if not _is_missing(text):
        try:
            style = _EXERCISE_ALIASES[text.strip().lower()]
        except KeyError:
            raise _RowError("exercise_style", "invalid_value",
                            f"exercise_style {text!r} must be european or american.") from None
    elif ctx.exercise_style is not None:
        style = ctx.exercise_style
    elif spec is not None:
        style = spec.exercise_style
        ctx.assume(f"{underlying}: exercise style {style.value} from the known-contract registry "
                   f"({spec.note}).")
    else:
        style = US_EQUITY_DEFAULT.exercise_style
        ctx.assume(f"{underlying}: exercise style assumed {style.value}, the US single-stock/ETF "
                   "convention; it was not in the file, not declared, and not a known index.")

    text = raw.get("settlement", "")
    if not _is_missing(text):
        try:
            settlement = _SETTLEMENT_ALIASES[text.strip().lower()]
        except KeyError:
            raise _RowError("settlement", "invalid_value",
                            f"settlement {text!r} must be cash or physical.") from None
    elif ctx.settlement is not None:
        settlement = ctx.settlement
    elif spec is not None:
        settlement = spec.settlement
        ctx.assume(f"{underlying}: settlement {settlement.value} from the known-contract registry.")
    else:
        settlement = US_EQUITY_DEFAULT.settlement
        ctx.assume(f"{underlying}: settlement assumed {settlement.value} (US equity convention).")

    multiplier = _number(raw, "multiplier")
    if multiplier is None:
        if ctx.multiplier is not None:
            multiplier = ctx.multiplier
        else:
            multiplier = (spec or US_EQUITY_DEFAULT).multiplier
            ctx.assume(f"{underlying}: multiplier assumed {multiplier:g}. Adjusted contracts after "
                       "corporate actions have non-standard deliverables and cannot be detected here.")
    elif multiplier <= 0:
        raise _RowError("multiplier", "invalid_value", f"multiplier must be > 0, got {multiplier}.")
    return style, settlement, multiplier


def _parse_row(raw: Mapping[str, str], line: int, ctx: _Context
               ) -> tuple[OptionQuote, list[ImportIssue]]:
    issues: list[ImportIssue] = []

    def note(severity: IssueSeverity, column: str | None, code: str, message: str) -> None:
        issues.append(ImportIssue(line, column, severity, code, message))

    # contract ---------------------------------------------------------------
    underlying = raw.get("underlying", "").strip().upper()
    if not underlying:
        raise _RowError("underlying", "missing_value", "underlying is blank.")
    type_text = raw.get("option_type", "")
    try:
        option_type = OptionType(_OPTION_TYPE_ALIASES[type_text.strip().lower()])
    except KeyError:
        raise _RowError("option_type", "invalid_value",
                        f"option_type {type_text!r} must be call/put (or C/P).") from None
    strike = _number(raw, "strike")
    if strike is None:
        raise _RowError("strike", "missing_value", "strike is blank.")
    if strike <= 0:
        raise _RowError("strike", "invalid_value", f"strike must be > 0, got {strike}.")
    if _is_missing(raw.get("expiry", "")):
        raise _RowError("expiry", "missing_value", "expiry is blank.")
    expiry = _expiry(raw["expiry"])
    style, settlement, multiplier = _resolve_spec(raw, underlying, ctx)
    currency_text = raw.get("currency", "")
    if _is_missing(currency_text):
        currency = ctx.currency or "USD"
        if ctx.currency is None:
            ctx.assume("currency assumed USD (no currency column and none declared).")
    else:
        currency = currency_text.strip().upper()

    # timestamp and provenance ----------------------------------------------
    stamp_text = raw.get("observed_at", "")
    if _is_missing(stamp_text):
        if ctx.observed_at is None:
            raise _RowError("observed_at", "missing_timestamp",
                            "observed_at is blank and no file-level snapshot time was supplied; "
                            "timestamps are never inferred.")
        observed_at = ctx.observed_at
    else:
        observed_at, assumed = _timestamp(stamp_text, ctx.assume_timezone)
        if assumed:
            ctx.assume(f"naive observed_at values interpreted in {ctx.assume_timezone} (caller-declared).")

    observed_ny = observed_at.astimezone(NEW_YORK)
    if observed_ny.date() > expiry:
        raise _RowError("expiry", "expired",
                        f"contract expired {expiry.isoformat()} before the observation "
                        f"({observed_ny:%Y-%m-%d %H:%M} ET).")
    if observed_ny >= expiry_datetime(expiry):
        note(IssueSeverity.WARNING, "observed_at", "post_expiry_observation",
             f"observed {observed_ny:%H:%M} ET on expiry day, at or after the 16:00 ET close; "
             "the quote is likely stale or no longer tradeable.")

    delayed = _boolean(raw, "is_delayed")
    if delayed is None:
        if ctx.is_delayed is None:
            raise _RowError("is_delayed", "missing_delay_flag",
                            "is_delayed is blank and the caller did not declare whether the feed is "
                            "delayed.")
        delayed = ctx.is_delayed
    synthetic = ctx.is_synthetic or bool(_boolean(raw, "is_synthetic"))

    # option quote --------------------------------------------------------------
    bid, ask = _number(raw, "bid"), _number(raw, "ask")
    if bid is None:
        raise _RowError("bid", "missing_value", "bid is blank; a missing bid is not treated as zero.")
    if ask is None:
        raise _RowError("ask", "missing_value", "ask is blank; a missing ask cannot be inferred.")
    if bid < 0 or ask < 0:
        raise _RowError("bid" if bid < 0 else "ask", "invalid_value", "bid and ask must be >= 0.")
    if ask == 0:
        raise _RowError("ask", "no_offer", "ask is 0: no offer on the book, so the option cannot be "
                                           "bought and the quote has no executable value.")
    if bid > ask:
        raise _RowError("bid", "crossed_quote", f"crossed quote: bid {bid} exceeds ask {ask}.")
    if bid == 0:
        note(IssueSeverity.WARNING, "bid", "zero_bid",
             "bid is 0: the option cannot be sold for a positive price and its mid is not a "
             "meaningful value estimate.")
    elif bid == ask:
        note(IssueSeverity.WARNING, "bid", "locked_market", f"locked market: bid equals ask ({bid}).")

    # underlying reference ------------------------------------------------------
    und_bid, und_ask = _number(raw, "underlying_bid"), _number(raw, "underlying_ask")
    if (und_bid is None) != (und_ask is None):
        note(IssueSeverity.WARNING, "underlying_bid", "one_sided_underlying",
             "only one side of the underlying quote is present; both sides ignored.")
        und_bid = und_ask = None
    if und_bid is not None and und_ask is not None:
        if und_bid <= 0 or und_ask <= 0:
            raise _RowError("underlying_bid", "invalid_value", "underlying bid/ask must be > 0.")
        if und_bid > und_ask:
            raise _RowError("underlying_bid", "crossed_quote",
                            f"crossed underlying quote: bid {und_bid} exceeds ask {und_ask}.")
    und_price = _number(raw, "underlying_price")
    if und_price is None:
        if und_bid is None or und_ask is None:
            raise _RowError("underlying_price", "missing_value",
                            "no underlying reference: supply underlying_price or both "
                            "underlying_bid and underlying_ask.")
        und_price = 0.5 * (und_bid + und_ask)
        note(IssueSeverity.INFO, "underlying_price", "derived_underlying",
             "underlying_price derived as the mid of underlying_bid and underlying_ask.")
    elif und_price <= 0:
        raise _RowError("underlying_price", "invalid_value",
                        f"underlying_price must be > 0, got {und_price}.")
    elif und_bid is not None and und_ask is not None and not und_bid <= und_price <= und_ask:
        note(IssueSeverity.WARNING, "underlying_price", "underlying_inconsistent",
             f"underlying_price {und_price} lies outside the underlying quote [{und_bid}, {und_ask}]; "
             "it may be a stale last trade.")

    try:
        quote = OptionQuote(
            contract=OptionContract(
                underlying=underlying, option_type=option_type, strike=strike, expiry=expiry,
                exercise_style=style, settlement=settlement, multiplier=multiplier, currency=currency,
            ),
            bid=bid, ask=ask, underlying_price=und_price,
            underlying_bid=und_bid, underlying_ask=und_ask,
            bid_size=_size(raw, "bid_size"), ask_size=_size(raw, "ask_size"),
            provenance=Provenance(source=ctx.source, observed_at=observed_at,
                                  is_synthetic=synthetic, is_delayed=delayed),
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        column = str(first["loc"][-1]) if first.get("loc") else None
        raise _RowError(column, "invalid_value", first["msg"]) from None
    return quote, issues


def _contract_key(q: OptionQuote) -> tuple[Any, ...]:
    c = q.contract
    return (c.underlying, c.option_type, c.strike, c.expiry, c.multiplier)


def parse_option_chain(
    frame: pd.DataFrame,
    *,
    source: str,
    is_delayed: bool | None,
    observed_at: datetime | None = None,
    assume_timezone: tzinfo | str | None = None,
    column_map: Mapping[str, str] | None = None,
    exercise_style: ExerciseStyle | str | None = None,
    settlement: SettlementType | str | None = None,
    multiplier: float | None = None,
    currency: str | None = None,
    is_synthetic: bool = False,
) -> ChainImport:
    """Validate a chain held in a DataFrame (all cells are read as text).

    ``source`` and ``is_delayed`` are deliberately keyword-only and have no
    default: the importer cannot know where data came from or whether the feed
    was delayed. Pass ``is_delayed=None`` only if the file has an
    ``is_delayed`` column. A ``source`` beginning with ``synthetic`` marks
    every row synthetic.
    """
    source = source.strip()
    if not source:
        raise InvalidInputError("source must describe where the data came from, e.g. 'IBKR TWS export'.")
    if observed_at is not None:
        require_aware(observed_at, "observed_at")
    zone = ZoneInfo(assume_timezone) if isinstance(assume_timezone, str) else assume_timezone
    if multiplier is not None and multiplier <= 0:
        raise InvalidInputError(f"multiplier must be > 0, got {multiplier}.")

    renames = {normalise_header(k): normalise_header(v) for k, v in (column_map or {}).items()}
    columns = [renames.get(normalise_header(c), normalise_header(c)) for c in frame.columns]
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise InvalidInputError(f"Columns appear more than once after normalisation: {duplicates}.")

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise InvalidInputError(
            f"Missing required column(s) {missing}. Found {columns}. Use column_map to rename "
            "your export's headers, e.g. column_map={'Strike Price': 'strike'}."
        )
    if "underlying_price" not in columns and not {"underlying_bid", "underlying_ask"} <= set(columns):
        raise InvalidInputError("No underlying reference column: supply underlying_price, or both "
                                "underlying_bid and underlying_ask.")
    if "observed_at" not in columns and observed_at is None:
        raise InvalidInputError("No observed_at column and no file-level observed_at: the snapshot time "
                                "must be recorded; it is never inferred.")
    if "is_delayed" not in columns and is_delayed is None:
        raise InvalidInputError("Declare is_delayed=True/False (or add an is_delayed column): whether "
                                "quotes are real-time cannot be inferred from the file.")

    ctx = _Context(
        source=source, is_delayed=is_delayed, observed_at=observed_at, assume_timezone=zone,
        exercise_style=ExerciseStyle(exercise_style) if exercise_style else None,
        settlement=SettlementType(settlement) if settlement else None,
        multiplier=multiplier, currency=currency.strip().upper() if currency else None,
        is_synthetic=is_synthetic or source.lower().startswith("synthetic"),
    )

    issues: list[ImportIssue] = []
    for col in columns:
        if col in CANONICAL_COLUMNS:
            continue
        if col in VENDOR_MODEL_COLUMNS:
            msg = (f"column {col!r} ignored: vendor model output (IV/Greeks/theoretical value) under "
                   "undisclosed assumptions is not an observation; Option Persona recomputes these "
                   "with stated assumptions.")
        elif col in LAST_TRADE_COLUMNS:
            msg = f"column {col!r} ignored: a last-trade price can be stale and is not executable."
        else:
            msg = f"column {col!r} is not part of the canonical schema and was ignored."
        issues.append(ImportIssue(None, col, IssueSeverity.INFO, "ignored_column", msg))

    text = frame.astype(object).where(frame.notna(), "")
    text.columns = columns
    quotes: list[OptionQuote] = []
    first_line: dict[tuple[Any, ...], int] = {}
    for index, record in enumerate(text.to_dict("records")):
        line = index + 2
        raw = {k: str(v) for k, v in record.items()}
        ctx.pending.clear()
        try:
            quote, row_issues = _parse_row(raw, line, ctx)
        except _RowError as err:
            issues.append(ImportIssue(line, err.column, IssueSeverity.ERROR, err.code, err.message))
            continue
        key = _contract_key(quote)
        if key in first_line:
            issues.append(ImportIssue(
                line, None, IssueSeverity.ERROR, "duplicate_contract",
                f"duplicate of the contract on line {first_line[key]} (kept); one file is one snapshot.",
            ))
            continue
        first_line[key] = line
        ctx.commit()
        quotes.append(quote)
        issues.extend(row_issues)

    return ChainImport(
        source=source, rows_read=len(text), quotes=tuple(quotes), issues=tuple(issues),
        assumptions=tuple(ctx.assumptions),
    )


def read_option_chain_csv(
    path_or_buffer: str | Path | IO[str] | IO[bytes],
    *,
    source: str,
    is_delayed: bool | None,
    **kwargs: Any,
) -> ChainImport:
    """Read a CSV file (or file-like object) and validate it with :func:`parse_option_chain`."""
    try:
        frame = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise InvalidInputError("The CSV file is empty.") from None
    return parse_option_chain(frame, source=source, is_delayed=is_delayed, **kwargs)
