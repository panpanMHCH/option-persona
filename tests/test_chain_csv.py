"""Option-chain CSV ingestion: schema, row-level rejection, provenance and stated assumptions."""

from __future__ import annotations

import io
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from option_persona.arbitrage import generate_synthetic_chain
from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionContract, OptionQuote, Provenance
from option_persona.core.time import BANGKOK, NEW_YORK, time_to_expiry
from option_persona.core.types import ExerciseStyle, OptionType, SettlementType
from option_persona.data import (
    SYNTHETIC_UNDERLYING,
    IssueSeverity,
    parse_option_chain,
    read_option_chain_csv,
    synthetic_canonical_chain,
)

HEADER = "underlying,option_type,strike,expiry,bid,ask,underlying_price,observed_at"
STAMP = "2026-11-05T15:59:58-05:00"


def _csv(*rows: str, header: str = HEADER) -> io.StringIO:
    return io.StringIO("\n".join([header, *rows]) + "\n")


def _load(*rows: str, header: str = HEADER, **kw):
    kw = {"source": "test export", "is_delayed": False} | kw
    return read_option_chain_csv(_csv(*rows, header=header), **kw)


def _codes(result, severity=None):
    return [i.code for i in result.issues if severity is None or i.severity is severity]


# ---------------------------------------------------------------------- happy path
def test_valid_rows_become_quotes_with_provenance():
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.50,{STAMP}",
                f"nvda,Call,210,2026-11-06,1.10,1.18,201.50,{STAMP}")
    assert res.rows_read == 2 and res.rows_accepted == 2 and res.rows_rejected == 0
    put = res.quotes[0]
    assert put.contract.underlying == "NVDA" and put.contract.option_type is OptionType.PUT
    assert put.contract.strike == 195.0 and put.contract.expiry == date(2026, 11, 6)
    assert (put.bid, put.ask, put.underlying_price) == (2.30, 2.46, 201.50)
    assert put.mid == pytest.approx(2.38)
    prov = put.provenance
    assert prov.source == "test export" and not prov.is_synthetic and not prov.is_delayed
    assert prov.observed_at == datetime(2026, 11, 5, 15, 59, 58, tzinfo=NEW_YORK)
    assert res.quotes[1].contract.option_type is OptionType.CALL
    assert not res.errors and not res.warnings


def test_unknown_equity_spec_is_an_explicit_assumption_not_silent():
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.50,{STAMP}")
    c = res.quotes[0].contract
    assert c.exercise_style is ExerciseStyle.AMERICAN and c.settlement is SettlementType.PHYSICAL
    assert c.multiplier == 100
    text = " ".join(res.assumptions)
    assert "NVDA: exercise style assumed american" in text
    assert "multiplier assumed 100" in text and "currency assumed USD" in text


@pytest.mark.parametrize("ticker", ["SPX", "SPXW", "XSP", "NDX", "RUT"])
def test_known_index_options_resolve_to_european_cash(ticker):
    res = _load(f"{ticker},C,100,2026-11-20,1.00,1.10,99.5,{STAMP}")
    c = res.quotes[0].contract
    assert c.exercise_style is ExerciseStyle.EUROPEAN and c.settlement is SettlementType.CASH
    assert any("known-contract registry" in a for a in res.assumptions)


def test_spec_precedence_column_over_argument_over_registry():
    header = HEADER + ",exercise_style,settlement,multiplier"
    res = _load(f"SPX,C,100,2026-11-20,1,1.1,99.5,{STAMP},american,physical,10",
                f"SPX,P,100,2026-11-20,1,1.1,99.5,{STAMP},,,",
                header=header, exercise_style="american", multiplier=50)
    first, second = (q.contract for q in res.quotes)
    assert (first.exercise_style, first.settlement, first.multiplier) == (
        ExerciseStyle.AMERICAN, SettlementType.PHYSICAL, 10)
    assert second.exercise_style is ExerciseStyle.AMERICAN  # argument beats registry
    assert second.settlement is SettlementType.CASH  # registry fills what the argument left open
    assert second.multiplier == 50


def test_column_map_and_header_normalisation():
    header = "Symbol,Put/Call,Strike Price,Expiration,BID,ASK,Underlying-Price,Quote Time"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.50,{STAMP}", header=header,
                column_map={"Symbol": "underlying", "Put/Call": "option_type",
                            "Strike Price": "strike", "Expiration": "expiry",
                            "Quote Time": "observed_at"})
    assert res.rows_accepted == 1


def test_thousands_separator_and_dollar_sign():
    res = _load(f'SPX,C,"6,000",2026-11-20,$12.30,$12.80,"5,990.25",{STAMP}')
    q = res.quotes[0]
    assert q.contract.strike == 6000 and q.bid == 12.30 and q.underlying_price == 5990.25


def test_underlying_mid_is_derived_and_recorded():
    header = "underlying,option_type,strike,expiry,bid,ask,underlying_bid,underlying_ask,observed_at"
    res = _load(f"SPY,C,600,2026-11-20,5.00,5.10,599.98,600.02,{STAMP}", header=header)
    q = res.quotes[0]
    assert q.underlying_price == pytest.approx(600.0)
    assert (q.underlying_bid, q.underlying_ask) == (599.98, 600.02)
    assert "derived_underlying" in _codes(res, IssueSeverity.INFO)


# ---------------------------------------------------------------------- file-level refusals
def test_missing_required_column_is_actionable():
    with pytest.raises(InvalidInputError, match=r"Missing required column\(s\) \['ask'\].*column_map"):
        _load(header="underlying,option_type,strike,expiry,bid,underlying_price,observed_at")


def test_missing_underlying_reference_column():
    with pytest.raises(InvalidInputError, match="No underlying reference"):
        _load(header="underlying,option_type,strike,expiry,bid,ask,observed_at")


def test_timestamp_is_never_inferred():
    with pytest.raises(InvalidInputError, match="never inferred"):
        _load(header="underlying,option_type,strike,expiry,bid,ask,underlying_price")


def test_delay_status_must_be_declared():
    with pytest.raises(InvalidInputError, match="is_delayed"):
        _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.50,{STAMP}", is_delayed=None)
    with pytest.raises(TypeError):
        read_option_chain_csv(_csv(), source="x")  # is_delayed has no default


def test_blank_source_and_empty_file():
    with pytest.raises(InvalidInputError, match="source"):
        _load(source="  ")
    with pytest.raises(InvalidInputError, match="empty"):
        read_option_chain_csv(io.StringIO(""), source="x", is_delayed=False)


def test_file_level_snapshot_time_must_be_aware():
    header = "underlying,option_type,strike,expiry,bid,ask,underlying_price"
    with pytest.raises(InvalidInputError, match="timezone-aware"):
        _load("NVDA,P,195,2026-11-06,2.30,2.46,201.50", header=header,
              observed_at=datetime(2026, 11, 5, 15))
    res = _load("NVDA,P,195,2026-11-06,2.30,2.46,201.50", header=header,
                observed_at=datetime(2026, 11, 5, 15, tzinfo=NEW_YORK))
    assert res.quotes[0].provenance.observed_at.hour == 15


# ---------------------------------------------------------------------- row-level rejection
@pytest.mark.parametrize(("row", "code", "column"), [
    (f"NVDA,P,195,2026-11-06,,2.46,201.5,{STAMP}", "missing_value", "bid"),
    (f"NVDA,P,195,2026-11-06,2.30,,201.5,{STAMP}", "missing_value", "ask"),
    (f"NVDA,P,195,2026-11-06,0,0,201.5,{STAMP}", "no_offer", "ask"),
    (f"NVDA,P,195,2026-11-06,2.50,2.40,201.5,{STAMP}", "crossed_quote", "bid"),
    (f"NVDA,P,195,2026-11-06,-1,2.40,201.5,{STAMP}", "invalid_value", "bid"),
    (f"NVDA,X,195,2026-11-06,2.30,2.46,201.5,{STAMP}", "invalid_value", "option_type"),
    (f"NVDA,P,abc,2026-11-06,2.30,2.46,201.5,{STAMP}", "unparseable", "strike"),
    (f"NVDA,P,0,2026-11-06,2.30,2.46,201.5,{STAMP}", "invalid_value", "strike"),
    (f"NVDA,P,195,06/11/2026,2.30,2.46,201.5,{STAMP}", "unparseable", "expiry"),
    (f"NVDA,P,195,2026-11-04,2.30,2.46,201.5,{STAMP}", "expired", "expiry"),
    (f",P,195,2026-11-06,2.30,2.46,201.5,{STAMP}", "missing_value", "underlying"),
    ("NVDA,P,195,2026-11-06,2.30,2.46,201.5,", "missing_timestamp", "observed_at"),
    ("NVDA,P,195,2026-11-06,2.30,2.46,201.5,2026-11-05 15:59:58", "naive_timestamp", "observed_at"),
    ("NVDA,P,195,2026-11-06,2.30,2.46,201.5,yesterday", "unparseable", "observed_at"),
    (f"NVDA,P,195,2026-11-06,2.30,2.46,,{STAMP}", "missing_value", "underlying_price"),
    (f"NVDA,P,195,2026-11-06,2.30,2.46,inf,{STAMP}", "unparseable", "underlying_price"),
])
def test_bad_rows_are_rejected_with_reason(row, code, column):
    res = _load(f"NVDA,C,210,2026-11-06,1.10,1.18,201.5,{STAMP}", row)
    assert res.rows_read == 2 and res.rows_accepted == 1 and res.rows_rejected == 1
    (err,) = res.errors
    assert (err.line, err.code, err.column) == (3, code, column)


def test_naive_timestamp_accepted_only_with_declared_zone():
    row = "NVDA,P,195,2026-11-06,2.30,2.46,201.5,2026-11-06 03:59:58"
    res = _load(row, assume_timezone="Asia/Bangkok")
    stamp = res.quotes[0].provenance.observed_at
    assert stamp.tzinfo == BANGKOK
    assert stamp.astimezone(NEW_YORK) == datetime(2026, 11, 5, 15, 59, 58, tzinfo=NEW_YORK)
    assert any("caller-declared" in a for a in res.assumptions)


def test_duplicate_contract_keeps_first_and_rejects_rest():
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP}",
                f"NVDA,P,195,2026-11-06,2.31,2.47,201.5,{STAMP}")
    assert res.rows_accepted == 1 and res.quotes[0].bid == 2.30
    (err,) = res.errors
    assert err.code == "duplicate_contract" and err.line == 3 and "line 2" in err.message


def test_rejected_rows_contribute_no_assumptions():
    """A row that fails late (after spec resolution) must not record assumptions about itself."""
    res = _load(f"NVDA,P,195,2026-11-06,,2.46,201.5,{STAMP}",
                f"SPX,C,100,2026-11-20,1,1.1,99.5,{STAMP}",
                f"SPX,C,100,2026-11-20,1,1.1,99.5,{STAMP}")  # duplicate: rejected
    assert res.rows_accepted == 1
    assert not any(a.startswith("NVDA") for a in res.assumptions)
    assert any(a.startswith("SPX") for a in res.assumptions)


# ---------------------------------------------------------------------- warnings (row kept)
def test_zero_bid_and_locked_market_are_warnings():
    res = _load(f"NVDA,C,300,2026-11-06,0,0.01,201.5,{STAMP}",
                f"NVDA,C,250,2026-11-06,0.05,0.05,201.5,{STAMP}")
    assert res.rows_accepted == 2
    assert _codes(res, IssueSeverity.WARNING) == ["zero_bid", "locked_market"]


def test_expiry_day_after_close_is_flagged():
    res = _load("NVDA,P,195,2026-11-06,0.00,0.05,201.5,2026-11-06T16:05:00-05:00")
    assert "post_expiry_observation" in _codes(res, IssueSeverity.WARNING)


def test_underlying_price_outside_its_quote_is_flagged():
    header = HEADER + ",underlying_bid,underlying_ask"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,203.00,{STAMP},201.40,201.60", header=header)
    assert "underlying_inconsistent" in _codes(res, IssueSeverity.WARNING)
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.50,{STAMP},201.40,", header=header)
    assert "one_sided_underlying" in _codes(res, IssueSeverity.WARNING)
    assert res.quotes[0].underlying_bid is None


def test_vendor_greeks_and_last_trade_are_ignored_deliberately():
    header = HEADER + ",IV,Delta,Last,Volume"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP},0.46,-0.29,2.35,1200", header=header)
    ignored = {i.column: i.message for i in res.issues if i.code == "ignored_column"}
    assert set(ignored) == {"iv", "delta", "last", "volume"}
    assert "not an observation" in ignored["iv"] and "not executable" in ignored["last"]


def test_sizes_parsed_and_validated():
    header = HEADER + ",bid_size,ask_size"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP},12,30",
                f"NVDA,C,210,2026-11-06,1.10,1.18,201.5,{STAMP},1.5,3", header=header)
    assert (res.quotes[0].bid_size, res.quotes[0].ask_size) == (12, 30)
    assert res.errors[0].code == "invalid_value" and res.errors[0].column == "bid_size"


# ---------------------------------------------------------------------- provenance flags
def test_synthetic_flag_cannot_be_cleared():
    header = HEADER + ",is_synthetic,is_delayed"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP},true,true",
                f"NVDA,C,210,2026-11-06,1.10,1.18,201.5,{STAMP},false,", header=header)
    assert [q.provenance.is_synthetic for q in res.quotes] == [True, False]
    assert [q.provenance.is_delayed for q in res.quotes] == [True, False]  # blank -> declared
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP},false,", header=header,
                source="synthetic:manual")
    assert res.quotes[0].provenance.is_synthetic  # source name forces the flag


def test_per_row_delay_flag_required_when_not_declared():
    header = HEADER + ",is_delayed"
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP},yes",
                f"NVDA,C,210,2026-11-06,1.10,1.18,201.5,{STAMP},", header=header, is_delayed=None)
    assert res.quotes[0].provenance.is_delayed
    assert res.errors[0].code == "missing_delay_flag"


# ---------------------------------------------------------------------- frames and round trip
def test_frames_have_stable_columns():
    res = _load(f"NVDA,P,195,2026-11-06,2.30,2.46,201.5,{STAMP}", f"NVDA,P,1,2026-11-06,,1,1,{STAMP}")
    frame = res.to_frame()
    assert list(frame.columns[:4]) == ["underlying", "option_type", "strike", "expiry"]
    assert {"source", "is_synthetic", "is_delayed", "observed_at"} <= set(frame.columns)
    assert len(frame) == 1
    assert list(res.issues_frame().columns) == ["line", "column", "severity", "code", "message"]
    empty = _load()
    assert empty.to_frame().empty and empty.rows_read == 0


def test_dataframe_input_with_native_types():
    frame = pd.DataFrame({
        "underlying": ["XSP"], "option_type": ["put"], "strike": [600.0],
        "expiry": [date(2026, 11, 20)], "bid": [3.1], "ask": [3.2], "underlying_price": [601.2],
        "observed_at": [pd.Timestamp("2026-11-05 15:00", tz="America/New_York")],
        "bid_size": [np.nan],
    })
    res = parse_option_chain(frame, source="dataframe", is_delayed=True)
    assert res.rows_accepted == 1 and res.quotes[0].bid_size is None
    assert res.quotes[0].provenance.is_delayed


def test_synthetic_export_round_trips_exactly():
    observed = datetime(2026, 11, 5, 15, 0, tzinfo=NEW_YORK)
    expiry = date(2026, 12, 18)
    canonical = synthetic_canonical_chain(observed, expiry, violations={90.0: 0.6})
    buf = io.StringIO()
    canonical.to_csv(buf, index=False)
    buf.seek(0)
    res = read_option_chain_csv(buf, source="synthetic:bsm", is_delayed=False)
    assert res.rows_accepted == len(canonical) == 18 and not res.errors
    assert all(q.provenance.is_synthetic for q in res.quotes)
    assert all(q.contract.underlying == SYNTHETIC_UNDERLYING for q in res.quotes)
    assert not res.assumptions  # every contract term is explicit in the file

    wide = generate_synthetic_chain(time_to_expiry=time_to_expiry(observed, expiry),
                                    observed_at=observed, violations={90.0: 0.6})
    calls = {q.contract.strike: q for q in res.quotes if q.contract.option_type is OptionType.CALL}
    for rec in wide.to_dict("records"):
        assert calls[rec["strike"]].bid == pytest.approx(rec["call_bid"], abs=1e-9)
        assert calls[rec["strike"]].ask == pytest.approx(rec["call_ask"], abs=1e-9)


# ---------------------------------------------------------------------- model change
def test_option_quote_underlying_quote_validation():
    contract = OptionContract(underlying="SPY", option_type="call", strike=600, expiry=date(2026, 11, 20))
    prov = Provenance(source="manual", observed_at=datetime(2026, 11, 5, 15, tzinfo=NEW_YORK))
    base = {"contract": contract, "bid": 1.0, "ask": 1.1, "underlying_price": 600, "provenance": prov}
    with pytest.raises(ValidationError, match="both underlying_bid and underlying_ask"):
        OptionQuote(**base, underlying_bid=599.9)
    with pytest.raises(ValidationError, match="Crossed underlying"):
        OptionQuote(**base, underlying_bid=600.1, underlying_ask=599.9)
    assert OptionQuote(**base).underlying_bid is None


# ---------------------------------------------------------------------- property: valid in, valid out
prices = st.decimals(min_value="0.01", max_value="500", places=2)


@settings(max_examples=150, deadline=None)
@given(bid=prices, extra=st.decimals(min_value="0", max_value="20", places=2), strike=prices,
       spot=prices, call=st.booleans())
def test_any_well_formed_row_round_trips(bid, extra, strike, spot, call):
    ask = bid + extra
    res = _load(f"ABC,{'C' if call else 'P'},{strike},2026-12-18,{bid},{ask},{spot},{STAMP}")
    assert res.rows_accepted == 1, res.issues
    q = res.quotes[0]
    assert q.bid == float(bid) and q.ask == float(ask) and q.contract.strike == float(strike)
    assert q.underlying_price == float(spot) and q.bid <= q.ask
