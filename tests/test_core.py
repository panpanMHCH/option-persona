"""Time conventions and data-model validation."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from option_persona.core.errors import InvalidInputError
from option_persona.core.models import OptionContract, OptionPosition, OptionQuote, Provenance
from option_persona.core.time import (
    BANGKOK,
    NEW_YORK,
    expiry_datetime,
    time_to_expiry,
    to_bangkok,
    year_fraction,
)


def test_naive_datetimes_rejected():
    with pytest.raises(InvalidInputError, match="timezone-aware"):
        year_fraction(datetime(2026, 1, 1), datetime(2026, 2, 1, tzinfo=NEW_YORK))


def test_year_fraction_act365():
    a = datetime(2026, 1, 1, tzinfo=NEW_YORK)
    b = datetime(2027, 1, 1, tzinfo=NEW_YORK)
    assert year_fraction(a, b) == pytest.approx(1.0)
    assert year_fraction(b, a) == 0.0


def test_expiry_is_4pm_eastern_and_dst_aware():
    assert expiry_datetime(date(2026, 11, 6)).utcoffset().total_seconds() == -5 * 3600  # EST
    assert expiry_datetime(date(2026, 10, 30)).utcoffset().total_seconds() == -4 * 3600  # EDT


def test_bangkok_and_new_york_agree_on_elapsed_time():
    bkk = datetime(2026, 11, 6, 21, 30, tzinfo=BANGKOK)  # = 09:30 ET (EST, UTC-5)
    assert to_bangkok(datetime(2026, 11, 6, 9, 30, tzinfo=NEW_YORK)) == bkk
    t = time_to_expiry(bkk, date(2026, 11, 6))
    assert t * 365 * 24 == pytest.approx(6.5)  # 6.5 hours to the 16:00 ET close


def _contract(**kw):
    return OptionContract(**({"underlying": "nvda", "option_type": "PUT", "strike": 195.0,
                              "expiry": date(2026, 11, 6)} | kw))


def test_contract_normalisation():
    c = _contract()
    assert c.underlying == "NVDA" and c.option_type == "put" and c.multiplier == 100


def test_quote_provenance_and_crossed_market():
    prov = Provenance(source="manual", observed_at=datetime(2026, 11, 5, 15, tzinfo=NEW_YORK))
    q = OptionQuote(contract=_contract(), bid=2.30, ask=2.46, underlying_price=201.5,
                    provenance=prov)
    assert q.mid == pytest.approx(2.38) and q.spread == pytest.approx(0.16)
    with pytest.raises(ValidationError, match="Crossed quote"):
        OptionQuote(contract=_contract(), bid=2.5, ask=2.4, underlying_price=201.5,
                    provenance=prov)
    with pytest.raises(ValidationError, match="timezone-aware"):
        Provenance(source="manual", observed_at=datetime(2026, 11, 5, 15))


def test_position_delta_sign_validated():
    with pytest.raises(ValidationError, match="long put cannot have positive delta"):
        OptionPosition(contract=_contract(), quantity=1, entry_premium=2.38,
                       entry_underlying_price=201.5, observed_entry_delta=0.29)


def test_realised_pnl_after_fees():
    p = OptionPosition(contract=_contract(), quantity=2, entry_premium=2.38,
                       entry_underlying_price=201.5, entry_fees=1.3, exit_premium=5.00,
                       exit_fees=1.3)
    assert p.realised_pnl() == pytest.approx((5.00 - 2.38) * 200 - 2.6)
