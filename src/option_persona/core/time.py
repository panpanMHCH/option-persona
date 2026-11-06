"""Time conventions.

* All timestamps must be timezone-aware. Naive datetimes are rejected because
  the platform is used from Asia/Bangkok to analyse US-listed contracts.
* Year fraction is ACT/365 Fixed measured in elapsed seconds:
  T = (t_expiry - t_valuation) / (365 * 86 400 s).
  Calendar time (not trading time) is used, consistent with how rates are
  quoted. This is a convention, not a law: implied volatility is only
  meaningful relative to the time measure used to compute it.
* Standard US equity and PM-settled index options are treated as expiring at
  16:00 America/New_York on the expiration date. AM-settled contracts (e.g.
  standard monthly SPX) stop trading the previous day and settle on the opening
  print; callers must supply the appropriate expiry time for those.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from option_persona.core.errors import InvalidInputError

NEW_YORK = ZoneInfo("America/New_York")
BANGKOK = ZoneInfo("Asia/Bangkok")
UTC = ZoneInfo("UTC")

SECONDS_PER_YEAR = 365.0 * 86_400.0
DAYS_PER_YEAR = 365.0
US_EQUITY_EXPIRY_TIME = time(16, 0)


def require_aware(dt: datetime, name: str = "timestamp") -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise InvalidInputError(
            f"{name} must be timezone-aware (e.g. America/New_York or Asia/Bangkok); "
            f"got naive datetime {dt.isoformat()}."
        )
    return dt


def expiry_datetime(expiry: date, expiry_time: time = US_EQUITY_EXPIRY_TIME) -> datetime:
    """Expiry instant in US Eastern Time (DST-aware via zoneinfo)."""
    return datetime.combine(expiry, expiry_time, tzinfo=NEW_YORK)


def year_fraction(start: datetime, end: datetime) -> float:
    """ACT/365F year fraction between two aware datetimes; never negative."""
    require_aware(start, "start")
    require_aware(end, "end")
    seconds = (end - start).total_seconds()
    return max(seconds, 0.0) / SECONDS_PER_YEAR


def time_to_expiry(
    valuation_time: datetime, expiry: date, expiry_time: time = US_EQUITY_EXPIRY_TIME
) -> float:
    """Years from ``valuation_time`` to the contract's expiry instant (>= 0)."""
    return year_fraction(valuation_time, expiry_datetime(expiry, expiry_time))


def to_new_york(dt: datetime) -> datetime:
    return require_aware(dt).astimezone(NEW_YORK)


def to_bangkok(dt: datetime) -> datetime:
    return require_aware(dt).astimezone(BANGKOK)
