"""Contract specifications for well-known underlyings.

Why this exists: a CSV export rarely states whether a contract is European or
American, or cash- or physically settled, yet that decides which identities
apply (e.g. put-call parity as an equality is only valid for European
exercise). The importer resolves each specification in this order:

1. a value in the row (``exercise_style`` / ``settlement`` / ``multiplier`` column);
2. a value the caller passed explicitly to the importer;
3. this registry of well-known index options;
4. the US single-stock/ETF convention (American, physically settled, 100
   shares), recorded as an explicit *assumption* in the import result.

The registry only contains contract terms that are standard and stable. It
deliberately does not encode AM versus PM settlement timing (standard monthly
SPX settles on the opening print; SPXW weeklies are PM-settled): the expiry
instant must be supplied by the caller for AM-settled analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from option_persona.core.types import ExerciseStyle, SettlementType


@dataclass(frozen=True)
class ContractSpec:
    exercise_style: ExerciseStyle
    settlement: SettlementType
    multiplier: float
    note: str = ""


_EUROPEAN_CASH_INDEX = (ExerciseStyle.EUROPEAN, SettlementType.CASH, 100.0)

KNOWN_CONTRACT_SPECS: dict[str, ContractSpec] = {
    "SPX": ContractSpec(*_EUROPEAN_CASH_INDEX, note="S&P 500 index options (Cboe)"),
    "SPXW": ContractSpec(*_EUROPEAN_CASH_INDEX, note="S&P 500 weeklies (PM-settled)"),
    "XSP": ContractSpec(*_EUROPEAN_CASH_INDEX, note="Mini-SPX, one tenth of SPX"),
    "NDX": ContractSpec(*_EUROPEAN_CASH_INDEX, note="Nasdaq-100 index options"),
    "RUT": ContractSpec(*_EUROPEAN_CASH_INDEX, note="Russell 2000 index options"),
}

US_EQUITY_DEFAULT = ContractSpec(
    ExerciseStyle.AMERICAN, SettlementType.PHYSICAL, 100.0,
    note="US single-stock/ETF convention (American, physically settled, 100 shares)",
)


def known_spec(underlying: str) -> ContractSpec | None:
    """Registry entry for ``underlying`` (case-insensitive), or ``None``."""
    return KNOWN_CONTRACT_SPECS.get(underlying.strip().upper())
