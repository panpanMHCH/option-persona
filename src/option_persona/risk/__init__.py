"""Module D (leverage & risk) and Module E foundations (position payoff & scenarios)."""

from option_persona.risk.leverage import (
    ExposureReport,
    RiskLimits,
    compare_with_shares,
    exposure_report,
)
from option_persona.risk.payoff import (
    ExpirationProfile,
    breakeven_at_expiry,
    expiration_pnl,
    expiration_profile,
    intrinsic_value,
)
from option_persona.risk.scenarios import (
    ModelAssumptions,
    model_caveats,
    model_value_per_unit,
    scenario_grid,
    spot_vol_stress,
)

__all__ = [
    "ExpirationProfile",
    "ExposureReport",
    "ModelAssumptions",
    "RiskLimits",
    "breakeven_at_expiry",
    "compare_with_shares",
    "expiration_pnl",
    "expiration_profile",
    "exposure_report",
    "intrinsic_value",
    "model_caveats",
    "model_value_per_unit",
    "scenario_grid",
    "spot_vol_stress",
]
