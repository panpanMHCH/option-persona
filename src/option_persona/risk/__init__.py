"""Module D (leverage & risk) and Module E foundations (position payoff & scenarios)."""

from option_persona.risk.exit_review import (
    ExitReview,
    ExitReviewThresholds,
    ReviewItem,
    ReviewStatus,
    exit_review,
)
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
from option_persona.risk.strike_comparison import StrikeCandidate, compare_strikes, expected_move

__all__ = [
    "ExitReview",
    "ExitReviewThresholds",
    "ExpirationProfile",
    "ExposureReport",
    "ModelAssumptions",
    "ReviewItem",
    "ReviewStatus",
    "RiskLimits",
    "StrikeCandidate",
    "breakeven_at_expiry",
    "compare_strikes",
    "compare_with_shares",
    "exit_review",
    "expected_move",
    "expiration_pnl",
    "expiration_profile",
    "exposure_report",
    "intrinsic_value",
    "model_caveats",
    "model_value_per_unit",
    "scenario_grid",
    "spot_vol_stress",
]
