"""Module B — Arbitrage Research Engine (European put-call parity in Phase 1)."""

from option_persona.arbitrage.parity import (
    ExecutionCosts,
    ParityAssessment,
    ParityClassification,
    ParityQuoteSet,
    assess_parity,
    implied_forward,
    put_call_parity_residual,
)
from option_persona.arbitrage.synthetic import generate_synthetic_chain, scan_chain_for_parity

__all__ = [
    "ExecutionCosts",
    "ParityAssessment",
    "ParityClassification",
    "ParityQuoteSet",
    "assess_parity",
    "generate_synthetic_chain",
    "implied_forward",
    "put_call_parity_residual",
    "scan_chain_for_parity",
]
