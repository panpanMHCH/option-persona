"""Market-data ingestion (Phase 2: option-chain CSV import with provenance)."""

from option_persona.data.chain_csv import (
    CANONICAL_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    ChainImport,
    ImportIssue,
    IssueSeverity,
    parse_option_chain,
    read_option_chain_csv,
)
from option_persona.data.contract_specs import KNOWN_CONTRACT_SPECS, ContractSpec, known_spec
from option_persona.data.examples import SYNTHETIC_UNDERLYING, synthetic_canonical_chain

__all__ = [
    "CANONICAL_COLUMNS",
    "KNOWN_CONTRACT_SPECS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SYNTHETIC_UNDERLYING",
    "ChainImport",
    "ContractSpec",
    "ImportIssue",
    "IssueSeverity",
    "known_spec",
    "parse_option_chain",
    "read_option_chain_csv",
    "synthetic_canonical_chain",
]
