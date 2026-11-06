"""Enumerations describing option contract characteristics."""

from enum import StrEnum


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: "OptionType | str") -> "OptionType":
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Option type must be 'call' or 'put', got {value!r}.") from exc


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class SettlementType(StrEnum):
    PHYSICAL = "physical"  # delivery of the underlying (e.g. US single-stock options)
    CASH = "cash"  # cash settlement against a reference value (e.g. SPX, XSP)
