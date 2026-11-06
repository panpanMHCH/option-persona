"""Domain-specific exceptions with messages intended for end users."""


class OptionPersonaError(Exception):
    """Base class for all Option Persona errors."""


class InvalidInputError(OptionPersonaError, ValueError):
    """A model input is outside its mathematically valid domain."""


class ArbitrageBoundError(OptionPersonaError, ValueError):
    """An observed price violates model-free no-arbitrage bounds.

    Raised, for example, when no implied volatility can exist because the
    price is below discounted intrinsic value or above its upper bound.
    """


class ConvergenceError(OptionPersonaError, RuntimeError):
    """A numerical solver failed to converge to the requested tolerance."""


class NotApplicableError(OptionPersonaError, ValueError):
    """A model was requested for an instrument whose assumptions it violates
    (for example, European put-call parity applied to American options)."""
