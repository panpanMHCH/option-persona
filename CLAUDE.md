# CLAUDE.md — Option Persona (OP)

## Purpose

Option Persona is a quantitative options research platform with three aims:

- Mathematically rigorous pricing.
- Honest assessment of arbitrage feasibility.
- Personal trade risk analysis.

It is both a learning project and a portfolio piece for quant, data-science and fintech roles. It is independent of every other project; in particular, never touch the SlabSiam repository.

The differentiation it pursues: **explainable valuation, arbitrage feasibility assessment and personal risk intelligence in one research workflow.** Payoff diagrams and option calculators are commoditised. The value here is the transparency, verification and honest separation of observation from model.

## Non-negotiable rules

1. **Read-only.** No broker order placement, automatic execution or autonomous trading. Any future AI assistant explains verified calculations; it never manufactures prices, risk numbers or recommendations.
2. **Never present synthetic or model values as market observations.** Every market number carries provenance (source, aware timestamp, `is_synthetic`, `is_delayed`).
3. **Mid ≠ executable.** Executable analysis buys at the ask and sells at the bid.
4. **Never call a trade risk-free.** Arbitrage output always has three tiers: theoretical residual, apparent executable edge, and execution/operational risks.
5. **American ≠ European.** European identities (parity equality, BSM value) must not be applied to American contracts without an explicit caveat or a `NotApplicableError`.
6. **Risk-neutral ≠ real-world.** Label N(d2)-style probabilities as Q-measure.
7. **Leverage is never presented as improving expected return.**
8. **No invented data.** Missing historical quotes, timestamps and exits stay missing. Reconstructions are labelled HYPOTHETICAL.
9. **No paid APIs** without reviewing costs and getting the owner's approval. No hard-coded credentials; use environment variables.
10. **No placeholders posing as finished features.** If something isn't implemented, say so.
11. Keep to the **phase plan** in ROADMAP.md. Don't start a phase until the previous one is validated.

## Architecture

- `src/option_persona/core`: enums, validation (`InvalidInputError` with actionable messages), time conventions, Pydantic models (`OptionContract`, `Provenance`, `OptionQuote`, `OptionPosition`).
- `pricing`: pure, deterministic, NumPy-broadcasting functions. No state, I/O or randomness.
- `arbitrage`: model-free identities plus the cost-aware assessment, and a seeded, labelled synthetic data generator.
- `risk`: model-free payoff (`payoff.py`), model-dependent scenarios (`scenarios.py`, with `ModelAssumptions` always surfaced), and exposure/leverage (`leverage.py`).
- `app/`: Streamlit UI only, with **no financial maths**. It calls library functions and labels outputs as observation, model or assumption.
- Pydantic is used only at the data boundary. Pricing takes plain floats/arrays for speed and testability.
- QuantLib is a **test-time oracle only**, never a runtime dependency.

Key decisions and trade-offs:

- **Brent rather than Newton for IV:** robust bracketing where vega is tiny. Jäckel's "Let's Be Rational" is faster but would mean re-implementing mature infrastructure.
- **ITM IV is solved on the OTM side via parity** for numerical conditioning.
- **Degenerate limits (v = σ√T ≤ 1e-12)** use d → ±∞ and φ terms = 0. Gamma, vega and the decay part of theta are reported as 0 there (the true ATM-at-expiry gamma is a Dirac delta).
- **Parity classification** uses an absolute mid tolerance (default 2 ticks) and a materiality threshold `min_edge` (default $1 per package).
- **Altair for charts** (bundled with Streamlit), so there's no Plotly dependency.

## Mathematical conventions

- r and q are continuously compounded annual decimals; σ is an annualised decimal (0.46 = 46%; values > 20 are rejected as unit errors).
- T is in years, ACT/365F calendar seconds from aware datetimes. US equity/PM-settled expiry is 16:00 America/New_York. Supported display time zones are US Eastern and Asia/Bangkok.
- Prices are per unit of underlying. Dollar values = per-unit × multiplier × contracts.
- Greeks: delta and gamma are spot derivatives. Vega and rho are per 1.00. Theta = ∂V/∂t = −∂V/∂T per year. Helpers convert to per day, per vol point and per rate point.
- Parity: C − P = S e^{−qT} − K e^{−rT}.

## Development rules

- Python 3.12+, managed with uv. Use `uv sync --extra app`, `uv run pytest` and `uv run ruff check .`.
- Every financial function gets tests alongside it: known analytical values, QuantLib cross-checks, finite differences, identities (hypothesis), and edge cases.
- `pytest` turns `RuntimeWarning` into errors, so a silent NumPy overflow or divide-by-zero fails the suite.
- Use type hints and small pure functions. Names should be readable financial terms. Avoid premature optimisation.
- Documentation uses British English. For each model, document its intuition, equations, assumptions, validation, trader use and limitations (see `docs/models/`).
- After changing maths, update `docs/verification.md` by re-running `scripts/verify_phase1.py`.
- Keep `ROADMAP.md` current.
