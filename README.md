# Option Persona (OP)

**Quantitative Options Intelligence Platform** — *Understand the price. Own the risk.*

Option Persona is a research platform for options valuation, arbitrage feasibility and personal risk analysis. It is built around mathematically verified pricing functions, and it keeps three kinds of number visibly separate: market observations, model outputs and assumptions.

It is **read-only and analytical**. It does not connect to brokers, place orders or make trading decisions, and nothing in it is investment advice.

## Status: Phase 1 complete; Phase 2 (arbitrage feasibility) in progress

| Area | Phase 1 contents |
|---|---|
| Module A — Pricing | European Black-Scholes-Merton with continuous dividend yield; analytical delta, gamma, vega, theta, rho; implied volatility (bracketed Brent with OTM conditioning); model-free price bounds; explicit T→0 and σ→0 limits |
| Module B — Arbitrage | Put-call parity residuals; three-tier assessment (theoretical residual → apparent executable edge at bid/ask after fees and asymmetric financing → execution risks); synthetic chains with labelled injected violations |
| Module D — Leverage & risk | Share-equivalent exposure, delta-adjusted notional, elasticity (λ), premium at risk, % of equity, spot × vol stress grid, comparison with shares, concentration/leverage/expiry/liquidity warnings |
| Module E — Trade analyser | Long call/put positions: breakeven, max loss/gain, expiry P&L after fees, BSM scenario grid through time, realised P&L |
| Data ingestion (Phase 2) | Option-chain CSV import into validated quotes with provenance; row-level rejection with reasons; contract terms from the file, a known-index registry, or a stated assumption ([notes](docs/data/chain_csv_import.md)) |
| Dashboard | Streamlit app connecting the modules above, including a chain-import tab |

See [ROADMAP.md](ROADMAP.md) for what comes next.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra app
```

This installs the library, the Streamlit app and the development tools (pytest, hypothesis, ruff), plus **QuantLib**, which the tests use only as an independent reference implementation.

Without uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[app]" pytest hypothesis ruff QuantLib
```

## Usage

Run the dashboard:

```bash
uv run streamlit run app/streamlit_app.py
```

Run the tests:

```bash
uv run pytest
```

Reproduce the verification report in [docs/verification.md](docs/verification.md):

```bash
PYTHONPATH=. uv run python scripts/verify_phase1.py
```

Reproduce the NVDA case study:

```bash
uv run python scripts/case_study_nvda.py
```

Library use:

```python
from option_persona.pricing import bsm_greeks, implied_volatility

g = bsm_greeks("put", S=201.5, K=195, T=7/365, r=0.04, sigma=0.46)
g.price, g.delta, g.theta_per_day, g.vega_per_vol_point

implied_volatility(2.38, "put", S=201.5, K=195, T=7/365, r=0.04)  # ≈ 0.46
```

## Repository layout

```
src/option_persona/
  core/        types, validation, timezone-aware time conventions, Pydantic data models
  pricing/     Black-Scholes-Merton, Greeks, implied volatility          (Module A)
  arbitrage/   put-call parity assessment, synthetic chain generator     (Module B)
  risk/        expiry payoff, scenario engine, leverage & exposure       (Modules D, E)
  data/        option-chain CSV ingestion with provenance and stated assumptions
app/           Streamlit dashboard (no financial maths in the UI layer)
tests/         pytest + hypothesis; QuantLib cross-validation
scripts/       reproducible verification and case-study scripts
docs/          model notes, benchmark of existing tools, verification log, case studies
```

## Conventions and assumptions (summary)

Full detail is in [docs/models/](docs/models/) and [CLAUDE.md](CLAUDE.md).

- **Rates and yields** are continuously compounded annual decimals (0.04 = 4%). Negative rates are allowed.
- **Time** is ACT/365F calendar time, computed from timezone-aware datetimes. US equity options expire at 16:00 America/New_York. Naive datetimes are rejected.
- **Greek units:** per share; theta per year (`theta_per_day` = /365); vega and rho per 1.00 (`*_per_vol_point`, `*_per_rate_point` = /100).
- **Probabilities** from N(d2) are risk-neutral (Q-measure). They are not real-world forecasts.
- **Mid-quotes** are indicative. Executable analysis buys at the ask and sells at the bid.
- **Synthetic data** is always flagged `is_synthetic=True` and `source="synthetic:bsm"`.

## Limitations (Phase 1)

- **European exercise only.** American options valued with BSM are approximations: for puts the value is a lower bound, because BSM omits the early-exercise premium. US single-stock options are American, so the NVDA case study carries this caveat. A binomial American pricer is planned for Phase 3.
- **Constant volatility, rate and dividend yield.** There's no smile, term structure or stochastic volatility in scenarios yet. In a sell-off, implied volatility usually rises, so constant-vol scenarios understate put values in crashes.
- **Continuous dividend yield** approximates discrete dividends.
- **Basic cost model.** Fees are per contract, with flat borrow/lend spreads and a flat stock-borrow fee. There's no slippage model, no quote-size checks and no margin.
- **No market-data feed.** Chains can be imported from CSV files you supply. Storage, quote-freshness and liquidity checks, and vertical/box spreads are still to come in Phase 2.
- **Long positions only.** No short options, spreads, margin borrowing or portfolio aggregation in the MVP.

## Licence

Proprietary — all rights reserved (personal research and portfolio project).
