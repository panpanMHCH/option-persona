# Roadmap

## Phase 1: Mathematical foundation and working MVP ✅ (6 November 2026)

- [x] Repository structure, `CLAUDE.md`, README, model documentation, benchmark of existing tools
- [x] European Black-Scholes-Merton pricing with continuous dividend yield
- [x] Analytical delta, gamma, vega, theta, rho, and risk-neutral ITM probability
- [x] Implied volatility: bracketed Brent, OTM-side conditioning, no-arbitrage bound errors
- [x] Verification: published values, QuantLib (648 cases), finite differences, BSM PDE, hypothesis identities
- [x] European put-call parity: residuals, implied forward, three-tier assessment, synthetic chains with injected violations
- [x] Long call/put expiration analysis: breakeven, max loss/gain, P&L after fees
- [x] Scenario grid (spot × time) and spot × vol stress using explicit `ModelAssumptions`
- [x] Leverage: share-equivalent, delta notional, elasticity, % equity, share comparison, warnings
- [x] Minimal Streamlit dashboard (pricer, IV, trade analyser, parity lab)
- [x] NVDA $195P case study (model-free facts versus labelled hypothetical reconstruction)

**Acceptance criteria:** all met. See [docs/verification.md](docs/verification.md).

## Phase 2: Arbitrage feasibility (in progress)

- [x] Option-chain CSV ingestion into validated `OptionQuote` records, with source and timestamp (9 November 2026; see [docs/data/chain_csv_import.md](docs/data/chain_csv_import.md))
- [ ] Local storage in DuckDB/Parquet, plus a provider interface for future data feeds
- [ ] Quote freshness, staleness and leg-synchronisation checks; displayed size; zero-bid handling
- [ ] Executable bid-ask parity screening over real chain snapshots (XSP/SPX first: European, cash-settled)
- [ ] Implied-forward and implied-rate extraction from parity, avoiding a dividend assumption for indices
- [ ] Vertical spread bounds (0 ≤ C(K1) − C(K2) ≤ (K2 − K1)e^{−rT}), strike monotonicity and butterfly convexity checks
- [ ] Box spreads: implied financing rate versus benchmark rates, after costs
- [ ] Richer cost model: exchange/regulatory fees, tiered commissions, slippage, margin financing

## Added at the owner's request (9 November 2026)

These sit outside the original phase order. They reuse only validated Phase 1 maths.

- [x] Strike comparison: cost, breakeven, move needed in expected moves, Q-probabilities, theta and λ across candidate strikes; observed rows enter at the ask with their own IV ([notes](docs/models/05_strike_comparison_and_exit_review.md))
- [x] Hold-or-sell review: a checklist of factors (re-buy test, time left, theta, intrinsic versus time value, auto-exercise and delivery, liquidity, the reason for the trade) that never outputs a recommendation

## Phase 3: Research and personal trading analytics

- [ ] American options: Cox-Ross-Rubinstein binomial tree (convergence study versus QuantLib), early-exercise boundary
- [ ] Volatility surface: smile/skew, term structure, realised versus implied volatility, expected move
- [ ] Defined-risk spreads in the trade analyser (verticals) and multi-leg payoff
- [ ] Trade journal persisted locally; P&L attribution (delta, gamma, theta, vega, residual) using recorded observations
- [ ] Historical strategy simulation without look-ahead bias
- [ ] Portfolio-level exposure aggregation and concentration analytics

## Phase 4: Product expansion (only after earlier phases are validated)

- [ ] FastAPI service and a Next.js/TypeScript interface
- [ ] Supported market-data connections (after a cost review and approval)
- [ ] AI research assistant that explains verified outputs only, with human control of all decisions
