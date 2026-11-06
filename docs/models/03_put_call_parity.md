# Put-call parity and arbitrage feasibility

Implementation: `src/option_persona/arbitrage/parity.py`, `synthetic.py`

## Intuition and equation

A long European call plus a short European put with the same K and T pays S_T − K in every state of the world. That's a forward contract. Two portfolios with identical payoffs must cost the same today:

$$C - P = S e^{-qT} - K e^{-rT}$$

This is **model-free**: it needs no volatility, no lognormality and no BSM. A volatility smile doesn't affect it, because the call and put at the same strike share one IV.

## Assumptions

- **European exercise**, the same underlying, strike and expiry for both legs.
- **A known dividend yield** (or, equivalently, known discrete dividends).
- **A single borrowing and lending rate.**
- **No transaction costs.**

## American options

For American options parity weakens to the bounds S − D − K ≤ C − P ≤ S − Ke^{−rT}, where D is the present value of dividends. `assess_parity` raises `NotApplicableError` for American contracts rather than silently applying the equality.

## Three tiers of evidence

| Tier | Quantity | Meaning |
|---|---|---|
| 1. Theoretical | Mid residual (C_mid − P_mid) − (S_mid e^{−qT} − Ke^{−rT}) | A pricing observation. Mids aren't executable. Residuals under ~2 ticks are rounding noise. |
| 2. Apparent executable edge | **Conversion:** C_bid − P_ask − S_ask e^{−qT} + K e^{−r_borrow T}<br>**Reversal:** P_bid − C_ask + S_bid e^{−qT} − K e^{−r_lend T} − borrow fee<br>(both × multiplier, minus fees) | What survives quoted spreads, fees and asymmetric financing. Labelled "apparent", never "risk-free". |
| 3. Execution and operational risks | Text notes | Legging and fill risk, stale or unsynchronised quotes, dividend uncertainty, settlement style (cash/AM/PM), pin risk, stock recall. |

**Classification:**

- `apparent_executable_edge` if the best edge is above `min_edge` (default $1 per package).
- `theoretical_only` if the mid residual is above `mid_tolerance` (default $0.02).
- `no_violation` otherwise.

The `half_spread_cost` field is the residual you need just to cross the three spreads, before fees and financing.

## Synthetic data

`generate_synthetic_chain` builds BSM-consistent quotes with a smile and realistic spreads, then shifts the call quotes at chosen strikes. The shifts are recorded in `injected_call_shift`.

Every row is labelled `is_synthetic=True`, `source="synthetic:bsm"`, and every assessment carries a "SYNTHETIC DATA" risk note.

## Validation

- The residual is ≤ 4 × 10⁻¹⁶ × max(S, K) over 100,000 random BSM prices.
- A zero-spread, zero-cost fair quote gives zero edges.
- A hand-computed conversion edge is reproduced exactly: (0.50 − 3 × 0.05) × 100 − 2 × 0.65.
- On the synthetic chain, +0.60 at K = 90 is detected as a conversion and −0.50 at K = 105 as a reversal. The +0.03 injection at K = 110 is visible at the mid but correctly **not** executable.
- Clean strikes never produce an executable edge.

## How a trader uses it

Parity violations at mid prices are common and almost never tradeable: they come from wide or stale markets and zero-bid wings.

The useful research question is how large a residual must be, given a specific broker's fees and financing, to survive execution, and how often real chains exceed that. That's the Phase 2 agenda, on European cash-settled index options (XSP/SPX).

For SPX, the more natural use of parity is **extracting the implied forward** (`implied_forward`). It avoids needing a dividend forecast and reveals the market's implied financing rate.

## Limitations (Phase 1)

- Flat rate spreads.
- Continuous dividend yield.
- No quote-size check.
- Freshness checking is limited to a leg-timestamp gap.
- No futures or ETF basis model for indices.
