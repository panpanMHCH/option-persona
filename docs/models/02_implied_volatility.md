# Implied volatility

Implementation: `src/option_persona/pricing/implied_vol.py`

## Intuition

Implied volatility (IV) is the σ that makes the BSM price equal an observed price. It's a **quoting convention**: a way of expressing a price in units that can be compared across strikes, expiries and underlyings. It is not a forecast of realised volatility. IV usually exceeds subsequent realised volatility on average (the volatility risk premium) and jumps ahead of known events.

## Existence and uniqueness

For T > 0, vega > 0, so the price is strictly increasing in σ. It runs from the lower no-arbitrage bound (σ → 0) to the upper bound (σ → ∞):

- Call: max(Se^{−qT} − Ke^{−rT}, 0) < C < Se^{−qT}
- Put: max(Ke^{−rT} − Se^{−qT}, 0) < P < Ke^{−rT}

A price strictly inside these bounds has exactly one IV. A price outside them has none, and the solver raises `ArbitrageBoundError` with the bound value.

In practice that error usually means one of the following, not an arbitrage:

- a stale quote;
- a mid-quote of a wide market;
- a wrong dividend or rate input;
- an American option's early-exercise value.

## Method

1. Validate inputs and check the bounds. A price at the lower bound returns 0.
2. **Conditioning:** convert an ITM option's price to the OTM option at the same strike using parity, which is exact in-model. An ITM option's time value is a small difference of large numbers, so solving on the OTM side preserves precision.
3. Bracket the root: start at [10⁻⁸, 1] and double the upper end until it brackets (cap 2000%). Then apply Brent's method with xtol = 10⁻¹².

Brent is guaranteed to converge for a continuous monotone function. Newton-Raphson fails where vega → 0: deep OTM and very short expiries.

## Validation

- Over a grid of 350 (type, strike, T, σ) combinations, the price-space round trip error is ≤ 10⁻¹¹ × S. The vol-space error is ≤ 10⁻¹¹ wherever vega > 10⁻³.
- 84 grid points are skipped because the option has **no time value at double precision** (for example K = 50, T = 1 day, σ = 3%). There, σ is genuinely not identifiable from the price. That's information loss, not a solver failure.
- Call and put at the same strike return identical IV under parity.
- Out-of-bounds prices raise with clear messages.

## Conditioning: what the error really is

The IV error is approximately δP / vega. A $0.01 price error on an option with vega = 0.01 per 1.00 σ, i.e. $0.0001 per vol point, produces a 1.00 (100 vol point) IV error.

That's why IVs of far-OTM, short-dated options are noisy even when the solver is exact. Solving the bid and the ask separately shows the IV band the market actually implies; the dashboard does this.

## Limitations

- **European model.** For American puts, BSM-IV slightly overstates volatility.
- **Sensitive to the time measure.** Calendar time and trading-day time give different IVs, especially across weekends and around events.
- **Sensitive to inputs.** The rate and dividend inputs shift IV, most of all for long-dated options.
