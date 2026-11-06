# Black-Scholes-Merton pricing and Greeks

Implementation: `src/option_persona/pricing/black_scholes.py`

## Intuition

Black-Scholes doesn't forecast where a stock is going. It prices an option by **replication**: under the model's assumptions, a continuously rebalanced portfolio of Δ shares plus cash reproduces the option's payoff exactly. The cost of that portfolio must be the option's price, or there's an arbitrage.

Because the hedge removes exposure to the stock's direction, the stock's expected return μ drops out. The price is then the expected discounted payoff under the **risk-neutral measure Q**, where the underlying drifts at r − q.

## Equations

Under Q, dS = (r − q) S dt + σ S dW. Writing F = S e^{(r−q)T}, D = e^{−rT} and v = σ√T:

$$d_1 = \frac{\ln(F/K) + v^2/2}{v}, \qquad d_2 = d_1 - v$$

$$C = D\,[F N(d_1) - K N(d_2)], \qquad P = D\,[K N(-d_2) - F N(-d_1)]$$

Writing ω = +1 for a call and −1 for a put, and φ for the standard normal density:

| Greek | Formula | Code units |
|---|---|---|
| Δ | ω e^{−qT} N(ω d₁) | per share |
| Γ | e^{−qT} φ(d₁) / (S σ √T) | per share per $1 |
| Vega | S e^{−qT} φ(d₁) √T | per 1.00 σ (`vega_per_vol_point` = /100) |
| Θ | −S e^{−qT} φ(d₁) σ / (2√T) − ω r K e^{−rT} N(ω d₂) + ω q S e^{−qT} N(ω d₁) | per year (`theta_per_day` = /365) |
| ρ | ω K T e^{−rT} N(ω d₂) | per 1.00 r (`rho_per_rate_point` = /100) |

N(d₂) is the **risk-neutral** probability that a call finishes in the money. In the real world that probability also depends on μ and on the volatility risk premium; the model has no view on either.

## Assumptions

- **European exercise.**
- **Frictionless markets:** no costs, continuous trading and short selling allowed.
- **Constant r, q and σ.**
- **Lognormal underlying with no jumps.**
- **Continuous dividend yield.**

Each of these fails in practice. Implied-volatility smiles are the market's way of correcting the constant-σ, no-jump assumptions.

## Numerical choices

- Puts use N(−d) directly, not parity, so deep-OTM values keep full relative precision.
- ln(F/K) is computed as ln(S/K) + (r − q)T.
- **Degenerate limit (v ≤ 10⁻¹²):** the terminal distribution is a point mass at F. The price is D·max(ω(F − K), 0), Δ is an indicator function, and Γ and vega are 0. Exactly at the money at expiry, Γ is formally a Dirac delta, which no finite number represents.
- Inputs are validated with actionable messages. σ > 20 is rejected as a probable percentage-versus-decimal error.

## Validation

The full record is in [docs/verification.md](../verification.md):

- Hull Example 15.6 and Haug's (2007) BSM and Merton-1973 examples reproduce to the published precision.
- QuantLib `AnalyticEuropeanEngine` agrees on price and all five Greeks over 648 cases to ≤ 3 × 10⁻¹².
- Central finite differences agree within O(h²) truncation error.
- The BSM PDE residual Θ + ½σ²S²Γ + (r − q)SΔ − rV is ≤ 10⁻¹⁴.
- Hypothesis property tests check parity, no-arbitrage bounds, monotonicity and convexity in strike, and the digital identity N(d₂) = −e^{rT} ∂C/∂K.

## How a trader uses it

- **Δ** is your share-equivalent exposure.
- **Γ** tells you how fast that exposure changes. Γ is largest near the money close to expiry, which is why short-dated options swing violently.
- **Θ** is the daily cost of holding that convexity. Θ and Γ are linked by the PDE: for a delta-hedged option, Θ ≈ −½σ²S²Γ, so you pay for gamma through time decay.
- **Vega** is exposure to implied volatility, which often matters more than direction for longer-dated options.

## Limitations

- **Not valid for American puts.** The early-exercise premium is missing.
- **No vol smile or skew:** each strike's IV must be supplied separately.
- **No jump or event risk** (earnings, for example).
- **Model value is a mid-type theoretical value, not a tradeable price.**
