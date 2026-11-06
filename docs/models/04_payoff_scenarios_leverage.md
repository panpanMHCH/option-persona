# Payoff, scenarios and leverage

Implementation: `src/option_persona/risk/{payoff,scenarios,leverage}.py`

## 1. Expiration payoff (model-free, exact)

Payoff per share: call max(S_T − K, 0), put max(K − S_T, 0).

For a long position of n contracts with multiplier m, premium p per share and total fees f:

- P&L(S_T) = (payoff − p)·m·n − f
- Breakeven: K ± (p + f/(m·n)) (+ for a call, − for a put)
- Maximum loss: p·m·n + f
- Maximum gain: unbounded for a call; (K − p)·m·n − f for a put (at S_T = 0)

**Assumptions:** held to expiry, settled at intrinsic value against the official settlement price, and exercise fees and taxes excluded unless recorded as `exit_fees`.

## 2. Pre-expiration scenarios (model-dependent)

`scenario_grid` and `spot_vol_stress` revalue the position with BSM using an explicit `ModelAssumptions` (σ, r, q, model name). Its `describe()` text is shown next to every result.

Horizons past expiry collapse to intrinsic value, so the model curve meets the exact payoff at T = 0. This is tested.

**What these numbers are:** theoretical mid-market values *conditional* on a constant volatility.

**What they are not:**

- Forecasts.
- Executable prices.
- A description of how IV will move. IV typically rises when equities fall; a spot × vol stress grid exists for exactly this reason.

`model_caveats` adds position-specific warnings, such as American puts valued with a European model.

## 3. Leverage and exposure

| Measure | Definition | Interpretation |
|---|---|---|
| Share-equivalent | Δ·m·n | Shares with the same P&L for a small move |
| Delta-adjusted notional | Δ·m·n·S | Market exposure in currency |
| Elasticity λ | Δ·S/V = ∂ln V / ∂ln S | % change in option value per 1% move in the underlying |
| Premium at risk | current model value V·m·n | The most a long option holder can still lose |
| Gamma dollars (1%) | Γ·m·n·S²·0.01 | Change in delta-notional for a 1% move |

**Validation:**

- The identities are tested directly, and λ is checked against a finite difference of ln V.
- For tiny moves, option P&L matches the delta-equivalent shares.
- For ±10% moves, the long option beats the delta-equivalent shares (convexity) but never loses more than its value.

## Why leverage does not raise expected return

- **Under Q**, every asset (options included) earns r in expectation. Leverage changes the *amount* of risk, not the reward per unit of risk.
- **Under the real-world measure P**, an option's expected excess return is roughly λ times the underlying's. For a put on a stock with a positive risk premium, that is *negative* before costs, and long options also typically pay a volatility risk premium.
- **λ is local.** It changes with spot, time and vol, and high λ magnifies forecast errors just as much as correct forecasts.
- **Position sizing follows from this.** If your edge estimate is uncertain, the effective size you should take shrinks. Sizing on λ-magnified expected gains is the classic way leveraged traders blow up.

**Warnings** (thresholds in `RiskLimits` are personal choices, not universal rules):

- Premium at risk above 2% of equity.
- Delta notional above 25% of equity.
- |λ| above 10.
- Fewer than 7 days to expiry.
- Relative bid-ask spread above 10%.

## Limitations

- Single-position analysis; no portfolio aggregation or correlations yet.
- Constant-σ BSM; American puts are under-valued (the result is a lower bound).
- Share comparison ignores dividends, financing and margin on the share leg. Short-share benchmarks are hypothetical.
