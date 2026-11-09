# Strike comparison and hold-or-sell review

Implementation: `src/option_persona/risk/{strike_comparison,exit_review}.py`
Tests: `tests/test_strike_and_exit.py`
Dashboard: the **Strike comparison** tab, and the **Hold-or-sell review** section of the Trade analyser.

These two tools answer the practical questions "which strike?" and "should I close before expiry?". Neither tool answers the question for you. Each lays out the factors with their numbers, because the deciding inputs are your view, horizon and risk budget, and those belong to you (rule 1: the tools explain, they never recommend).

## 1. Strike comparison

### Intuition

Every strike is a different bet on the same view:

- **A lower-strike put** (further out of the money) is cheaper but needs a bigger fall.
- **A higher-strike put** costs more, needs less of a fall, decays more dollars per day and is less leveraged in percentage terms.

The table puts these numbers side by side.

### Price basis

| Row type | Entry price | Volatility used |
|---|---|---|
| Observed (bid and ask entered) | **Ask**: a buyer pays the offer | BSM IV of the ask, so the market's skew at that strike is reflected |
| Model (no quote) | BSM value | The single reference σ (a flat smile) |

Model rows understate the cost of lower-strike equity puts, because real chains price them at higher IVs (skew). Enter real quotes when you have them.

### Columns

All dollar figures are for the chosen number of contracts.

- **Cost** = entry × multiplier × contracts + entry fees. This is also the maximum loss.
- **Breakeven at expiry** = K ± (entry + fees / units). Entry fees only, because exit costs are unknown before the trade.
- **Move to breakeven** = breakeven / S − 1.
- **Breakeven in expected moves:** (breakeven − S) / (S σ√T), with σ the reference volatility.
  - S σ√T is the one-standard-deviation move implied by σ. It's a scale, not a forecast.
  - A value of −0.5 is well inside a typical move; −1.5 needs an unusual one.
- **Q-prob ITM** = N(±d2). **Q-prob past breakeven** is the same quantity with the breakeven in place of the strike.
  - Both are risk-neutral (Q-measure) probabilities. They describe the market's pricing, not real-world odds (rule 6).
- **Theta $/day**, and **theta % of value/day**: the share of the option lost per calendar day if nothing else changes.
- **λ (elasticity)** = Δ S / V. It is the % change in option value per 1% move in the underlying.

### What the table shows, and what it doesn't

Here are your NVDA case-study inputs: S = 201.50, 7.25 days, σ = 46%, r = 4%, all model rows.

| Strike | Cost | Breakeven | Move needed | In expected moves | Q-prob ITM | Theta $/day |
|---|---|---|---|---|---|---|
| 185 | $54 | 184.46 | −8.5% | −1.30 | 10% | −14.0 |
| 190 | $124 | 188.76 | −6.3% | −0.97 | 19% | −22.5 |
| 195 | $248 | 192.52 | −4.5% | −0.69 | 31% | −30.2 |
| 200 | $441 | 195.59 | −2.9% | −0.45 | 46% | −34.5 |

Under the model all four strikes are **fairly priced**. None is "cheap", because a lower cost buys a lower chance. Whether one is better for you depends on whether you believe the real-world odds differ from the Q-odds, and on how much you can afford to lose. The table can't know either.

## 2. Hold-or-sell review

### Intuition

1. **The re-buy test.** Holding an option is economically the same as selling it at the bid and buying it straight back. Your entry price is a sunk cost: it has no bearing on what the option is worth or can do from here. So the useful question is: would you open this trade today, at today's prices?
2. **Theta.** Each day you hold, theta is the price of keeping the remaining chance alive.
3. **Physical settlement.** At expiry, US single-stock options become share positions.

### The checklist

Each factor gets a status: `ok`, `watch`, `alert` or `info`. Thresholds are in `ExitReviewThresholds` and are personal settings.

| Factor | Finding | Default thresholds |
|---|---|---|
| Reason for the trade | Your answer: holds, no longer holds, or not sure | no → alert, not sure → watch |
| Re-buy test | Breakeven and move needed if bought now at the ask (or at model value) | info |
| Time left | Days to 16:00 ET expiry, and the same instant in Bangkok | ≤ 2 days alert, ≤ 7 days watch |
| Time decay | Theta $/day and as a share of value | ≥ 10%/day alert, ≥ 5%/day watch |
| Intrinsic vs time value | Value = intrinsic + time value; selling captures the time value | watch if an American ITM bid is below intrinsic |
| Expiry settlement | Whether it's in the money, how close to the strike, and which shares would be delivered | ITM in the final 2 days → alert; within one expected move → watch |
| Liquidity | Spread as a % of mid, and the cost of selling at the bid versus mid | zero bid or ≥ 10% → alert, ≥ 5% → watch |
| Unrealised P&L | At the bid (or the model value), after entry fees | info |

**Automatic exercise.** US equity options that finish in the money by $0.01 or more are normally exercised automatically. Broker policies vary.

- **A long put** that is exercised sells 100 shares per contract at the strike. Without shares to deliver, that becomes a **short stock position**.
- **A long call** that is exercised buys 100 shares per contract at the strike, for the full cash amount.

The resulting position appears after the 16:00 ET close, which is 03:00–04:00 in Bangkok, and can be carried over a weekend.

### Limitations

- **Constant volatility and European exercise.** IV usually rises when equities fall. The early-exercise premium of American options is ignored.
- **The thresholds are personal choices,** not market facts.
- **Without a live bid/ask, values fall back to the model.** The review says so wherever this happens.
- **The review knows nothing outside the position:** earnings dates, news, portfolio context or your account margin.
