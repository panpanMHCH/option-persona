# Verification record — Phase 1

Generated on 6 November 2026 with Python 3.12, NumPy 2, SciPy and QuantLib 1.44 on macOS arm64. Regenerate with:

```bash
PYTHONPATH=. uv run python scripts/verify_phase1.py
uv run pytest
```

**Test suite:** 1,272 passed, 84 skipped. The skips are IV round trips where the option has no time value at double precision, so σ cannot be identified (see [02_implied_volatility.md](models/02_implied_volatility.md)). `RuntimeWarning` is promoted to an error, so no silent overflows or divide-by-zero occurred.

## Checks and tolerances

| # | Check | Independent of our code? | Tolerance asserted in tests | Observed worst case |
|---|---|---|---|---|
| 1 | Published values: Hull Ex. 15.6, Haug (2007) | Yes (textbooks) | Published rounding (±0.005, ±0.00005) | Matches to all published digits |
| 2 | QuantLib `AnalyticEuropeanEngine`, 648 cases (S, K, T from 1 day to 5 years, r including negative, q, σ from 5% to 120%) | Yes (independent C++ library) | rel 1e-9, abs 1e-10 | 2.9e-12 (theta), ≤ 1.6e-13 for the others |
| 3 | Central finite differences, 144 cases | Partly (uses our price function) | First-order Greeks rel 1e-5 / abs 1e-6; gamma rel 1e-4 / abs 1e-5 | Rel ≤ 2.7e-6, consistent with O(h²) truncation |
| 3b | BSM PDE: Θ + ½σ²S²Γ + (r−q)SΔ − rV = 0 | Yes (an identity linking all Greeks) | 1e-9 × max(S, K) | 6.9e-15 |
| 4 | Put-call parity on random BSM prices (hypothesis: 500 per property; script: 100,000) | Yes (model-free identity) | 1e-10 × max(S, K) | 3.7e-16 × max(S, K) |
| 4b | No-arbitrage bounds; strike monotonicity and convexity; N(d₂) = −e^{rT} ∂C/∂K; Δ_C − Δ_P = e^{−qT} | Yes | 1e-10 × max(S, K) / 1e-7 | Pass |
| 5 | IV round trip, 350 grid points | Yes (inverse problem) | Price 1e-11 × S; vol max(1e-8, 1e-11·S/vega) | Price 7.5e-12, vol 1.2e-12 |
| 6 | Synthetic parity scan with injected violations | Yes (ground-truth labels) | Correct class and direction; residual within ±0.03 of injection | Pass |
| 7 | Payoff, breakeven, max loss, NVDA expiry P&L points | Yes (hand calculation) | 1e-9 | Pass |
| 8 | Scenario grid equals exact payoff at expiry; elasticity = ∂ln V/∂ln S | Yes | 1e-6 relative | Pass |
| 9 | Time: DST-aware 16:00 ET expiry; Bangkok ↔ New York; naive datetimes rejected | Yes | Exact | Pass |

## Raw output

```
1. Published reference values
   Hull Ex. 15.6 call   OP = 4.759422   published = 4.76
   Hull Ex. 15.6 put    OP = 0.808599   published = 0.81
   Haug BSM call        OP = 2.133368   published = 2.1334
   Haug Merton-73 put   OP = 2.464788   published = 2.4648
2. QuantLib AnalyticEuropeanEngine, 648 cases — max absolute difference
   price  4.263e-14
   delta  1.255e-14
   gamma  4.996e-16
   vega   1.563e-13
   theta  2.851e-12
   rho    5.826e-13
3. Central finite differences, 144 cases
   greek   max abs err   max rel err (|greek| > 1e-4)
   delta     1.190e-07                      2.549e-07
   gamma     3.706e-07                      1.931e-06
   vega      4.818e-08                      2.666e-06
   theta     1.102e-07                      3.056e-07
   rho       9.487e-08                      1.094e-08
   BSM PDE residual theta + 1/2 s^2 S^2 gamma + (r-q) S delta - rV: max abs 6.883e-15
4. Put-call parity on 100,000 random BSM prices — max |residual| / max(S,K) = 3.706e-16
5. Implied volatility round trip: 266 solved, 84 with no identifiable time value
   max price error 7.471e-12;  max vol error (vega > 1e-3) 1.198e-12
6. Synthetic chain (SYNTHETIC DATA) — injected call shifts {90.0: 0.6, 105.0: -0.5, 110.0: 0.03}
   K=   80  mid residual -0.0154  conv   -29.62  rev   -28.59  no_violation
   K=   85  mid residual -0.0018  conv   -24.46  rev   -26.15  no_violation
   K=   90  mid residual +0.6068  conv   +40.69  rev   -82.71  apparent_executable_edge
   K=   95  mid residual +0.0053  conv   -15.16  rev   -18.28  no_violation
   K=  100  mid residual -0.0061  conv   -13.00  rev   -13.84  no_violation
   K=  105  mid residual -0.5025  conv   -65.85  rev   +32.59  apparent_executable_edge
   K=  110  mid residual +0.0261  conv   -17.69  rev   -24.97  theoretical_only
   K=  115  mid residual +0.0147  conv   -22.54  rev   -27.53  no_violation
   K=  120  mid residual +0.0133  conv   -27.38  rev   -32.10  no_violation
```
