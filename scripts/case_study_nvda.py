"""NVDA $195 put (6 Nov 2026) case study — reproducible calculations.

Run:  uv run python scripts/case_study_nvda.py

RECORDED FACTS (supplied by the user): 1 contract, NVDA 195 put, expiry
2026-11-06, premium 2.38/share, underlying ~201-202 at entry, delta ~-0.29,
IV ~46%. NOT RECORDED: entry timestamp, quotes after entry, exit price.

Everything below that depends on time to expiry, rates or later prices is a
HYPOTHETICAL reconstruction under stated assumptions, not historical data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from scipy.optimize import brentq

from option_persona.core.models import OptionContract, OptionPosition
from option_persona.core.time import expiry_datetime
from option_persona.pricing import bsm_delta, bsm_gamma, bsm_greeks, bsm_put
from option_persona.risk import (
    ModelAssumptions,
    compare_with_shares,
    expiration_pnl,
    expiration_profile,
    exposure_report,
    scenario_grid,
)

R = 0.04  # ASSUMPTION: continuously compounded USD rate
Q = 0.0  # ASSUMPTION: NVDA's dividend yield is ~0.02%; treated as zero
S0, K, PREMIUM, IV, DELTA = 201.5, 195.0, 2.38, 0.46, -0.29

contract = OptionContract(underlying="NVDA", option_type="put", strike=K, expiry=date(2026, 11, 6))
position = OptionPosition(contract=contract, quantity=1, entry_premium=PREMIUM,
                          entry_underlying_price=S0, observed_entry_iv=IV,
                          observed_entry_delta=DELTA)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


print("## 1. Model-free expiration profile (exact)\n")
prof = expiration_profile(position)
print(md_table(["Quantity", "Value"], [
    ["Premium paid", f"${prof.premium_paid:,.2f}"],
    ["Maximum loss", f"${prof.max_loss:,.2f} ({prof.max_loss_condition})"],
    ["Breakeven at expiry", f"${prof.breakeven:.2f}"],
    ["Maximum gain", f"${prof.max_gain:,.2f} ({prof.max_gain_condition})"],
]))
pts = [210, 200, 195, 192.62, 190, 185, 180, 170]
print("\n" + md_table(["NVDA at expiry", "P&L"],
                      [[f"${s:g}", f"${float(expiration_pnl(position, s)):+,.2f}"] for s in pts]))

print("\n## 2. What time to expiry are the recorded observations consistent with? (model inference)\n")
rows = []
for S in (201.0, 201.5, 202.0):
    t_delta = brentq(lambda T, S=S: bsm_delta("put", S, K, T, R, IV, Q) - DELTA, 1e-5, 1.0)
    t_prem = brentq(lambda T, S=S: bsm_put(S, K, T, R, IV, Q) - PREMIUM, 1e-5, 1.0)
    rows.append([f"${S:.1f}", f"{t_delta * 365:.2f}", f"{t_prem * 365:.2f}"])
print(md_table(["Spot", "Days implied by delta -0.29 @ 46% IV", "Days implied by $2.38 @ 46% IV"],
               rows))


print("\nAlong the curve of (sigma, T) pairs that reproduce the $2.38 premium at S = 201.5, delta "
      "barely moves:\n")
rows = []
for sig in (0.30, 0.40, 0.46, 0.50, 0.60, 0.80):
    t = brentq(lambda T, sig=sig: bsm_put(S0, K, T, R, sig, Q) - PREMIUM, 1e-6, 2.0)
    rows.append([f"{sig:.0%}", f"{t * 365:.2f}", f"{bsm_delta('put', S0, K, t, R, sig, Q):+.4f}"])
print(md_table(["sigma", "Days to expiry reproducing $2.38", "Delta"], rows))
print("\nFor an out-of-the-money option both price and delta depend mainly on total volatility "
      "sigma*sqrt(T), so price and delta together cannot separate sigma from T. The ~7-day "
      "inference rests on the premium *and* the recorded 46% IV; delta -0.29 is consistent "
      "with it (model: -0.288) but adds almost no independent information. Only the recorded "
      "entry timestamp can settle it.")
expiry_dt = expiry_datetime(contract.expiry)
early = expiry_dt - timedelta(days=7.43)  # S = 202, from the table above
late = expiry_dt - timedelta(days=6.49)  # S = 201
print(f"\nAcross S = 201–202 the premium-implied horizon spans {early:%a %d %b %H:%M} to "
      f"{late:%a %d %b %H:%M} ET (calendar time), which points to the **Friday 30 October 2026 "
      "session** as the most plausible entry. This is an inference, not a recorded timestamp: "
      "calendar-day versus trading-day time conventions alone move it by about a day, because "
      "a weekend sits inside the window.")

# Hypothetical entry time used for the remaining sections.
t0 = expiry_datetime(contract.expiry) - timedelta(days=7)
A = ModelAssumptions(volatility=IV, rate=R, dividend_yield=Q)
print(f"\n**Assumption for sections 3–5:** entry at {t0:%a %d %b %Y %H:%M} ET (7.0 days before "
      f"expiry), {A.describe()}.")

print("\n## 3. Greeks at (hypothetical) entry\n")
g = bsm_greeks("put", S0, K, 7 / 365, R, IV, Q)
rep = exposure_report(position, S0, t0, A)
print(md_table(["Measure", "Per share", "Position (1 contract)"], [
    ["Model value", f"{g.price:.3f}", f"${rep.position_value:,.2f}"],
    ["Delta", f"{g.delta:.3f}", f"{rep.share_equivalent:.1f} share-equivalents"],
    ["Delta-adjusted notional", "", f"${rep.delta_notional:,.0f}"],
    ["Gamma", f"{g.gamma:.4f}", f"delta changes by {g.gamma * 100:.2f} shares per $1"],
    ["Theta (per day)", f"{g.theta_per_day:.3f}", f"${rep.theta_per_day:,.2f}/day"],
    ["Vega (per vol point)", f"{g.vega_per_vol_point:.3f}", f"${rep.vega_per_vol_point:,.2f}"],
    ["Elasticity (lambda)", f"{rep.elasticity:.1f}", "% option move per +1% NVDA"],
]))
for w in rep.warnings:
    print(f"\n> Warning: {w}")

print("\n## 4. Hypothetical decline scenarios: gamma at work\n")
rows = []
for move in (0.0, -0.02, -0.04, -0.06, -0.08, -0.10):
    S = S0 * (1 + move)
    for days_left in (7, 4, 1):
        v = bsm_put(S, K, days_left / 365, R, IV, Q)
        d = bsm_delta("put", S, K, days_left / 365, R, IV, Q)
        rows.append([f"{move:+.0%}", f"${S:.2f}", str(days_left), f"{v:.2f}", f"{d:+.2f}",
                     f"${(v - PREMIUM) * 100:+,.0f}"])
print(md_table(["NVDA move", "NVDA", "Days left", "Model put value", "Delta", "P&L vs $238"], rows))
print("\nThese are model values at 46% constant IV. In a sharp equity decline implied volatility "
      "usually rises, so market prices would likely have been *higher* than shown; a calm drift "
      "lower with falling IV would be lower.")

print("\n## 5. Leverage: option versus shares (instantaneous moves, model-based)\n")
cmp = compare_with_shares(position, S0, t0, A, spot_moves=[-0.10, -0.05, -0.02, 0.0, 0.02, 0.05])
rows = [[f"{r.spot_move:+.0%}", f"${r.option_pnl:+,.0f}", f"{r.option_return_on_capital:+.0%}",
         f"${r.delta_equiv_shares_pnl:+,.0f}", f"${r.capital_equiv_shares_pnl:+,.2f}"]
        for r in cmp.itertuples()]
print(md_table(["NVDA move", "Put P&L", "Put return", f"Short {abs(cmp.delta_equiv_shares[0]):.0f} "
                "shares P&L", f"Short ${cmp.option_capital[0]:.0f} of shares P&L"], rows))
print(f"\nThe delta-equivalent short would tie up ≈ ${cmp.delta_equiv_capital[0]:,.0f} of exposure "
      f"(and need margin); the put controls it for ${cmp.option_capital[0]:,.0f}, which is also the "
      "most it can lose.")

grid = scenario_grid(position, [S0], [0, 3, 5, 7], t0, A)
print("\nPure time decay at unchanged spot: " + ", ".join(
    f"day {int(r.days_forward)}: ${r.position_value:,.0f}" for r in grid.itertuples()))
_ = (datetime, bsm_gamma)
