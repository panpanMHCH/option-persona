"""Option Persona — research dashboard (Phase 1, plus Phase 2 chain import).

Run:  uv run streamlit run app/streamlit_app.py

The UI contains no financial mathematics: every number comes from the
validated ``option_persona`` library. Model outputs, user-entered observations
and assumptions are labelled separately.
"""

from __future__ import annotations

from datetime import date, datetime, time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from option_persona.arbitrage import ExecutionCosts, generate_synthetic_chain, scan_chain_for_parity
from option_persona.core.errors import OptionPersonaError
from option_persona.core.models import OptionContract, OptionPosition
from option_persona.core.time import BANGKOK, DAYS_PER_YEAR, NEW_YORK, time_to_expiry
from option_persona.core.types import ExerciseStyle, OptionType
from option_persona.data import CANONICAL_COLUMNS, read_option_chain_csv, synthetic_canonical_chain
from option_persona.pricing import (
    bsm_greeks,
    bsm_price,
    european_price_bounds,
    implied_volatility,
    risk_neutral_prob_itm,
)
from option_persona.risk import (
    ModelAssumptions,
    ReviewStatus,
    StrikeCandidate,
    compare_strikes,
    compare_with_shares,
    exit_review,
    expiration_pnl,
    expiration_profile,
    exposure_report,
    model_caveats,
    scenario_grid,
    spot_vol_stress,
)

st.set_page_config(page_title="Option Persona", page_icon="◆", layout="wide")


def label(kind: str) -> str:
    return {"model": "🧮 Model output", "obs": "👁 Your observation / input",
            "assume": "⚙ Assumption", "synthetic": "🧪 Synthetic data"}[kind]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Option Persona")
st.sidebar.caption("Quantitative Options Intelligence · *Understand the price. Own the risk.*")
st.sidebar.subheader(label("assume"))
rate = st.sidebar.number_input("Risk-free rate r (continuous, %)", value=4.0, step=0.25) / 100
div_yield = st.sidebar.number_input("Dividend yield q (continuous, %)", value=0.0, step=0.1) / 100
st.sidebar.divider()
st.sidebar.caption(
    "Read-only research tool. Not investment advice. Model values are theoretical estimates "
    "under stated assumptions, not executable prices or real-world forecasts."
)

st.title("Option Persona")
tab_price, tab_iv, tab_strikes, tab_trade, tab_parity, tab_import = st.tabs(
    ["Pricer & Greeks", "Implied volatility", "Strike comparison", "Trade analyser", "Parity lab",
     "Chain import"])

# ------------------------------------------------------------------ pricer
with tab_price:
    st.subheader("European Black-Scholes-Merton pricer")
    c1, c2, c3, c4, c5 = st.columns(5)
    otype = c1.selectbox("Type", [OptionType.CALL, OptionType.PUT], format_func=str.capitalize)
    S = c2.number_input("Underlying S", value=100.0, min_value=0.01)
    K = c3.number_input("Strike K", value=100.0, min_value=0.01)
    days = c4.number_input("Calendar days to expiry", value=30.0, min_value=0.0)
    vol = c5.number_input("Volatility σ (%)", value=25.0, min_value=0.0) / 100
    T = days / DAYS_PER_YEAR
    try:
        g = bsm_greeks(otype, S, K, T, rate, vol, div_yield)
        st.caption(label("model") + f" · T = {T:.6f} years (ACT/365F)")
        m = st.columns(7)
        m[0].metric("Price", f"{g.price:.4f}")
        m[1].metric("Delta", f"{g.delta:.4f}")
        m[2].metric("Gamma", f"{g.gamma:.5f}")
        m[3].metric("Vega / vol pt", f"{g.vega_per_vol_point:.4f}")
        m[4].metric("Theta / day", f"{g.theta_per_day:.4f}")
        m[5].metric("Rho / 1%", f"{g.rho_per_rate_point:.4f}")
        m[6].metric("Q-prob ITM", f"{risk_neutral_prob_itm(otype, S, K, T, rate, vol, div_yield):.1%}",
                    help="Risk-neutral probability N(±d2). Not a real-world forecast.")
        spots = np.linspace(0.6 * K, 1.4 * K, 161)
        curves = []
        for d in sorted({days, max(days / 2, 0), 0.0}, reverse=True):
            curves.append(pd.DataFrame({"S": spots, "value": bsm_price(
                otype, spots, K, d / DAYS_PER_YEAR, rate, vol, div_yield),
                "horizon": f"{d:g} days left"}))
        chart = alt.Chart(pd.concat(curves)).mark_line().encode(
            x=alt.X("S", title="Underlying price"), y=alt.Y("value", title="Model value per share"),
            color=alt.Color("horizon", title=None))
        st.altair_chart(chart, width="stretch")
        st.caption("Conventions: Greeks per share; theta per calendar day; vega per 1 vol point; "
                   "rho per 1 percentage point of r. 0-day curve = intrinsic value.")
    except OptionPersonaError as e:
        st.error(str(e))

# ------------------------------------------------------------------ implied vol
with tab_iv:
    st.subheader("Implied volatility solver (Brent, bracketed)")
    st.caption(label("obs") + " — enter a price you observed. A mid-quote is indicative, not "
               "executable; solving bid and ask separately shows the IV range the market implies.")
    c1, c2, c3, c4 = st.columns(4)
    iv_type = c1.selectbox("Type ", [OptionType.CALL, OptionType.PUT], format_func=str.capitalize)
    iv_S = c2.number_input("Underlying S ", value=201.5, min_value=0.01)
    iv_K = c3.number_input("Strike K ", value=195.0, min_value=0.01)
    iv_days = c4.number_input("Calendar days to expiry ", value=7.0, min_value=0.0)
    c5, c6 = st.columns(2)
    bid = c5.number_input("Bid", value=2.30, min_value=0.0)
    ask = c6.number_input("Ask", value=2.46, min_value=0.0)
    iv_T = iv_days / DAYS_PER_YEAR
    rows = []
    for name, px in (("Bid", bid), ("Mid", 0.5 * (bid + ask)), ("Ask", ask)):
        try:
            iv = implied_volatility(px, iv_type, iv_S, iv_K, iv_T, rate, div_yield)
            rows.append({"Quote": name, "Price": px, "Implied vol": f"{iv:.2%}",
                         "Note": "indicative only" if name == "Mid" else "executable side"})
        except OptionPersonaError as e:
            rows.append({"Quote": name, "Price": px, "Implied vol": "—", "Note": str(e)})
    st.caption(label("model"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if iv_T > 0:
        b = european_price_bounds(iv_type, iv_S, iv_K, iv_T, rate, div_yield)
        st.caption(f"Model-free no-arbitrage bounds: {b.lower:.4f} < price < {b.upper:.4f}")
    st.info("BSM implied volatility assumes European exercise. For American puts (e.g. US single "
            "stocks), it slightly overstates volatility because the price includes an early-exercise "
            "premium the European model ignores.")

# ------------------------------------------------------------------ strike comparison
with tab_strikes:
    st.subheader("Strike comparison — before you trade")
    st.caption("Lays out the trade-off between strikes: cost, the move needed to break even, time "
               "decay and leverage. It does not rank strikes; which trade-off suits you is your call.")
    st.markdown(f"**{label('obs')}**")
    c = st.columns(5)
    sc_type = c[0].selectbox("Type   ", [OptionType.PUT, OptionType.CALL], format_func=str.capitalize)
    sc_spot = c[1].number_input("Underlying now ", value=201.5, min_value=0.01)
    sc_expiry = c[2].date_input("Expiry ", value=date(2026, 11, 6))
    sc_contracts = c[3].number_input("Contracts ", value=1, min_value=1, step=1)
    sc_fee = c[4].number_input("Fee per contract ($)", value=0.65, min_value=0.0)
    st.markdown("Strikes to compare. Fill in **bid** and **ask** from your broker where you have them "
                "(the row then enters at the ask, with the ask's own IV); leave them blank to use the "
                "model at the reference volatility.")
    sc_edit = st.data_editor(
        pd.DataFrame({"strike": [185.0, 190.0, 195.0, 200.0], "bid": [None] * 4, "ask": [None] * 4},
                     dtype="float64"),
        num_rows="dynamic", hide_index=True, key="strike_editor")
    st.markdown(f"**{label('assume')}**")
    c = st.columns(4)
    sc_date = c[0].date_input("Valuation date ", value=date(2026, 10, 30))
    sc_time = c[1].time_input("Valuation time ", value=time(10, 0))
    sc_tz = c[2].selectbox("Time zone ", ["America/New_York", "Asia/Bangkok"])
    sc_vol = c[3].number_input("Reference volatility σ (%)", value=46.0, min_value=0.1,
                               help="Used for rows without a quote, and for the expected-move scale.") / 100
    sc_now = datetime.combine(sc_date, sc_time, tzinfo=NEW_YORK if sc_tz == "America/New_York" else BANGKOK)
    try:
        cands = []
        for r_ in sc_edit.dropna(subset=["strike"]).itertuples():
            has = not (pd.isna(r_.bid) or pd.isna(r_.ask))
            cands.append(StrikeCandidate(float(r_.strike), float(r_.bid) if has else None,
                                         float(r_.ask) if has else None))
        table = compare_strikes(sc_type, sc_spot, sc_expiry, sc_now, cands,
                                ModelAssumptions(volatility=sc_vol, rate=rate, dividend_yield=div_yield),
                                contracts=int(sc_contracts), fee_per_contract=sc_fee)
        em = table.attrs["expected_move"]
        st.caption(f"{table.attrs['days_to_expiry']:.2f} days to expiry. One expected move "
                   f"(S·σ·√T at {sc_vol:.0%}) = ±{em:,.2f}, i.e. {sc_spot - em:,.2f} to {sc_spot + em:,.2f}: "
                   "a scale reference, not a forecast.")
        view = table[["strike", "price_basis", "entry_price", "cost", "breakeven", "move_to_breakeven",
                      "breakeven_in_expected_moves", "q_prob_itm", "q_prob_beyond_breakeven",
                      "theta_per_day", "theta_pct_of_value_per_day", "elasticity", "volatility",
                      "relative_spread"]].rename(columns={
            "price_basis": "price from", "entry_price": "entry / share", "cost": "cost = max loss ($)",
            "move_to_breakeven": "move to breakeven", "breakeven_in_expected_moves": "…in expected moves",
            "q_prob_itm": "Q-prob ITM", "q_prob_beyond_breakeven": "Q-prob past breakeven",
            "theta_per_day": "theta $/day", "theta_pct_of_value_per_day": "theta % of value/day",
            "elasticity": "λ", "volatility": "σ used", "relative_spread": "spread % of mid"})
        st.dataframe(view.style.format({
            "strike": "{:g}", "entry / share": "{:.2f}", "cost = max loss ($)": "{:,.0f}",
            "breakeven": "{:,.2f}", "move to breakeven": "{:+.1%}", "…in expected moves": "{:+.2f}",
            "Q-prob ITM": "{:.0%}", "Q-prob past breakeven": "{:.0%}", "theta $/day": "{:+,.1f}",
            "theta % of value/day": "{:+.1%}", "λ": "{:.1f}", "σ used": "{:.1%}",
            "spread % of mid": "{:.1%}"}, na_rep="—"), hide_index=True, width="stretch")
        for note in table.loc[table.note != "", "note"]:
            st.warning(note)
        st.markdown(
            "**How to read it**\n"
            "- **Cost** is also the most you can lose. **Move to breakeven** is how far the stock must "
            "go by expiry just to get your money back.\n"
            "- **…in expected moves** compares that to the market's typical move: −0.5 is well within "
            "it, −1.5 needs an unusually large move.\n"
            "- **Q-prob** columns are risk-neutral (the market's pricing), **not** real-world odds.\n"
            "- **theta % of value/day** is the share of the option you lose each day if nothing moves.\n"
            "- **λ** is how many % the option moves per 1% move in the stock. Further out of the money "
            "is cheaper in dollars but more fragile in percent.")
        st.caption(label("model") + " — " + table.attrs["assumptions"] + ". Model rows assume one "
                   "volatility for every strike; real chains usually charge more for lower-strike puts. "
                   "US single-stock options are American; BSM ignores early exercise.")
    except (OptionPersonaError, ValueError) as e:
        st.error(str(e))

# ------------------------------------------------------------------ trade analyser
with tab_trade:
    st.subheader("Personal trade analyser — long calls and puts")
    st.caption("Defaults: the NVDA $195 put case study. Entry timestamp was not recorded; the "
               "default valuation time (30 Oct 2026) is a model inference, i.e. hypothetical.")
    st.markdown(f"**{label('obs')}**")
    c = st.columns(6)
    ticker = c[0].text_input("Ticker", "NVDA")
    p_type = c[1].selectbox("Type  ", [OptionType.PUT, OptionType.CALL], format_func=str.capitalize)
    strike = c[2].number_input("Strike", value=195.0, min_value=0.01)
    expiry = c[3].date_input("Expiry", value=date(2026, 11, 6))
    qty = c[4].number_input("Contracts", value=1, min_value=1, step=1)
    mult = c[5].number_input("Multiplier", value=100.0, min_value=1.0)
    c = st.columns(6)
    premium = c[0].number_input("Premium paid / share", value=2.38, min_value=0.01)
    entry_S = c[1].number_input("Underlying at entry", value=201.5, min_value=0.01)
    fees = c[2].number_input("Total fees ($)", value=0.0, min_value=0.0)
    obs_iv = c[3].number_input("Observed IV at entry (%)", value=46.0, min_value=0.1) / 100
    style = c[4].selectbox("Exercise style", [ExerciseStyle.AMERICAN, ExerciseStyle.EUROPEAN],
                           format_func=str.capitalize)
    equity = c[5].number_input("Portfolio equity ($, optional)", value=0.0, min_value=0.0)

    st.markdown(f"**{label('assume')}**")
    c = st.columns(5)
    val_date = c[0].date_input("Valuation date", value=date(2026, 10, 30))
    val_time = c[1].time_input("Valuation time", value=time(15, 0))
    tz_name = c[2].selectbox("Time zone", ["America/New_York", "Asia/Bangkok"])
    spot_now = c[3].number_input("Underlying now", value=201.5, min_value=0.01)
    model_vol = c[4].number_input("Model volatility σ (%)", value=obs_iv * 100, min_value=0.1) / 100
    tz = NEW_YORK if tz_name == "America/New_York" else BANGKOK
    valuation = datetime.combine(val_date, val_time, tzinfo=tz)

    try:
        contract = OptionContract(underlying=ticker, option_type=p_type, strike=strike,
                                  expiry=expiry, exercise_style=style, multiplier=mult)
        pos = OptionPosition(contract=contract, quantity=int(qty), entry_premium=premium,
                             entry_underlying_price=entry_S, entry_fees=fees,
                             observed_entry_iv=obs_iv)
        A = ModelAssumptions(volatility=model_vol, rate=rate, dividend_yield=div_yield)
        T_now = time_to_expiry(valuation, expiry)
        st.caption(f"Valuation {valuation:%d %b %Y %H:%M %Z} = "
                   f"{valuation.astimezone(NEW_YORK):%d %b %H:%M} ET; "
                   f"{T_now * DAYS_PER_YEAR:.2f} calendar days to the 16:00 ET expiry.")

        st.markdown("#### Expiration outcome (model-free, exact)")
        prof = expiration_profile(pos)
        m = st.columns(4)
        m[0].metric("Premium paid", f"${prof.premium_paid:,.2f}")
        m[1].metric("Maximum loss", f"${prof.max_loss:,.2f}", help=prof.max_loss_condition)
        m[2].metric("Breakeven at expiry", f"${prof.breakeven:,.2f}")
        m[3].metric("Maximum gain", "Unbounded" if np.isinf(prof.max_gain) else f"${prof.max_gain:,.0f}",
                    help=prof.max_gain_condition)

        lo, hi = 0.75 * min(strike, spot_now), 1.25 * max(strike, spot_now)
        spots = np.linspace(lo, hi, 201)
        days_left = T_now * DAYS_PER_YEAR
        horizons = sorted({0.0, round(days_left / 2, 2)})
        grid = scenario_grid(pos, spots, horizons, valuation, A)
        grid["curve"] = grid["days_forward"].map(lambda d: f"Model, +{d:g} days")
        exp_df = pd.DataFrame({"spot": spots, "pnl": expiration_pnl(pos, spots), "curve": "At expiry"})
        plot = pd.concat([grid[["spot", "pnl", "curve"]], exp_df])
        chart = alt.Chart(plot).mark_line().encode(
            x=alt.X("spot", title=f"{ticker.upper()} price"),
            y=alt.Y("pnl", title="P&L after fees ($)"),
            color=alt.Color("curve", title=None),
            strokeDash=alt.condition(alt.datum.curve == "At expiry", alt.value([1, 0]),
                                     alt.value([5, 3])))
        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y")
        st.altair_chart(chart + rule, width="stretch")
        st.caption("Solid: exact expiry payoff. Dashed: " + label("model") + " — " + A.describe())

        st.markdown("#### Exposure and leverage " + "(" + label("model") + ")")
        rep = exposure_report(pos, spot_now, valuation, A,
                              portfolio_equity=equity if equity > 0 else None)
        m = st.columns(6)
        m[0].metric("Model value", f"${rep.position_value:,.2f}")
        m[1].metric("Share-equivalent", f"{rep.share_equivalent:,.1f}")
        m[2].metric("Delta notional", f"${rep.delta_notional:,.0f}")
        m[3].metric("Elasticity λ", f"{rep.elasticity:.1f}")
        m[4].metric("Theta / day", f"${rep.theta_per_day:,.2f}")
        m[5].metric("Vega / vol pt", f"${rep.vega_per_vol_point:,.2f}")
        if rep.pct_equity_at_risk is not None:
            st.caption(f"Premium at risk = {rep.pct_equity_at_risk:.2%} of equity; "
                       f"delta notional = {rep.pct_equity_delta_notional:.1%} of equity.")
        for w in rep.warnings:
            st.warning(w)

        left, right = st.columns(2)
        with left:
            st.markdown("**Stress test: P&L ($) at valuation time**")
            stress = spot_vol_stress(pos, spot_now, valuation, A,
                                     spot_moves=[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
                                     vol_shifts=[-0.15, 0, 0.15])
            st.dataframe(stress.style.format("{:+,.0f}"), width="stretch")
        with right:
            st.markdown("**Option vs shares (instantaneous moves)**")
            cmp = compare_with_shares(pos, spot_now, valuation, A,
                                      spot_moves=[-0.1, -0.05, 0, 0.05, 0.1])
            view = cmp[["spot_move", "option_pnl", "delta_equiv_shares_pnl",
                        "capital_equiv_shares_pnl"]].rename(columns={
                "spot_move": "Move", "option_pnl": "Option P&L",
                "delta_equiv_shares_pnl": f"{cmp.delta_equiv_shares[0]:+.0f} shares P&L",
                "capital_equiv_shares_pnl": "Same-capital shares P&L"})
            st.dataframe(view.style.format({"Move": "{:+.0%}"} | {
                col: "{:+,.0f}" for col in view.columns[1:]}), hide_index=True,
                width="stretch")
            st.caption("Negative share counts are hypothetical short positions, shown only as "
                       "benchmarks. Leverage magnifies gains and mistakes alike; it does not raise "
                       "expected return by itself.")
        with st.expander("Model caveats for this position"):
            for note in model_caveats(pos):
                st.markdown(f"- {note}")

        st.markdown("#### Hold-or-sell review")
        st.caption("A checklist of the factors that matter for closing before expiry. It never says "
                   "hold or sell: that decision is yours.")
        c = st.columns(3)
        cur_bid = c[0].number_input("Current option bid (optional)", value=None, min_value=0.0,
                                    format="%.2f")
        cur_ask = c[1].number_input("Current option ask (optional)", value=None, min_value=0.0,
                                    format="%.2f")
        thesis = c[2].radio("Does your reason for the trade still hold?", ["Not sure", "Yes", "No"],
                            horizontal=True)
        try:
            review = exit_review(pos, spot_now, valuation, A, bid=cur_bid, ask=cur_ask,
                                 thesis_intact={"Yes": True, "No": False}.get(thesis))
        except OptionPersonaError as e:
            st.info(str(e))
        else:
            icon = {ReviewStatus.OK: "🟢", ReviewStatus.WATCH: "🟡", ReviewStatus.ALERT: "🔴",
                    ReviewStatus.INFO: "ℹ️"}
            for item in review.items:
                st.markdown(f"{icon[item.status]} **{item.factor}** — {item.finding}  \n"
                            f"<span style='opacity:0.7'>{item.why}</span>", unsafe_allow_html=True)
            st.caption(review.disclaimer)
    except (OptionPersonaError, ValueError) as e:
        st.error(str(e))

# ------------------------------------------------------------------ parity lab
with tab_parity:
    st.subheader("Put-call parity lab")
    st.warning(label("synthetic") + ": every quote on this tab is generated from a BSM model "
               "with deliberately injected mispricings. None of it is market data.")
    c = st.columns(5)
    p_spot = c[0].number_input("Spot", value=100.0, min_value=1.0)
    p_days = c[1].number_input("Days to expiry", value=30.0, min_value=1.0)
    p_vol = c[2].number_input("ATM vol (%)", value=20.0, min_value=1.0) / 100
    p_fee = c[3].number_input("Fee per option contract ($)", value=0.65, min_value=0.0)
    p_fin = c[4].number_input("Borrow/lend spread (bp)", value=50.0, min_value=0.0) / 10_000
    st.markdown("**Injected call mispricings (per share)**")
    inj = st.data_editor(pd.DataFrame({"strike": [90.0, 105.0, 110.0], "call_shift": [0.60, -0.50, 0.03]}),
                         num_rows="dynamic", hide_index=True)
    strikes = np.arange(round(0.8 * p_spot / 5) * 5, round(1.2 * p_spot / 5) * 5 + 1, 5.0)
    violations = {float(r.strike): float(r.call_shift) for r in inj.dropna().itertuples()}
    try:
        chain = generate_synthetic_chain(spot=p_spot, strikes=strikes, time_to_expiry=p_days / 365,
                                         rate=rate, dividend_yield=div_yield, sigma_atm=p_vol,
                                         violations=violations)
        costs = ExecutionCosts(option_fee_per_contract=p_fee, borrow_rate_spread=p_fin,
                               lend_rate_spread=p_fin)
        scan = scan_chain_for_parity(chain, rate, div_yield, costs)
        table = chain[["strike", "call_bid", "call_ask", "put_bid", "put_ask"]].merge(
            scan.drop(columns=["risk_notes", "relative_mid_residual"]), on="strike")
        table = table[["strike", "classification", "best_strategy", "mid_residual",
                       "half_spread_cost", "conversion_edge", "reversal_edge",
                       "call_bid", "call_ask", "put_bid", "put_ask"]]
        st.dataframe(table.style.format(precision=3), hide_index=True, width="stretch")
        st.markdown(
            "- **theoretical_only**: mid-market residual exceeds tolerance, but no edge survives "
            "the quoted spreads, fees and financing.\n"
            "- **apparent_executable_edge**: positive at quoted bid/ask after modelled costs — "
            "still *not* risk-free (see risks below).")
        with st.expander("Execution and operational risks (tier 3)"):
            for note in max(scan["risk_notes"], key=len):
                st.markdown(f"- {note}")
    except (OptionPersonaError, ValueError) as e:
        st.error(str(e))

# ------------------------------------------------------------------ chain import
with tab_import:
    st.subheader("Option-chain CSV import")
    st.caption(
        "Validates a chain snapshot into quotes that carry their provenance. Rows that cannot be "
        "trusted are rejected with a reason; nothing missing is filled in. Canonical columns: "
        + ", ".join(f"`{c}`" for c in CANONICAL_COLUMNS) + "."
    )
    example = synthetic_canonical_chain(
        datetime(2026, 11, 5, 15, 0, tzinfo=NEW_YORK), date(2026, 12, 18),
        violations={90.0: 0.60, 105.0: -0.50})
    st.download_button(
        "Download a synthetic example CSV (" + label("synthetic") + ")",
        example.to_csv(index=False), file_name="synthetic_example_chain.csv", mime="text/csv")

    upload = st.file_uploader("Chain CSV", type=["csv"])
    c = st.columns(3)
    src = c[0].text_input("Source (required)", placeholder="e.g. IBKR TWS export, Cboe delayed quotes")
    delay_choice = c[1].radio("Quote timeliness", ["Delayed", "Real-time", "Per-row is_delayed column"],
                              help="Cannot be inferred from a file: you must declare it.")
    tz_choice = c[2].selectbox("Time zone for timestamps without an offset",
                               ["Reject them", "America/New_York", "Asia/Bangkok"])
    if upload is not None:
        if not src.strip():
            st.info("Enter where this file came from before it can be imported.")
        else:
            delayed = {"Delayed": True, "Real-time": False}.get(delay_choice)
            try:
                result = read_option_chain_csv(
                    upload, source=src, is_delayed=delayed,
                    assume_timezone=None if tz_choice == "Reject them" else tz_choice)
            except (OptionPersonaError, ValueError) as e:
                st.error(str(e))
            else:
                m = st.columns(4)
                m[0].metric("Rows read", result.rows_read)
                m[1].metric("Accepted", result.rows_accepted)
                m[2].metric("Rejected", result.rows_rejected)
                m[3].metric("Warnings", len(result.warnings))
                frame = result.to_frame()
                if not frame.empty and frame["is_synthetic"].any():
                    st.warning(label("synthetic") + ": some or all rows are synthetic, not market data.")
                if result.assumptions:
                    st.markdown("**" + label("assume") + "s made during import**")
                    for a in result.assumptions:
                        st.markdown(f"- {a}")
                if result.issues:
                    st.markdown("**Issues** (error = row rejected; warning = row kept, check it)")
                    st.dataframe(result.issues_frame(), hide_index=True, width="stretch")
                if not frame.empty:
                    st.markdown("**" + label("obs") + ": accepted quotes**")
                    st.caption("Mid is an indicative reference only; executable prices are bid and ask.")
                    st.dataframe(frame, hide_index=True, width="stretch")
