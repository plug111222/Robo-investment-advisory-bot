# Robo Advisor – Streamlit App (ASCII-safe version)
# -----------------------------------------------------------
# Features:
# - Collects user demographics, finances, goals, and risk tolerance
# - Produces a recommended asset allocation & ETF mix
# - Projects accumulation & retirement spending with Monte Carlo simulation
# - Displays percentile wealth paths and probability of success
# -----------------------------------------------------------

import math
import io
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Robo Advisor", layout="wide")

# ---------------------------
# Helper dataclasses & constants
# ---------------------------
@dataclass
class MarketAssumptions:
    eq_mu: float = 0.08   # Equity nominal annual return
    eq_sigma: float = 0.18
    bd_mu: float = 0.035  # Bonds nominal annual return
    bd_sigma: float = 0.06
    cash_mu: float = 0.02
    cash_sigma: float = 0.005
    rho_eq_bd: float = -0.10
    rho_eq_cash: float = 0.00
    rho_bd_cash: float = 0.10
    inflation: float = 0.025

ASSUMP = MarketAssumptions()

ETF_MAP = {
    "US Equity": {"ticker": "VTI", "expense": 0.0003},
    "Intl Equity": {"ticker": "VXUS", "expense": 0.0007},
    "US Bonds": {"ticker": "BND", "expense": 0.0003},
    "Cash": {"ticker": "BIL", "expense": 0.0001},
}

# ---------------------------
# Utility functions
# ---------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def risk_score_to_equity_boost(score: int) -> int:
    # Map 0-20 risk score to an equity tilt in percentage points
    return int(round((score - 10) * (10 / 10)))

def base_equity_share(age: int) -> int:
    # Classic glide path: 110 - age (in %)
    return clamp(110 - age, 30, 95)

def recommend_allocation(age: int, risk_score: int):
    base = base_equity_share(age)
    boost = risk_score_to_equity_boost(risk_score)
    equity = clamp(base + boost, 20, 95)

    cash = 5
    bonds = clamp(100 - equity - cash, 0, 75)
    us_eq = round(equity * 0.70)
    intl_eq = equity - us_eq

    total = us_eq + intl_eq + bonds + cash
    if total != 100:
        bonds += 100 - total

    return {
        "US Equity": us_eq,
        "Intl Equity": intl_eq,
        "US Bonds": bonds,
        "Cash": cash,
    }

def simulate_portfolio(
    current_age: int,
    retire_age: int,
    horizon_age: int,
    current_assets: float,
    monthly_contrib: float,
    annual_raise: float,
    allocation: dict,
    assump: MarketAssumptions,
    initial_salary: float = 0.0,
    goal_spend_annual: float = 0.0,
    n_paths: int = 2000,
):
    np.random.seed(42)
    years_acc = max(0, retire_age - current_age)
    years_dec = max(0, horizon_age - retire_age)
    total_years = years_acc + years_dec
    steps = total_years * 12

    w_eq = (allocation["US Equity"] + allocation["Intl Equity"]) / 100.0
    w_bd = allocation["US Bonds"] / 100.0
    w_cs = allocation["Cash"] / 100.0

    mu = np.array([assump.eq_mu/12, assump.bd_mu/12, assump.cash_mu/12])
    sig = np.array([assump.eq_sigma/np.sqrt(12), assump.bd_sigma/np.sqrt(12), assump.cash_sigma/np.sqrt(12)])

    corr = np.array([
        [1.0, assump.rho_eq_bd, assump.rho_eq_cash],
        [assump.rho_eq_bd, 1.0, assump.rho_bd_cash],
        [assump.rho_eq_cash, assump.rho_bd_cash, 1.0],
    ])
    cov = np.outer(sig, sig) * corr
    L = np.linalg.cholesky(cov)

    wealth = np.zeros((n_paths, steps + 1))
    wealth[:, 0] = current_assets
    m_contrib = monthly_contrib
    contrib_track = np.zeros(steps)
    spend_track = np.zeros(steps)

    for t in range(steps):
        if t % 12 == 0 and t > 0:
            m_contrib *= (1 + annual_raise)
            goal_spend_annual *= (1 + assump.inflation)

        is_acc = (t < years_acc * 12)
        Z = np.random.normal(size=(n_paths, 3))
        shocks = Z @ L.T
        monthly_rets = mu + shocks
        port_ret = w_eq * monthly_rets[:,0] + w_bd * monthly_rets[:,1] + w_cs * monthly_rets[:,2]

        cashflow = m_contrib if is_acc else -(goal_spend_annual/12)
        contrib_track[t] = m_contrib if is_acc else 0.0
        spend_track[t] = 0.0 if is_acc else (goal_spend_annual/12)

        wealth[:, t] = np.maximum(wealth[:, t], 0) + cashflow
        wealth[:, t] = np.maximum(wealth[:, t], 0)
        wealth[:, t+1] = wealth[:, t] * (1 + port_ret)

    end_of_acc = years_acc * 12
    success = np.all(wealth[:, end_of_acc:] > 0, axis=1) if years_dec > 0 else (wealth[:,-1] > 0)
    p_success = success.mean()

    median = np.percentile(wealth, 50, axis=0)
    p10 = np.percentile(wealth, 10, axis=0)
    p90 = np.percentile(wealth, 90, axis=0)
    months = np.arange(steps + 1)
    ages = current_age + months/12

    df = pd.DataFrame({
        "Month": months,
        "Age": ages,
        "Wealth_p10": p10,
        "Wealth_p50": median,
        "Wealth_p90": p90,
    })

    return df, p_success, contrib_track, spend_track

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Robo Advisor - Personalized Plan")

with st.sidebar:
    st.header("Your Profile")
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Current age", min_value=18, max_value=100, value=30)
    with colB:
        retire_age = st.number_input("Target retirement age", min_value=age, max_value=100, value=65)

    horizon_age = st.number_input("Plan to model until age", min_value=retire_age, max_value=110, value=95)

    st.subheader("Financials")
    current_assets = st.number_input("Current investment assets ($)", min_value=0.0, step=1000.0, value=50000.0, format="%0.2f")
    annual_income = st.number_input("Annual gross income ($)", min_value=0.0, step=1000.0, value=100000.0, format="%0.2f")
    savings_rate = st.slider("Savings rate (% of income)", 0, 50, 15)
    monthly_contrib = (annual_income * (savings_rate / 100)) / 12
    annual_raise = st.slider("Annual raise assumption (%)", 0, 10, 2) / 100

    st.subheader("Goals")
    goal_type = st.selectbox("Primary goal", ["Retirement income", "College fund", "Wealth accumulation only"])
    if goal_type == "Retirement income":
        goal_spend = st.number_input("Desired retirement spending ($/yr)", min_value=0.0, step=1000.0, value=50000.0, format="%0.2f")
    else:
        goal_spend = 0.0

    st.subheader("Risk Tolerance (5 questions)")
    q1 = st.slider("Market drop of 20%: you... (0=sell all, 4=buy more)", 0, 4, 2)
    q2 = st.slider("Sleep with volatility? (0=poorly, 4=great)", 0, 4, 2)
    q3 = st.slider("Preference: (0=stable, 4=high growth)", 0, 4, 2)
    q4 = st.slider("Time horizon comfort (0=short, 4=very long)", 0, 4, 2)
    q5 = st.slider("Loss tolerance over 1y (0=<5%, 4=>30%)", 0, 4, 2)
    risk_score = q1 + q2 + q3 + q4 + q5

    st.caption(f"Risk score: {risk_score} / 20")

# ---------------------------
# Core
# ---------------------------
allocation = recommend_allocation(age, risk_score)

col_left, col_right = st.columns([1,1])
with col_left:
    st.subheader("Recommended Asset Allocation")
    df_alloc = pd.DataFrame({"Asset Class": list(allocation.keys()), "Weight %": list(allocation.values())})
    st.dataframe(df_alloc, use_container_width=True)

with col_right:
    st.subheader("Allocation Chart")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(df_alloc["Weight %"], labels=df_alloc["Asset Class"], autopct="%1.0f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# Simulation
with st.expander("Run Projection (Monte Carlo)", expanded=True):
    n_paths = st.slider("Number of simulation paths", 200, 5000, 2000, step=200)
    df_mc, p_success, contrib_track, spend_track = simulate_portfolio(
        current_age=age,
        retire_age=retire_age,
        horizon_age=horizon_age,
        current_assets=current_assets,
        monthly_contrib=monthly_contrib,
        annual_raise=annual_raise,
        allocation=allocation,
        assump=ASSUMP,
        initial_salary=annual_income,
        goal_spend_annual=goal_spend,
        n_paths=n_paths,
    )

    st.subheader("Projected Wealth Percentiles")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df_mc["Age"], df_mc["Wealth_p10"], label="p10")
    ax2.plot(df_mc["Age"], df_mc["Wealth_p50"], label="median")
    ax2.plot(df_mc["Age"], df_mc["Wealth_p90"], label="p90")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Projected Wealth ($)")
    ax2.legend()
    st.pyplot(fig2)

    st.metric(label="Probability of Funding Success (not running out)", value=f"{p_success * 100:0.1f}%")

    csv = df_mc.to_csv(index=False).encode("utf-8")
    st.download_button("Download projection CSV", data=csv, file_name="projection_percentiles.csv", mime="text/csv")

# ---------------------------
# Methodology Section
# ---------------------------
st.markdown("""
### Methodology (for your write-up)
**Inputs gathered:** age, current assets, income, savings rate, retirement age/horizon, primary goal, and a 5-question risk score (0-20).

**Mapping to recommendation:**
- Start with a classic age-based glide path (110 - age) and adjust +/- 10 percentage points with the risk score to set equity exposure.
- The balance goes to bonds and a strategic 5% to cash.
- Split equity 70/30 between U.S. and international to diversify across regions. Provide a low-cost ETF lineup for implementability.

**Projections:**
- Two-phase Monte Carlo: accumulation (contributions grow with raises) then retirement (inflation-adjusted withdrawals).
- Asset returns use nominal assumptions you can edit in the sidebar; monthly vol and correlations add realism.
- Display percentile bands (p10/p50/p90) and success probability.

**Interpretation:**
- If success probability is low: raise savings, retire later, reduce spending, or accept higher risk if consistent with tolerance.
""")

st.info("This educational tool is for coursework only and not investment advice.")
