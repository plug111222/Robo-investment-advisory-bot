# app.py
# Robo Advisor - Streamlit App (ASCII-safe with error surfacing)

import traceback
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Show full error details in the UI (helps avoid blank screen)
st.set_option("client.showErrorDetails", True)

try:
    st.set_page_config(page_title="Robo Advisor", layout="wide")

    # ---------------------------
    # Assumptions and constants
    # ---------------------------
    @dataclass
    class MarketAssumptions:
        eq_mu: float = 0.08     # Equity nominal annual return
        eq_sigma: float = 0.18
        bd_mu: float = 0.035    # Bonds nominal annual return
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
    # Utilities
    # ---------------------------
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def risk_score_to_equity_boost(score: int) -> int:
        # Map 0-20 risk score to an equity tilt in percentage points
        return int(round((score - 10) * (10 / 10)))

    def base_equity_share(age: int) -> int:
        # Classic glide path: 110 - age (in percent)
        return clamp(110 - age, 30, 95)

    def recommend_allocation(age: int, risk_score: int):
        base = base_equity_share(age)
        boost = risk_score_to_equity_boost(risk_score)
        equity = clamp(base + boost, 20, 95)

        cash = 5
        bonds = clamp(100 - equity - cash, 0, 75)

        us_eq = round(equity * 0.70)
        intl_eq = equity - us_eq

        # Adjust rounding drift
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
        # Seed for reproducibility in class demos
        np.random.seed(42)

        years_acc = max(0, retire_age - current_age)
        years_dec = max(0, horizon_age - retire_age)
        total_years = years_acc + years_dec
        steps = total_years * 12

        w_eq = (allocation["US Equity"] + allocation["Intl Equity"]) / 100.0
        w_bd = allocation["US Bonds"] / 100.0
        w_cs = allocation["Cash"] / 100.0

        # Monthly means and vols (approx)
        mu = np.array([assump.eq_mu/12, assump.bd_mu/12, assump.cash_mu/12])
        sig = np.array([
            assump.eq_sigma/np.sqrt(12),
            assump.bd_sigma/np.sqrt(12),
            assump.cash_sigma/np.sqrt(12)
        ])

        # Correlation and covariance
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

        for t in range(st
