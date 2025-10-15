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
    ret

