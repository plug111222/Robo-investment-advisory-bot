# app.py
# Robo Advisor - Streamlit App (stable, ASCII-safe, with error surfacing)

import traceback
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Always show errors instead of blank screen
st.set_option("client.showErrorDetails", True)

try:
    st.set_page_config(page_title="Robo Advisor", layout="wide")

    # ---------------------------
    # Market assumptions
    # ---------------------------
    @dataclass
    class MarketAssumptions:
        eq_mu: float = 0.08     # Equity nominal annual return
        eq_sigma: float = 0.18
        bd_mu: float = 0.035    # Bonds nominal annual return
        bd_sigma: float = 0.06
        cash_mu: float = 0.02
        cash_sigma: float = 0.0_
