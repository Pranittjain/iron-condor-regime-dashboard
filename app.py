import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Iron Condor Regime Dashboard (NIFTY)", layout="wide")

st.title("📊 Iron Condor Regime Dashboard (NIFTY)")
st.caption("Data source: Yahoo Finance via yfinance. This version includes a MultiIndex column fix for Streamlit Cloud.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

default_start = date(2015, 1, 1)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=date.today())

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="Use 1d for trading metrics. 1wk/1mo for longer regime overview."
)

# Yahoo tickers
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_yf(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns (e.g., ('Close','^NSEI')) in some environments
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # keep OHLCV only

    df = df.reset_index()

    # Standardize column names safely
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = c[0]
        c = str(c).strip().title()
        new_cols.append(c)
    df.columns = new_cols

    # Some intervals return "Dateti

