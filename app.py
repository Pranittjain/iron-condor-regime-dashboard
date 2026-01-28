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
st.caption("Data source: Yahoo Finance via yfinance. This is Step 2: reliable data extraction + basic metrics.")

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
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Standardize column names
    df.columns = [c.strip().title() for c in df.columns]
    # Some intervals return "Datetime" instead of "Date"
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df

def plot_line(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_col,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def basic_return_stats(close: pd.Series) -> dict:
    rets = close.pct_change().dropna()
    if rets.empty:
        return {"Last": np.nan, "Ann. Vol": np.nan, "Ann. Return": np.nan, "Max DD": np.nan}

    ann_factor = 252  # daily assumption; for weekly/monthly this is approximate, but ok for Step 2
    ann_vol = rets.std() * np.sqrt(ann_factor)
    ann_ret = (1 + rets.mean()) ** ann_factor - 1

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = dd.min()

    return {
        "Last": float(close.iloc[-1]),
        "Ann. Vol": float(ann_vol),
        "Ann. Return": float(ann_ret),
        "Max DD": float(max_dd),
    }

# -----------------------------
# Fetch data
# -----------------------------
with st.spinner("Fetching data..."):
    nifty = fetch_yf(NIFTY_TICKER, str(start_date), str(end_date), interval)
    vix = fetch_yf(VIX_TICKER, str(start_date), str(end_date), interval)

# -----------------------------
# Validations
# -----------------------------
if nifty.empty:
    st.error("Could not fetch NIFTY data. Try changing dates or interval.")
    st.stop()

if vix.empty:
    st.warning("Could not fetch India VIX data (sometimes Yahoo is flaky). App will still work with NIFTY only.")

# -----------------------------
# Display
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("NIFTY 50 (Spot Proxy)")
    st.write(f"Ticker: `{NIFTY_TICKER}`")
    st.plotly_chart(plot_line(nifty, "Date", "Close", "NIFTY Close"), use_container_width=True)

with col2:
    st.subheader("India VIX")
    st.write(f"Ticker: `{VIX_TICKER}`")
    if not vix.empty:
        st.plotly_chart(plot_line(vix, "Date", "Close", "India VIX"), use_container_width=True)
    else:
        st.info("No VIX chart because VIX data was not returned.")

# -----------------------------
# Quick metrics
# -----------------------------
st.divider()
st.subheader("Quick Metrics (Step 2)")

m1, m2, m3, m4 = st.columns(4)
stats_nifty = basic_return_stats(nifty["Close"])

m1.metric("NIFTY Last", f"{stats_nifty['Last']:.2f}")
m2.metric("Ann. Vol (approx)", f"{stats_nifty['Ann. Vol']*100:.2f}%")
m3.metric("Ann. Return (approx)", f"{stats_nifty['Ann. Return']*100:.2f}%")
m4.metric("Max Drawdown", f"{stats_nifty['Max DD']*100:.2f}%")

# -----------------------------
# Raw data (optional)
# -----------------------------
with st.expander("Show raw data tables"):
    st.write("NIFTY data")
    st.dataframe(nifty, use_container_width=True)
    st.write("VIX data")
    st.dataframe(vix, use_container_width=True)

st.caption("Next: Step 3 = compute regime metrics (trend strength, realized vol windows, IV-RV spread, range & mean-reversion signals) + a Condor Regime Score.")
