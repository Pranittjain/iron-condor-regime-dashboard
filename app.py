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
st.caption("Yahoo Finance via yfinance. This build adds strong debugging + safer parsing so you can see what's happening.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

default_start = date(2015, 1, 1)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=date.today())

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="Use 1d for regime metrics. 1wk/1mo for higher-level view."
)

debug_mode = st.sidebar.toggle("Debug mode (show data diagnostics)", value=True)

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

# -----------------------------
# Data fetch
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
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    # Standardize columns safely
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = c[0]
        cols.append(str(c).strip())
    df.columns = cols

    # Normalize Date column name
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "date" in [c.lower() for c in df.columns] and "Date" not in df.columns:
        # just in case weird casing
        for c in df.columns:
            if c.lower() == "date":
                df = df.rename(columns={c: "Date"})
                break

    # Parse Date properly
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")

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

    ann_factor = 252  # ok for 1d; we'll refine later for 1wk/1mo
    ann_vol = rets.std() * np.sqrt(ann_factor)
    ann_ret = (1 + rets.mean()) ** ann_factor - 1

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
# Fetch with visible errors
# -----------------------------
try:
    with st.spinner("Fetching NIFTY + VIX data..."):
        nifty = fetch_yf(NIFTY_TICKER, str(start_date), str(end_date), interval)
        vix = fetch_yf(VIX_TICKER, str(start_date), str(end_date), interval)
except Exception as e:
    st.error("Data fetch crashed. Here’s the exact error:")
    st.exception(e)
    st.stop()

# -----------------------------
# Debug diagnostics (always shows something)
# -----------------------------
if debug_mode:
    st.subheader("🧪 Debug diagnostics")
    st.write("NIFTY ticker:", NIFTY_TICKER, "| rows:", len(nifty), "| cols:", list(nifty.columns))
    st.write("VIX ticker:", VIX_TICKER, "| rows:", len(vix), "| cols:", list(vix.columns) if not vix.empty else [])
    if not nifty.empty:
        st.write("NIFTY tail:")
        st.dataframe(nifty.tail(5), use_container_width=True)
    if not vix.empty:
        st.write("VIX tail:")
        st.dataframe(vix.tail(5), use_container_width=True)

# -----------------------------
# Validations
# -----------------------------
if nifty.empty:
    st.error("Could not fetch NIFTY data (empty dataframe). Try changing the dates or interval.")
    st.stop()

if "Close" not in nifty.columns:
    st.error("NIFTY data fetched but 'Close' column is missing. Columns returned:")
    st.write(list(nifty.columns))
    st.stop()

if vix.empty or "Close" not in vix.columns:
    st.warning("India VIX not available from Yahoo right now (common issue). App will still run with NIFTY only.")

# -----------------------------
# Display charts
# -----------------------------
st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("NIFTY 50 (Spot Proxy)")
    st.plotly_chart(plot_line(nifty, "Date", "Close", "NIFTY Close"), use_container_width=True)

with col2:
    st.subheader("India VIX")
    if not vix.empty and "Close" in vix.columns:
        st.plotly_chart(plot_line(vix, "Date", "Close", "India VIX"), use_container_width=True)
    else:
        st.info("No VIX chart (data not returned).")

# -----------------------------
# Quick metrics
# -----------------------------
st.divider()
st.subheader("Quick Metrics")

stats_nifty = basic_return_stats(nifty["Close"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("NIFTY Last", f"{stats_nifty['Last']:.2f}")
m2.metric("Ann. Vol (approx)", f"{stats_nifty['Ann. Vol'] * 100:.2f}%")
m3.metric("Ann. Return (approx)", f"{stats_nifty['Ann. Return'] * 100:.2f}%")
m4.metric("Max Drawdown", f"{stats_nifty['Max DD'] * 100:.2f}%")

st.caption("Next: we’ll add regime metrics for iron condors (RV windows, ATR%, trend vs range, VIX percentile, IV-RV spread proxy, and a Condor Score).")
