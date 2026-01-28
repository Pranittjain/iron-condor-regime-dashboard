import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Iron Condor Study Dashboard (NIFTY)", layout="wide")
st.title("📊 Iron Condor Study Dashboard (NIFTY)")
st.caption("Simple, study-focused metrics for iron condors. No recommendations. Data: Yahoo Finance via yfinance.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

view_start = st.sidebar.date_input("View start date", value=date(2026, 1, 1))
view_end = st.sidebar.date_input("View end date", value=date.today())

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="Use 1d for daily regime study."
)

lookback_days = st.sidebar.slider(
    "Extra lookback for indicator calculation (days)",
    min_value=100,
    max_value=1200,
    value=500,
    step=50,
    help="We fetch earlier history so MA200/RV60/ADX work even if your view starts in 2026."
)

show_help = st.sidebar.toggle("Show metric explanations (click-to-read)", value=True)

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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]

    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")

    return df

# -----------------------------
# Indicators
# -----------------------------
def realized_vol(close: pd.Series, window: int, ann_factor: int = 252) -> pd.Series:
    r = close.pct_change()
    return r.rolling(window).std() * np.sqrt(ann_factor)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    tr_n = pd.Series(tr, index=high.index).rolling(window).sum()
    plus_dm_n = pd.Series(plus_dm, index=high.index).rolling(window).sum()
    minus_dm_n = pd.Series(minus_dm, index=high.index).rolling(window).sum()

    plus_di = 100 * (plus_dm_n / tr_n)
    minus_di = 100 * (minus_dm_n / tr_n)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window).mean()

def label_trend(adx_val: float) -> str:
    if np.isnan(adx_val):
        return "Trend: —"
    if adx_val >= 25:
        return "Trend: Strong"
    if adx_val >= 18:
        return "Trend: Mixed"
    return "Trend: Weak (Range-ish)"

def label_vol(rv20: float) -> str:
    if np.isnan(rv20):
        return "Vol: —"
    if rv20 >= 0.18:
        return "Vol: High"
    if rv20 >= 0.12:
        return "Vol: Medium"
    return "Vol: Low"

def label_premium(iv_minus_rv: float) -> str:
    if np.isnan(iv_minus_rv):
        return "IV-RV: —"
    if iv_minus_rv >= 0.06:
        return "IV-RV: High"
    if iv_minus_rv >= 0.02:
        return "IV-RV: Medium"
    return "IV-RV: Low"

# -----------------------------
# Fetch with lookback
# -----------------------------
fetch_start = view_start - timedelta(days=int(lookback_days))

with st.spinner("Fetching NIFTY + VIX..."):
    nifty = fetch_yf(NIFTY_TICKER, str(fetch_start), str(view_end), interval)
    vix = fetch_yf(VIX_TICKER, str(fetch_start), str(view_end), interval)

if nifty.empty:
    st.error("NIFTY data is empty. Try changing dates/interval.")
    st.stop()

needed = {"Date", "Open", "High", "Low", "Close"}
if not needed.issubset(set(nifty.columns)):
    st.error(f"NIFTY missing required columns. Got: {list(nifty.columns)}")
    st.stop()

# Merge VIX
if not vix.empty and "Close" in vix.columns:
    vix = vix[["Date", "Close"]].rename(columns={"Close": "Vix"})
    df = pd.merge(nifty, vix, on="Date", how="left")
else:
    df = nifty.copy()
    df["Vix"] = np.nan
    st.warning("India VIX not available from Yahoo right now (IV proxy will be blank).")

# Compute metrics
df["Ma20"] = df["Close"].rolling(20).mean()
df["Ma50"] = df["Close"].rolling(50).mean()
df["Ma200"] = df["Close"].rolling(200).mean()

df["Adx14"] = adx(df["High"], df["Low"], df["Close"], window=14)

df["Rv10"] = realized_vol(df["Close"], 10)
df["Rv20"] = realized_vol(df["Close"], 20)
df["Rv60"] = realized_vol(df["Close"], 60)

df["Atr14"] = atr(df["High"], df["Low"], df["Close"], window=14)
df["Atr_pct"] = df["Atr14"] / df["Close"]

df["Iv_proxy"] = df["Vix"] / 100.0
df["Iv_minus_rv20"] = df["Iv_proxy"] - df["Rv20"]

# View window filter
view_df = df[(df["Date"].dt.date >= view_start) & (df["Date"].dt.date <= view_end)].copy()
if view_df.empty:
    st.error("No rows in your view window. Widen the date range.")
    st.stop()

latest = view_df.iloc[-1]

# -----------------------------
# MAIN WINDOW: key metrics + regime label
# -----------------------------
st.divider()
st.subheader("Key Iron Condor Study Metrics (Latest)")

regime_line = " | ".join([
    label_trend(latest["Adx14"]),
    label_vol(latest["Rv20"]),
    label_premium(latest["Iv_minus_rv20"])
])

st.
