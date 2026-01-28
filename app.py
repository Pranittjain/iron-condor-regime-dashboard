import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Iron Condor – Live Regime Dashboard", layout="wide")
st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Right-now market regime metrics for iron condor planning. No trade execution. Informational only.")

# -----------------------------
# Fetch data (stable sources only)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_daily(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    return df

nifty = fetch_daily("^NSEI", "3mo")
vix   = fetch_daily("^INDIAVIX", "3mo")

if nifty.empty or vix.empty:
    st.error("Data not available right now.")
    st.stop()

# -----------------------------
# Core calculations
# -----------------------------
nifty["ret"] = nifty["Close"].pct_change()

# Realized volatility
rv_5d  = nifty["ret"].tail(5).std() * np.sqrt(252)
rv_20d = nifty["ret"].tail(20).std() * np.sqrt(252)

# Trend & range metrics
nifty["MA20"] = nifty["Close"].rolling(20).mean()
nifty["TR"] = np.maximum(
    nifty["High"] - nifty["Low"],
    np.maximum(
        abs(nifty["High"] - nifty["Close"].shift(1)),
        abs(nifty["Low"] - nifty["Close"].shift(1))
    )
)
nifty["ATR14"] = nifty["TR"].rolling(14).mean()
atr_pct = (nifty["ATR14"].iloc[-1] / nifty["Close"].iloc[-1]) * 100

# ADX (simplified but correct)
def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = df["TR"]
    atr = tr.rolling(n).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(n).sum() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean()

nifty["ADX14"] = adx(nifty)
adx_val = nifty["ADX14"].iloc[-1]

# VIX (implied volatility proxy)
vix_now = vix["Close"].iloc[-1]
iv_proxy = vix_now / 100

iv_minus_rv5 = iv_proxy - rv_5d

spot = nifty["Close"].iloc[-1]
ma20 = nifty["MA20"].iloc[-1]

# -----------------------------
# Regime labels
# -----------------------------
if adx_val < 18:
    trend_label = "Weak / Range-bound"
elif adx_val < 25:
    trend_label = "Moderate"
else:
    trend_label = "Strong Trend"

if rv_5d < 0.10:
    vol_label = "Low realized volatility"
elif rv_5d < 0.16:
    vol_label = "Medium realized volatility"
else:
    vol_label = "High realized volatility"

# Delta guidance (NOT advice)
if adx_val < 18 and iv_minus_rv5 > 0:
    delta_band = "≈ 10–15 delta (typical for calm, range-bound regimes)"
elif adx_val < 25:
    delta_band = "≈ 15–20 delta (mixed regime)"
else:
    delta_band = "≈ 20–30 delta (trend risk elevated)"

# -----------------------------
# Dashboard
# -----------------------------
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

st.caption(f"Last update: **{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"{spot:,.2f}")
c2.metric("India VIX (IV)", f"{vix_now:.2f}%")
c3.metric("RV (last 5 days)", f"{rv_5d*100:.2f}%")
c4.metric("RV (last 20 days)", f"{rv_20d*100:.2f}%")
c5.metric("IV − RV(5d)", f"{iv_minus_rv5*100:.2f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("ADX14", f"{adx_val:.1f}")
c7.metric("Trend regime", trend_label)
c8.metric("ATR% (14)", f"{atr_pct:.2f}%")
c9.metric("Spot vs MA20", "Above MA20" if spot > ma20 else "Below MA20")

st.divider()

st.subheader("📌 Regime Summary (informational)")
st.markdown(f"""
- **Trend:** {trend_label}  
- **Realized Volatility (5d):** {rv_5d*100:.2f}%  
- **Implied Volatility (VIX):** {vix_now:.2f}%  
- **IV − RV:** {iv_minus_rv5*100:.2f}%  
- **Typical iron-condor short-strike delta band:** **{delta_band}**
""")

with st.expander("What this means (quick explanation)"):
    st.markdown("""
- **RV (5 days)** shows how much NIFTY actually moved recently  
- **VIX** shows what options are pricing for the *next ~30 days*  
- **IV > RV** → option premiums richer than realized movement  
- **ADX** tells you if the market is trending or ranging  
- **Delta band** is a *historical regime-based range*, not a trade signal
""")

st.caption("This dashboard is for study and regime awareness only. Final strategy selection is your responsibility.")
