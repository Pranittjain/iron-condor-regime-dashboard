import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# --------------------------------
# Page setup
# --------------------------------
st.set_page_config(page_title="Iron Condor – Live Metrics", layout="wide")
st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Real-time regime metrics for iron-condor planning. No trade execution.")

# --------------------------------
# Data fetch (stable only)
# --------------------------------
@st.cache_data(ttl=300)
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    return df

nifty = fetch_data("^NSEI", "3mo")
vix   = fetch_data("^INDIAVIX", "3mo")

if nifty.empty or vix.empty:
    st.error("Market data not available.")
    st.stop()

# --------------------------------
# Core calculations
# --------------------------------
nifty["ret"] = nifty["Close"].pct_change()

# Realized volatility
rv_5d  = nifty["ret"].tail(5).std() * np.sqrt(252)
rv_20d = nifty["ret"].tail(20).std() * np.sqrt(252)

# Moving averages
nifty["MA20"] = nifty["Close"].rolling(20).mean()

# ATR %
nifty["TR"] = np.maximum(
    nifty["High"] - nifty["Low"],
    np.maximum(
        abs(nifty["High"] - nifty["Close"].shift(1)),
        abs(nifty["Low"] - nifty["Close"].shift(1))
    )
)
nifty["ATR14"] = nifty["TR"].rolling(14).mean()
atr_pct = (nifty["ATR14"].iloc[-1] / nifty["Close"].iloc[-1]) * 100

# Spot & IV
spot = nifty["Close"].iloc[-1]
ma20 = nifty["MA20"].iloc[-1]
vix_now = vix["Close"].iloc[-1]
iv_proxy = vix_now / 100

iv_minus_rv5 = iv_proxy - rv_5d

# --------------------------------
# Trend logic (simple & reliable)
# --------------------------------
ma_slope = nifty["MA20"].iloc[-1] - nifty["MA20"].iloc[-5]
dist_ma = abs(spot - ma20) / spot * 100

if dist_ma < 0.6 and abs(ma_slope) < 20:
    trend_label = "Range-bound"
elif dist_ma < 1.2:
    trend_label = "Mild trend"
else:
    trend_label = "Strong trend"

# --------------------------------
# Delta band logic (informational)
# --------------------------------
if trend_label == "Range-bound" and iv_minus_rv5 > 0:
    delta_band = "10–15 delta (tight condors preferred)"
elif trend_label == "Mild trend":
    delta_band = "15–20 delta (balanced width)"
else:
    delta_band = "20–30 delta (trend risk elevated)"

# --------------------------------
# Display
# --------------------------------
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.caption(f"Last update: **{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"{spot:,.2f}")
c2.metric("India VIX (IV)", f"{vix_now:.2f}%")
c3.metric("RV (5 days)", f"{rv_5d*100:.2f}%")
c4.metric("RV (20 days)", f"{rv_20d*100:.2f}%")
c5.metric("IV − RV(5d)", f"{iv_minus_rv5*100:.2f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("ATR % (14)", f"{atr_pct:.2f}%")
c7.metric("Trend regime", trend_label)
c8.metric("Spot vs MA20", "Above MA20" if spot > ma20 else "Below MA20")
c9.metric("Typical delta band", delta_band)

st.divider()

st.subheader("📌 Regime Summary")
st.markdown(f"""
- **Trend:** {trend_label}  
- **Realized Volatility (5d):** {rv_5d*100:.2f}%  
- **Implied Volatility (VIX):** {vix_now:.2f}%  
- **IV − RV Spread:** {iv_minus_rv5*100:.2f}%  
- **ATR (range proxy):** {atr_pct:.2f}%  
- **Typical iron-condor delta band:** **{delta_band}**
""")

with st.expander("What these metrics mean"):
    st.markdown("""
- **RV (5d)** = actual recent movement  
- **VIX** = market-priced forward volatility  
- **IV − RV** > 0 → options richer than realized moves  
- **ATR %** = how wide price is swinging  
- **Trend regime** = helps avoid selling tight condors in trends  
- **Delta bands** are historical regime-based ranges, not trade advice
""")

st.caption("Educational dashboard only. Final strategy selection remains discretionary.")
