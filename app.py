import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Iron Condor – Live Metrics", layout="wide")
st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Real-time regime metrics for iron-condor study. Informational only.")

# =========================================================
# DATA FETCH (STABLE ONLY)
# =========================================================
@st.cache_data(ttl=300)
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    return df.reset_index()

nifty = fetch_data("^NSEI", "3mo")
vix   = fetch_data("^INDIAVIX", "3mo")

if nifty.empty or vix.empty:
    st.error("Market data unavailable right now.")
    st.stop()

# =========================================================
# CORE CALCULATIONS
# =========================================================
nifty["ret"] = nifty["Close"].pct_change()

# ---- Realized Volatility ----
rv_5d  = float(nifty["ret"].tail(5).std() * np.sqrt(252))
rv_20d = float(nifty["ret"].tail(20).std() * np.sqrt(252))

# ---- Moving Average ----
nifty["MA20"] = nifty["Close"].rolling(20).mean()

# ---- ATR % (range proxy) ----
nifty["TR"] = np.maximum(
    nifty["High"] - nifty["Low"],
    np.maximum(
        abs(nifty["High"] - nifty["Close"].shift(1)),
        abs(nifty["Low"] - nifty["Close"].shift(1))
    )
)
nifty["ATR14"] = nifty["TR"].rolling(14).mean()

# =========================================================
# FORCE SAFE SCALARS (NO PANDAS AMBIGUITY)
# =========================================================
spot = float(nifty["Close"].iloc[-1])
ma20_series = nifty["MA20"].dropna()

if len(ma20_series) < 6:
    st.error("Not enough data to compute trend metrics yet.")
    st.stop()

ma20 = float(ma20_series.iloc[-1])
ma_slope = float(ma20_series.iloc[-1] - ma20_series.iloc[-5])
atr_pct = float((nifty["ATR14"].iloc[-1] / spot) * 100)

# ---- Implied Volatility Proxy ----
vix_now = float(vix["Close"].iloc[-1])
iv_proxy = vix_now / 100
iv_minus_rv5 = float(iv_proxy - rv_5d)

# ---- Distance from MA ----
dist_ma = float(abs(spot - ma20) / spot * 100)

# =========================================================
# TREND REGIME (SIMPLE, ROBUST)
# =========================================================
if dist_ma < 0.6 and abs(ma_slope) < 20:
    trend_label = "Range-bound"
elif dist_ma < 1.2:
    trend_label = "Mild trend"
else:
    trend_label = "Strong trend"

# =========================================================
# DELTA BAND (INFORMATIONAL, NOT ADVICE)
# =========================================================
if trend_label == "Range-bound" and iv_minus_rv5 > 0:
    delta_band = "10–15 delta (calm / range regime)"
elif trend_label == "Mild trend":
    delta_band = "15–20 delta (balanced regime)"
else:
    delta_band = "20–30 delta (trend risk elevated)"

# =========================================================
# DISPLAY
# =========================================================
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.caption(f"Last update: **{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"{spot:,.2f}")
c2.metric("India VIX (IV)", f"{vix_now:.2f}%")
c3.metric("RV (last 5 days)", f"{rv_5d*100:.2f}%")
c4.metric("RV (last 20 days)", f"{rv_20d*100:.2f}%")
c5.metric("IV − RV(5d)", f"{iv_minus_rv5*100:.2f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("ATR % (14)", f"{atr_pct:.2f}%")
c7.metric("Trend Regime", trend_label)
c8.metric("Spot vs MA20", "Above MA20" if spot > ma20 else "Below MA20")
c9.metric("Typical Delta Band", delta_band)

st.divider()

# =========================================================
# EXPLANATION
# =========================================================
st.subheader("📌 Regime Summary")
st.markdown(f"""
- **Trend:** {trend_label}  
- **Realized Volatility (5d):** {rv_5d*100:.2f}%  
- **Implied Volatility (VIX):** {vix_now:.2f}%  
- **IV − RV Spread:** {iv_minus_rv5*100:.2f}%  
- **ATR % (range proxy):** {atr_pct:.2f}%  
- **Typical iron-condor delta band:** **{delta_band}**
""")

with st.expander("What these metrics mean"):
    st.markdown("""
- **RV (5 days)** = how much NIFTY actually moved recently  
- **VIX** = market-priced forward (ATM) volatility  
- **IV − RV > 0** → options priced richer than realized moves  
- **ATR %** = how wide the daily range is  
- **Trend regime** helps avoid tight condors in trending markets  
- **Delta bands** are regime-based reference ranges, not trade advice
""")

st.caption("Educational dashboard only. Final strategy selection is discretionary.")
