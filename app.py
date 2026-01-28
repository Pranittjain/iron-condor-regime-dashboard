import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="Iron Condor – Live Market Metrics (NIFTY)", layout="wide")
st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Real-time regime metrics for iron condor planning. Informational only.")

# ===============================
# DATA FETCH
# ===============================
@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    return df.reset_index()

nifty = fetch_data("^NSEI", "1y")
vix   = fetch_data("^INDIAVIX", "1y")

if nifty.empty or vix.empty:
    st.error("Market data unavailable.")
    st.stop()

# ===============================
# CORE METRICS
# ===============================
nifty["ret"] = nifty["Close"].pct_change()

# Realized volatility
rv_5d  = float(nifty["ret"].tail(5).std() * np.sqrt(252))
rv_20d = float(nifty["ret"].tail(20).std() * np.sqrt(252))

# Avg daily move in POINTS
last_5_closes = nifty["Close"].tail(6)
avg_daily_move_pts = float(last_5_closes.diff().abs().dropna().mean())

# Moving average
nifty["MA20"] = nifty["Close"].rolling(20).mean()

# ATR
nifty["TR"] = np.maximum(
    nifty["High"] - nifty["Low"],
    np.maximum(
        abs(nifty["High"] - nifty["Close"].shift(1)),
        abs(nifty["Low"] - nifty["Close"].shift(1))
    )
)
nifty["ATR14"] = nifty["TR"].rolling(14).mean()

spot = float(nifty["Close"].iloc[-1])
ma20 = float(nifty["MA20"].dropna().iloc[-1])
atr_pct = float(nifty["ATR14"].iloc[-1] / spot * 100)

# VIX metrics
vix_now = float(vix["Close"].iloc[-1])
iv_minus_rv = float(vix_now/100 - rv_5d)

# VIX percentile
vix_percentile = float(
    (vix["Close"] < vix_now).mean() * 100
)

# Trend regime
dist_ma = abs(spot - ma20) / spot * 100
if dist_ma < 0.6:
    trend = "Range-bound"
elif dist_ma < 1.2:
    trend = "Mild trend"
else:
    trend = "Strong trend"

# ===============================
# DISPLAY METRICS
# ===============================
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.caption(f"Last update: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"{spot:,.2f}")
c2.metric("India VIX (IV)", f"{vix_now:.2f}%")
c3.metric("RV (5 days)", f"{rv_5d*100:.2f}%")
c4.metric("Avg Daily Move (5d)", f"≈ {avg_daily_move_pts:.0f} pts")
c5.metric("VIX Percentile (1Y)", f"{vix_percentile:.0f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("ATR % (14)", f"{atr_pct:.2f}%")
c7.metric("Trend Regime", trend)
c8.metric("Spot vs MA20", "Below MA20" if spot < ma20 else "Above MA20")
c9.metric("IV − RV (5d)", f"{iv_minus_rv*100:.2f}%")

st.divider()

# ===============================
# NIFTY CHART
# ===============================
st.subheader("📈 NIFTY Price (1 Year)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=nifty["Date"], y=nifty["Close"], name="Close"))
fig.add_trace(go.Scatter(x=nifty["Date"], y=nifty["MA20"], name="MA20"))
fig.update_layout(height=400, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ===============================
# INTERPRETATION
# ===============================
st.subheader("🧠 Market Interpretation")

st.markdown(f"""
- **NIFTY has moved ~{avg_daily_move_pts:.0f} points per day on average over the last 5 sessions.**
- **Implied volatility (VIX)** is above recent realized volatility → option premiums are relatively rich.
- Market structure is **{trend.lower()}**, with spot trading **{'below' if spot < ma20 else 'above'} MA20**.
- **ATR suggests intraday swings of ~{atr_pct*spot/100:.0f} points**, so tight structures are risky.
- VIX percentile of **{vix_percentile:.0f}%** shows IV is {'elevated' if vix_percentile > 60 else 'moderate'} relative to the last year.
""")

st.caption("This dashboard is for regime awareness and study only. No trade recommendations.")
