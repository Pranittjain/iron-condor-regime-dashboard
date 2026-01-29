import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(
    page_title="Iron Condor – Live Market Metrics (NIFTY)",
    layout="wide"
)

st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Metrics & regime interpretation for iron condor planning. Informational only.")

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

# Realized Volatility (5D)
rv_5d = float(nifty["ret"].tail(5).std() * np.sqrt(252))

# Avg daily close-to-close move (points)
last_5_closes = nifty["Close"].tail(6)
avg_daily_move_pts = float(last_5_closes.diff().abs().dropna().mean())

# Moving Average
nifty["MA20"] = nifty["Close"].rolling(20).mean()

# MA20 slope (direction)
ma20_slope = nifty["MA20"].diff().iloc[-1]

# ATR (14)
nifty["TR"] = np.maximum(
    nifty["High"] - nifty["Low"],
    np.maximum(
        abs(nifty["High"] - nifty["Close"].shift(1)),
        abs(nifty["Low"] - nifty["Close"].shift(1))
    )
)
nifty["ATR14"] = nifty["TR"].rolling(14).mean()

# Scalars
spot = float(nifty["Close"].iloc[-1])
ma20 = float(nifty["MA20"].dropna().iloc[-1])
atr_pct = float(nifty["ATR14"].iloc[-1] / spot * 100)

# VIX
vix_now = float(vix["Close"].iloc[-1])
iv_minus_rv = float(vix_now / 100 - rv_5d)

# ===============================
# TREND STRENGTH (REGIME)
# ===============================
dist_ma = abs(spot - ma20) / spot * 100

if dist_ma < 0.6:
    trend_strength = "Range-bound"
elif dist_ma < 1.2:
    trend_strength = "Mild trend"
else:
    trend_strength = "Strong trend"

# ===============================
# TREND DIRECTION
# ===============================
if (spot > ma20) and (ma20_slope > 0):
    trend_direction = "Bullish"
elif (spot < ma20) and (ma20_slope < 0):
    trend_direction = "Bearish"
else:
    trend_direction = "Neutral / Mixed"

trend_regime = f"{trend_strength} ({trend_direction})"

# ===============================
# DELTA STRUCTURE MAPPING
# ===============================
if trend_strength == "Range-bound" and iv_minus_rv > 0:
    short_put = "10–15 Δ"
    long_put  = "5–10 Δ"
    short_call = "10–15 Δ"
    long_call  = "5–10 Δ"

elif trend_strength == "Mild trend":
    short_put = "15–20 Δ"
    long_put  = "8–12 Δ"
    short_call = "15–20 Δ"
    long_call  = "8–12 Δ"

else:  # Strong trend
    short_put = "25–30 Δ"
    long_put  = "10–15 Δ"
    short_call = "20–25 Δ"
    long_call  = "8–12 Δ"

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
c5.metric("IV − RV (5d)", f"{iv_minus_rv*100:.2f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("ATR % (14)", f"{atr_pct:.2f}%")
c7.metric("Trend Regime", trend_strength)
c8.metric("Trend Direction", trend_direction)
c9.metric("Spot vs MA20", "Below MA20" if spot < ma20 else "Above MA20")

st.divider()

# ===============================
# INTERPRETATION
# ===============================
st.subheader("🧠 Market Interpretation")

st.markdown(f"""
- Over the last **5 trading sessions**, NIFTY has moved **~{avg_daily_move_pts:.0f} points per day on average** (close-to-close).
- **Implied volatility (VIX)** is {'above' if iv_minus_rv > 0 else 'below'} recent realized volatility, indicating option premiums are {'rich' if iv_minus_rv > 0 else 'thin'}.
- Market structure shows a **{trend_strength.lower()} with a {trend_direction.lower()} bias**.
- Spot is trading **{'below' if spot < ma20 else 'above'} the 20-day moving average**.
- **ATR implies intraday swings**
