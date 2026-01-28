import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Iron Condor – Live Market Metrics (NIFTY)", layout="wide")
st.title("⚡ Iron Condor – Live Market Metrics (NIFTY)")
st.caption("Real-time regime metrics for iron-condor study. Informational only.")

# =========================================================
# DATA FETCH (STABLE SOURCES ONLY)
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

# ---- Realized Volatility (annualized) ----
rv_5d  = float(nifty["ret"].tail(5).std() * np.sqrt(252))
rv_20d = float(nifty["ret"].tail(20).std() * np.sqrt(252))

# ---- Average DAILY MOVE in POINTS (last 5 days) ----
last_5_closes = nifty["Close"].tail(6)   # need 6 prices for 5 moves
daily_moves = last_5_closes.diff().abs().dropna()
avg_daily_move_points_5d = float(daily_moves.mean())

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
# FORCE SAFE SCALARS (NO PANDAS TRUTH ERRORS)
# =========================================================
spot = float(nifty["Close"].iloc[-1])

ma20_series = nifty["MA20"].dropna()
if len(ma20_series) < 6:
    st.error("Not enough data to compute trend metrics yet.")
    st.stop()

ma20 = float(ma20_series.iloc[-1])
ma_slope = float(ma20_series.iloc[-1] - ma20_series.iloc[-5])

atr_pct = float((nifty["ATR14"].iloc[-1] / spot) * 100)

# ---- Implied Volatility (VIX proxy) ----
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
# DELTA STRUCTURE (ANALYSIS, NOT ADVICE)
# =========================================================
if trend_label == "Range-bound" and iv_minus_rv5 > 0:
    delta_structure = (
        "Short Put: 10–15Δ | Long Put: 5–10Δ\n"
        "Short Call: 10–15Δ | Long Call: 5–10Δ"
    )
elif trend_label == "Mild trend":
    delta_structure = (
        "Short Put: 15–20Δ | Long Put: 8–12Δ\n"
        "Short Call: 15–20Δ | Long Call: 8–12Δ"
    )
else:
    delta_structure = (
        "Short Put: 25–30Δ | Long Put: 10–15Δ\n"
        "Short Call: 20–25Δ | Long Call: 8–12Δ"
    )

# =========================================================
# DISPLAY METRICS
# =========================================================
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.caption(f"Last update: **{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"{spot:,.2f}")
c2.metric("India VIX (IV)", f"{vix_now:.2f}%")
c3.metric("RV (last 5 days)", f"{rv_5d*100:.2f}%")
c4.metric("RV (last 20 days)", f"{rv_20d*100:.2f}%")
c5.metric("IV − RV (5d)", f"{iv_minus_rv5*100:.2f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Avg Daily Move (5d)", f"≈ {avg_daily_move_points_5d:.0f} pts")
c7.metric("ATR % (14)", f"{atr_pct:.2f}%")
c8.metric("Trend Regime", trend_label)
c9.metric("Spot vs MA20", "Below MA20" if spot < ma20 else "Above MA20")

st.divider()

# =========================================================
# INTERPRETATION
# =========================================================
st.subheader("🧠 Market Read (Interpretation)")

st.markdown(f"""
- Over the **last 5 trading days**, NIFTY has moved **~{avg_daily_move_points_5d:.0f} points per day on average**.
- **Implied volatility (VIX)** is **higher than realized volatility**, indicating option premiums are relatively rich.
- The market is currently in a **{trend_label.lower()}**, trading **{'below' if spot < ma20 else 'above'} its 20-day average**.
- This environment favors **wider, more conservative iron condor structures**, with increased respect for directional risk.
""")

st.subheader("🧩 Regime-Based Structure Insight (Analysis Only)")
st.markdown(delta_structure)

with st.expander("How to use this for ATM straddle comparison"):
    st.markdown("""
- Compare **Avg Daily Move (points)** with **ATM straddle price**
- If straddle price > avg move → premium relatively rich
- If straddle price < avg move → movement underpriced
- Trend regime tells you whether range assumptions are fragile
""")

st.caption("Educational dashboard only. No trade recommendations.")
