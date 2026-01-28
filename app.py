import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timezone, timedelta

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Iron Condor Live Metrics (NIFTY)", layout="wide")
st.title("⚡ Iron Condor Live Metrics (NIFTY)")
st.caption("Live/near-live market state metrics for decision support. No recommendations. Data: Yahoo Finance (NIFTY, India VIX) + optional NSE option chain snapshot.")

# -----------------------------
# Controls (simple)
# -----------------------------
c0, c1, c2 = st.columns([1.2, 1, 1])
refresh_sec = c0.selectbox("Auto refresh (seconds)", [15, 30, 60, 120], index=2)
show_explain = c1.toggle("Explain metrics", value=True)
try_nse_chain = c2.toggle("Try LIVE NSE ATM IV", value=False)

# Auto refresh
st.markdown(
    f"<meta http-equiv='refresh' content='{refresh_sec}'>",
    unsafe_allow_html=True
)

# Tickers
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

# -----------------------------
# Helpers: indicators
# -----------------------------
def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

def atr(high, low, close, window=14):
    return true_range(high, low, close).rolling(window).mean()

def adx(high, low, close, window=14):
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

def annualized_vol(returns, periods_per_year):
    r = returns.dropna()
    if len(r) < 5:
        return np.nan
    return float(r.std() * np.sqrt(periods_per_year))

def label_trend_strength(adx_val):
    if np.isnan(adx_val):
        return "—"
    if adx_val >= 25:
        return "Strong"
    if adx_val >= 18:
        return "Medium"
    return "Weak"

def label_direction(close, ma20):
    # purely descriptive
    if np.isnan(close) or np.isnan(ma20):
        return "—"
    if close > ma20:
        return "Above MA20"
    if close < ma20:
        return "Below MA20"
    return "On MA20"

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_5m(ticker: str) -> pd.DataFrame:
    # 5m data limited window; yfinance supports up to ~60d
    df = yf.download(
        ticker,
        period="60d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        threads=True
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_2y(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

# -----------------------------
# NSE Option Chain (today snapshot only, may fail on Streamlit Cloud)
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_nse_chain(symbol="NIFTY"):
    base = "https://www.nseindia.com"
    url = f"{base}/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }
    s = requests.Session()
    s.get(base, headers=headers, timeout=8)  # warm cookies
    r = s.get(url, headers=headers, timeout=8)
    r.raise_for_status()
    return r.json()

def parse_atm_iv(chain_json: dict):
    rec = chain_json.get("records", {})
    underlying = rec.get("underlyingValue", None)
    expiries = rec.get("expiryDates", [])
    data = rec.get("data", [])
    if underlying is None or not expiries or not data:
        return None

    expiry = expiries[0]
    rows = [x for x in data if x.get("expiryDate") == expiry]
    if not rows:
        return None

    strikes = np.array([x.get("strikePrice") for x in rows if x.get("strikePrice") is not None], dtype=float)
    if strikes.size == 0:
        return None

    atm = float(strikes[np.argmin(np.abs(strikes - float(underlying)))])
    row = next((x for x in rows if float(x.get("strikePrice", -1)) == atm), None)
    if not row:
        return None

    ce_iv = row.get("CE", {}).get("impliedVolatility", np.nan)
    pe_iv = row.get("PE", {}).get("impliedVolatility", np.nan)
    ivs = [v for v in [ce_iv, pe_iv] if isinstance(v, (int, float)) and not np.isnan(v)]
    atm_iv = float(np.mean(ivs)) if ivs else np.nan

    return {
        "underlying": float(underlying),
        "expiry": expiry,
        "atm_strike": atm,
        "ce_iv": ce_iv,
        "pe_iv": pe_iv,
        "atm_iv": atm_iv
    }

# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading live data..."):
    intraday_nifty = fetch_intraday_5m(NIFTY_TICKER)
    intraday_vix = fetch_intraday_5m(VIX_TICKER)   # may be sparse/blank intraday
    daily_nifty = fetch_daily_2y(NIFTY_TICKER)
    daily_vix = fetch_daily_2y(VIX_TICKER)

if daily_nifty.empty:
    st.error("Daily NIFTY data not available right now.")
    st.stop()

# Latest values (use intraday close if available, else daily close)
now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
last_update = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")

spot = float(intraday_nifty["Close"].iloc[-1]) if not intraday_nifty.empty else float(daily_nifty["Close"].iloc[-1])
spot_prev = float(daily_nifty["Close"].iloc[-2]) if len(daily_nifty) >= 2 else np.nan
chg = (spot / spot_prev - 1) if not np.isnan(spot_prev) else np.nan

vix_now = np.nan
if not intraday_vix.empty:
    vix_now = float(intraday_vix["Close"].iloc[-1])
elif not daily_vix.empty:
    vix_now = float(daily_vix["Close"].iloc[-1])

# Daily indicators (trend + medium-term vol)
daily_nifty["Ret"] = daily_nifty["Close"].pct_change()
daily_nifty["MA20"] = daily_nifty["Close"].rolling(20).mean()
daily_nifty["MA50"] = daily_nifty["Close"].rolling(50).mean()
daily_nifty["MA200"] = daily_nifty["Close"].rolling(200).mean()
daily_nifty["ADX14"] = adx(daily_nifty["High"], daily_nifty["Low"], daily_nifty["Close"], 14)
daily_nifty["ATR14"] = atr(daily_nifty["High"], daily_nifty["Low"], daily_nifty["Close"], 14)
daily_nifty["ATR_PCT"] = daily_nifty["ATR14"] / daily_nifty["Close"]
rv20_daily = annualized_vol(daily_nifty["Ret"].tail(20), 252)
rv60_daily = annualized_vol(daily_nifty["Ret"].tail(60), 252)
adx14 = float(daily_nifty["ADX14"].iloc[-1]) if not np.isnan(daily_nifty["ADX14"].iloc[-1]) else np.nan
atr_pct = float(daily_nifty["ATR_PCT"].iloc[-1]) if not np.isnan(daily_nifty["ATR_PCT"].iloc[-1]) else np.nan
ma20 = float(daily_nifty["MA20"].iloc[-1]) if not np.isnan(daily_nifty["MA20"].iloc[-1]) else np.nan
ma50 = float(daily_nifty["MA50"].iloc[-1]) if not np.isnan(daily_nifty["MA50"].iloc[-1]) else np.nan
ma200 = float(daily_nifty["MA200"].iloc[-1]) if not np.isnan(daily_nifty["MA200"].iloc[-1]) else np.nan

# Intraday realized vol (5m) – for “today’s regime feel”
rv_intraday = np.nan
if not intraday_nifty.empty:
    # last ~1 trading day (5m bars). Annualize with 5m periods:
    # ~75 bars/day, ~252 days => ~18900 5m bars/year
    intraday_nifty["Ret"] = intraday_nifty["Close"].pct_change()
    rv_intraday = annualized_vol(intraday_nifty["Ret"].tail(75), periods_per_year=18900)

# IV proxy from VIX (convert % -> decimal)
iv_proxy = (vix_now / 100.0) if not np.isnan(vix_now) else np.nan
iv_minus_rv20 = (iv_proxy - rv20_daily) if (not np.isnan(iv_proxy) and not np.isnan(rv20_daily)) else np.nan

# NSE chain snapshot (optional)
atm_iv = np.nan
atm_expiry = None
atm_strike = None
if try_nse_chain:
    try:
        chain = fetch_nse_chain("NIFTY")
        parsed = parse_atm_iv(chain)
        if parsed:
            atm_iv = parsed["atm_iv"] /_]()_
