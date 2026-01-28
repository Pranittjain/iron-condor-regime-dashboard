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
st.caption("A simple dashboard to study direction, trend strength, and volatility for iron condors. No recommendations.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

view_start = st.sidebar.date_input("View start date", value=date(2026, 1, 1))
view_end = st.sidebar.date_input("View end date", value=date.today())

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0
)

lookback_days = st.sidebar.slider(
    "Extra lookback for indicator calculation (days)",
    min_value=100,
    max_value=1200,
    value=500,
    step=50
)

show_help = st.sidebar.toggle("Show metric explanations", value=True)

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

# -----------------------------
# Fetch data
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

    # Flatten MultiIndex if needed
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
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

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

def slope(series: pd.Series, lookback: int = 20) -> float:
    """
    Simple slope of last N points using linear regression on index 0..N-1.
    Returns slope per step.
    """
    y = series.dropna().tail(lookback)
    if len(y) < max(5, lookback // 2):
        return np.nan
    x = np.arange(len(y))
    m = np.polyfit(x, y.values, 1)[0]
    return float(m)

def direction_label(start_close: float, end_close: float, ma_slope: float) -> str:
    if np.isnan(start_close) or np.isnan(end_close):
        return "Direction: —"
    ret = (end_close / start_close) - 1

    # simple thresholds
    if ret > 0.01 and (np.isnan(ma_slope) or ma_slope >= 0):
        return "Direction: Up"
    if ret < -0.01 and (np.isnan(ma_slope) or ma_slope <= 0):
        return "Direction: Down"
    return "Direction: Sideways / Mixed"

def trend_strength_label(adx_val: float) -> str:
    if np.isnan(adx_val):
        return "Trend strength: —"
    if adx_val >= 25:
        return "Trend strength: Strong"
    if adx_val >= 18:
        return "Trend strength: Medium"
    return "Trend strength: Weak"

# -----------------------------
# Fetch with extra lookback
# -----------------------------
fetch_start = view_start - timedelta(days=int(lookback_days))

with st.spinner("Fetching data..."):
    nifty = fetch_yf(NIFTY_TICKER, str(fetch_start), str(view_end), interval)
    vix = fetch_yf(VIX_TICKER, str(fetch_start), str(view_end), interval)

if nifty.empty:
    st.error("No NIFTY data returned. Try changing your dates/interval.")
    st.stop()

required = {"Date", "Open", "High", "Low", "Close"}
if not required.issubset(set(nifty.columns)):
    st.error(f"NIFTY missing required columns. Got: {list(nifty.columns)}")
    st.stop()

# Merge VIX
if not vix.empty and "Close" in vix.columns:
    vix = vix[["Date", "Close"]].rename(columns={"Close": "Vix"})
    df = pd.merge(nifty, vix, on="Date", how="left")
else:
    df = nifty.copy()
    df["Vix"] = np.nan

# Compute metrics
df["Ma20"] = df["Close"].rolling(20).mean()
df["Ma50"] = df["Close"].rolling(50).mean()
df["Ma200"] = df["Close"].rolling(200).mean()

df["Adx14"] = adx(df["High"], df["Low"], df["Close"], 14)
df["Rv20"] = realized_vol(df["Close"], 20)
df["Atr14"] = atr(df["High"], df["Low"], df["Close"], 14)
df["Atr_pct"] = df["Atr14"] / df["Close"]

df["Iv_proxy"] = df["Vix"] / 100.0
df["Iv_minus_rv20"] = df["Iv_proxy"] - df["Rv20"]

# Filter to view window
view_df = df[(df["Date"].dt.date >= view_start) & (df["Date"].dt.date <= view_end)].copy()
if view_df.empty:
    st.error("No rows in the selected view window. Widen the date range.")
    st.stop()

latest = view_df.iloc[-1]
first = view_df.iloc[0]

# Direction based on selected window
ma20_slope = slope(view_df["Ma20"], lookback=20)
dir_text = direction_label(float(first["Close"]), float(latest["Close"]), ma20_slope)
trend_text = trend_strength_label(float(latest["Adx14"]))

# -----------------------------
# MAIN DASHBOARD (values you need)
# -----------------------------
st.divider()
st.subheader("Snapshot for Selected Time Window")

st.info(f"**{dir_text}**  |  **{trend_text}**  |  Window: {view_start} → {view_end}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Start Close", f"{float(first['Close']):.2f}")
c2.metric("End Close", f"{float(latest['Close']):.2f}")
c3.metric("RV20 (ann.)", "-" if np.isnan(latest["Rv20"]) else f"{float(latest['Rv20'])*100:.2f}%")
c4.metric("VIX", "-" if np.isnan(latest["Vix"]) else f"{float(latest['Vix']):.2f}")
c5.metric("IV − RV20", "-" if np.isnan(latest["Iv_minus_rv20"]) else f"{float(latest['Iv_minus_rv20'])*100:.2f}%")
c6.metric("ADX14", "-" if np.isnan(latest["Adx14"]) else f"{float(latest['Adx14']):.1f}")

d1, d2, d3 = st.columns(3)
window_return = (float(latest["Close"]) / float(first["Close"])) - 1
d1.metric("Return (window)", f"{window_return*100:.2f}%")
d2.metric("ATR% (14)", "-" if np.isnan(latest["Atr_pct"]) else f"{float(latest['Atr_pct'])*100:.2f}%")
d3.metric("MA20 slope (recent)", "-" if np.isnan(ma20_slope) else f"{ma20_slope:.2f}")

if show_help:
    with st.expander("📌 What do these metrics mean? (click)"):
        st.markdown(
            """
**Direction (Up / Down / Sideways):** Based on return over your selected window + recent MA20 slope.

**Trend strength (ADX14):** Measures how strong the trend is (not direction).
- < 18: weak trend (more chop/range)
- 18–25: medium
- > 25: strong trend

**RV20 (annualized):** Realized volatility over last ~20 sessions (how much NIFTY actually moved).

**India VIX:** Implied volatility gauge (proxy for how rich option premiums are).

**IV − RV20:** A simple “premium gap” proxy (purely descriptive).

**ATR% (14):** Average daily range relative to price (helps you see how “wide” daily movement is).
"""
        )

# -----------------------------
# Charts (tidy)
# -----------------------------
st.divider()
st.subheader("Charts")

tab1, tab2, tab3 = st.tabs(["Price + MAs", "ADX", "Volatility (RV vs VIX)"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Ma20"], name="MA20"))
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Ma50"], name="MA50"))
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Ma200"], name="MA200"))
    fig.update_layout(title="NIFTY Close + MA20/50/200", height=520, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Adx14"], name="ADX14"))
    fig.add_hline(y=25, line_dash="dot")
    fig.add_hline(y=18, line_dash="dot")
    fig.update_layout(title="ADX14 (Trend Strength)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Rv20"] * 100, name="RV20 (%)"))
    fig.add_trace(go.Scatter(x=view_df["Date"], y=view_df["Vix"], name="India VIX"))
    fig.update_layout(title="RV20 vs India VIX", yaxis_title="Percent", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Table (clean + downloadable)
# -----------------------------
st.divider()
st.subheader("Metrics Table (download-ready)")

cols_to_show = [
    "Date", "Open", "High", "Low", "Close",
    "Vix", "Iv_proxy",
    "Rv20", "Iv_minus_rv20",
    "Adx14",
    "Ma20", "Ma50", "Ma200",
    "Atr_pct"
]

table_df = view_df[cols_to_show].copy().tail(400)
st.dataframe(table_df, use_container_width=True)

csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="iron_condor_metrics.csv", mime="text/csv")
