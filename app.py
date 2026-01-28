import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Iron Condor Study Dashboard (NIFTY)", layout="wide")
st.title("📊 Iron Condor Study Dashboard (NIFTY)")
st.caption("Simple regime + volatility + trend metrics for studying iron condors. No recommendations.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

# This is what you WANT to view (e.g., 2026 -> today)
view_start = st.sidebar.date_input("View start date", value=date(2026, 1, 1))
view_end = st.sidebar.date_input("View end date", value=date.today())

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="Use 1d for daily regime metrics."
)

# Extra history to compute indicators properly even if you view only 2026+
lookback_days = st.sidebar.slider(
    "Extra lookback (days) for indicator calculation",
    min_value=100,
    max_value=1200,
    value=500,
    step=50,
    help="We fetch more history so indicators like MA200, RV60, ADX14 work even for short view windows."
)

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
# Indicators (simple + standard)
# -----------------------------
def realized_vol(close: pd.Series, window: int, ann_factor: int = 252) -> pd.Series:
    r = close.pct_change()
    return r.rolling(window).std() * np.sqrt(ann_factor)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(window).mean()

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

# -----------------------------
# Plots
# -----------------------------
def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure(
        data=[go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
        )]
    )
    fig.update_layout(title=title, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_metrics(df: pd.DataFrame, title: str):
    # Simple “study” view: Close+MAs, ADX, RV, VIX
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.45, 0.18, 0.19, 0.18],
        subplot_titles=("Close + Moving Averages", "ADX (Trend Strength)", "Realized Vol (Annualized)", "India VIX (IV proxy)")
    )

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    for col, name in [("Ma20", "MA20"), ("Ma50", "MA50"), ("Ma200", "MA200")]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=name), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Adx14"], mode="lines", name="ADX14"), row=2, col=1)

    for col, name in [("Rv10", "RV10"), ("Rv20", "RV20"), ("Rv60", "RV60")]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=name), row=3, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Vix"], mode="lines", name="VIX"), row=4, col=1)

    fig.update_layout(
        title=title, height=950,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# -----------------------------
# Fetch with lookback buffer
# -----------------------------
# We fetch from earlier so indicators work, but we DISPLAY only the view window.
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

# -----------------------------
# Compute metrics (simple set)
# -----------------------------
df["Ma20"] = df["Close"].rolling(20).mean()
df["Ma50"] = df["Close"].rolling(50).mean()
df["Ma200"] = df["Close"].rolling(200).mean()

df["Adx14"] = adx(df["High"], df["Low"], df["Close"], window=14)

df["Rv10"] = realized_vol(df["Close"], 10)
df["Rv20"] = realized_vol(df["Close"], 20)
df["Rv60"] = realized_vol(df["Close"], 60)

df["Atr14"] = atr(df["High"], df["Low"], df["Close"], window=14)
df["Atr_pct"] = df["Atr14"] / df["Close"]

# VIX is %; convert to decimal and compare with RV20
df["Iv_proxy"] = df["Vix"] / 100.0
df["Iv_minus_rv20"] = df["Iv_proxy"] - df["Rv20"]

# -----------------------------
# Filter to VIEW window (this is what you asked for)
# -----------------------------
view_df = df[(df["Date"].dt.date >= view_start) & (df["Date"].dt.date <= view_end)].copy()

if view_df.empty:
    st.error("No rows in your view window. Try widening the date range.")
    st.stop()

latest = view_df.iloc[-1]

# -----------------------------
# Top snapshot numbers (simple + interpretable)
# -----------------------------
st.divider()
st.subheader("Latest Snapshot (in your view window)")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Close", f"{latest['Close']:.2f}")
c2.metric("Open", f"{latest['Open']:.2f}")
c3.metric("VIX", "-" if np.isnan(latest["Vix"]) else f"{latest['Vix']:.2f}")
c4.metric("RV20 (ann.)", "-" if np.isnan(latest["Rv20"]) else f"{latest['Rv20']*100:.2f}%")
c5.metric("ADX14", "-" if np.isnan(latest["Adx14"]) else f"{latest['Adx14']:.1f}")
c6.metric("ATR% (14)", "-" if np.isnan(latest["Atr_pct"]) else f"{latest['Atr_pct']*100:.2f}%")

# -----------------------------
# Charts (simple)
# -----------------------------
tab1, tab2 = st.tabs(["Candles (OHLC)", "Study Metrics (MA, ADX, RV, VIX)"])

with tab1:
    st.plotly_chart(plot_candles(view_df, "NIFTY OHLC (Index = Spot Proxy)"), use_container_width=True)

with tab2:
    # Need columns present even if early rows have NaN
    plot_cols = ["Date", "Close", "Ma20", "Ma50", "Ma200", "Adx14", "Rv10", "Rv20", "Rv60", "Vix"]
    plot_df = view_df[plot_cols].copy()
    st.plotly_chart(plot_metrics(plot_df, "Core Iron Condor Study Metrics"), use_container_width=True)

# -----------------------------
# Table (keep this style like you said)
# -----------------------------
st.divider()
st.subheader("Metrics Table (download-ready)")

cols_to_show = [
    "Date", "Open", "High", "Low", "Close",
    "Vix", "Iv_proxy",
    "Rv10", "Rv20", "Rv60",
    "Iv_minus_rv20",
    "Adx14",
    "Ma20", "Ma50", "Ma200",
    "Atr_pct"
]

table_df = view_df[cols_to_show].copy()

# Keep it readable: last 300 rows
table_df = table_df.tail(300)

st.dataframe(table_df, use_container_width=True)

csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="iron_condor_metrics.csv", mime="text/csv")

