"""Research page — ticker search, quotes, historical charts, sentiment."""

import json
import time
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import inject_css, pnl_color

inject_css()

st.set_page_config(page_title="Research", layout="wide")
st.title("🔬 Market Research")

auto_refresh = st.toggle("Auto-refresh", value=True, key="res_autorefresh")
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Sidebar Search ──────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Ticker Search")
    symbol_input = st.text_input("Symbol", value="AAPL", max_chars=10).upper().strip()
    
    # Date range for chart
    end_date = st.date_input("End Date", value=date.today())
    start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    
    # Chart type
    chart_type = st.selectbox("Chart Type", ["OHLC Candlestick", "Line"])
    
    # Interval
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    
    if st.button("Search & Load", type="primary", use_container_width=True):
        st.session_state["ticker"] = symbol_input or None
        st.session_state["chart_start"] = start_date
        st.session_state["chart_end"] = end_date

# Get ticker from sidebar state or input
ticker = st.session_state.get("ticker", symbol_input.upper() if symbol_input else None)

if not ticker:
    st.info("Enter a ticker symbol above and click 'Search & Load'")
    st.stop()

# ── Fetch Data ──────────────────────────────────────────────────
try:
    from bot.data import fetch_history
    df = fetch_history(ticker, start_date.strftime("%Y-%m-%d"), 
                       end_date.strftime("%Y-%m-%d"), interval=interval)
except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()

if df is None or df.empty:
    st.warning(f"No data found for {ticker} in this date range.")
    st.stop()

# ── Header Info ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close * 100) if prev_close != 0 else 0

with col1:
    st.metric(label=f"{ticker} Price", value=f"${last_close:.2f}")
with col2:
    change_color = pnl_color(change)
    st.metric(label="Day Change", value=f"${'+' if change > 0 else ''}{change:.2f}", delta=f"{change_pct:+.2f}%")
with col3:
    high_52w = float(df["High"].max())
    low_52w = float(df["Low"].min())
    st.metric(label="Range", value=f"{low_52w:.2f} - {high_52w:.2f}")
with col4:
    avg_vol = int(df["Volume"].mean())
    st.metric(label="Avg Volume", value=f"{avg_vol:,}")

st.divider()

# ── Chart ───────────────────────────────────────────────────────
st.subheader("📈 Price Chart")

fig = go.Figure()

if chart_type == "OHLC Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"], name="Price"))
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price", line=dict(color="#00C896")))

# Add EMA overlays
try:
    from bot.indicators import ema
    ema9 = ema(df["Close"], period=9)
    ema21 = ema(df["Close"], period=21)
    fig.add_trace(go.Scatter(x=df.index, y=ema9, mode="lines", name="EMA 9", line=dict(color="#FFB74D", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ema21, mode="lines", name="EMA 21", line=dict(color="#E57373", width=1)))
except Exception:
    pass

# Bollinger Bands
try:
    from bot.indicators import bollinger
    bands = bollinger(df["Close"], period=20, num_std=2)
    fig.add_trace(go.Scatter(x=df.index, y=bands["upper"], mode="lines", name="BB Upper", line=dict(color="rgba(158,158,158,0.2)", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=bands["lower"], mode="lines", fill="tonexty", name="BB Lower", line=dict(color="rgba(158,158,158,0.1)", width=1)))
except Exception:
    pass

fig.update_layout(template="plotly_dark", height=450, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Fundamentals ────────────────────────────────────────────────
st.subheader("📊 Key Metrics")

try:
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info
    cols_fund = st.columns(4)
    fund_items = [
        ("Sector", info.get("sector", "N/A")),
        ("Industry", info.get("industry", "N/A")),
        ("Market Cap", info.get("marketCap") and f"${info['marketCap']/1e9:.1f}B" or "N/A"),
        ("P/E Ratio", info.get("trailingPE") and f"{info['trailingPE']:.1f}" or "N/A"),
    ]
    for i, (label, val) in enumerate(fund_items):
        cols_fund[i].metric(label, val)
except Exception as e:
    st.caption("Fundamentals unavailable — run `python main.py backtest --symbols AAPL` to test data pipeline.")

# ── Historical Returns ──────────────────────────────────────────
st.divider()
st.subheader("📊 Return Statistics")

returns = df["Close"].pct_change().dropna()
if len(returns) > 1:
    r_cols = st.columns(5)
    stats_list = [
        ("Mean Daily %", f"{returns.mean()*100:.3f}%"),
        ("Std Dev Daily %", f"{returns.std()*100:.3f}%"),
        ("Best Day %", f"{returns.max()*100:.2f}%"),
        ("Worst Day %", f"{returns.min()*100:.2f}%"),
        ("Skewness", f"{returns.skew():.3f}"),
    ]
    for i, (label, val) in enumerate(stats_list):
        r_cols[i % 5].metric(label, val)

# ── Sentiment Preview ───────────────────────────────────────────
st.divider()
st.subheader("💬 Social Sentiment")

try:
    from bot.sentiment import SentimentEngine
    engine = SentimentEngine()
    score = engine.score(ticker, hours=24)
    
    s_cols = st.columns(4)
    s_cols[0].metric("Mentions", str(score.mentions))
    bullish_pct = round(score.bullish / max(score.mentions, 1) * 100, 1) if score.mentions > 0 else 0
    bearish_pct = round(score.bearish / max(score.mentions, 1) * 100, 1) if score.mentions > 0 else 0
    s_cols[1].metric("Bullish %", f"{bullish_pct}%")
    s_cols[2].metric("Bearish %", f"{bearish_pct}%")
    s_cols[3].metric("Net Score", f"{score.net_score:.3f}")
    
    # Gauge chart
    import plotly.express as px
    gauge_fig = px.gauge(score.net_score, min=-1, max=1, title=f"{ticker} Net Sentiment")
    gauge_fig.update_layout(height=200)
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Top posts
    if score.top_posts:
        st.caption("Top recent posts:")
        for p in score.top_posts[:3]:
            emoji = "🟢" if p.score > 0.25 else "🔴" if p.score < -0.25 else "⚪"
            st.markdown(f"{emoji} **{p.source}** ({p.timestamp.strftime('%m/%d %H:%M')}): {p.text[:150]}...")
except Exception as e:
    st.caption("Sentiment data unavailable (requires live network access). Error: {e}")

# ── Quick Backtest ──────────────────────────────────────────────
st.divider()
if st.button("🧪 Run Quick Backtest on this Ticker →"):
    st.session_state["quick_bt_ticker"] = ticker
    st.switch_page("pages/2_🧪_Backtest.py")
