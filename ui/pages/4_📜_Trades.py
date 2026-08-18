"""Trades page — full trade journal with P&L colors, filtering, export."""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.theme import inject_css, pnl_color

inject_css()

st.set_page_config(page_title="Trades", layout="wide")
st.title("📜 Trade Journal")

auto_refresh = st.toggle("Auto-refresh", value=True, key="trades_autorefresh")
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Load Trade Data ────────────────────────────────────────────
trades_path = Path("logs/trades.csv")
signals_path = Path("logs/signals.csv")

df_trades = None
df_signals = None

if trades_path.exists() and trades_path.stat().st_size > 0:
    try:
        df_trades = pd.read_csv(trades_path)
        if "timestamp" in df_trades.columns:
            df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
        # Calculate realized PnL (simplified: net of matching buy/sell pairs)
        if not df_trades.empty:
            df_trades["amount"] = df_trades.apply(
                lambda r: -float(r["price"]) * int(r["qty"]) 
                    if str(r["side"]).upper() == "BUY" 
                    else float(r["price"]) * int(r["qty"]), 
                axis=1
            )
    except Exception as e:
        st.error(f"Error loading trades: {e}")

if signals_path.exists() and signals_path.stat().st_size > 0:
    try:
        df_signals = pd.read_csv(signals_path)
    except Exception:
        pass

# ── Summary Metrics ─────────────────────────────────────────────
if df_trades is not None and not df_trades.empty:
    cols_sum = st.columns(6)
    
    total_trades = len(df_trades)
    buys = df_trades[df_trades["side"].str.upper() == "BUY"]
    sells = df_trades[df_trades["side"].str.upper() == "SELL"]
    net_invested = buys["amount"].sum() if len(buys) > 0 else 0
    net_realized = sells["amount"].sum() if len(sells) > 0 else 0
    
    unique_symbols = df_trades["symbol"].nunique()
    traded_symbols = list(df_trades["symbol"].unique())
    
    cols_sum[0].metric(label="Total Trades", value=str(total_trades))
    cols_sum[1].metric(label="Buys", value=str(len(buys)))
    cols_sum[2].metric(label="Sells", value=str(len(sells)))
    gross_pnl = df_trades["amount"].sum()
    cols_sum[3].metric(label="Gross P&L", value=f"${gross_pnl:,.0f}", delta=f"${gross_pnl:+,.0f}")
    cols_sum[4].metric(label="Symbols Traded", value=str(unique_symbols))
    cols_sum[5].metric(label="Avg Trade Size", value=f"${df_trades['amount'].abs().mean():,.0f}")
    
    st.divider()
else:
    cols_sum = st.columns(5)
    for c in cols_sum:
        c.metric(label="—", value="N/A")
    st.info("No trades recorded yet. Start the engine or run a backtest to generate trade data.")

# ── Filters ─────────────────────────────────────────────────────
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    symbol_filter = st.selectbox("Filter by Symbol", options=["ALL"] + traded_symbols if traded_symbols else ["ALL"], index=0)
with col_f2:
    side_filter = st.selectbox("Filter by Side", options=["ALL", "BUY", "SELL"], index=0)
with col_f3:
    reason_filter = st.text_input("Filter by Reason", placeholder="e.g., signal, stop_loss")
with col_f4:
    st.markdown("**Actions:**")
    if df_trades is not None and not df_trades.empty:
        csv_data = df_trades.to_csv(index=False).encode()
        st.download_button("📥 Export CSV", data=csv_data, file_name=f"trades_{Path('logs/bot.log').stat().st_mtime:.0f}.csv", mime="text/csv")

# ── Signals Log ─────────────────────────────────────────────────
if df_signals is not None and not df_signals.empty:
    st.divider()
    st.subheader("📡 Signal Log")
    sig_cols = st.columns(4)
    sig_cols[0].metric("Total Signals", str(len(df_signals)))
    entry_count = len(df_signals[df_signals["signal"] == 1]) if "signal" in df_signals.columns else 0
    exit_count = len(df_signals[df_signals["signal"] == -1]) if "signal" in df_signals.columns else 0
    sig_cols[1].metric("Entry Signals (1)", str(entry_count))
    sig_cols[2].metric("Exit Signals (-1)", str(exit_count))
    last_signal = df_signals.iloc[-1]
    sig_cols[3].metric("Last Signal", f"{last_signal.get('signal', 'N/A')} @ {last_signal.get('timestamp', 'N/A')}")
    
    st.dataframe(df_signals.tail(30), use_container_width=True, hide_index=True)

# ── Trades Table ────────────────────────────────────────────────
st.divider()
st.subheader("📋 Trade History")

if df_trades is not None and not df_trades.empty:
    display_df = df_trades.copy()
    
    # Apply filters
    if symbol_filter != "ALL":
        display_df = display_df[display_df["symbol"].str.upper() == symbol_filter.upper()]
    if side_filter != "ALL":
        display_df = display_df[display_df["side"].str.upper() == side_filter.upper()]
    if reason_filter:
        display_df = display_df[display_df["reason"].str.contains(reason_filter, case=False, na=False)]
    
    # Color-code columns
    styled = display_df.style.map(
        lambda v: f"color: #FF5252" if isinstance(v, (int, float)) and v < 0 
                  else f"color: #00C896" if isinstance(v, (int, float)) and v > 0 
                  else "", 
        subset=["amount"] if "amount" in display_df.columns else []
    )
    
    # Format timestamp for readability
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # Symbol summary
    st.divider()
    st.subheader("📊 Summary by Symbol")
    
    sym_summary = display_df.groupby("symbol").agg(
        trade_count=("symbol", "count"),
        net_invested=("amount", lambda x: x[x < 0].sum()),
        net_realized=("amount", lambda x: x[x >= 0].sum()),
        avg_price=("price", "mean"),
        avg_qty=("qty", "mean"),
    ).round(2)
    
    sym_summary["net_pnl"] = sym_summary["net_invested"] + sym_summary["net_realized"]
    st.dataframe(sym_summary, use_container_width=True)
else:
    st.info("No trade data available. Run the engine or backtest to populate this table.")
