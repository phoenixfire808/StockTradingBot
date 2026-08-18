"""Dashboard page — equity metrics, positions, trades overview."""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Any

from ui.theme import inject_css, kill_switch_banner, get_positions_file, pnl_color

inject_css()

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Stock Trading Bot — Dashboard")

# ── Auto-refresh toggle ───────────────────────────────────────
auto_refresh = st.toggle("Auto-refresh", value=True, key="dashboard_autorefresh")
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Kill switch status ────────────────────────────────────────
flag_file = Path("logs/kill_switch.flag")
is_tripped = flag_file.exists()
kill_switch_banner(is_tripped)

# ── Load state files ──────────────────────────────────────────
engine_state_path = Path("logs/engine_state.json")
positions_path = Path("logs/positions_state.json")
equity_path = Path("logs/equity_history.csv")
trades_path = Path("logs/trades.csv")
watchlist_path = Path("logs/watchlist.json")

def load_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

state = load_json(str(engine_state_path))
positions = load_json(str(positions_path)) or {}
watchlist_data = load_json(str(watchlist_path)) or {"symbols": []}

# ── Metric Cards Row ──────────────────────────────────────────
eq_value = "N/A"
buying_power = "N/A"
day_pnl = "N/A"
open_pos_count = len(positions)
mode_label = "DRY-RUN"
strategy_label = "None"

if state:
    mode_label = state.get("mode", "unknown").upper()
    strategy_label = state.get("strategy", "none")
    eq_value = f"${state.get('day_start_equity', 0):,.2f}" if state.get('day_start_equity') else "N/A"
    day_pnl = "— N/A (no closed P&L yet)"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="💰 Equity", value=eq_value)
with col2:
    st.metric(label="📊 Open Positions", value=str(open_pos_count))
with col3:
    st.metric(label="⚡ Mode", value=mode_label)
with col4:
    st.metric(label="🎯 Strategy", value=strategy_label)

# Broker badge
from bot.broker import MockBroker, RobinhoodMcpBroker
from bot.config import load_settings
settings = load_settings()
has_auth = bool(settings.robinhood_mcp_auth_header)
badge_html = ""
if has_auth:
    badge_html = '<span class="dot-green"></span><b>LIVE — AGENTIC ACCOUNT</b>'
else:
    badge_html = '<span class="dot-amber"></span><b>MOCK DATA (NO BROKER)</b>'
st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)

# Watchlist summary
if watchlist_data.get("symbols"):
    st.caption(f"Watchlist ({len(watchlist_data['symbols'])} symbols): {', '.join(watchlist_data['symbols'])}")

st.divider()

# ── Equity Curve Chart ────────────────────────────────────────
st.subheader("Equity Curve")
eq_data = None
try:
    if equity_path.exists() and equity_path.stat().st_size > 0:
        df_eq = pd.read_csv(equity_path, names=["timestamp", "equity"], header=0)
        if len(df_eq) > 1:
            fig = go.Figure(go.Scatter(x=df_eq["timestamp"], y=df_eq["equity"], mode="lines+markers"))
            fig.update_layout(height=280, title="", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.warning("No equity data available yet — start the engine to begin tracking.")

st.divider()

# ── Positions Table ───────────────────────────────────────────
st.subheader("Open Positions")
if positions:
    pos_rows = []
    for sym, pos in positions.items():
        entry_price = pos.get("entry_price", 0)
        qty = pos.get("qty", 0)
        stop = pos.get("stop", 0)
        target = pos.get("target", 0)
        entry_ts = pos.get("entry_ts", "")
        pos_rows.append({
            "Symbol": sym,
            "Qty": qty,
            "Entry Price": f"${entry_price:.2f}",
            "Stop Loss": f"${stop:.2f}" if stop else "—",
            "Target": f"${target:.2f}" if target else "—",
        })
    pos_df = pd.DataFrame(pos_rows)
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
else:
    st.info("No open positions yet — they appear once the engine starts trading.")

st.divider()

# ── Recent Trades ─────────────────────────────────────────────
st.subheader("Recent Trades")
recent_trades = None
total_trades = 0
if trades_path.exists() and trades_path.stat().st_size > 0:
    try:
        df_trade = pd.read_csv(trades_path, header=0)
        total_trades = len(df_trade)
        recent = df_trade.tail(20).reset_index(drop=True)
        # Convert timestamps for display
        if "timestamp" in recent.columns:
            recent["timestamp"] = pd.to_datetime(recent["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        recent_trades = recent
        st.dataframe(recent[["timestamp", "symbol", "side", "qty", "price", "reason"]], 
                     use_container_width=True, hide_index=True)
    except Exception:
        pass

if not recent_trades and total_trades == 0:
    st.info("No trade history yet — start a backtest or engine run to generate trades.")
elif recent_trades:
    st.caption(f"Showing last 20 of {total_trades} total trades.")

st.divider()

# ── Engine Status ─────────────────────────────────────────────
st.subheader("Engine Info")
if state:
    last_cycle = state.get("last_cycle_ts", "")
    cols_e = st.columns(4)
    cols_e[0].metric("Strategy", state.get("strategy", "N/A"))
    cols_e[1].metric("Kill Switch", "TRIPPED" if is_tripped else "ARMED")
    cols_e[2].metric("Last Cycle", last_cycle[:19] if last_cycle else "Never")
    cols_e[3].metric("Day Start Equ.", f"${state.get('day_start_equity', 0):,.0f}")
else:
    st.info("Engine hasn't been started yet. Run `python main.py dry-run` or `python main.py live` to begin.")
