"""Settings page — view/edit engine settings, strategy confirmation, MCP info."""

import json
import time
from pathlib import Path

import streamlit as st
from typing import Any

from ui.theme import inject_css

inject_css()

st.set_page_config(page_title="Settings", layout="wide")
st.title("⚙️ Settings & Configuration")

auto_refresh = st.toggle("Auto-refresh", value=True, key="settings_autorefresh")
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Current Settings ────────────────────────────────────────────
with st.spinner("Loading configuration..."):
    try:
        from bot.config import load_settings
        settings = load_settings()
        
        st.subheader("📋 Active Settings")
        s_cols = st.columns(4)
        
        setting_items = [
            ("MCP URL", settings.robinhood_mcp_url),
            ("Symbols", ", ".join(settings.symbols)),
            ("Risk/Trade", f"{settings.risk_per_trade*100:.1f}%"),
            ("Max Daily Loss", f"{settings.max_daily_loss_pct:.1f}%"),
            ("Engine Interval", f"{settings.engine_interval_minutes} min"),
            ("Log Level", settings.log_level),
            ("Cash (Backtest)", f"${settings.cash:,}"),
            ("Auth Mode", "SSE" if settings.robinhood_mcp_auth_header else ("stdio" if settings.robinhood_mcp_command else "none")),
        ]
        
        for i, (label, val) in enumerate(setting_items):
            col = s_cols[i % len(s_cols)]
            col.metric(label, str(val))
            
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        settings = None

st.divider()

# ── Strategy Confirmation ───────────────────────────────────────
st.subheader("🎯 Strategy Confirmation")

confirmed_path = Path("logs/strategy_confirmed.json")
if confirmed_path.exists():
    try:
        confirmation = json.loads(confirmed_path.read_text())
        c_cols = st.columns(4)
        c_cols[0].metric("Strategy", confirmation.get("strategy", "N/A"))
        c_cols[1].metric("Symbols", ", ".join(confirmation.get("symbols", [])))
        params = confirmation.get("params", {})
        param_summary = ", ".join([f"{k}={v}" for k, v in list(params.items())[:4]])
        c_cols[2].metric("Key Params", param_summary or "—")
        c_cols[3].metric("Confirmed At", confirmation.get("confirmed_at", "unknown")[:16])
        
        st.success("✅ Live trading enabled — strategy is active.")
        st.caption("To change strategy, stop the engine and re-run `python main.py live --strategy <new>`")
    except Exception as e:
        st.error(f"Could not read confirmation: {e}")
else:
    st.warning("⚠️ No strategy confirmed yet. Run `python main.py live --strategy ema_cross_rsi` to confirm.")

st.divider()

# ── Registered Strategies ───────────────────────────────────────
st.subheader("📊 Registered Strategies")

try:
    from bot.core.plugins import discover_all
    from bot.core import STRATEGIES
    
    discover_all()
    
    strat_names = STRATEGIES.names()
    if strat_names:
        st.write(f"**{len(strat_names)} strategies** registered:")
        for name in strat_names:
            obj = STRATEGIES.get(name)
            params = getattr(obj, 'params', {})
            param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
            st.markdown(f"- `{name}` params: {param_str or '(default)'}, has `to_backtesting_strategy`: {hasattr(obj, 'to_backtesting_strategy')}")
    else:
        st.info("No strategies discovered.")
except Exception as e:
    st.error(f"Error listing strategies: {e}")

st.divider()

# ── Data Sources ────────────────────────────────────────────────
st.subheader("🔌 Data Sources")

try:
    from bot.core import DATASOURCES
    
    ds_names = DATASOURCES.names()
    if ds_names:
        d_cols = st.columns(len(ds_names))
        for i, name in enumerate(ds_names):
            ds_obj = DATASOURCES.get(name)
            priority = getattr(ds_obj, 'priority', 'N/A')
            supports = hasattr(ds_obj, 'supports')
            d_cols[i % len(d_cols)].metric(f"Source: {name}", f"Priority: {priority}")
    else:
        st.info("No data sources registered.")
except Exception as e:
    st.error(f"Error listing data sources: {e}")

st.divider()

# ── Engine State ────────────────────────────────────────────────
st.subheader("🖥️ Engine Status")

state_path = Path("logs/engine_state.json")
if state_path.exists() and state_path.stat().st_size > 0:
    try:
        state = json.loads(state_path.read_text())
        e_cols = st.columns(4)
        e_cols[0].metric("Mode", state.get("mode", "unknown").upper())
        e_cols[1].metric("Strategy", state.get("strategy", "N/A"))
        e_cols[2].metric("Last Cycle", state.get("last_cycle_ts", "never")[:19] if state.get("last_cycle_ts") else "never")
        e_cols[3].metric("Kill Switch", "TRIPPED 🔴" if state.get("kill_switch") else "ARMED 🟢")
        
        eq = state.get("day_start_equity", 0)
        if eq:
            st.metric("Day Start Equity", f"${eq:,.2f}")
    except Exception as e:
        st.error(f"Error reading engine state: {e}")
else:
    st.info("Engine hasn't been started yet.")

st.divider()

# ── MCP Server Info ─────────────────────────────────────────────
st.subheader("🤖 MCP Server Info")

try:
    from bot.fastmcp.server import mcp
    import asyncio
    
    tool_count = len(asyncio.run(mcp.list_tools()))
    mc_cols = st.columns(3)
    mc_cols[0].metric("Total MCP Tools", str(tool_count))
    mc_cols[1].metric("Transport", "stdio")
    mc_cols[2].metric("Server", "stocktradingbot v0.1")
    
    # List key tool categories
    tools = asyncio.run(mcp.list_tools())
    categories = {}
    for t in tools:
        cat = t.name.split("_")[0].title()
        categories.setdefault(cat, []).append(t.name)
    
    with st.expander("📋 Tool Categories"):
        for cat, names in sorted(categories.items()):
            st.write(f"**{cat}** ({len(names)} tools): {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
except Exception as e:
    st.error(f"Error loading MCP info: {e}")

st.divider()

# ── Watchlist Management ────────────────────────────────────────
st.subheader("📝 Watchlist Management")

wl_path = Path("logs/watchlist.json")
try:
    wl_data = json.loads(wl_path.read_text()) if wl_path.exists() else {"symbols": []}
    wl_symbols = wl_data.get("symbols", [])
    
    w_cols = st.columns(2)
    w_cols[0].selectbox("Add Symbol", options=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META"], key="wl_add", index=0)
    w_cols[1].button("➕ Add to Watchlist", use_container_width=True, type="primary")
    
    if wl_symbols:
        st.markdown("**Current watchlist:**")
        for sym in wl_symbols:
            st.markdown(f"- `{sym}`")
    else:
        st.caption("Watchlist is empty.")
except Exception:
    st.caption("Watchlist file could not be read.")

st.divider()

# ── Logs & Files ────────────────────────────────────────────────
st.subheader("📁 Project Files")

root_dir = Path(__file__).parent.parent.parent
subdirs = ["bot/", "ui/", "docs/", "references/", "tests/", "data/", "reports/", "logs/"]

for subdir in subdirs:
    p = root_dir / subdir
    if p.exists():
        count = sum(1 for _ in p.glob("**/*") if _.is_file())
        total_size = sum(_.stat().st_size for _ in p.glob("**/*") if _.is_file())
        size_str = f"{total_size/1024:.0f} KB" if total_size > 0 else "—"
        st.metric(subdir.rstrip("/"), f"{count} files • {size_str}")

# ── System Info ─────────────────────────────────────────────────
st.divider()
st.subheader("💻 System Information")

import sys
s_cols = st.columns(4)
s_cols[0].metric("Python Version", sys.version.split()[0])
s_cols[1].metric("Executable", sys.executable)
s_cols[2].metric("Platform", sys.platform)
s_cols[3].metric("Working Dir", str(Path.cwd()).replace("\\", "/"))

st.caption(f"Project root: {str(root_dir).replace(chr(92), '/')}")
st.caption("Built with Python 3.13 | Streamlit | Plotly | backtesting.py | FastMCP")
