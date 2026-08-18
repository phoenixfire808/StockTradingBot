"""Backtest page — configure strategies/symbols/dates and run backtests."""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Any

from ui.theme import inject_css, pnl_color

inject_css()

st.set_page_config(page_title="Backtest", layout="wide")
st.title("🧪 Backtest Runner")

auto_refresh = st.toggle("Auto-refresh", value=False, key="bt_autorefresh")
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Sidebar Configuration ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Symbols
    all_symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM"]
    selected_symbols = st.multiselect("Symbols", options=all_symbols, default=["AAPL", "MSFT"])
    
    # Dates
    from datetime import date
    start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    end_date = st.date_input("End Date", value=date(2024, 6, 1))
    
    # Cash
    starting_cash = st.number_input("Starting Cash ($)", value=100_000, min_value=1000, step=10000)
    
    # Strategy selection
    from bot.core.plugins import discover_all
    from bot.core import STRATEGIES
    
    discover_all()
    strategy_names = STRATEGIES.names() if STRATEGIES.names() else ["ema_cross_rsi"]
    selected_strategy = st.selectbox("Strategy", options=strategy_names, index=0 if "ema_cross_rsi" in strategy_names else 0)
    
    # Strategy params sliders
    try:
        strat_obj = STRATEGIES.get(selected_strategy)
        default_params = getattr(strat_obj, 'params', {})
        
        st.subheader(f"Parameters: {selected_strategy}")
        param_sliders = {}
        for pname, pdefault in default_params.items():
            if isinstance(pdefault, int):
                param_slivers[pname] = st.slider(pname, int(pdefault * 0.5), int(pdefault * 2), int(pdefault))
            elif isinstance(pdefault, float):
                param_sliders[pname] = st.slider(pname, float(pdefault * 0.5), float(pdefault * 2), float(pdefault), 0.01)
            else:
                param_sliders[pname] = pdefault
    except Exception:
        param_sliders = {}
    
    # Compare mode
    st.divider()
    compare_mode = st.checkbox("Compare with another strategy", value=False)
    if compare_mode:
        compare_strategy = st.selectbox("Compare against", options=[s for s in strategy_names if s != selected_strategy], 
                                         index=0 if strategy_names and len(strategy_names) > 1 else 0)
    
    # Run button
    st.divider()
    run_clicked = st.button("🚀 RUN BACKTEST", type="primary", use_container_width=True)

# ── Results Display ─────────────────────────────────────────────
if run_clicked:
    st.session_state["bt_running"] = True
    st.info("Running backtest... this may take a moment.")
    
    try:
        from bot.backtest import run_backtest, print_backtest_table
        
        results = run_backtest(
            symbols=selected_symbols or ["AAPL"],
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            cash=starting_cash,
            strategy_name=selected_strategy,
            strategy_params=param_sliders,
        )
        
        st.success("✅ Backtest complete!")
        
        # Metrics table
        if results:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Summary Metrics")
                
                headers = ["Symbol", "Total %", "BH %", "Sharpe", "Sortino", "Calmar", "MaxDD %", "Trades", "WinRate %", "P.Factor"]
                data_rows = []
                for sym, res in sorted(results.items()):
                    m = res.get("metrics", {})
                    data_rows.append({
                        "Symbol": sym,
                        "Total Return %": m.get("total_return_pct", 0),
                        "Buy&Hold %": m.get("buy_hold_pct", 0),
                        "Sharpe Ratio": m.get("sharpe_ratio", 0),
                        "Sortino Ratio": m.get("sortino_ratio", 0),
                        "Calmar Ratio": m.get("calmar_ratio", 0),
                        "Max Drawdown %": m.get("max_dd_pct", 0),
                        "Trades": m.get("trades", 0),
                        "Win Rate %": m.get("win_rate_pct", 0),
                        "Profit Factor": m.get("profit_factor", 0),
                    })
                
                if data_rows:
                    df_metrics = pd.DataFrame(data_rows)
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("📁 Artifacts")
                for sym, res in results.items():
                    report_path = Path(f"reports/{sym}_backtest.html")
                    if report_path.exists():
                        size_kb = report_path.stat().st_size / 1024
                        st.metric(label=f"{sym} Report", value=f"{size_kb:.0f} KB")
                
                summary_path = Path("logs/backtest_summary.json")
                if summary_path.exists():
                    st.caption("Summary saved to `logs/backtest_summary.json`")
        
        # Equity curve charts per symbol
        st.divider()
        st.subheader("📈 Equity Curves")
        
        for sym, res in results.items():
            eq_curve = res.get("equity_curve")
            if eq_curve is not None and hasattr(eq_curve, '__len__') and len(eq_curve) > 1:
                fig = go.Figure(go.Scatter(x=eq_curve.index, y=eq_curve["Equity"] if "Equity" in eq_curve.columns else eq_curve.iloc[:, 0], mode="lines"))
                fig.update_layout(title=f"{sym} — {selected_strategy}", height=250, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True, key=f"eq_{sym}")
        
        # Trades table if any trades exist
        st.divider()
        st.subheader("📋 Trade Details")
        
        for sym, res in results.items():
            trades_df = res.get("trades_df")
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                st.markdown(f"**{sym}** ({len(trades_df)} trades)")
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        import traceback
        st.code(traceback.format_exc())

st.session_state["bt_running"] = False

# ── Comparison Mode ─────────────────────────────────────────────
if compare_mode and "compare_strategy" in dir():
    st.divider()
    st.subheader(f"⚔️ Head-to-Head: {selected_strategy} vs {compare_strategy}")
    
    try:
        from bot.backtest import run_backtest
        results_a = run_backtest(selected_symbols, start_date, end_date, starting_cash, selected_strategy, param_sliders)
        results_b = run_backtest(selected_symbols, start_date, end_date, starting_cash, compare_strategy, {})
        
        headers = ["Metric", f"{selected_strategy}", f"{compare_strategy}", "Difference"]
        rows = []
        metric_keys = ["total_return_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_dd_pct", "trades"]
        
        for sym in (set(results_a.keys()) | set(results_b.keys())):
            m_a = results_a.get(sym, {}).get("metrics", {})
            m_b = results_b.get(sym, {}).get("metrics", {})
            for mk in metric_keys:
                val_a = m_a.get(mk, 0) or 0
                val_b = m_b.get(mk, 0) or 0
                diff = round(val_a - val_b, 4)
                color = "green" if diff > 0 else "red" if diff < 0 else "gray"
                rows.append([sym, val_a, val_b, f"[{color}] {diff:+.4f}"])
        
        comp_df = pd.DataFrame(rows, columns=headers)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Comparison error: {e}")
