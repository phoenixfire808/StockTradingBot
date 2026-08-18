"""Stock Trading Bot Dashboard — Home Page."""

import streamlit as st

from ui.theme import inject_css, broker_badge, get_engine_status, status_dot

inject_css()

st.set_page_config(page_title="StockTradingBot", layout="wide", page_icon="📈")

st.title("Stock Trading Bot")

# Header row
status = get_engine_status()
engine_mode = ""
if status:
    mode = status.get("mode", "unknown")
    if mode == "live":
        engine_mode = f'<div style="text-align:right">{status_dot("red")} Live</div>'
    elif mode == "dry-run":
        engine_mode = f'<div style="text-align:right">{status_dot("amber")} Dry-Run</div>'
    else:
        engine_mode = '<div style="text-align:right">Engine not running</div>'
else:
    engine_mode = '<div style="text-align:right">Engine not running</div>'

st.markdown(f"**Broker:** {broker_badge()} &nbsp;&nbsp; **State:** {engine_mode}", unsafe_allow_html=True)

st.divider()

# Navigation
st.subheader("Navigation")
cols = st.columns(7)
urls = {
    "📊 Dashboard": "pages/1_📊_Dashboard.py",
    "🧪 Backtest": "pages/2_🧪_Backtest.py",
    "🔬 Research": "pages/3_🔬_Research.py",
    "📜 Trades": "pages/4_📜_Trades.py",
    "📄 Logs": "pages/5_📄_Logs.py",
    "🥧 Portfolio": "pages/7_📊_Portfolio.py",
    "⚙️ Settings": "pages/6_⚙️_Settings.py",
}
for col, (label, _) in enumerate(urls.items()):
    cols[col].page_link(_, label=label.replace("&nbsp;", ""))

st.info("Select a page above to get started.")
