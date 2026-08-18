"""Dashboard page — equity metrics, positions, trades overview."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from ui.theme import inject_css, kill_switch_banner, get_positions_file

inject_css()

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard")

# Kill switch check
flag_file = Path("logs/kill_switch.flag")
is_tripped = flag_file.exists()
kill_switch_banner(is_tripped)

st.divider()

# Equity curve
eq_path = Path("logs/equity_history.csv")
if eq_path.exists() and eq_path.stat().st_size > 0:
    try:
        df = pd.read_csv(eq_path, names=["timestamp", "equity"])
        fig = go.Figure(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines+markers"))
        fig.update_layout(height=300, title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Could not render equity history")
else:
    st.info("No equity data yet. Start the engine to begin tracking.")

st.divider()
st.write("Data from backend modules only — no market prices available yet.")
