"""Log viewer page with emergency stop."""

import time
import streamlit as st
from pathlib import Path

from ui.theme import inject_css, auto_refresh

inject_css()

st.set_page_config(page_title="Logs", layout="wide")
st.title("📄 Logs")

level_filter = st.selectbox("Log Level", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

log_path = Path("logs/bot.log")
if log_path.exists():
    lines = log_path.read_text().splitlines()[-500:]
    if level_filter != "ALL":
        lines = [l for l in lines if f"| {level_filter}| " in l]
    
    st.code("\n".join(lines), language="text")
else:
    st.info("No log file found yet.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    if st.button("EMERGENCY STOP ⛔", type="primary"):
        Path("logs/kill_switch.flag").touch()
        st.success("Kill-switch engaged. Trading halted.")
with col2:
    if st.button("Re-arm ✅"):
        flag = Path("logs/kill_switch.flag")
        if flag.exists():
            flag.unlink()
            st.success("Kill-switch re-armed.")
        else:
            st.info("Already armed.")
