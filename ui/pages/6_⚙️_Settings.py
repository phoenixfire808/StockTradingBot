"""Settings & configuration page."""

import json
import streamlit as st

from ui.theme import inject_css, broker_badge

inject_css()

st.set_page_config(page_title="Settings", layout="wide")
st.title("⚙️ Settings")

st.info("Configuration loaded from `.env`. Modify `.env.example` and copy to `.env`.")

st.divider()

# Plugin registry info
st.subheader("Discovered Plugins")
st.info("Plugins are auto-discovered from bot/plugins/*/. Add new plugins by dropping a Python file there.")

st.divider()

# Strategy confirmation display
confirm_path = __import__('pathlib').Path("logs/strategy_confirmed.json")
if confirm_path.exists():
    try:
        data = json.loads(confirm_path.read_text())
        st.json(data)
    except Exception:
        st.warning("Could not read strategy confirmation file.")
else:
    st.info("No live strategy confirmed yet. Use: `python main.py live --strategy ema_cross_rsi`")

st.divider()

# Broker badge
st.markdown(f"**Broker Mode:** {broker_badge()}")
