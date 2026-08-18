"""Backtest runner page."""

import streamlit as st

from ui.theme import inject_css

inject_css()

st.set_page_config(page_title="Backtest", layout="wide")
st.title("🧪 Backtest Runner")

st.info("Use `python main.py backtest --symbols AAPL MSFT --start 2022-01-01` via CLI, or select symbols below:")

sym_sel = st.multiselect("Symbols", options=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"], default=["AAPL"])
start_date = st.date_input("Start Date", value=None)
cash = st.number_input("Starting Cash", value=100_000, min_value=1000)

if st.button("Run Backtest"):
    st.info("Use CLI command: python main.py backtest --symbols %s --cash %d" % (", ".join(sym_sel), cash))
