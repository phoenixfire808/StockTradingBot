"""Market research & sentiment page."""

import streamlit as st

from ui.theme import inject_css

inject_css()

st.set_page_config(page_title="Research", layout="wide")
st.title("🔬 Market Research")

ticker = st.text_input("Ticker Symbol", value="AAPL")

if ticker:
    st.info(f"Use `python main.py sentiment {ticker}` via CLI for sentiment analysis, or run backtests for historical data.")
