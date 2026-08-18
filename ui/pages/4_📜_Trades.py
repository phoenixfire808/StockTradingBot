"""Trade history page."""

import pandas as pd
import streamlit as st
from pathlib import Path

from ui.theme import inject_css

inject_css()

st.set_page_config(page_title="Trades", layout="wide")
st.title("📜 Trade History")

trades_path = Path("logs/trades.csv")
if trades_path.exists() and trades_path.stat().st_size > 0:
    df = pd.read_csv(trades_path)
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv_bytes, file_name="trades.csv", mime="text/csv")
else:
    st.info("No trades recorded yet. Start the engine to log trades.")
