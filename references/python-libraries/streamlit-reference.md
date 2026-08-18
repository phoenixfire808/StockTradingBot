# Streamlit Reference

**Package**: `streamlit >= 1.36.0`

## Multi-Page App Layout

Place pages in `ui/pages/` directory — each file becomes a nav link:

```
ui/app.py           → Home page (main entry point)
ui/pages/1_📊_Dashboard.py
ui/pages/2_🧪_Backtest.py
ui/pages/3_🔬_Research.py
```

File naming convention: `N_Shortname.py` — sort order determines nav sidebar order. Use emoji prefixes for visual distinction.

## Dark Theme Configuration

Create `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#00C896"       # Robinhood green accent
backgroundColor = "#0E1117"    # Very dark background
secondaryBackgroundColor = "#1A1F2E"   # Card backgrounds
textColor = "#E6E9EF"         # Light text
font = "monospace"            # Terminal feel

[server]
headless = true
port = 8501
```

## Auto-Refresh Pattern
```python
import time
enabled = st.toggle("Auto-refresh", value=True)
if enabled:
    time.sleep(5)
    st.rerun()
```

## Common UI Components Used

| Component | Purpose |
|-----------|---------|
| `st.title()`, `st.subheader()` | Page headings |
| `st.metric(label, value, delta)` | KPI cards (equity, returns) |
| `st.dataframe(df)` | Sortable/filterable table view |
| `st.table(df)` | Read-only table (faster for static) |
| `st.plotly_chart(fig)` | Interactive Plotly charts |
| `st.code(text)` | Monospaced code/log output |
| `st.button(..., type="primary")` | Action buttons (red for danger) |
| `st.tabs([...])` | Tabbed sections |
| `st.columns(n)` | Horizontal layout columns |
| `st.selectbox()`, `st.multiselect()` | Dropdown selectors |
| `st.date_input()` | Date pickers |
| `st.number_input()` | Numeric inputs |
| `st.download_button()` | CSV export/download |
| `st.page_link()` | Navigation between pages |
| `st.toggle()` | Toggle switches |
| `st.progress(value)` | Progress bars during long ops |
| `st.spinner()` | Loading indicator |
| `st.warning()`, `st.success()`, `st.error()` | Status messages |

## CSS Injection
```python
st.markdown("""
<style>
.pnl-pos { color: #00C896; font-weight: bold; }
.pnl-neg { color: #FF5252; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
```

## Session State
```python
val = st.session_state.get("key", default_value)
st.session_state["key"] = new_value
```

Useful for persisting filter selections across autorefresh cycles.

## Launch
```bash
streamlit run ui/app.py --server.port 8501 --server.headless=true
```

Or via CLI: `python main.py ui` (subprocess invocation).
