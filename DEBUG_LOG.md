# Debug Log

Build session started: 2026-08-18

## Environment

| Component | Version |
|-----------|---------|
| OS | Windows 11 Pro (NT 10.0.26100) |
| Python | 3.13.3 |
| pip | 25.3 |
| git | 2.49.0.windows.1 |
| gh CLI | 2.74.1 |

## Installed Packages (to be filled after `pip install -r requirements.txt`)

| Package | Version | Status |
|---------|---------|--------|
| mcp | _pending_ | |
| httpx[sse] | _pending_ | |
| yfinance | _pending_ | |
| backtesting | _pending_ | |
| pandas-ta | _pending_ | Python 3.13 compatible (need 0.4.4b0+) |
| pandas | _pending_ | |
| numpy | _pending_ | |
| python-dotenv | _pending_ | |
| APScheduler | _pending_ | |
| pytest | _pending_ | |
| streamlit | _pending_ | |
| plotly | _pending_ | |
| vaderSentiment | _pending_ | |
| requests | _pending_ | |

## Indicator Path Taken

_TBD — determined at first import of bot.indicators: whether pandas_ta was available or manual fallback used._

## Robinhood MCP Discovered Schema

_TBD — captured on first call to `session.list_tools()` during implementation._

## Issue Log

| Date | Issue | Resolution |
|------|-------|------------|
| 2026-08-18 | Git "dubious ownership" error on Windows file system | Added `git config --global --add safe.directory D:/StockTradingBot` |
| 2026-08-18 | Scout workers failed with "No model selected" | All work executed directly instead of delegated |
| 2026-08-18 | pandas-ta version conflict on Python 3.13 | Conditional requirements in requirements.txt; manual fallback in indicators.py |

## Decisions Recorded

1. Broker: Robinhood Agentic Trading MCP (official server at `https://agent.robinhood.com/mcp/trading`)
2. Auth method: Platform OAuth token extracted from AI platform session (no standalone API key)
3. Execution mode: Live trading from day one, strategy-confirmation-only gate (per Drew's explicit instruction)
4. Framework: Modular plugin system with auto-discovery via importlib scan
5. UI: Streamlit multi-page app, dark trading-terminal theme (#00C896 accent)
6. Data: yfinance as primary free datasource; Robinhood MCP historical as premium fallback
