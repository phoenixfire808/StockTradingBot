# Project Tracker

## Status: ✅ Scaffold Complete

All source code, documentation, references, and GitHub staging committed. Ready for dependency installation and smoke testing.

| Item | Status | Notes |
|------|--------|-------|
| Repository staged | ✅ | https://github.com/phoenixfire808/StockTradingBot (private) |
| Plugin framework | ✅ | Registry + auto-discovery across strategies/datasources/sentiment sources |
| Config + logging | ✅ | `.env` loading, rotating file logs |
| Data layer | ✅ | yfinance + Robinhood MCP datasource plugins + CSV cache |
| Indicators | ✅ | pandas-ta + pure-pandas fallback |
| Strategy engine | ✅ | Strategy ABC + EmaCrossRsi plugin |
| Risk management | ✅ | Position sizing, stops/targets, KillSwitch |
| Broker abstraction | ✅ | RobinhoodMcpBroker (SSE/stdio) + MockBroker |
| Backtester | ✅ | backtesting.py integration returning metrics dict |
| Sentiment engine | ✅ | VADER scoring + StockTwits/Reddit plugins |
| Live engine | ✅ | Full account management with position reconciliation |
| CLI | ✅ | backtest, dry-run, live, sentiment, ui subcommands |
| Streamlit UI | ✅ | 6 pages: Dashboard, Backtest, Research, Trades, Logs, Settings |
| References vault | ✅ | 20+ reference files covering all APIs, tools, commands |
| Docs | ✅ | Architecture, setup, troubleshooting |
| Tests | 🔄 | Stub directory ready — unit tests to be written during smoke testing |

## Last Session Summary

- Built complete modular plugin framework with auto-discovery
- Implemented Robinhood MCP broker (both SSE remote and local stdio modes)
- Created sentiment crawler with StockTwits and Reddit public endpoints
- Built Streamlit dark-terminal UI with emergency stop functionality
- Committed comprehensive `references/` vault with all API docs
- Staged private GitHub repo at phoenixfire808/StockTradingBot

## Next Session First Action

1. Install dependencies: `pip install -r requirements.txt`
2. Verify imports: `python -c "import bot.config; print('OK')"`
3. Dry-run test: `python main.py dry-run` (Ctrl+C after ~30s)
4. UI launch test: `python main.py ui` → check http://localhost:8501

## Known Issues

- pandas-ta version compatibility with Python 3.13 + numpy 2: resolved via conditional requirement + manual fallback
- Robinhood MCP granularity parameters unverified: implementer should call `session.list_tools()` at first use and log discovered schema
- Subagent capacity limited to ~2 workers: build executed directly rather than delegated
