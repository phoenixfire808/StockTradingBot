# Project Tracker

## Status: ✅ Feature Expansion Complete — Production Hardening Next

All P0–P17, P19–P25, P28–P29 modules are implemented (P15 is partial). P18 autostart, P26 capital review, and P27 per-strategy budget limits remain. Source code, documentation, references, and GitHub staging committed.

| Item | Status | Notes |
|------|--------|-------|
| Repository staged | ✅ | https://github.com/phoenixfire808/StockTradingBot (private) |
| Plugin framework | ✅ | Registry + auto-discovery across strategies/datasources/sentiment sources |
| Config + logging | ✅ | `.env` loading, rotating file logs |
| Data layer | ✅ | yfinance + Robinhood MCP datasource plugins + CSV cache |
| Datasources — premium | ✅ | Polygon.io (`bot/plugins/datasources/polygon.py`), Databento (`bot/plugins/datasources/databento.py`), yfinance crypto (`yfinance_crypto.py`) |
| Indicators | ✅ | pandas-ta + pure-pandas fallback |
| Strategy engine | ✅ | Strategy ABC + EmaCrossRsi, VWAP Breakout, Bollinger Reversion, Momentum Scanner plugins + ML signal plugins (`ml_hybrid`, `ml_signal_filter`) |
| Strategies — advanced | ✅ | Sentiment-filtered (`sentiment_filtered.py`), multi-timeframe (`multi_timeframe.py`), sector rotation (`sector_rotation.py`), ML-driven `ml_hybrid` + `ml_signal_filter` (gradient-boosted signal + filter plugins) |
| Risk management | ✅ | Position sizing, stops/targets, KillSwitch |
| Broker abstraction | ✅ | RobinhoodMcpBroker (SSE/stdio) + MockBroker + `OptionOrder`/`OptionChain`/`CryptoOrder`/`CryptoQuote` dataclasses |
| Options & crypto orders | ✅ | `submit_option_order` / `get_option_chain` / `submit_crypto_order` / `get_crypto_quotes` implemented in `bot/broker.py` |
| Backtester | ✅ | backtesting.py integration returning metrics dict |
| Walk-forward optimization | ✅ | `bot/optimization.py` + MCP `walk_forward_optimize` |
| Portfolio allocation | ✅ | `bot/portfolio.py` — Kelly (fractional), equal-weight, risk-parity |
| Per-strategy equity tracking | ⚠️ Partial | `bot/equity_tracker.py` (CSV+JSON persistence) + Streamlit `Portfolio` page; auto-feeding the snapshot every engine cycle still being wired |
| ML signals | ✅ | `bot/ml/model.py` (gradient-boosted classifier), `features.py` (tech+fund+sent pipeline), `validation.py` (out-of-sample gate) |
| Alerts | ✅ | `bot/alerts.py` — Discord / email on fills, kill-switch, drawdowns |
| Trade store | ✅ | `bot/trade_store.py` — SQLite, concurrent-safe, queryable |
| Sentiment engine | ✅ | VADER scoring + StockTwits/Reddit plugins |
| Live engine | ✅ | Full account management with position reconciliation |
| Docker | ✅ | Multi-stage `Dockerfile` + `docker-compose.yml` (healthcheck, volumes) |
| CLI | ✅ | backtest, dry-run, live, sentiment, ui subcommands |
| Streamlit UI | ✅ | 7 pages: Dashboard, Backtest, Research, Trades, Logs, Settings, Portfolio |
| References vault | ✅ | 20+ reference files covering all APIs, tools, commands |
| Docs | ✅ | Architecture, setup, troubleshooting |
| Tests | ✅ | `tests/` covers each new module: portfolio, options, crypto, trade_store, polygon, databento, alerts, ml_features, ml_model, ml_validation, multi_timeframe, sector_rotation, optimization, ui_portfolio, docker, fastmcp_server, dry_run_symbols, dry_run_duration, ml_hybrid_strategy, ml_signal_strategy |

## Last Session Summary

- **P12** Portfolio allocation — `bot/portfolio.py`: Kelly (fractional, mean/var with min-samples guard), equal-weight, risk-parity; integrates with `bot.equity_tracker`
- **P13** Sentiment-filtered strategy plugin — `bot/plugins/strategies/sentiment_filtered.py`
- **P14** Walk-forward optimization — `bot/optimization.py` (rolling-window grid search, IS/OOS split, equity-curve output) + MCP `walk_forward_optimize` wired
- **P15** Per-strategy equity tracker — `bot/equity_tracker.py` (per-strategy CSV + aggregate JSON); **partial** because auto-feed into live engine not yet on the hot path
- **P16/P17** Options + crypto — dataclasses + broker methods (`submit_option_order`, `get_option_chain`, `submit_crypto_order`, `get_crypto_quotes`) in `bot/broker.py`; re-export shim `bot/options.py`
- **P19** Alerts — `bot/alerts.py`: Discord webhook + SMTP email, triggered on fills / kill-switch / drawdown thresholds
- **P20** Trade store — `bot/trade_store.py`: SQLite (WAL mode, parameterized queries, replacement for CSV `trades.csv`)
- **P21/P22** Premium datasources — Polygon + Databento plugins under `bot/plugins/datasources/`
- **P22b** Docker — multi-stage build, compose healthcheck, persistent volumes for logs/data/reports
- **P23–P25** ML signals — `bot/ml/` (model, features, validation); validation gates deployment on hold-out metrics
- **P28** Multi-timeframe — `bot/multi_timeframe.py` + matching strategy plugin
- **P29** Sector rotation — `bot/sector_rotation.py` + matching strategy plugin (re-ranks symbols by sector strength)
- Streamlit `Portfolio` page added (`ui/pages/7_📊_Portfolio.py`)
- **ML strategy plugins** — `bot/plugins/strategies/ml_hybrid.py` (gradient-boosted signal fused with technicals) and `ml_signal_filter.py` (ML confidence gates an underlying base strategy); tests `tests/test_ml_hybrid_strategy.py`, `tests/test_ml_signal_strategy.py`
- **Dry-run CLI overrides** — `bot/engine.py` honours `--symbols` and `--duration` overrides from CLI; covered by `tests/test_dry_run_symbols.py` and `tests/test_dry_run_duration.py`
- **`.gitignore` hygiene** — added root-level session/report/markdown exclusions (`STATUS.md`, `TODO.md`, `SESSION_*.md`, `*_REPORT.md`, etc.) and `logs/`, `tests_run*.log`, root `*.log` are now excluded

## Next Session First Action

1. Wire `EquityTracker.record(...)` into the live engine cycle so `logs/equity_by_strategy.json` updates automatically each loop (closes P15).
2. Verify full test suite passes after the auto-feed change.
3. Decide on P18 autostart mechanism (Windows scheduled task vs NSSM service) and start drafting.
4. Run `python main.py ui` to smoke-test the Portfolio page once P15 auto-feed is live.

## Known Issues

- pandas-ta version compatibility with Python 3.13 + numpy 2: resolved via conditional requirement + manual fallback.
- Robinhood MCP granularity parameters unverified: implementer should call `session.list_tools()` at first use and log discovered schema.
- Subagent capacity limited to ~2 workers: build executed directly rather than delegated (2026-08-18 log).
- **P15 partial**: `EquityTracker` is implemented but the live engine does not yet auto-call `record()` per cycle; the Portfolio UI relies on manually-written or older snapshots.
- Docker image build path unverified on Windows host (Dockerfile targets linux/amd64 via BuildKit syntax).