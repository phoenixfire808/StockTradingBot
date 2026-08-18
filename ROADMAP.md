# Roadmap — Stock Trading Bot

## Completed

- [x] **P0** Scaffold + repo staging + research vault (`references/`)
- [x] **P1** Core framework: registry, plugin auto-discovery, config, logging
- [x] **P2** Backend modules: data hub (yfinance + Robinhood MCP), indicators, strategy engine, risk manager, broker abstraction, sentiment engine, live engine with full account management
- [x] **P3** Streamlit UI — Dashboard, Backtest, Research, Trades, Logs, Settings pages
- [x] **P4** Docs: architecture, setup, troubleshooting, all API/tool references saved to `references/`
- [x] **P5** GitHub repo created at https://github.com/phoenixfire808/StockTradingBot (private)
- [x] **P6** Strategy expansion: VWAP Breakout, Bollinger Reversion, Momentum Scanner plugins
- [x] **P7** Engine enhancements: daily rebalance at market open, trade journal P&L calculations, signal logging
- [x] **P8** MCP tool suite: backtest_compare, walk_forward_optimize, daily_rebalance, trade_journal_pnl, signal_export_csv, performance_dashboard (85 total tools)
- [x] **P9** Streamlit UI pages fully implemented: Backtest (with Run button + comparison mode), Research (ticker search + chart + fundamentals + sentiment gauge), Trades (journal with PnL colors + filters), Settings (engine status + strategy confirmation + MCP info), Portfolio (allocation dashboard)
- [x] **P10** Integration test suite: 98 tests covering strategy compare, walk-forward optimization, multi-symbol batch, engine lifecycle, sentiment aggregation, watchlist management chain, all MCP tools structure verification, data pipeline caching
- [x] **P11** CodeGraph indexing enabled for project (966 nodes, 1,659 edges)
- [x] **P12** Multi-strategy portfolio allocation with Kelly criterion weighting — `bot/portfolio.py` (Kelly / equal-weight / risk-parity allocators)
- [x] **P13** Sentiment-filtered strategy — `bot/plugins/strategies/sentiment_filtered.py`
- [x] **P14** Walk-forward optimization — `bot/optimization.py` + MCP `walk_forward_optimize`
- [~] **P15** Per-strategy equity tracking & portfolio allocation dashboard — `bot/equity_tracker.py` + Streamlit `Portfolio` page shipped; pie chart, Kelly fractions, risk-parity breakdown. **Partial**: dashboard links to `logs/equity_by_strategy.json` written by manual/optional path; auto-feeding the snapshot every engine cycle still being wired.
- [x] **P16** Options trading support — `OptionOrder` / `OptionChain` dataclasses + `submit_option_order` / `get_option_chain` in `bot/broker.py`; re-exported via `bot/options.py`
- [x] **P17** Crypto trading support — `CryptoOrder` / `CryptoQuote` + `submit_crypto_order` / `get_crypto_quotes` in `bot/broker.py`; yfinance crypto datasource `bot/plugins/datasources/yfinance_crypto.py`
- [x] **P19** Discord / email alerts on fills, kill-switch events, large drawdowns — `bot/alerts.py`
- [x] **P20** SQLite trade store replacing CSV files — `bot/trade_store.py` (concurrent-safe, queryable)
- [x] **P21** Polygon.io datasource plugin — `bot/plugins/datasources/polygon.py`
- [x] **P22** Databento datasource plugin — `bot/plugins/datasources/databento.py`
- [x] **P22b** Docker containerization — multi-stage `Dockerfile` + `docker-compose.yml` (Streamlit healthcheck, volumes for logs/data/reports)
- [x] **P23** Gradient-boosted feature model as signal filter — `bot/ml/model.py`
- [x] **P24** Feature pipeline: technical + fundamental + sentiment features — `bot/ml/features.py`
- [x] **P25** Out-of-sample validation gate — `bot/ml/validation.py` (walk-forward / hold-out check before deploy)
- [x] **P28** Multi-timeframe analysis — `bot/multi_timeframe.py` + `bot/plugins/strategies/multi_timeframe.py`
- [x] **P29** Sector rotation detection — `bot/sector_rotation.py` + `bot/plugins/strategies/sector_rotation.py`

## In Progress

- [~] **P15** (carry-over from Completed list — see partial note above)

_Next session priorities below. P18 / P26 / P27 still pending._

## Future — Infrastructure Hardening

- [ ] **P18** Service autostart (Windows scheduled task / service) for persistent engine operation

## Future — Scaling

- [ ] **P26** Capital review process after ≥30 active trading days
- [ ] **P27** Per-strategy budget limits

## Live Trading Protocol

Live trading shipped in P2. No paper-first gating by design. The following apply:

1. **Strategy confirmation**: One-time prompt at `live` startup confirming strategy name, params, symbols, risk settings. Saved to `logs/strategy_confirmed.json`.
2. **Blast radius**: Robinhood Agentic account isolation — agent writes confined to dedicated sub-account with user-funded capital only.
3. **Risk controls always active**: Daily loss kill switch (configurable, default 3%), position sizing cap (25% equity), stop-loss per trade.
4. **Emergency stop**: UI button in Logs page writes `logs/kill_switch.flag`; engine checks this every cycle.
5. **Scaling gates**: When expanding to additional strategies or larger capital allocations, run walk-forward optimization and paper-test new additions in dry-run mode first.