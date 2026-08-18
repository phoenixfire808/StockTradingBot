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
- [x] **P9** Streamlit UI pages fully implemented: Backtest (with Run button + comparison mode), Research (ticker search + chart + fundamentals + sentiment gauge), Trades (journal with PnL colors + filters), Settings (engine status + strategy confirmation + MCP info)
- [x] **P10** Integration test suite: 98 tests covering strategy compare, walk-forward optimization, multi-symbol batch, engine lifecycle, sentiment aggregation, watchlist management chain, all MCP tools structure verification, data pipeline caching
- [x] **P11** CodeGraph indexing enabled for project (966 nodes, 1,659 edges)

## In Progress

_Next session priorities below._

## Next Phase — Advanced Features

- [ ] **P12** Multi-strategy portfolio allocation with Kelly criterion weighting
- [ ] **P13** Sentiment-filtered strategy (use VADER scores as signal modifier)
- [ ] **P14** Walk-forward optimization integration with CLI and MCP tool
- [ ] **P15** Per-strategy equity tracking & portfolio allocation dashboard in Streamlit
- [ ] **P16** Options trading support (Robinhood Agentic account supports options)
- [ ] **P17** Crypto trading support (Robinhood Agentic account supports crypto)

## Future — Infrastructure Hardening

- [ ] **P18** Service autostart (Windows scheduled task / service) for persistent engine operation
- [ ] **P19** Discord / email alerts on fills, kill-switch events, large drawdowns
- [ ] **P20** SQLite trade store replacing CSV files (concurrent-safe, queryable)
- [ ] **P21** True tick data plugin (Polygon.io or Databento)
- [ ] **P22** Docker containerization for cross-platform deployment

## Future — ML Signals

- [ ] **P23** Gradient-boosted feature model as signal filter (XGBoost/LightGBM)
- [ ] **P24** Feature pipeline: technical + fundamental + sentiment features
- [ ] **P25** Out-of-sample validation only — never deploy untested models

## Future — Scaling

- [ ] **P26** Capital review process after ≥30 active trading days
- [ ] **P27** Per-strategy budget limits
- [ ] **P28** Multi-timeframe analysis (1m, 5m, 1h candles across same symbols)
- [ ] **P29** Sector rotation detection and automatic symbol re-ranking

## Live Trading Protocol

Live trading shipped in P2. No paper-first gating by design. The following apply:

1. **Strategy confirmation**: One-time prompt at `live` startup confirming strategy name, params, symbols, risk settings. Saved to `logs/strategy_confirmed.json`.
2. **Blast radius**: Robinhood Agentic account isolation — agent writes confined to dedicated sub-account with user-funded capital only.
3. **Risk controls always active**: Daily loss kill switch (configurable, default 3%), position sizing cap (25% equity), stop-loss per trade.
4. **Emergency stop**: UI button in Logs page writes `logs/kill_switch.flag`; engine checks this every cycle.
5. **Scaling gates**: When expanding to additional strategies or larger capital allocations, run walk-forward optimization and paper-test new additions in dry-run mode first.
