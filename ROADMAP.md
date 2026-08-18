# Roadmap — Stock Trading Bot

## Completed

- [x] **P0** Scaffold + repo staging + research vault (`references/`)
- [x] **P1** Core framework: registry, plugin auto-discovery, config, logging
- [x] **P2** Backend modules: data hub (yfinance + Robinhood MCP), indicators, strategy engine, risk manager, broker abstraction, sentiment engine, live engine with full account management
- [x] **P3** Streamlit UI — Dashboard, Backtest, Research, Trades, Logs, Settings pages
- [x] **P4** Docs: architecture, setup, troubleshooting, all API/tool references saved to `references/`
- [x] **P5** GitHub repo created at https://github.com/phoenixfire808/StockTradingBot (private)

## In Progress

- [ ] **P6** Dependency install + integration testing on target machine
- [ ] **P7** Smoke test each mode (dry-run, backtest, sentiment, UI launch)

## Next Phase — Operational Hardening

- [ ] **P8** Service autostart (Windows scheduled task / service) for persistent engine operation
- [ ] **P9** Discord / email alerts on fills, kill-switch events, large drawdowns
- [ ] **P10** SQLite trade store replacing CSV files (concurrent-safe, queryable)
- [ ] **P11** True tick data plugin (Polygon.io or Databento)
- [ ] **P12** Per-strategy equity tracking & portfolio allocation dashboard

## Future — Strategy Expansion

- [ ] **P13** Mean-reversion strategy (Bollinger Bands + RSI-2)
- [ ] **P14** Momentum scanner over S&P 100 universe
- [ ] **P15** Sentiment-filtered strategy (use VADER scores as signal modifier)
- [ ] **P16** Walk-forward optimization to fight overfitting
- [ ] **P17** Multi-strategy portfolio allocation with Kelly criterion

## Future — ML Signals

- [ ] **P18** Gradient-boosted feature model as signal filter (XGBoost/LightGBM)
- [ ] **P19** Feature pipeline: technical + fundamental + sentiment features
- [ ] **P20** Out-of-sample validation only — never deploy untested models

## Future — Scaling

- [ ] **P21** Capital review process after ≥30 active trading days
- [ ] **P22** Per-strategy budget limits
- [ ] **P23** Options trading (Robinhood supports options in Agentic account)
- [ ] **P24** Crypto trading (Robinhood supports crypto in Agentic account)

## Live Trading Protocol

Live trading shipped in P2. No paper-first gating by design. The following apply:

1. **Strategy confirmation**: One-time prompt at `live` startup confirming strategy name, params, symbols, risk settings. Saved to `logs/strategy_confirmed.json`.
2. **Blast radius**: Robinhood Agentic account isolation — agent writes confined to dedicated sub-account with user-funded capital only.
3. **Risk controls always active**: Daily loss kill switch (configurable, default 3%), position sizing cap (25% equity), stop-loss per trade.
4. **Emergency stop**: UI button in Logs page writes `logs/kill_switch.flag`; engine checks this every cycle.
5. **Scaling gates**: When expanding to additional strategies or larger capital allocations, run walk-forward optimization and paper-test new additions in dry-run mode first.
