# StockTradingBot Production-Readiness Audit

**Date:** 2026-08-18
**Target:** `D:/StockTradingBot/bot/` (Python 3.13, APScheduler engine, Streamlit UI)
**Scope:** 10 audit dimensions → prioritized actionable improvements

---

## Findings Summary

| # | Dimension | Severity | Issue Count | Status |
|---|-----------|----------|-------------|--------|
| 1 | Risk Management | **CRITICAL** | 3 | Partially implemented, broken in places |
| 2 | Position Sizing | **HIGH** | 2 | Basic; Kelly underutilised |
| 3 | Data Quality | **CRITICAL** | 3 | No validation pipeline whatsoever |
| 4 | Paper Trading | **MEDIUM** | 2 | Exists but flawed |
| 5 | Event-Driven Architecture | **LOW** | 1 | Polling-only (acceptable baseline) |
| 6 | Portfolio Rebalancing | **MEDIUM** | 3 | Wires exist but incomplete |
| 7 | Monitoring/Alerting | **MEDIUM** | 2 | Good foundations, gaps remain |
| 8 | Test Coverage Gaps | **HIGH** | 7+ modules under-tested |
| 9 | Error Handling | **CRITICAL** | 6 critical paths | Silent failures abound |
| 10 | Broken TODOs / Tech Debt | **MEDIUM** | 12+ scattered markers |

---

## DETAILED FINDINGS

### 1. RISK MANAGEMENT — CRITICAL

#### Finding 1.1: Stop-Loss & Take-Profit are static ATR-based only
- **File:** `bot/risk.py:33-40`, `bot/engine.py`
- **Issue:** Stop-loss = `entry - 2×ATR`, take-profit = `entry + 3×ATR`. Hardcoded multipliers with no trailing stop, time-based exit, or ROI tiers.
- **Impact:** Positions can run deeply negative or leave money on table. No dynamic adjustment based on profit progress.
- **Benchmark:** FreqTrade supports custom_stoploss(), trailing stops, ROI tables. This bot has none of these.

```python
# Current code (risk.py):
def stop_loss(entry_price: float, atr: float) -> float:
    return entry_price - 2 * atr  # Static, never adjusts

def take_profit(entry_price: float, atr: float) -> float:
    return entry_price + 3 * atr  # Static target
```

#### Finding 1.2: KillSwitch tracks daily drawdown but not max historical drawdown
- **File:** `bot/risk.py:54-93`
- **Issue:** `KillSwitch.check()` resets every day (`reset_day`). It cannot detect if the account is below its all-time peak by more than X%. A prolonged losing streak won't trigger it after day N.
- **Impact:** Account can bleed from $100k to $60k over weeks without triggering the kill switch.
- **Fix:** Add `max_drawdown_from_peak` tracking across days via `logs/equity_history.csv` readback.

#### Finding 1.3: No per-trade loss limit beyond risk_per_trade percentage
- **File:** `bot/config.py:33` — `risk_per_trade: float = 0.01`
- **Issue:** Only one parameter controls exposure. No stop-loss at portfolio level (e.g., "halt trading if any single trade loses > 5%").
- **Impact:** A single bad signal can still lose too much if ATR estimation is wrong or price gaps through stop.

#### Finding 1.4: Missing trailing-stop logic entirely
- **File:** `bot/engine.py` exit-check loop (~lines 320-345)
- **Issue:** Engine checks stop/target only once per cycle against current quote. If a position goes +20% then drops back to entry before the next cycle, nothing sells at profit.
- **Fix:** Track `highest_price_since_entry` per position; add trailing-stop distance check.

---

### 2. POSITION SIZING — HIGH

#### Finding 2.1: Kelly criterion exists in portfolio.py but is NOT wired into the engine
- **File:** `bot/portfolio.py:80-146` has full fractional-Kelly implementation (`allocate_kelly`).
- **File:** `bot/engine.py` — uses raw `position_size()` which is fixed-risk-per-trade (line-size method), never calls Kelly.
- **Impact:** The well-implemented Kelly module sits unused. Strategy capital allocation is purely equal-weight.
- **Files involved:** `bot/portfolio.py` (implements ✅), `bot/engine.py` (doesn't call it ❌), `bot/core.py` STRATEGIES registry (no Kelly-aware allocation strategy).

#### Finding 2.2: ATR fallback when indicator returns empty
- **File:** `bot/engine.py:368-382`
- **Issue:** When ATR indicator fails (column missing), falls back to `last_close * 0.02` (arbitrary 2%). No logging that this fallback happened.
- **Impact:** Position sizing silently degrades to guesswork during data-quality issues.

```python
# From engine.py:
if atr_val <= 0:
    atr_val = last_close * 0.02  # Silent fallback — no log warning
```

---

### 3. DATA QUALITY — CRITICAL

#### Finding 3.1: Zero data validation pipeline
- **File:** `bot/data.py` — `fetch_history()` and `fetch_latest_bars()` return raw DataFrame from datasource plugin.
- **Issue:** No checks for: missing candles (gaps in timestamps), NaN values in OHLCV columns, impossible prices (High < Low, Close < 0), duplicate timestamps, stale data.
- **Impact:** Bad data → bad signals → bad trades. The entire pipeline is trust-no-validation.

#### Finding 3.2: Outlier detection absent
- **File:** `bot/plugins/datasources/yfinance_source.py`, `bot/data.py`
- **Issue:** Price spikes from data glitches (Yahoo Finance penny-bar events like GOOGL 2021 split adjustment) pass straight through to strategy signal generation.
- **Impact:** A single spike can produce a false crossover signal and trigger a buy/sell.

#### Finding 3.3: MultiTimeframe data aggregator swallows exceptions
- **File:** `bot/multi_timeframe.py:102-103`
- **Issue:** `except Exception: pass` silences fetch errors for higher timeframe data. Weekly/monthly bars may be missing without anyone knowing.
- **Impact:** Multi-Timeframe strategies silently degrade to lower-timeframe-only behavior.

---

### 4. PAPER TRADING WORKFLOW — MEDIUM

#### Finding 4.1: MockBroker deterministic pricing undermines testing
- **File:** `bot/broker.py:139`
- **Issue:** MockBroker computes fake prices as `100.0 + hash(symbol) % 200`. Same symbol always same price. Does not simulate realistic fill latency, partial fills, or order rejections.
- **Impact:** Backtests/dry-runs pass even with strategies that would fail in production due to slippage, latency, or broker rejection.

#### Finding 4.2: No explicit dry-run → live transition gate
- **File:** `main.py:182-210`
- **Issue:** While there is a confirmation file mechanism (`strategy_confirmed.json`), the `live` command accepts any strategy name without checking if a successful dry-run exists first.
- **Impact:** Operator can accidentally deploy untested strategies to live trading.
- **Note:** The confirmation gate exists but is fragile — relies on exact param matching of hardcoded defaults, not actual dry-run results.

#### Finding 4.3: No simulation realism features
- Missing: bid-ask spread simulation, market-impact modeling, partial-fill handling, order-book depth simulation, circuit-breaker emulation (halting on rapid price moves).

---

### 5. EVENT-DRIVEN ARCHITECTURE — LOW

#### Finding 5.1: Engine uses polling exclusively (APScheduler interval)
- **File:** `bot/engine.py:30` — `scheduler.add_job(_cycle, "interval", minutes=settings.engine_interval_minutes, id="engine_cycle")`
- **Issue:** All state changes happen at fixed intervals (default 5 min). No event-driven triggers (e.g., price-change callback, volume spike alert, sentiment flash).
- **Assessment:** Acceptable for a 5-minute swing-trading regime. Not an issue unless user adds high-frequency strategies.
- **Suggestion:** Document this design decision and ensure `engine_interval_minutes` >= exchange rate limits for chosen timeframes.

---

### 6. PORTFOLIO REBALANCING — MEDIUM

#### Finding 6.1: Daily rebalance tied to day-change, not drift threshold
- **File:** `bot/engine.py:478-506` (`_daily_rebalance`)
- **Issue:** Rebalance fires ONCE per UTC day regardless of whether positions drifted outside tolerance. The 15% tolerance band is checked only when rebalancing runs; missed cycles mean drift accumulates.
- **Impact:** On slow-moving stocks, positions can exceed 50% weight drift before rebalance fires.

#### Finding 6.2: Rebalance ignores transaction costs
- **Issue:** `_daily_rebalance` submits orders whenever diff exceeds 15% of target value, with no cost-benefit analysis. Small deviations may trigger costly trades that eat into returns.

#### Finding 6.3: Kelly weights not used for position-level sizing
- **File:** `bot/portfolio.py` implements Kelly, HRP, and equal-weight allocation.
- **Issue:** These are utility functions only — nothing in `engine.py` or `run_multi_strategy()` calls `allocate_kelly()`, `allocate_risk_parity()`, or reads `PortfolioState`.
- **Impact:** Multi-strategy engine allocates capital by fixed weight (from `strategy_allocations` config JSON), ignoring historical performance evidence that Kelly would use.

---

### 7. MONITORING / ALERTING — MEDIUM

#### Finding 7.1: Alert system is well-designed but transport-gated
- **File:** `bot/alerts.py:1-421`
- **Assessment:** Discord webhook + SMTP email coverage is comprehensive. Fill alerts, kill-switch alerts, drawdown alerts, daily summaries are all implemented.
- **Gap:** No health-check alerts (e.g., "datasource unreachable for 3 consecutive cycles" or "sentiment API down for 1 hour").

#### Finding 7.2: No local dashboard beyond Streamlit UI
- **File:** `ui/pages/1_📊_Dashboard.py`
- **Issue:** Dashboard reads from `logs/*.json` / `logs/*.csv`. If logs directory is corrupted/empty, dashboard shows blanks. No built-in anomaly detection on the displayed metrics.
- **Addition:** Add "system health" panel: last cycle timestamp, datasource availability count, ML model freshness check.

#### Finding 7.3: Logging format consistent but insufficient for debugging
- **File:** `bot/config.py:143-154`
- **Issue:** RotatingFileHandler with DEBUG level is good, but no structured logging (JSON lines) for machine parsing. No log correlation IDs per trade/cycle.
- **Impact:** Debugging requires manual grep across log files.

---

### 8. TEST COVERAGE GAPS — HIGH

#### Finding 8.1: Module-to-test mapping
Based on 30 test files vs 40+ source files, significant gaps:

| Source Module | Has Tests? | Notes |
|---------------|-----------|-------|
| `bot/risk.py` | ✅ `test_risk.py` | Well-covered |
| `bot/alerts.py` | ✅ `test_alerts.py`, `test_engine_alerts.py` | Good |
| `bot/config.py` | ⚠️ Indirect (via fixtures) | No dedicated unit tests |
| `bot/broker.py` | ⚠️ `test_docker.py` covers Docker, not broker logic | **GAP** |
| `bot/engine.py` | ⚠️ `test_multi_strategy.py`, `test_engine_alerts.py` | **No pure-engine-unit tests** |
| `bot/data.py` | ⚠️ `test_polygon_datasource.py`, `test_databento_datasource.py` | **No core data-fetch validation tests** |
| `bot/strategy.py` | ⚠️ `test_strategy.py` | Basic only |
| `bot/indicators.py` | ✅ `test_indicators.py` | OK |
| `bot/portfolio.py` | ✅ `test_portfolio.py` | OK |
| `bot/ml/model.py` | ✅ `test_ml_model.py`, `test_ml_validation.py` | OK |
| `bot/ml/features.py` | ✅ `test_ml_features.py` | OK |
| `bot/ml/validation.py` | ⚠️ Covered by `test_ml_validation.py` | OK |
| `bot/backtest.py` | ✅ `test_backtest_integration.py` | OK |
| `bot/trade_store.py` | ✅ `test_trade_store.py` | OK |
| `bot/equity_tracker.py` | ✅ `test_engine_equity_tracker.py` | OK |
| `bot/sentiment.py` | ✅ `test_sentiment.py`, `test_sentiment_strategy.py` | OK |
| `bot/multi_timeframe.py` | ✅ `test_multi_timeframe.py` | OK |
| `bot/optimization.py` | ✅ `test_optimization.py` | OK |
| `bot/sector_rotation.py` | ✅ `test_sector_rotation.py` | OK |
| `bot/options.py` | ✅ `test_options.py`, `test_crypto.py` | OK |
| `bot/plugins/strategies/*` | ⚠️ Partial | Non-EMA strategies lightly tested |
| `bot/plugins/datasources/*` | ✅ `test_polygon_datasource.py`, `test_databento_datasource.py` | Partial (yfinance missing) |
| `bot/utils/fixtures.py` | ✅ `test_fastmcp_server.py` indirect | OK |
| `bot/fastmcp/server.py` | ✅ `test_fastmcp_server.py` | OK |

#### Finding 8.2: Missing integration tests
- **No end-to-end dry-run test:** Nothing validates a full engine cycle from signal generation through order submission to position reconciliation with a real strategy.
- **No multi-strategy allocation test:** Weight normalisation and per-strategy equity splitting not directly verified.
- **No kill-switch lifecycle test:** Reset-day → tripping → re-arm → re-trip chain not fully tested.

---

### 9. ERROR HANDLING IN CRITICAL PATHS — CRITICAL

#### Finding 9.1: Bare `except: pass` in hot path — silent failure
- **File:** `bot/engine.py:472-474` (SIGINT handler)
- **Code:** 
```python
try:
    async def _cancel():
        await broker.cancel_all()
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_cancel())
    finally:
        loop.close()
except Exception:
    pass  # Silently swallows cancellation failures on shutdown
```
- **Impact:** On Ctrl+C, orders may remain open. No confirmation to user that orders were cancelled.

#### Finding 9.2: Equity fetch failure silently continues
- **File:** `bot/engine.py:~310-315`
- **Code:**
```python
try:
    equity = await broker.get_equity()
    last_equity = equity
except Exception:
    logger.warning("Could not fetch equity, continuing with previous value")
    equity = last_equity
```
- **Impact:** If broker connection drops, engine keeps using stale equity. KillSwitch drawsdown calculation becomes meaningless. After 24h of disconnection, the kill switch could trip on stale data or miss a real drawdown.

#### Finding 9.3: Quote fetch failure skips exit checks
- **File:** `bot/engine.py:~326-330`
- **Impact:** If `broker.get_quotes([sym])` fails for one symbol, ALL exit checks for that position are skipped for that cycle. Stop-losses are not enforced.

#### Finding 9.4: Subprocess/env load errors silently swallowed
- **File:** `bot/plugins/datasources/yfinance_source.py` — bare `import yfinance` wrapped in try/except but error message raised via `RuntimeError("yfinance not installed...")` which gets caught by `except Exception:` in plugins.py `_scan_kind()` and logged only.
- **Impact:** Datasource appears "not available" rather than giving actionable feedback.

#### Finding 9.5: No circuit breaker pattern for repeated broker failures
- **Issue:** Engine cycles continue regardless of how many consecutive failures occur. After 12 failures in an hour (2-hour outage at 5-min intervals), the engine still tries.
- **Impact:** Resource exhaustion from retry storms; silent misbehavior from stale data.

#### Finding 9.6: ATR column resolution bug
- **File:** `bot/engine.py:361`
- **Code:**
```python
atr_col = "ATRr14_14" if "ATRr14_14" in atr_result else "ATRr14_14"
```
- **Issue:** Both branches check the same string. The `else` should be a different fallback. This is dead-code that also means if the column really isn't present, it will raise KeyError later.

---

### 10. BROKEN TODOS / TECHNICAL DEBT — MEDIUM

#### Finding 10.1: Strategy scaffold template has unresolved placeholders
- **File:** `bot/utils/strategies.py:56-82`
- **Content:** Generated plugin contains `TODO: Document entry/exit logic`, `# TODO: Add constructor parameters here`, `# TODO: Implement signal logic`.
- **Impact:** Developer who uses the scaffold starts with a broken file. While intentional as a template, it should be clearer that it's incomplete.

#### Finding 10.2: `engine_excluded_symbols` referenced but undefined
- **File:** `bot/engine.py:383`, `bot/engine.py:844`
- **Issue:** `engine_excluded_symbols` is used in two loops but never defined in either function scope. This is either dead code (the condition is always evaluated with an undefined variable → NameError at runtime) or was intended to reference `settings.symbol_exclusions`.
- **Severity:** BUG — causes `NameError: name 'engine_excluded_symbols' is not defined` at runtime. Every BUY/SELL signal processing hits this.

#### Finding 10.3: Attribute name mismatch — `min_buying` vs `min_buying_amount_usd`
- **File:** `bot/engine.py:429-430, 465-466, 941-942, 973-974`
- **Issue:** Uses `getattr(settings, 'min_buying', 0)` and `getattr(settings, 'max_buying', float('inf'))`. The actual Settings attributes are `min_buying_amount_usd` and `max_buying_amount_usd`.
- **Impact:** `getattr` returns the default (0 and inf respectively), so these bounds are NEVER applied. Trade amount guidelines are effectively disabled.

#### Finding 10.4: Duplicate entry-price update bug in multi-strategy
- **File:** `bot/engine.py:~770-775`
- **Code:**
```python
for sym in tracked_syms:
    q = quotes_result.get(sym, {})
    internal_pos[_k(s)]["entry_price"] = q.get("last", q.get("price", 0))  # ← 's' undefined!
```
- **Issue:** Loop variable is `sym` but body references `s`. This is a `NameError` that would crash the multi-strategy engine's entry-price update phase.

#### Finding 10.5: Backtest TransactionCostModel gross_pnl bug
- **File:** `bot/backtest.py:~67`
- **Code:**
```python
gross_pnl = n_shares * (exit_price - entry_price) if side == "long" else -gross_pnl
```
- **Issue:** `else -gross_pnl` — `gross_pnl` hasn't been computed yet in the short case. This is an `UnboundLocalError`. Short-side cost estimation is broken.

#### Finding 10.6: SentimentEngine uses pd.read_csv without importing pandas
- **File:** `bot/sentiment.py:~178`
- **Code:**
```python
cached = pd.read_csv(csv_path, parse_dates=["timestamp"])
```
- **Issue:** `pandas` is not imported in the module. Only `hashlib`, `csv`, `logging`, `dataclasses`, `datetime`, `pathlib` are imported. This is a `NameError` that would crash cached sentiment reading.

---

## PRIORITIZED ACTIONABLE IMPROVEMENTS

### P0 — CRITICAL (must fix before any production deployment)

| # | Title | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| C1 | Fix `engine_excluded_symbols` NameError | `bot/engine.py` | 15 min | Replace undefined var with `getattr(settings, 'symbol_exclusions', [])` |
| C2 | Fix attribute name mismatch `min_buying` → `min_buying_amount_usd` | `bot/engine.py` | 15 min | 4 occurrences in both engine functions |
| C3 | Fix `s` undefined variable in multi-strategy entry-price update | `bot/engine.py` | 10 min | Change `_k(s)` to `_k(sym)` |
| C4 | Fix ATR column resolution — dead-branch logic | `bot/engine.py:361` | 10 min | Else branch should fall back to High/Low diff method |
| C5 | Fix `TransactionCostModel.gross_pnl` UnboundLocalError for shorts | `bot/backtest.py:67` | 10 min | Compute base pnl first, then negate for shorts |
| C6 | Add missing `import pandas` to `bot/sentiment.py` | `bot/sentiment.py` | 5 min | Used via `pd.read_csv` but never imported |
| C7 | Add circuit breaker for repeated broker failures | `bot/engine.py`, `bot/broker.py` | 2h | After N consecutive failures, pause engine and alert |

### P1 — HIGH (significantly improves reliability)

| # | Title | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| H1 | Wire Kelly allocation into multi-strategy engine | `bot/engine.py`, `bot/portfolio.py` | 4h | Call `allocate_kelly()` to dynamically adjust weights |
| H2 | Add trailing stop-loss logic | `bot/risk.py`, `bot/engine.py` | 4h | Track highest price since entry; trail stop up |
| H3 | Implement data validation pipeline | `bot/data/validation.py` | 6h | Missing candles, NaN clipping, outlier detection, timestamp alignment |
| H4 | KillSwitch: track max historical drawdown | `bot/risk.py` | 2h | Cross-day peak tracking using persisted equity history |
| H5 | Add graceful SIGINT handler verification | `bot/engine.py` | 2h | Log confirmation of order cancellation; don't swallow errors |
| H6 | Add datasoure availability monitoring | `bot/data.py` | 2h | Track consecutive failures per datasource; alert when unhealthy |

### P2 — MEDIUM (production polish)

| # | Title | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| M1 | Replace equal-weight rebalance with HRP | `bot/portfolio.py`, `bot/engine.py` | 4h | Hierarchical Risk Parity from Stefan Jansen ML4T patterns |
| M2 | Add ROI tier-based exits | `bot/config.py`, `bot/engine.py` | 2h | Configurable `{profit_pct: max_hold_minutes}` dict |
| M3 | Enhance MockBroker with realistic fill simulation | `bot/broker.py` | 3h | Bid-ask spread, partial fills, latency jitter |
| M4 | Add system-health panel to Streamlit dashboard | `ui/pages/1_📊_Dashboard.py` | 2h | Cycle age, datasource health, ML model freshness |
| M5 | Structured JSON logging | `bot/config.py` | 2h | Add optional JSON-lines formatter for log parsing |
| M6 | Improve strategy scaffold template clarity | `bot/utils/strategies.py` | 30 min | Add clear "INCOMPLETE TEMPLATE" banner + working example |
| M7 | End-to-end dry-run integration test | `tests/` | 4h | Full engine cycle: signal → order → position → exit |

### P3 — LOW (nice-to-have, future roadmap)

| # | Title | Description |
|---|-------|-------------|
| L1 | Event-driven architecture upgrade | WebSocket price feeds replace polling for intraday strategies |
| L2 | Survivorship bias detection | Flag datasets that exclude delisted securities |
| L3 | Point-in-time data validation | Prevent lookahead bias from retroactive data updates |
| L4 | Cross-provider data validation | Yahoo vs Polygon comparison for price consistency |
| L5 | Volatility-targeted position sizing | Scale position sizes inversely to current market volatility |
| L6 | Paper-to-live migration checklist | Structured validation before enabling live mode |

---

## FILE REFERENCES INDEX

All file paths relative to `D:/StockTradingBot/`:

| File | Purpose | Key Line Numbers |
|------|---------|-----------------|
| `bot/engine.py` | Main trading engine (single + multi-strategy) | Full file, esp. 300-500 (exit/entry), 700-900 (multi-strat) |
| `bot/risk.py` | Position sizing, stop/TP calc, KillSwitch | Lines 1-93 |
| `bot/config.py` | Settings dataclass, logging setup | Lines 1-155 |
| `bot/broker.py` | Broker abstraction (Mock + Robinhood MCP) | Lines 1-613 |
| `bot/data.py` | Market data fetching + cache | Lines 1-77 |
| `bot/portfolio.py` | Kelly, risk-parity allocation utilities | Lines 1-195 |
| `bot/alerts.py` | Discord + SMTP alert notifications | Lines 1-421 |
| `bot/sentiment.py` | Social sentiment aggregation | Lines 1-207 |
| `bot/indicators.py` | EMA, RSI, ATR, Bollinger indicators | Lines 1-86 |
| `bot/strategy.py` | Strategy base class + EMA-Cross-RSI plugin | Lines 1-123 |
| `bot/backtest.py` | Backtesting runner + TransactionCostModel | Lines 1-668 |
| `bot/ml/model.py` | Gradient-boosted signal classifier | Lines 1-220 |
| `bot/ml/features.py` | Feature engineering + TripleBarrier labels | Lines 1-642 |
| `bot/equity_tracker.py` | Per-strategy equity curve persistence | Lines 1-189 |
| `bot/trade_store.py` | SQLite-backed trade/position store | Lines 1-463 |
| `bot/plugins/datasources/yfinance_source.py` | Yahoo Finance datasource | Lines 1-61 |
| `bot/plugins/strategies/__init__.py` | Strategy plugin directory (empty) | Empty |
| `bot/core/plugins.py` | Auto-discovery scanner | Lines 1-55 |
| `ui/pages/1_📊_Dashboard.py` | Streamlit dashboard UI | Lines 1-~200 |
| `bot/utils/strategies.py` | Strategy scaffold generator | Lines 56-82 (TODO-filled template) |

## CONCLUSION

The codebase has solid foundational infrastructure — the engine skeleton, broker abstraction, alert system, portfolio allocation utilities, and ML pipeline are all well-designed. However, there are **6 hard bugs** that prevent correct operation in production (C1-C6) and **significant feature gaps** in risk management (no trailing stops, no historical drawdown tracking, no Kelly wiring) and data quality (zero validation pipeline). 

The recommended sequence is:
1. **Immediate:** Fix C1-C6 (all NameErrors/AttributeErrors, ~1h total)
2. **Week 1:** H1-H4 (Kelly wiring, trailing stops, data validation, drawdown tracking)
3. **Week 2:** M1-M7 (HRP rebalance, ROI exits, improved testing, structured logging)
4. **Ongoing:** L1-L6 as capacity allows
