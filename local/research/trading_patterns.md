# Trading Patterns Research — Production Readiness for StockTradingBot

**Date:** 2026-08-18  
**Goal:** Identify high-value patterns from production trading systems that enhance StockTradingBot's readiness  
**Target Systems Analyzed:** [FreqTrade](https://github.com/freqtrade/freqtrade) (53.4k⭐), [Stefan Jansen ML4T](https://github.com/stefan-jansen/machine-learning-for-trading) (20.5k⭐), Robinhood-AI-Bot

---

## 1. Kelly Criterion Position Sizing

### Sources
- **ML4T** — `machine-learning-for-trading/17_portfolio_construction/04_kelly_criterion.py` (913 lines, ~25.9KB)
- **ML4T** Chapter 23 portfolio construction notebooks + `case_studies/etfs/`
- **FreqTrade** — `plugins/pairlist/WeightPairList.py`: weight-based allocation
- **StockTradingBot current** — `bot/portfolio.py`: Fractional Kelly (`f* = mean(r) / var(r)`) + Risk Parity + Equal Weight

### Key Patterns

#### 1a. Full Kelly Formula
```
f* = (p × b - q) / b       ← Win probability formulation
f* = μ / σ²                ← Return variance formulation (ML4T)
```
Where `p = win probability`, `q = 1-p`, `b = avg_win/avg_loss ratio`

#### 1b. Fractional Kelly (Quarter-Kelly default)
- Applies fraction `α ∈ {0.25, 0.5}` to full Kelly output
- Rationale: Full Kelly maximizes geometric growth but produces extreme volatility drag (~50% drawdowns common)
- Quarter-Kelly achieves ~70% of geometric growth with ~25% of volatility/drawdown
- ML4T recommends starting at 0.25 for live trading; scale up as confidence grows

#### 1c. Confidence-Weighted Kelly
```python
def confidence_adjusted_kelly(kelly_fraction, signal_confidence):
    """Scale Kelly by signal quality metric (e.g., model probability, Sharpe estimate)."""
    return kelly_fraction * signal_confidence * min(1.0, max(0.0, signal_confidence))
```
When ML model confidence < threshold → position size shrinks proportionally.

#### 1d. Cap & Floor Constraints
- **Max single-position cap:** 25% of equity (prevents concentration risk)
- **Floor:** Minimum $1 trade or minimum 0.01 weight
- **Negative edge handling:** f* < 0 → position_size = 0 (no shorting unless explicitly coded)

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Fractional Kelly | ✅ Implemented in `portfolio.py` | Partial — no confidence-weighted scaling |
| Kelly estimation window | Fixed `_MIN_SAMPLES = 10` | Too low; ML4T uses rolling 30-90 day windows |
| Kelly cap per position | No explicit cap | Missing — add 25% max |
| Negative edge handling | Returns 0.0 | ✅ Correct |
| Rolling vs staticKelly | Static computation | ML4T uses expanding window with decay weighting |

### Recommendation
**Priority: HIGH** — Add confidence-weighted Kelly (ML4T pattern) and increase estimation window. Integrate signal confidence from `bot/ml/model.py` into position sizing.

---

## 2. Stop-Loss & Take-Profit Management

### Sources
- **FreqTrade** — `docs/stoploss/`, `docs/strategy-callbacks/` (freqtrade.io documentation)
- **FreqTrade** `strategy/interface.py`: `custom_stoploss()`, `custom_exit()`, `minimal_roi`
- **FreqTrade** `optimize/backtesting.py`: trailing stop-loss computation engine

### Key Patterns

#### 2a. Dynamic Stop-Loss via Callback (FreqTrade)
```python
# FreqTrade interface pattern — the gold standard:
def custom_stoploss(self, pair, trade, current_time, current_rate, 
                    current_profit, **kwargs) -> tuple[bool, float] | None:
    """Return (True, new_stop_price) to update stop, or None to leave unchanged."""
    if current_profit >= 0.05:     # At 5% profit
        return True, current_rate * 0.98   # Move to breakeven+2%
    elif current_profit >= 0.10:   # At 10% profit
        return True, current_rate * 0.95   # Trail 5% below
    return None   # Leave current stop unchanged
```
Returns `(should_update, new_stopprice)` tuple — elegant hook architecture.

#### 2b. Three Stop-Loss Modes (FreqTrade)
| Mode | Mechanism | Config Parameter |
|------|-----------|------------------|
| **Fixed %** | Entry price × (1 + stoploss_value) | `stoploss = -0.10` (10% below entry) |
| **Trailing** | Cancel previous order, set new at current × offset | `trailing_stop=True`, `trailing_stop_positive=0.05` |
| **Time-based** | Auto-exit if N bars pass without ROI target hit | Time thresholds in `minimal_roi` |

#### 2c. Trailing Stop-Loss Variants
- **Standard trailing:** Stops trail behind highest reached price at fixed % distance
- **Positive offset trailing:** Only starts trailing AFTER price reaches positive offset from entry
- **Multi-level trailing:** Tighter stop after higher profits (e.g., 10% buffer at +5%, then 5% at +10%, then 3% at +20%)
- **Trailing only when profitable:** `trailing_only_offset_is_reached` flag prevents premature trailing

#### 2d. ROI-Based Exit Tiers (FreqTrade `minimal_roi`)
```json
{
    "100": 0,      // At 100% profit: exit immediately
    "30": 60,      // At 30% profit: exit after 60 minutes
    "10": 120,     // At 10% profit: exit after 120 minutes
    "-0.9": 1440   // Otherwise: hold up to 24 hours before forced exit
}
```
Format: `{"profit_threshold_pct": "max_hold_minutes"}`

#### 2e. Custom Exit Signals (FreqTrade `custom_exit()`)
```python
def custom_exit(self, pair, trade, current_time, current_rate, 
                current_profit, **kwargs) -> str | None:
    """Return string signal ('take_profit', 'stop_loss', etc.) to trigger exit."""
    if current_profit > 0.15 and trade.max_profit > 0.18:
        # Profit fell 3 percentage points from peak → take profit
        return 'peak_reversal'
    return None  // No exit signal
```

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Basic stop-loss | ✅ `risk.stop_loss()` returns entry - 2×ATR | Hardcoded formula; no dynamic adjustment |
| Trailing stop | ❌ Not implemented | Critical gap — FreqTrade pattern is battle-tested |
| ROI tiers | ❌ Not implemented | Missing — enables automated profit-taking |
| Custom exit hooks | ❌ Not implemented | Missing — `custom_exit()` callback pattern |
| Time-based exits | ❌ Not implemented | Missing |
| Multi-tier trailing | ❌ Not implemented | Missing |

### Recommendation
**Priority: CRITICAL (P0)** — Stop-loss management is foundational risk control. Implement trailing stops + ROI tiers first (highest impact for lowest effort).

**Implementation steps:**
1. Create `bot/risk/stop_manager.py` with three modes: fixed %, trailing %, multi-tier trailing
2. Integrate `custom_stoploss()` callback into engine tick loop (following FreqTrade pattern)
3. Add `minimal_roi` dict to config schema
4. Extend Streamlit dashboard with real-time P&L + stop-loss status per position

---

## 3. Portfolio Rebalancing

### Sources
- **ML4T** — `ch07_signal_generation/` HRP notebooks, `07_signal_generation/portfolio_optimization.ipynb`
- **FreqTrade** — `plugins/pairlist/*.py`: VolumePairList, PerformancePairList, WeightPairList
- **StockTradingBot current** — `bot/portfolio.py`: Kelly + Risk Parity + Equal Weight + `PortfolioState` persistence

### Key Patterns

#### 3a. Hierarchical Risk Parity (HRP) — ML4T
Algorithm steps:
1. Compute correlation matrix of asset returns
2. Apply Ward's hierarchical clustering (linkage='ward') on inverse correlation distance
3. Recursively bisect clusters: assign more capital to lower-volatility sub-clusters
4. Within-cluster: allocate by inverse variance (standard risk parity)

```python
# HRP pseudo-code from ML4T:
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp_allocation(cov_matrix):
    """Hierarchical Risk Parity allocation."""
    corr = correlation_matrix(cov_matrix)
    dist = inverse_correlation_distance(corr)
    link = linkage(squareform(dist), method='ward')
    clust_order = leaves_list(link)
    
    weights = pd.Series(1.0, index=clust_order)
    for cluster in range(len(clust_order) - 1):
        subset_idx = clust_order[cluster : (len(clust_order) - 1)]
        w_sub = weights.iloc[subset_idx].sum()
        # Allocate based on volatility within each branch
        w_a = get_variance_branch(weights, subset_idx[:split])
        alpha = 1 - w_a / w_sub
        # Update weights...
    return normalize(weights)
```

#### 3b. Convex Optimization with Constraints (ML4T)
Minimize `-w·μ + λ·wᵀΣw` subject to:
- `sum(w) == 1` (fully invested)
- `w_i >= 0` (long-only)
- `w_i <= max_weight` (concentration limit, e.g., 25%)
- Sector exposure constraints

Uses `scipy.optimize.minimize` with SLSQP method.

#### 3c. Drift-Based Rebalancing (FreqTrade pattern adapted)
Instead of periodic rebalancing, rebalance when any position deviates > X% from target:
```python
if any(abs(pos.current_weight - pos.target_weight) > THRESHOLD):
    rebalance_portfolio()
```

#### 3d. Performance-Based Weight Adjustment (FreqTrade)
`PerformancePairList` ranks symbols by past performance over configurable lookback, adjusts weights dynamically. Re-sorts every refresh period.

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Kelly allocation | ✅ In `portfolio.py` | Good base, lacks HRP fallback |
| Risk parity | ✅ In `portfolio.py` | Implemented, works well |
| HRP clustering | ❌ Not present | High-value addition from ML4T |
| Drift detection | ❌ Not present | Missing — rebalances on demand only |
| Performance-based reweighting | ❌ Not present | Missing |
| Concentration limits | ⚠️ 25% equity cap in risk.py | Could be extended to weight-level caps |

### Recommendation
**Priority: MEDIUM (P2)** — HRP would significantly improve diversification beyond current approach. Add drift-detection as immediate value-add.

---

## 4. Data Validation Pipelines

### Sources
- **ML4T** — `02_financial_data_universe/13_data_quality_framework.ipynb`
- **ML4T** — `02_financial_data_universe/14_point_in_time_validation.ipynb`
- **ML4T** — `02_financial_data_universe/16_provider_comparison.ipynb`
- **ML4T** library `ml4t-data`: unified data acquisition from 19+ providers
- **FreqTrade** — built-in data download validation (checks candle count, NaN gaps, OHLC consistency)
- **StockTradingBot current** — `bot/validation.py`: OOSValidator, `train_test_split_by_date()`, metric calculations

### Key Patterns

#### 4a. Complete Data Quality Pipeline (ML4T)
| Stage | Operation | Threshold |
|-------|-----------|-----------|
| **Missing values** | Forward-fill price data; flag gaps > N bars as missing | Max 5 consecutive missing bars for 1-min |
| **Outlier detection** | Z-score clipping (±3σ), IQR method, winsorization | Rolling window of 60 periods |
| **Duplicate removal** | Deduplicate by timestamp+symbol; keep first | Strict equality |
| **Session alignment** | Align timestamps to exchange boundaries | Market open/close times |
| **OHLC consistency** | Verify `high ≥ open, close, low` and `low ≤ open, close, high` | Always |
| **Corporate actions** | Adjust for splits, dividends, ticker changes | When applicable |

```python
class DataValidator:
    def validate_timestamps(self, df):
        """Ensure continuous timestamps aligned to exchange sessions."""
        expected = pd.date_range(df.index[0], df.index[-1], freq='1min')
        missing = expected.difference(df.index)
        if len(missing) > 0:
            logger.warning(f"{len(missing)} missing timestamps")
        return df.reindex(expected).ffill(limit=5)

    def clip_outliers(self, df, col='close', window=60, z_threshold=3.0):
        """Rolling z-score outlier clipping."""
        mean = df[col].rolling(window).mean()
        std = df[col].rolling(window).std()
        return df[col].clip(mean - z_threshold*std, mean + z_threshold*std)

    def ensure_ohlc_validity(self, df):
        """Fix OHLC violations: high must be max, low must be min."""
        df['high'] = df[['Open','High','Close','Low']].max(axis=1)
        df['low'] = df[['Open','High','Close','Low']].min(axis=1)
        return df

    def detect_survivorship_bias(self, universe_df):
        """Check if delisted securities are excluded from dataset."""
        delisted = universe_df.get('status') == 'delisted'
        return not delisted.all()
```

#### 4b. Point-in-Time Validation (ML4T)
Problem: Financial data updates retroactively (earnings revisions, analyst upgrades). Each data point tagged with publication timestamp. Features only see data available at decision time.

Solution: Maintain version history; query feature snapshot as-of a given timestamp. Prevents lookahead bias in backtests.

#### 4c. Provider Cross-Validation (ML4T)
Cross-validate same instrument across multiple providers (Yahoo, Polygon, Alpha Vantage):
- Price difference % (tolerance: < 0.1%)
- Volume discrepancy % (tolerance: < 1%)
- Missing data % (target: < 0.5%)
- Timestamp alignment
- Fallback: use most complete provider; weighted average if no dominant source

#### 4d. FreqTrade Built-in Validation
- Checks candle count sufficiency before analysis
- Detects NaN gaps in OHLCV data
- Validates OHLC consistency (high >= all, low <= all)
- Auto-retry downloads with exponential backoff

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Basic validation | ⚠️ Some in `validation.py` (OOS-focused) | Not general-purpose data validation |
| Outlier clipping | ❌ Not present | Missing |
| Session alignment | ❌ Not present | Missing |
| OHLC consistency check | ❌ Not present | Missing |
| Survivorship bias detection | ❌ Not present | Missing |
| Point-in-time validation | ❌ Not present | Missing |
| Provider cross-validation | ❌ Not present | Missing |
| FreqTrade-style OHLC sanity | ❌ Not present | Missing |

### Recommendation
**Priority: HIGH (P0)** — Bad data = bad signals = blown accounts. Most impactful investment in reliability.

**Implementation:**
1. Create `bot/data/data_validator.py` implementing ML4T `DataValidator` class
2. Wire validator into data fetch pipeline (`yfinance_source.py`)
3. Add OHLC sanity checks that fix minor violations rather than fail entirely
4. Flag significant issues (missing >5 bars, survivorship risk) as warnings

---

## 5. Event-Driven Architecture

### Sources
- **FreqTrade** — `worker.py` worker loop, `strategy/interface.py` callback system
- **ML4T** — `ml4t-backtest` library: event queue with ordered events
- **FreqTrade** callback chain: `populate_indicators()` → `populate_buy_trend()` → `analyze_pair()` → `confirm_trade_entry()` → `custom_sell()` → `custom_stoploss()` → `adjust_stoploss()`

### Key Patterns

#### 5a. Worker Tick Loop (FreqTrade)
```python
class Worker:
    def __init__(self, config, exchange, strategy):
        self.config = config
        self.exchange = exchange
        self.strategy = strategy
    
    def run(self):
        while True:
            state = self._check_tick()
            if state == RUNNING:
                self._process_open_trades()  # Evaluate existing positions
                self._enter_new_positions()  # Scan for entries
            self._notify_status()           # Update UI/dashboard
            sleep(self.config.ticker_interval)
```

#### 5b. Strategy Callback Chain (FreqTrade)
Called in strict order during each tick:
1. `populate_indicators()` — batch compute indicators on new bar
2. `populate_buy_trend()` — batch generate buy/sell signals on new bar
3. `analyze_pair()` — per-symbol per-tick evaluation
4. `confirm_trade_entry()` — pre-entry validation hook
5. `custom_exit(pair, trade, ...)` — exit decision override
6. `custom_stoploss(pair, trade, ...)` — dynamic stop-loss update
7. `adjust_stoploss(trade, ...)` — post-entry adjustment

Each callback can veto/override the default behavior.

#### 5c. Event Queue Pattern (ML4T `ml4t-backtest`)
Ordered event types:
1. `bar_start` — new candle begins
2. `indicator_update` — indicator values refreshed
3. `signal_generation` — signals computed from indicators
4. `position_adjust` — entry/exit decisions executed
5. `bar_end` — candle closes

Strategies transition between states: `idle → long → exiting → closed`

#### 5d. Market Hours Guard (Robinhood Bot)
```python
while True:
    if is_market_hours():  # 9:30-16:00 ET Mon-Fri
        execute_analysis_cycle()
    wait_until_next_minute()
```

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Engine loop | APScheduler cron jobs | Rigid schedule; not event-driven |
| Strategy callbacks | ⚠️ Basic strategy framework | No callback chain like FreqTrade |
| Entry confirmation | ❌ Not present | Missing validation gate |
| State machine | ❌ Not present | Missing — strategies don't track state transitions |
| Market hours guard | ❌ Not present | Missing |
| Event queue | ❌ Not present | Missing |

### Recommendation
**Priority: MEDIUM (P2)** — Would enable real-time responsiveness, cleaner signal lifecycle. Can coexist alongside current APScheduler during migration.

---

## 6. Risk Metrics Dashboard

### Sources
- **FreqTrade** — RPC/Telgram commands: `/performance`, `/profit`, `/daily`, `/balance`
- **FreqTrade** — `optimize/hyperopt_loss/*.py`: CalmarHyperOptLoss, SortinoHyperOptLoss, SharpeHyperOptLoss
- **ML4T** — `diagnostic` library: Deflated Sharpe Ratio (DSR), tear sheet analysis
- **ML4T** — Conformal prediction confidence bands for position sizing
- **StockTradingBot current** — `bot/risk.py`: KillSwitch, basic metrics in `validation.py`

### Key Patterns

#### 6a. FreqTrade Performance Commands
| Command | Output |
|---------|--------|
| `/performance` | Per-trade profit breakdown by pair |
| `/profit` | Cumulative profit (abs, %, fiat equivalent) |
| `/daily` | Daily P&L rollup with charts |
| `/balance` | Account balance distribution |
| `/stats` | Duration, win rate, best/worst trades, total trades |

#### 6b. Deflated Sharpe Ratio (ML4T Diagnostic Library)
Standard Sharpe penalizes heavily for multiple testing and limited simulation length. DSR adjusts downward:

```
DSR = Φ(Φ⁻¹(SR) - (τ + ω) / √T)
```
Where:
- `τ = track record record coefficient` (multiple testing penalty)
- `ω = simulation bias` (selection bias penalty)
- `T = number of independent simulations`
- `Φ = standard normal CDF`

Interpretation: SR appears genuine only if DSR > 0.5 (adjusted for tests performed)

#### 6c. Hyperopt Loss Functions (FreqTrade)
| Loss Function | Objective | Use Case |
|--------------|-----------|----------|
| `SharpeHyperOptLoss` | Maximize Sharpe Ratio | General optimization |
| `SortinoHyperOptLoss` | Maximize Sortino (downside-only) | Asymmetric return preferences |
| `CalmarHyperOptLoss` | Maximize Calmar (return / max DD) | Drawdown-sensitive strategies |
| `MaxDrawDownHyperOptLoss` | Minimize max drawdown | Capital preservation |

#### 6d. Kill Switch / Circuit Breaker (FreqTrade)
- Max open trades limit
- Max drawdown threshold
- Consecutive loss limit
- On breach: halt ALL new entries; optionally exit all positions

#### 6e. Tear Sheet Outputs (ML4T)
- Cumulative returns vs benchmark
- Rolling Sharpe, rolling beta
- Return distribution histogram
- Monthly heat map
- Top/bottom contributors attribution
- Factor exposure decomposition

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| Sharpe Ratio | ✅ In `validation.py` | Present but not on dashboard |
| Sortino Ratio | ✅ Approximation in `validation.py` | Present |
| Max Drawdown | ✅ In `validation.py` | Present |
| Calmar Ratio | ❌ Not present | Missing |
| DSR | ❌ Not present | Missing — critical for avoiding false positives |
| Daily P&L Rollup | ❌ Not present | Missing |
| Per-strategy attribution | ❌ Not present | Missing |
| Kill switch | ⚠️ Basic in `risk.py` | Has daily loss cap but no max DD/consecutive losses |
| Rolling metrics | ❌ Not present | Missing |
| Return distribution visualization | ❌ Not present | Missing |

### Recommendation
**Priority: HIGH (P1)** — Comprehensive metrics are essential for monitoring and compliance. Start with Calmar + DSR + kill switch hardening.

---

## 7. Paper Trading / Dry-Run Workflow

### Sources
- **FreqTrade** — dry-run mode (identical code path, SQLite `tradesv3.dryrun.sqlite`)
- **ML4T** — Walk-forward analysis, Champion-Challenger evaluation
- **Robinhood-AI-Bot** — Three-mode execution: Demo → Manual → Auto

### Key Patterns

#### 7a. FreqTrade Dry-Run Architecture
Identical execution path to live trading:
- Same order types handled identically (limit, market, stop-limit)
- Same fee model applied
- Positions tracked in SQLite (`tradesv3.dryrun.sqlite`)
- Balance simulated from initial cash parameter
- Transition: change one flag `dry_run: true → false`, restart worker

Testing workflow: Backtest → Dry-run weeks → Monitor convergence → Go live

#### 7b. Walk-Forward Paper Trading (ML4T)
```
Rolling train/test windows on historical data:
Train on T₀→T₁ → Test on T₁→T₂ → Slide forward by Δt → Repeat

Key metric: OOS hit rate matches IS accuracy within statistical tolerance
Production proxy: If walk-forward diverges from backtest → strategy retirement recommended
```

#### 7c. Champion-Challenger Evaluation (ML4T)
Keep deployed strategy (champion) running while challenger runs in shadow mode:
- Shadow: generates signals but does NOT execute
- After evaluation period: compare champion vs challenger live P&L
- Swap if challenger significantly outperforms

#### 7d. Hallucination Filter (Robinhood Bot)
Post-AI, pre-execution validation gate:
- Rejects trades for excluded symbols
- Rejects zero-quantity trades
- Checks PDT restrictions
- Validates symbol exists in portfolio/watchlist

#### 7e. Three-Mode Execution Ladder (Robinhood Bot)
```
MODE = "demo"  →  Simulated, no API calls
MODE = "manual" →  Real orders, terminal confirmation required
MODE = "auto"   →  Fully autonomous
```
Safety recommendation: Demo → review logs → Manual → Auto

### StockTradingBot Gap Analysis
| Feature | Current State | Gap |
|---------|--------------|-----|
| MockBroker | ✅ Exists in `broker.py` | Functional but minimal |
| Dry-run toggle | ⚠️ Implied through mock broker | Not formalized in config |
| Separate DB per mode | ❌ Not present | Missing — could corrupt live data |
| Hallucination filter | ❌ Not present | Missing |
| Shadow mode | ❌ Not present | Missing |
| Walk-forward analysis | ⚠️ OOSValidator in `validation.py` | Covers split/testing but not walk-forward sliding |
| Champion-challenger | ❌ Not present | Missing |
| Mode ladder | ❌ Not present | Missing |
| P&L convergence monitor | ❌ Not present | Missing |

### Recommendation
**Priority: MEDIUM (P2)** — Safe deployment path. Formalize dry-run/live toggle, add hallucination filter, implement shadow mode for new strategies.

---

## Summary & Priority Matrix

| Priority | Pattern | Impact | Effort | Source(s) | First Step |
|----------|---------|--------|--------|-----------|------------|
| **P0-Critical** | Stop-loss + take-profit management | Prevents catastrophic loss | Medium | FreqTrade callbacks | Add trailing stop-loss + ROI tiers to engine |
| **P0-High** | Data validation/cleaning pipeline | Foundation for all signals | Medium | ML4T DataValidator | Create `bot/data/data_validator.py` |
| **P1-High** | Risk metrics dashboard (DSR + Calmar) | Monitoring + compliance | Medium | ML4T diagnostic + FreqTrade RPC | Extend `bot/analytics/performance.py` |
| **P1-High** | Enhanced Kelly (confidence-weighted) | Improves risk-adjusted returns | High | ML4T Ch.17 | Integrate model confidence into `portfolio.py` |
| **P1-Medium** | Kill switch hardening | Prevents blow-up scenarios | Low | FreqTrade circuit breaker | Add max-DD + consecutive-loss limits to `risk.py` |
| **P2-Medium** | Portfolio rebalancing (HRP) | Diversification improvement | High | ML4T Ch.23 | Add HRP allocator as alternative to Kelly |
| **P2-Medium** | Event-driven architecture | Real-time strategies, cleaner code | High | FreqTrade worker + ML4T backtest | Build event queue alongside APScheduler |
| **P2-Low-Med** | Paper trading workflow | Safe deployment path | Low-Med | All sources | Formalize dry-run toggle + shadow mode |
| **P3-Low** | WebSocket real-time feeds | Needed for intraday only | High | ML4T microstructure + FreqTrade WS | Future upgrade after core risk patterns |
| **P3-Low** | Provider cross-validation | Data reliability audit | Medium | ML4T provider comparison | Optional—depends on data source diversity |

---

## Implementation Roadmap (Suggested Order)

```
Phase 1 (Week 1-2): P0 — Core Risk Controls
├── P0: Stop-loss manager with trailing + ROI tiers
├── P0: Data validator (outliers, session alignment, OHLC sanity)
└── P1: Kill switch hardening (max DD, consecutive losses)

Phase 2 (Week 3-4): P1 — Measurement + Sizing
├── P1: Risk metrics dashboard (DSR, Calmar, rolling metrics)
├── P1: Enhanced Kelly (confidence-weighted + proper estimation window)
└── P2: Paper trading formalization (dry-run toggle + hallucination filter)

Phase 3 (Week 5-6): P2 — Advanced Allocation + Architecture
├── P2: HRP portfolio allocator
├── P2: Event-driven engine (incremental migration from APScheduler)
├── P2: Champion-challenger shadow mode
└── P3: WebSocket feed integration (future)
```

---

## Cross-Repo Comparison

| Pattern Area | ML4T | FreqTrade | Robinhood AI | StockTradingBot Best Fit |
|-------------|------|-----------|--------------|------------------------|
| **Kelly Sizing** | Full + fractional + confidence-weighted | Weight pairlist | Constraint limits | ML4T fractional Kelly (already partially here) |
| **Stop-Loss** | Implicit via backtester | Custom hooks + trailing + ROI | None | FreqTrade callback architecture |
| **Take-Profit** | Triple-barrier labeling | ROI tiers + custom_exit | None | FreqTrade ROI tiers |
| **Rebalancing** | HRP + convex optimization | Performance-based pairlists | Manual calc | ML4T HRP |
| **Data Quality** | Full framework + PIT validation | Basic sanity checks | Minimal length checks | ML4T DataValidator |
| **Architecture** | ml4t-backtest event queue | Worker loop + callbacks | Scheduled loops | Hybrid (FreqTrade callbacks + APScheduler) |
| **Risk Dashboard** | DSR + tear sheets | Telegram commands | Console logging | Combine all three |
| **Paper Trading** | Walk-forward + champion/challenger | Identical dry-run code path | Demo/manual/auto ladder | FreqTrade dry-run + ML4T champion/challenger |
| **WebSocket** | ITCH/LOB reconstruction | CCXT exchange layer | REST polling | Future: WebSocket-client |
