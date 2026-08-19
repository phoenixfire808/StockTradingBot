# VectorBT Research Findings

**Source:** github.com/polakowo/vectorbt (8720 stars, 1119 forks, Python, since Nov 2017)
**License:** Apache 2.0 with Commons Clause (Fair Code — free to use, not to resell as a product)
**Research Date:** 2026-08-18

---

## 1. Vectorized Backtesting Architecture

### Core Design Principle: "Thinks in Matrices, Backtests at Scale"

VectorBT takes a fundamentally different approach from traditional event-driven backtesters:

| Traditional OOP Backtester | VectorBT |
|---|---|
| Strategies are classes; each instance is a separate object | Every strategy instance is a **column** in a 2D NumPy array |
| Processes one bar × one strategy at a time | Packs thousands of configurations into matrices, processes ALL simultaneously |
| Hard to analyze at scale without optimization | Ultra-fast via vectorized NumPy + Numba JIT + optional precompiled Rust kernels |

### Three-Layer Performance Stack

1. **Pandas/NumPy Layer** — All data structures are pandas Series/DataFrames backed by NumPy arrays. Operations are vectorized across the entire time axis (rows) and asset/config axis (columns).

2. **Numba JIT Layer** — Path-dependent logic (the core challenge of vectorization) is solved by Numba-compiled functions (`@njit`). These compile hot paths on-the-fly, achieving C-like performance while keeping Python readability. Most portfolio computations live here.

3. **Rust Kernel Layer** (optional `vectorbt-rust`) — For the most performance-critical paths, precompiled Rust kernels provide speed without JIT overhead. Enabled via `engine="rust"` parameter.

### The Time-Series Path Dependency Solution

```
Problem: Each bar depends on prior state (position, cash, PnL).
       Naive vectorization breaks because you can't parallelize across time.

Solution: Row-by-row traversal in Numba, but column-paralllel (all strategies per bar together).
          This gives you parallelism across strategies while respecting temporal dependencies.
```

### Broadcasting System

The key enabler for large-scale experimentation is a sophisticated broadcasting model:

- Accepts inputs as scalars, 1-D arrays (per-column or per-row), or 2-D matrices
- Many parameters (fees, size, direction) can be passed at any granularity level
- Flexible indexing keeps original shapes instead of expensive full broadcasting — only selects relevant elements on access
- Memory footprint scales linearly, not with the product of broadcast dimensions

```python
# Size can be: scalar (applied everywhere), per-column (per-symbol), per-row (per-bar), or full matrix
size = pd.Series([1, -1, 1, -1])   # per row (bar-level)
# OR
size = pd.DataFrame({'a': [1,2], 'b': [1,2]})  # per element
# OR
size = np.inf  # constant across all bars/symbols/strategies
```

### Portfolio Simulation Modes (3 tiers)

From fastest/least-flexible to slowest/most-flexible:

1. **`Portfolio.from_orders()`** — Fastest. Takes raw order arrays (size, price, fees, direction) broadcasts them against each other, creates Order tuples per element. No signal processing — assumes orders are already generated. Ideal when you need pure speed.

2. **`Portfolio.from_signals()`** — Default entry point. Adds abstraction: won't re-enter if already in position, implements stop-loss/take-profit orders, uses `MA.ma_crossed_above()` type signals. Follows broadcasting principles but auto-handles state transitions.

3. **`Portfolio.from_order_func()`** — Most powerful. Custom callback executed during simulation loop. Realistic event-driven approach; can inspect current PnL, position state, generate multiple orders per bar. Uses `flex_simulate_nb` variant for multi-order-per-bar capability. Loses broadcasting — user handles manual broadcasts.

### Four-Phase Workflow

Every portfolio creation follows this pipeline:

1. **Preparation** — Resolve defaults from global settings, unify shapes, validate inputs, convert pandas→numpy
2. **Simulation** — Numba function traverses broadcasted shape element-by-element (row=time, col=asset); generates/fills/rejects orders; updates state (cash, balances)
3. **Construction** — Creates Portfolio object from returned order records
4. **Analysis** — Metrics computed from order records (trades, positions, drawdowns, returns)

### Signal Infrastructure

- Indicator factory pattern: `vbt.MA.run(price, windows)` generates moving averages per window
- Cross signals: `fast_ma.ma_crossed_above(slow_ma)` produces boolean entry/exits
- Pattern recognition integration: TA-Lib patterns can be mapped to order sizes
- Signal distribution analysis: ranking, mapping, percentile decomposition available

---

## 2. Portfolio Analysis Patterns

### Comprehensive Metrics Suite

The `pf.stats()` method produces a rich report including (verified from README example):

| Category | Metrics |
|---|---|
| **Period** | Start, End, Period duration, Benchmark Return |
| **Returns** | Total Return [%], Total Profit, Best/Worst Trade [%] |
| **Exposure** | Max Gross Exposure [%] |
| **Costs** | Total Fees Paid |
| **Drawdown** | Max Drawdown [%], Max Drawdown Duration |
| **Trades** | Total/Open/Closed Trades count |
| **Win Rate** | Win Rate [%], Avg Winning Trade [%], Avg Losing Trade [%] |
| **Duration** | Avg Winning Trade Duration, Avg Losing Trade Duration |
| **Risk-Adjusted** | Profit Factor, Expectancy, Sharpe Ratio, Calmar Ratio, Omega Ratio, Sortino Ratio |

### Per-Configuration Inspectability

Key design decision: every strategy configuration remains accessible:

```python
# Index into ANY config combination from the full sweep
pf[(10, 20, "ETH-USD")].stats()  # Returns stats for THAT specific config
pf[(10, 20, "ETH-USD")].plot()   # Interactive Plotly chart for that config
```

This enables post-hoc deep-dive into individual strategies after bulk screening.

### Drawdown Analysis

- Drawdowns tracked as structured records with start/end/duration metadata
- Equity curves computed continuously from trade records
- QuantStats integration available for additional risk analytics

### Visualization System

- **Plotly-based** interactive charts renderable in Jupyter widgets or standalone browser
- Heatmaps: `pf.total_return().vbt.heatmap(x_level="fast_window", y_level="slow_window", slider_level="symbol")`
- Scatter plots: Mean expectancy across random strategies grouped by parameter
- Time-series heatmaps: `%B` and Bandwidth animated across Bollinger Bands params
- Subplots: Multi-panel layouts for indicator comparison
- Animation support: `vbt.save_animation()` for param sweeps over time

---

## 3. Performance Attribution Methodology

### Per-Trade Trade Records

VectorBT maintains detailed trade records accessible via `pf.trades`:

```python
pf.trades.records_readable        # Full trade log with all fields
pf.trades.expectancy()            # Average profit per trade, grouped by any dimension
pf.trades.size()                  # Position sizes
pf.trades.closed()                # Filter to closed trades only
```

### Distribution-Level Statistics

Built-in grouping/cross-tabulation capabilities:

```python
# Group expectancy by parameter AND symbol
mean_expectancy = pf.trades.expectancy().groupby(["randnx_n", "symbol"]).mean()

# Visualize as heatmap across all parameter combinations
fig = mean_expectancy.unstack().vbt.scatterplot(
    xaxis_title="randnx_n", 
    yaxis_title="mean_expectancy")
```

### Walk-Forward Optimization

```python
vbt.WalkForwardOptimize.run(...)
# Rolling train/test windows for out-of-sample validation
```

Provides robustness testing against overfitting — train on one window, test on next.

### Label Generation for ML Workflows

VectorBT integrates with ML pipelines:

- Generates forward-looking labels from price data
- Can tag each bar with outcome (win/loss/break-even) for supervised training
- Compatible with scikit-learn/XGBoost/etc. feature pipelines

---

## 4. Position Sizing Approaches Beyond Kelly/HRP

### Built-in Sizing Modes

From `portfolio/base.py` and source analysis:

1. **Fixed Fractional** — Constant dollar or unit size per trade (`size=1`, `size=np.inf`)
2. **Percentage of Cash** — Size proportional to available cash balance
3. **Signal-Mapped Sizing** — Use indicator values directly as order sizes:
   ```python
   # Convert TA-Lib pattern scores to USD order value
   size = result / ohlcv['Open']  # result = pattern score, divide by price
   ```
4. **Custom Function Sizing** — Through `from_order_func()`, any algorithmic sizing logic
5. **Dynamic Fee-Aware Sizing** — Transaction costs factored into optimal position calculation

### Transaction Cost Modeling

Comprehensive cost tracking built into the engine:

- **Fees** — Configurable rate per trade (e.g., `fees=0.001` = 10bps)
- **Slippage** — Configurable price impact (e.g., `slippage=0.001`)
- Costs accumulated throughout simulation, reflected in final equity curve

### Risk Controls

- **Stop Loss** — Absolute price level or percentage
- **Take Profit** — Target price or percentage gain
- **Trailing Stops** — Dynamic trailing distance that locks in gains
- **Max Position Size** — Caps exposure per symbol

### Parameter Grid Sweeps

```python
# Test ALL window combinations in parallel
windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, 
                                     short_names=["fast", "slow"])
# Tests 100^2 = 10,000 strategy configs in seconds
```

### Random Strategy Testing

```python
# Generate N random strategies and compare expectancy distribution
pf = vbt.Portfolio.from_random_signals(price, n=n_strategies, init_cash=100, seed=42)
# Grouped analysis reveals which strategy complexity ranges are most profitable
```

---

## 5. Factor Analysis Pipelines

### Multi-Symbol Factor Testing

VectorBT natively supports multi-asset factor analysis:

```python
symbols = ["BTC-USD", "ETH-USD", "XRP-USD"]
data = vbt.YFData.download(symbols, missing_index="drop")
price = data.get("Close")

# MA with cross-symmetry: each window tested against every other
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# Result: 3 symbols × 100 fast_windows × 100 slow_windows = 30,000 strategies
pf = vbt.Portfolio.from_signals(price, entries, exits, size=np.inf, fees=0.001, freq="1D")

# Aggregate by factor dimension
fig = pf.total_return().vbt.heatmap(
    x_level="fast_window", y_level="slow_window", 
    slider_level="symbol", symmetric=True)
```

### Signal Ranking & Distribution Analysis

- Signals can be ranked across assets/timeframes
- Percentile decomposition shows how signal strength correlates with outcomes
- Mapping tools allow sorting signals into quantiles for stratified analysis

### Data Pipeline Integration

```python
# YFinance data with preprocessing
data = vbt.YFData.download("BTC-USD", start=start, end=end)
price = data.get("Close")

# Synthetic data generation for stress testing
synthetic_data = vbt.generate_ohlcv(...)

# Missing index handling
data = vbt.YFData.download(symbols, missing_index="drop")  # or "fill", "forward_fill"
```

### Automation Tools

- Scheduled data updates via cron-like mechanisms
- Telegram notifications for signal alerts or portfolio milestones
- Docker image available for reproducible deployments

---

## Reusable Patterns for Our Trading Bot

### High-Priority Patterns Worth Adopting

#### Pattern 1: Broadcast-Based Parameter Sweeping
**Current gap:** Our grid search works, but doesn't leverage the same broadcast elegance.
**From vectorbt:** `run_combs()` with automatic MultiIndex creation per parameter → direct heatmap visualization of result surfaces.

```python
# Reusable concept: Instead of dict-style param grids, use tensor broadcast semantics
# Parameters become array axes → results indexed by parameter names
```

#### Pattern 2: Per-Config Selectability
**Current gap:** After a sweep, hard to drill into individual configs.
**From vectorbt:** Every config has a natural key (tuples of param values) → `pf[(params)].stats()` for instant deep-dive.

#### Pattern 3: Three-Tier Simulation Flexibility
**Current gap:** We have one simulation path.
**From vectorbt:** Offer `from_signals` (easy), `from_orders` (fastest), and `from_order_func` (maximum control) — users pick their fidelity/complexity tradeoff.

#### Pattern 4: Heatmap Visualization for Param Surfaces
**Current gap:** Metrics are computed but not visualized as spatial maps.
**From vectorbt:** `.vbt.heatmap(x_level="param_a", y_level="param_b", slider_level="dim_c")` → interactive spatial understanding of performance landscapes.

#### Pattern 5: Trade Record Abstraction
**Current gap:** Our trade journal is CSV-based.
**From vectorbt:** `Trades` class wrapping records with methods like `.expectancy()`, `.size()`, `.closed()` — groupby-aware statistics computed lazily from stored records.

#### Pattern 6: Walking-Fold Robustness Testing
**Current gap:** We have PurgedKFold for ML validation but not for backtest robustness.
**From vectorbt:** Rolling train/test windows that sweep date ranges, producing out-of-sample degradation curves.

#### Pattern 7: Flexible Broadcasting for Inputs
**Current gap:** All inputs are uniform-size arrays.
**From vectorbt:** Scalars → 1D (per-column or per-row) → 2D matrices, auto-broadcasted. Enables passing per-symbol fees, per-bar slippage, or global constants uniformly.

### Medium-Priority Patterns

- **QuantStats integration** — Additional risk metrics beyond Sharpe/Sortino/Calmar
- **Walk-forward label generation** — For training ML models with realistic forward-looking targets
- **Pattern recognition pipeline** — TA-Lib pattern scores → order size mapping

### Lower-Priority Patterns

- **Animation of param sweeps** — Nice for presentations, less critical for production
- **Telegram automation** — Domain-specific notification system

---

## Comparison Matrix: VectorBT vs Current Bot Capabilities

| Feature | VectorBT | Our Bot | Gap Assessment |
|---|---|---|---|
| Vectorized param sweeps | ✅ `run_combs()` | ✅ Grid search | Similar capability |
| Per-config selectability | ✅ `pf[(k)].stats()` | Partial (dict lookup) | Moderate improvement possible |
| Three-tier simulation | ✅ Orders/Signals/Func | One path (signals) | Easy to add `from_orders` mode |
| Multi-asset broadcasting | ✅ Native tensors | Loop-based iteration | Major architectural gap |
| Interactive heatmaps | ✅ Plotly-backed | Static CSV/tables | Visual improvement opportunity |
| Trade record class | ✅ Methods on records | CSV-only | Good refactor target |
| Walk-forward backtesting | ✅ Rolling windows | Purged K-fold (ML only) | Should integrate into backtest engine |
| Kelly sizing | ❌ (not mentioned) | ✅ Implemented | We're ahead |
| HRP allocation | ❌ (not mentioned) | ✅ Ward linkage clustering | We're ahead |
| Trailing stops | ✅ Built-in | ✅ Fixed/dynamic/multi-tier | Comparable |
| Correlation/Risk modeling | Basic | ✅ Full correlation matrix + lookahead detection | We're more rigorous |
| Data validation layer | Basic preprocessing | ✅ Timestamp/outlier/survivorship checks | We're more thorough |
| Transaction cost modeling | ✅ Fees + slippage | ✅ Commission + slippage + spread | Comparable |
| ML-ready labeling | ✅ Forward labels | ✅ TripleBarrier | Comparable |

---

## Implementation Recommendations

### Phase 1: Quick Wins (Low Effort, High Impact)

1. **Broadcast-friendly input parsing** — Refactor signal acceptance to accept scalars, 1D arrays, and DataFrames simultaneously with auto-broadcasting. Improves usability of `multi_timeframe.py` and existing strategies.

2. **Per-strategy-key selection** — Assign hashable keys to each strategy config (tuple of all params), enable `config_key → stats` lookup dictionary for instant post-hoc inspection.

3. **Interactive return surface visualization** — Add heatmap plotting for any 2-parameter grid using Plotly. Extends our existing metrics computation with spatial context.

### Phase 2: Architectural Enhancements

4. **Multi-asset broadcast backtester** — Extend current single-asset engine to handle symbol-tensor representations, enabling true simultaneous multi-symbol testing (replaces our loop-based rotation pattern).

5. **Walk-forward robustness scoring** — Wrap existing `evaluate_oos()` with rolling window sweep, produce degradation curves for each param combo. Directly feeds our optimization loop.

### Phase 3: Advanced Pipeline Features

6. **Trade record abstractions** — Build `Trades` class with lazy-statistics methods (`expectancy()`, `size()`, `distribution()`) computed from stored records rather than recomputed CSV queries.

7. **Factor-analytical tooling** — Expose signal-to-outcome quantile analysis for rapid strategy classification (alpha decay, regime dependency, capacity constraints).

---

## Key Architectural Insights

1. **"Matrix-first" mindset** — Every operation should ask: can this be expressed as an array operation? If yes, vectorize it. If no, use Numba. Never loop in pure Python when doing trading research.

2. **Lazy computation with cached results** — Stats are computed on-demand from stored trade records, not eagerly during simulation. This saves memory and enables post-hoc exploration.

3. **Broadcasting as a first-class citizen** — Rather than requiring all inputs to be shaped identically upfront, accept flexible dimensions and resolve at compute time. This dramatically reduces boilerplate.

4. **Separation of concerns across simulation depth** — Three modes serve three user mental models: "I have orders" (fastest), "I have signals" (balanced), "I have logic" (full control). This matches our own separation between strategy plugins and the engine.

5. **Visualization as an integral component** — Not bolted on after computation. The `.vbt.` accessor chain enables inline `.heatmap()`, `.scatterplot()`, `.plot()` calls on any array result. Visualization is part of the interface contract.
