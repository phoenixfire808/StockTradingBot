# FinRL Patterns — Research Findings

**Source:** [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) (16k+ stars, MIT)
**Date:** 2026-08-18
**Target:** Non-crypto stock trading patterns we can adapt

---

## 1. Market Data Pipeline Architecture

### 1.1 Three-Layer Processor Abstraction

FinRL uses a **strategy-pattern dispatch** for data sources:

```
DataProcessor (facade)
  ├── AlpacaProcessor      (API key + secret auth)
  ├── WrdsProcessor          (WRDS institutional data)
  └── YahooFinanceProcessor  (yfinance, free tier)
```

Each processor implements the same five methods:
1. `download_data(ticker_list, start_date, end_date, time_interval)` -> DataFrame
2. `clean_data(df)` -> handles missing values, fills gaps with previous close
3. `add_technical_indicator(df, indicator_list)` -> appends indicator columns
4. `add_vix(df)` / `add_turbulence(df)` -> enriches with market regime signals
5. `df_to_array(df, tech_list, if_vix)` -> numpy arrays for DRL training

### 1.2 Key Pattern: Full Timestamp Index Filling

From `processor_yahoofinance.py::clean_data()` — the most reusable pattern:

- Gets NYSE trading calendar via pandas_market_calendars
- Creates full timestamp series per ticker, filling missing rows with
  forward-filled price + volume=0 (not NaN!)

**Why this matters for our bot:** We use yfinance directly without trading day awareness. This pattern ensures no gap days slip through for stocks that trade on different schedules. The volume=0 fill for missing rows is smarter than our current NaN handling.

### 1.3 Multi-Timestep Download Workaround

yfinance has a 7-day limit for 1-min data. FinRL works around it:

```python
while current_tic_start_date <= end_date:
    temp_df = yf.download(tic, start=current, end=current+delta, interval)
    current_tic_start_date += delta   # 1-day chunks
```

---

## 2. Feature Extraction Approaches for Stocks

### 2.1 Technical Indicators via stockstats Library

FinRL delegates indicator computation entirely to [`stockstats`](https://pypi.org/project/stockstats/) (a ta-lib alternative):

```python
INDICATORS = [
    "macd",           # Moving Average Convergence Divergence
    "boll_ub",        # Bollinger Bands Upper
    "boll_lb",        # Bollinger Bands Lower
    "rsi_30",         # Relative Strength Index (period 30)
    "cci_30",         # Commodity Channel Index
    "dx_30",          # Directional Movement Index
    "close_30_sma",   # 30-day Simple Moving Average
    "close_60_sma",   # 60-day Simple Moving Average
]
```

Computation pattern:
```python
stock = Sdf.retype(df.copy())  # Retype as StockDataFrame
for indicator in INDICATORS:
    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
    df = df.merge(indicator_df, on=["tic", "date"], how="left")
```

### 2.2 GroupByScaler — Per-Ticker Normalization

```python
class GroupByScaler(BaseEstimator, TransformerMixin):
    """Sklearn scaler that fits independently per ticker group."""
    def fit(self, X, y=None):
        for value in X[self.by].unique():  # 'by' = ticker column name
            X_group = X.loc[X[self.by] == value, self.columns]
            self.scalers[value] = MaxAbsScaler().fit(X_group)
```

This avoids cross-contamination between tickers during scaling — critical when each ticker has different price ranges. **Our bot currently uses global normalization; this is an improvement.**

### 2.3 User-Defined Features (Optional)

```python
def add_user_defined_feature(self, data):
    df["daily_return"] = df.close.pct_change(1)
    # Commented out return_lag_1 through return_lag_4
```

### 2.4 Market Regime Features

Two regime signals are appended:

1. **VIX proxy** (`VIXY` ETF close) — downloaded separately and merged by date
2. **Turbulence Index** — Mahalanobis distance of daily returns from historical mean:

   turbulence_t = (r_t - r_bar)^T * Sigma^-1 * (r_t - r_bar)

Computed over a rolling 252-day window. When turbulence exceeds threshold -> liquidate all positions.

**Relevance:** Turbulence index directly parallels our own risk management layer. Could be adapted as an additional feature or as a position-limiting signal.

---

## 3. Portfolio Optimization Methods

### 3.1 Mean-Variance Optimization Baseline

FinRL uses [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt) for MVO baseline:

```python
from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1e6 * cleaned_weights_mean[i] for i in range(len(...))])
```

Key params: weight bounds `(0, 0.5)` enforce long-only, max 50% single stock.

### 3.2 Gym Environment State Structure for Multi-Asset

State vector layout (numpy variant `env_stocktrading_np.py`):

| Offset | Field                        | Dim   | Scale   |
|--------|------------------------------|-------|---------|
| 0      | cash amount                  | 1     | x 2^-12 |
| 1      | turbulence (sigmoid-scaled)  | 1     | x 2^-5  |
| 2      | turbulence_bool              | 1     | {0,1}   |
| 3..N   | prices                       | S     | x 2^-6  |
| N+1..2N| stock holdings               | S     | x 2^-6  |
| 2N+1.. | stock cooldown days          | S     | raw int |
| 3N+1.. | technical indicators         | SxM   | x 2^-7  |

Total state_dim = `1 + 2 + 3*S + M*S` where S = stock count, M = indicator count.

Action space: `Box(low=-1, high=1, shape=(S,))` — continuous, negative=sell, positive=buy.

### 3.3 Position Cooldown Mechanism

```python
self.stocks_cd = np.zeros_like(self.stocks)  # tracks days since last trade per ticker
# On trade: self.stocks_cd[index] = 0
# On step: self.stocks_cd += 1
# Enforce: min_action = max_stock * min_stock_rate  (prevents flip-flopping)
```

---

## 4. Evaluation Metrics Beyond Sharpe Ratio

### 4.1 Tracking Loop Metrics (collected at every timestep)

The env.step() loop tracks:
- **Account value memory** (`asset_memory`) — portfolio value history
- **Reward memory** (`rewards_memory`) — step-by-step rewards
- **Actions memory** (`actions_memory`) — which actions taken when
- **Date memory** (`date_memory`) — timestamps aligned
- **Total cost** (`cost`) — cumulative slippage + commissions
- **Total trades** (`trades`) — transaction count
- **Episode return** = final_total_asset / initial_total_asset

### 4.2 End-of-Episode Statistics

```python
df_total_value = pd.DataFrame(asset_memory)
df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
sharpe = sqrt(252) * mean(daily_return) / std(daily_return)
```

### 4.3 Validation Sharpe (Multi-Iteration Cross-Validation)

```python
@staticmethod
def get_validation_sharpe(iteration, model_name):
    df_total_value = pd.read_csv(f"results/account_value_validation_{model_name}_{iteration}.csv")
    if df_total_value["daily_return"].var() == 0:
        return np.inf if mean > 0 else 0.0
    return sqrt(4) * mean(daily_return) / std(daily_return)  # freq-dependent annualization
```

Note: They use `sqrt(4)` instead of `sqrt(252)` — frequency-dependent annualization.

### 4.4 Benchmark Comparisons

Backtest script compares against THREE baselines:
1. **Mean Variance Optimization** (PyPortfolioOpt max_sharpe)
2. **DJIA index** (Dow Jones Industrial Average)
3. **All other trained DRL agents** (A2C vs DDPG vs PPO vs TD3 vs SAC)

### 4.5 Paper Trading Metrics

Live/paper trading records:
- Cumulative return curve vs DJIA benchmark
- Per-trade order completion rate
- Latency measurement for data processing pipeline

---

## 5. Paper Trading Simulation Architecture

### 5.1 Sliding Window Training/Testing Cycle

```
[TRAIN WINDOW 6d] [TEST 2d] [RETRAIN FULL 8d] [LIVE LOOP]
       |               |              |             |
 train(agent)     test(agent)   retrain(agent)   run() infinite loop
       |               |              |         get_state() -> act() -> submitOrder()
 save actor.pth     validate perf overwrite prev cancel open orders at start
```

From `FinRL_PaperTrading_Demo_refactored.py`:

```python
# Train on sliding window
train(start=TRAIN_START, end=TRAIN_END, ...)

# Validate on holdout
account_value = test(start=TEST_START, end=TEST_END, ...)

# Retrain on full available data
train(start=TRAINFULL_START, end=TRAINFULL_END, break_step=2e5)

# Deploy
paper_trading.run()  # runs until killed
```

### 5.2 Live Trading Loop (`PaperTradingAlpaca.run()`)

```python
while True:
    # Auto-close 2 minutes before market close
    if self.timeToClose < 120:
        liquidate_all_positions()
        sleep(15 minutes)
        continue

    state = self.get_state()           # Fetch live OHLCV + indicators + turbulence
    action = model.predict(state)[0]   # DRL inference

    # Execute trades (threaded for parallelism)
    for sell_order: threading.Thread(target=self.submitOrder)
    for buy_order:  threading.Thread(target=self.submitOrder)

    sleep(time_interval)  # e.g., 60 seconds for 1Min bars
```

### 5.3 Turbulence Kill Switch in Live Trading

```python
if turbulence >= turbulence_thresh:
    # Sell ALL positions immediately (bypasses normal action logic)
    positions = self.alpaca.list_positions()
    for pos in positions:
        self.submitOrder(abs(pos.qty), pos.symbol, "sell")
else:
    # Normal DRL action execution
    execute_actions(actions)
```

### 5.4 State Dimension Calculation for Paper Trading

```python
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
# 1(cash) + 2(turbulence+bool) + 3*N(price,shares,cooldown) + M*N(tech)
```

This is calculated manually because paper trading loads a pre-trained model that expects exactly this state dimension.

---

## 6. Assessment: What is Worth Adapting for Our Bot

### High Priority

| Pattern | Effort | Value | Notes |
|---------|--------|-------|-------|
| **Full-timestamp index filling** (pandas_market_calendars) | Low | High | Fix gap day holes in our data pipeline |
| **PyPortfolioOpt MVO baseline** | Low | Medium | Add MVO comparison to backtest results |
| **Position cooldown mechanism** | Low | Medium | Prevents whipsaw trading; simple counter array |

### Medium Priority

| Pattern | Effort | Value | Notes |
|---------|--------|-------|-------|
| **GroupByScaler** (per-ticker normalization) | Medium | Medium | Replaces global scaler; prevents leakage |

### Low Priority / Skip

| Pattern | Why Skip |
|---------|----------|
| **Turbulence index** | Redundant with our correlation matrix + Kelly + trailing stops |
| **Bit-shift normalization** | Opaque debugging; sklearn scalers are cleaner |
| **stockstats dependency** | Obscures what indicators are computed; our explicit module is better |
| **Threaded order submission** | Only needed for real broker API integration |

