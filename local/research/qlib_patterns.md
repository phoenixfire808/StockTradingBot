# Microsoft Qlib — Research Findings & Reusable Patterns

**Source:** [github.com/microsoft/qlib](https://github.com/microsoft/qlib)
**Stars:** 47.7K | **Forks:** 7.6K | **Language:** Python
**License:** MIT
**Date Researched:** 2026-08-19

---

## 1. Alpha Signal Pipeline Architecture


### Core Pattern: Expression-Based Feature DSL + Processor Chain


Qlib uses a three-layer pipeline:


```
Raw OHLCV Data  -->  Expression DSL Parser  -->  Feature DataFrame  -->  Processor Chain  -->  ML-ready DatasetH
     │                     │                        │                    │                     │
   $open,$close     parse_field('Ref($close, 5)')   MultiIndex           DropnaLabel          Train/Valid/Test
   $volume            becomes Feature objects       df[datetime,        CSZScoreNorm         segments dict
                      $$fields become              instrument]]           ZScoreNorm         
                      PFeature objects             columns=['feature',    Fillna
                                                     'label']
```


#### Layer 1: Expression DSL (`qlib/data/ops.py`)

Qlib defines an expression language for alpha factors:
- `$close`, `$open`, `$high`, `$low`, `$volume`, `$vwap` -- raw price/volume fields
- `Ref($close, N)` -- N-day lookback reference
- `Mean($close, N)`, `Std($close, N)`, `Slope($close, N)` -- rolling operators
- `Corr($close, Log($volume), N)` -- cross-correlation over window
- `Greater`, `Less`, `Max`, `Min`, `Rank`, `Quantile` -- elementwise operations
- Operators compiled to Cython (`_libs/rolling.pyx`) for performance

**Key insight:** Features are defined as **declarative expressions**, not imperative code. This enables:

- JIT-style compilation of complex factor chains
- Automatic handling of forward-looking bias (lookback windows only)
- Lazy evaluation -- expressions parsed once, materialized per query


#### Layer 2: Data Handler + Processor Chain (`handler.py`, `processor.py`)


Two data key types separate concerns:

| Key | Purpose |
|-----|---------|
| DK_R = raw | Raw data (no processing) |
| DK_I = infer | Inference data (inference-only processors) |
| DK_L = learn | Training data (learn+inference processors) |


**Processor chain has two phases:**

1. **Learn processors** -- fit on training period only (e.g., ZScoreNorm computes mean/std from train data)
2. **Inference processors** -- run at inference time (e.g., Fillna, ProcessInf)


**Available processors:**

| Processor | Purpose | Fit Required? |
|-----------|---------|---------------|
| `DropnaLabel` | Remove rows with NaN labels | No |
| `DropCol` | Remove irrelevant columns | No |
| `FilterCol` | Keep only specified columns | No |
| `TanhaProcess` | Denoise noisy features via tanh | No |
| `ProcessInf` | Replace inf with group mean | No |
| `Fillna` | Fill NaN values | No |
| `MinMaxNorm` | Min-max normalization | Yes (fit_start/end) |
| `ZScoreNorm` | Global z-score normalization | Yes (fit_start/end) |
| `RobustZScoreNorm` | MAD-based z-score (outlier-resistant) | Yes |
| `CSZScoreNorm` | Cross-sectional z-score (per timestamp) | No |


**Critical design decision:** `fit_end_time` **must never include test data** to prevent lookahead bias.


#### Layer 3: DatasetH with Segments (`dataset/__init__.py`)


Explicit date-range segments stored alongside dataset object:

```
dataset = DatasetH(
    handler=Alpha158(instruments='csi300'),
    segments={
        'train': ('2008-01-01', '2014-12-31'),
        'valid': ('2015-01-01', '2016-12-31'),
        'test': ('2017-01-01', '2020-08-01'),
    }
)
# Usage: dataset.prepare('train', col_set='feature', data_key=DK_L)
```


### Applicable Patterns for Our Bot


1. **Expression-based factor library** -- Instead of hardcoding indicators, define them declaratively:
   - Create `factor_library.py` with named factor groups: kbar_body, rolling_returns, volume_profile
   - Each factor is a function that takes `(df, column, window)` -> returns Series

2. **Two-phase processor pipeline** (learn vs infer) -- Critical for avoiding data leakage:
   - Implement `PreprocessorChain` class with `fit(df_train)` and `transform(df_any)` methods
   - Track `fit_end_time` strictly to prevent future data bleed

3. **Cross-sectional normalization** (`CSZScoreNorm`) -- Normalize features across stocks at each timestamp:
   - Much better than global z-score for market-neutral strategies
   - Extend our `data_validator.py` with CS-level operations

4. **Segment-based data splitting** -- Train/validation/test with explicit date ranges stored alongside model:
   - Store segments dict with every trained model artifact
   - Enables reproducible OOS evaluation without manual specification


---
## 2. Data Management Patterns


### Storage: Custom Binary Format (Fast Random Access)


Qlib stores data in `.bin` files with a memory-mapped header:
- Header: 4 bytes for first index (float)
- Body: float32 data blocks packed contiguously

Access uses direct file seek -- no pandas overhead for point queries.


### Provider Backend Abstraction


```
CalendarProvider       --> list market trading days
InstrumentProvider   --> list instruments by market/filter
Provider             --> load features, prices, financials
```


Three backend types supported:

1. **FileStorage** -- Default; reads from `.bin` files on disk
2. **ArcticBackend** -- MongoDB-backed for large-scale deployments
3. **Custom backends** -- Via inheritance


### Point-in-Time (PIT) Database


Critical for avoiding survivorship bias and lookahead bias with financial reports:
- At date 20190102, reading Q3 2018 earnings returns what was ACTUALLY available on that date
- Not the revised version published months later
- Uses linked-list revision chains: `[date, period, value, next_ptr] x N revisions`


### Automatic Data Health Checks


Qlib provides `check_data_health.py` that validates:
- Missing data counts per stock
- Large volume/price steps (potential anomalies)
- Calendar continuity gaps
- Cross-stock consistency


### Applicable Patterns for Our Bot


1. **Binary feature cache** -- After computing factors, store as numpy `.npy` or feather files:
   - Pre-computation phase -> compute all factors -> save to disk
   - Training phase -> load pre-computed features (fast)
   - Avoids recomputing on every training run

2. **PIT-aware fundamental data loading** -- For Earnings Date / PE Ratio / etc.:
   - Store financials with publication dates, not reporting periods
   - When querying, filter by what was actually known at query date

3. **Data health check hook** -- Add to pipeline before training:
   ```python
   def validate_dataset(df, min_dates=200, max_gap_days=10):
       """Check for missing data, outliers, continuity"""
       ...
   ```

4. **Provider interface** -- Abstract away data source:
   - `DataProvider.fetch(symbol, start, end, fields)` -> uniform API
   - Backends: Yahoo Finance, CSV files, SQL database, WebSocket feed


---

## 3. Portfolio Construction Patterns


### TopkDropoutStrategy (Signal -> Trade)


The canonical Qlib strategy converts ML predictions into portfolio orders:


```
class TopkDropoutStrategy(BaseSignalStrategy):
    'Start with TopK highest-scored stocks'
    'Each day, replace N worst performers with N best outsiders'
    'Minimum holding threshold prevents churn'

    def generate_trade_decision(self):
        pred_score = self.signal.get_signal()  # pd.Series[stock_id -> score]
        current_stocks = self.trade_position.stock_list
        
        # New candidates: buy top scores NOT currently held
        candidates = pred_score[~pred_score.index.isin(current_stocks)].nlargest(n_drop + topk - len(current_stocks))
        
        # Sell: drop bottom scores from combined list
        comb = pred_score.reindex(current_stocks.union(candidates.index)).sort_values(ascending=False).index
        sell_list = comb[-n_drop:]  # lowest scoring
        
        # Execute sell/buy orders with tradability checks (limit up/down)
```


**Key architectural decisions:**
- **Decoupled signal extraction** -- Strategy doesn't know about models; consumes `pd.Series[symbol -> score]`
- **Explicit tradability checking** -- Skips stocks that can't be traded (halting, settlement pending)
- **Position lifecycle** -- `hold_thresh` enforces minimum holding period -> reduces transaction costs
- **Nested execution levels** -- Same framework supports daily + intraday with `LevelInfrastructure`


### Simulated Executor Pattern


```
SimulatorExecutor(
    time_per_step='day',              # Granularity
    generate_portfolio_metrics=True,  # Auto-calculate Sharpe, drawdown
    track_data=False                  # For RL training data collection
)
```


Execution loop:
```
for trade_step in calendar.steps():
    trade_decision = strategy.generate_trade_decision()
    execute(trade_decision)          # Place orders, match fills
    account.update_bar_end()         # Mark-to-market positions
```


### Risk Analysis Framework (`qlib/contrib/evaluate.py`)


Two accumulation modes:
- **"sum" mode** (used by default): arithmetic accumulation
  - annualized_return = mean_daily_return * N_days
  - max_drawdown = (cumsum_returns).min()
- **"product" mode** (compound CAGR): geometric accumulation
  - cagr = (final_value / initial_value) ^ (N / total_days) - 1
  - max_drawdown uses cumprod-based peak-to-trough


Metrics tracked per backtest:
- Information ratio (mean/std * sqrt(N))
- Annualized return (both modes)
- Max drawdown
- Price advantage per trade
- Position turnover rate


### Applicable Patterns for Our Bot


1. **Pure signal interface** -- Strategies consume `dict[symbol -> prediction]`:
   - Our existing strategies already do this (predict -> rank -> trade)
   - Make the interface more explicit: `SignalPipeline.predict()` -> `OrderGenerator.plan()` -> `Executor.execute()`

2. **Hold threshold / cooldown** -- Prevents excessive churn:
   - Add `min_hold_days` parameter to our engine's position manager
   - Skip selling symbol if held < N days, even if signal says sell

3. **Per-trade indicator tracking** -- Qlib tracks PA (price advantage), POS (long ratio), FFR (fulfill rate):
   - We already track entry/exit prices in `trade_journal.csv`
   - Extend with: fill_rate (% of intended orders actually filled), slippage tracker

4. **Nested execution levels** -- Currently we don't have multi-timeframe execution:
   - Could support: daily executor calls hourly executor for order placement
   - Allows fine-grained stop loss management within daily rebalance cadence


---
## 4. Risk Model Integration Approach


### Long-Short Backtester


Qlib includes a long-short backtest for evaluating signal quality:
- Long top K predicted winners, short bottom K predicted losers
- `shift=1` means trade T+1 on T predictions
- Equal-weight long/short, hedges market exposure


### Score IC Analysis


Qlib evaluates model quality via **Information Coefficient** (rank correlation between prediction and return):
- **IC** = Spearman correlation(pred_rank, actual_return_rank)
- **ICIR** = mean(IC) / std(IC) -- stability metric
- **Long-short return** -- spread between long basket and short basket returns

Visualization helpers: `analysis_position.score_ic_graph()` and `analysis_position.risk_analysis_graph()`


### Model Interpretability


Qlib integrates SHAP-based feature attribution for tree models:
```python
from qlib.model.interpret import LightGBMFInt
interpreter = LightGBMFInt(model)
interpreter.plot_feature_importance()  # SHAP-based feature importance
```


### Applicable Patterns for Our Bot


1. **IC tracking** -- We calculate Sharpe/Sortino but should also track:
   - Rolling 20-day IC of our signals vs actual returns
   - IC decay curve (correlation drops as prediction horizon extends)
   - Significance testing (t-stat of IC != 0)

2. **Long-short performance split** -- Separate analysis of long-side vs short-side accuracy:
   - Are our sell signals working as well as our buy signals?
   - Often asymmetry reveals label construction issues

3. **Feature attribution logging** -- After training, log which features drive predictions:
   - Use SHAP for tree models, coefficient magnitudes for linear models
   - Helps debug when model performance degrades (did feature relationships change?)

4. **Prediction horizon testing** -- Evaluate at multiple horizons (1d, 5d, 20d):
   - Different strategies optimize for different horizons
   - Kelly sizing should vary by horizon too


---

## 5. Workflow Automation & Pipeline Orchestration


### Experiment Tracking (`qlib/workflow/__init__.py`)


Qlib implements its own experiment tracker (MLflow-like but simpler):
```python
from qlib.workflow import R  # QlibRecorder singleton

with R.start(experiment_name='train_model'):
    R.log_params(**flatten_dict(task_config))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id  # Unique experiment ID

# Later: resume or retrieve
recorder = R.get_recorder(recorder_id=rid, experiment_name='train_model')
model = recorder.load_object('trained_model')
pred_df = recorder.load_object('pred.pkl')
```


Tracking capabilities:
- **Params** -- Hyperparameters logged as flat key-value pairs
- **Metrics** -- `R.log_metrics(mae=0.5, rmse=0.3, step=epoch)` with step tracking
- **Objects** -- Arbitrary Python objects saved/loaded (models, DataFrames)
- **Search** -- `R.search_records([exp_ids], order_by=['metrics.mae ASC'])`


### Record Types for Reproducible Pipelines


```python
# SignalRecord -- generates predictions, saves pred.pkl
sr = SignalRecord(model, dataset, recorder)
sr.generate()

# PortAnaRecord -- runs backtest, calculates metrics, saves reports
par = PortAnaRecord(recorder, port_analysis_config, 'day')
par.generate()
# Saves: report_normal_1day.pkl, positions_normal_1day.pkl, port_analysis_1day.pkl
```


### Workflow-by-Code Configuration


Complete pipelines defined in structured config dictionaries:
```python
task = {
    'model': {'class': 'LGBModel', 'module_path': 'qlib.contrib.model.gbdt', 'kwargs': {...}},
    'dataset': {'class': 'DatasetH', 'module_path': 'qlib.data.dataset', 'kwargs': {...}},
}

port_analysis_config = {
    'executor': {'class': 'SimulatorExecutor', 'kwargs': {'time_per_step': 'day'}},
    'strategy': {'class': 'TopkDropoutStrategy', 'kwargs': {'topk': 50, 'n_drop': 5}},
    'backtest': {'start_time': '...', 'end_time': '...', 'exchange_kwargs': {...}},
}
```


Config-driven instantiation via `init_instance_by_config()` -- avoids factory pattern boilerplate.


### Online Model Rolling


Qlib supports automatic model retraining on schedule:
- Train -> backtest -> evaluate -> if degraded, trigger retrain
- Models deployed to online serving endpoint
- Continuous monitoring of prediction drift


### Applicable Patterns for Our Bot


1. **Experiment directory per run** -- Instead of scatter/trial dirs, standardize on:
   ```
   experiments/
     2026-08-19_ema_cross_opt1/
       params.yaml          <- exact config used
       model.pkl            <- saved model
       metrics.json         <- backtest results
       logs/                <- training output
       pred.csv             <- out-of-sample predictions
   ```

2. **Standard record classes** -- Create reusable pipeline stages:
   - `SignalGenerator(model, dataset)` -> generates and saves predictions
   - `BacktestRunner(strategy_config, exchange_config)` -> runs and saves reports
   - `ReportAggregator()` -> compiles metrics across experiments

3. **Config-as-source-of-truth** -- Every run fully reproducible from config YAML:
   - Current approach uses code kwargs scattered across functions
   - Centralize all parameters in a single config file per run

4. **Metric comparison dashboard** -- Query experiments by criteria:
   - `find_best_experiment(metric='sharpe_ratio', direction='asc')`
   - Plot performance over time -> detect degradation early


---

## Summary: Priority Implementation Recommendations


| Priority | Pattern | Effort | Impact | Files Affected |
|----------|---------|--------|--------|----------------|
| **P0** | Two-phase processor pipeline (fit/transform) | Low | High | `data.py` -- add `ProcessorChain` class with strict fit/transform separation |
| **P0** | Factor library with configurable windows | Medium | High | `factors.py` -- add rolling operators (ROC, MA, STD, slope, rank) |
| **P1** | Cross-sectional normalization | Low | Medium | `processors.py` -- add `CSNorm` class (normalize per timestamp) |
| **P1** | Hold threshold / cooldown | Low | Medium | `engine.py` -- skip sells if held < N days |
| **P1** | IC tracking and reporting | Low | Medium | `analytics.py` -- add rolling IC computation |
| **P2** | Experiment tracking infrastructure | Medium | Medium | `experiments/` dir convention, `run_pipeline.py` runner script |
| **P2** | Config-driven pipeline runner | High | Medium | `config.py` extension + new runner module |
| **P3** | Binary feature caching | High | Low-Medium | `fetch_history.py` -- add caching layer |
