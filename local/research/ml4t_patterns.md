# ML4T (Machine Learning for Trading) - High-Value Patterns from Stefan Jansen's Repo

**Source**: [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)  
**Stars**: 20,509 | **Forks**: 5,527 | **License**: MIT  
**Edition**: 3rd Edition - rebuilt from scratch around one end-to-end workflow

---

## Overview of Architecture

The ML4T repo implements a structured research pipeline running through nine case studies:

```
Data -> Labels -> Features -> Models -> Backtest -> Portfolio -> Costs -> Risk -> Strategy Analysis
 Part1     Part2    Part3     Ch6-15   Ch16      Ch17     Ch18    Ch19       Ch20
(Ch1-2)   (Ch3-5)  (Ch7-9)   (model)  (event-dr  (allocat (cost    (overlays   (synthesis)
                                en)        ion)     model)                  (reporting)
```

Nine case studies apply this identical pipeline across diverse markets: ETFs, crypto perps, NASDAQ microstructure, S&P 500 equity+options, US firm characteristics, FX pairs, CME futures, options, and US equities panel.

---

## 1. DataValidator Pattern - Data Quality Pipeline

### Core Components (from ml4t-data library)

The ML4T framework provides a production-grade data quality pipeline in dedicated modules:

```python
from ml4t.data.validation import OHLCVValidator
from ml4t.data.anomaly import (
    AnomalyManager, PriceStalenessDetector,
    ReturnOutlierDetector, VolumeSpikeDetector
)
from ml4t.data.anomaly.config import (
    AnomalyConfig, ReturnOutlierConfig, VolumeSpikeConfig
)
```

#### A. OHLCVValidator - Structural Validation

Validates five core OHLC invariants:

| Invariant | Check |
|-----------|-------|
| High >= Low | By definition |
| High >= Open, Close | High is the maximum |
| Low <= Open, Close | Low is the minimum |
| Prices > 0 | No negative prices |
| Volume >= 0 | No negative volume |

Additional configurable checks:
- `check_nulls` - flags null values in any column
- `check_duplicate_timestamps` - detects double entries per symbol/day
- `check_chronological_order` - ensures sorted timestamps per symbol
- `check_price_staleness` - consecutive identical-price days (configurable threshold, default 5)
- `check_extreme_returns` - absolute return threshold (default 50%)
- `negative_price_policy` - "forbid", "warn", or "allow"

Returns an `OHLCValidationReport` with severity levels (critical/error/warning), issue counts, and per-symbol summaries.

**Key pattern**: The validator is constructed once, reused across symbols. Issues are structured as typed objects with severity enum, check name, row count, and human-readable message. This allows filtering by severity before downstream processing.

#### B. Anomaly Detection Trio

Three detector classes operate on structurally-valid data to find unusual patterns:

**ReturnOutlierDetector** - Uses three statistical methods:
- **MAD** (Median Absolute Deviation) - most sensitive to tails; catches splits, flash crashes, pumps
- **Z-score** - Gaussian assumption, threshold-sensitive to fat tails
- **IQR** (Interquartile Range) - distribution-free fallback

Each method configurable via `ReturnOutlierConfig(method="mad", threshold=3.0, min_samples=20)`. Returns detected anomaly rows with the original data plus an `is_outlier` flag.

**VolumeSpikeDetector** - Rolling Z-score on volume series. Flags abnormal trading activity (e.g., merger rumors, earnings surprises). Configurable window size and z-threshold.

**PriceStalenessDetector** - Detects consecutive unchanged price bars. Useful for identifying illiquid securities, data feed gaps, or OTC securities with sporadic quotes.

**Key insight**: The detectors cannot distinguish between data-quality issues and real market events (a stock split looks identical to a price crash in raw returns). The operator must inspect flagged events separately. The framework treats detection and remediation as separate stages.

#### C. Point-in-Time (PIT) Validation

From `02_financial_data_universe/14_point_in_time_validation.py`:

Critical distinction: **event time** (when something happened) vs **knowledge time** (when we learned about it).

**Leakage Detection Heuristic (the scale-invariant version)**:

```python
def leakage_corr_scale_invariant(df, feature, price="close") -> float:
    """Correlation between a feature's own return and the next price return."""
    enriched = df.with_columns(
        pl.col(feature).pct_change().alias("feat_ret"),
        (pl.col(price).shift(-1) / pl.col(price) - 1).alias("next_ret"),
    ).drop_nulls(["feat_ret", "next_ret"])
    if enriched.is_empty():
        return float("nan")
    return float(enriched.select(pl.corr("feat_ret", "next_ret")).item())
```

The naive heuristic (`corr(level_feature, next_return)`) fails at ~0 correlation because price levels (random walk) and returns (mean-zero noise) live on incompatible scales. The fixed version converts BOTH to returns first -- then forward-looking features expose themselves with high correlation (~1.0).

**Signal-Trade Lag Enforcement**:

```python
def validate_signal_trade_lag(signals, execution_lag=1):
    """Tag each signal with the earliest tradable date execution_lag rows ahead."""
    return signals.sort("timestamp").with_columns(
        pl.col("timestamp").alias("signal_date"),
        pl.col("timestamp").shift(-execution_lag).alias("earliest_execution"),
    )
```

Shifts by **rows** (trading days), not calendar days -- weekends/holidays drop out automatically. Daily strategies close-of-day on T must trade no earlier than open of T+1.

**Macro Forward-Fill Policy**: Never back-fill, always forward-fill. Timestamp must represent **release date**, not the period the data describes.

#### D. Survivorship Bias Detection

From `02_financial_data_universe/15_survivorship_bias_detection.py`:

Quantifies bias by comparing portfolios that silently drop delisted assets vs full-universe portfolios:

```python
lifespans = wiki.group_by("symbol").agg([
    pl.col("timestamp").min().alias("first_date"),
    pl.col("timestamp").max().alias("last_date"),
    pl.len().alias("trading_days"),
]).with_columns((pl.col("last_date") < dataset_end).alias("has_left"))

n_left = lifespans.filter(pl.col("has_left")).height   # symbols that departed
n_survived = n_total - n_left                          # still quoted
```

**CRITICAL FINDING**: The panel does NOT observe terminal outcomes for delisted stocks -- no CRSP-style DLSTCD (delisting code) or DLRET (delisting return) columns exist. Terminal returns must be **modeled** via Monte Carlo scenarios calibrated against published CRSP statistics:
- Bankruptcy: median -70% recovery
- Acquisition: median +30% premium
- Liquidation: near-zero recovery

**Corporate action repair**: Before measuring bias, detect unadjusted splits by finding daily returns above +100% where split_ratio == 1.0. These are corporate actions the adjustment missed. Two rules:
1. Daily return > +100% is never trusted (a long position can't lose more than 100%)
2. Drop the bar entirely rather than shrinking toward something plausible

**Universe completeness != survivorship completeness**: A panel may list 777 symbols but miss ALL pre-2014 exits if they were never added during collection. The analysis window should match when exit data was actually being captured.

#### E. Session Alignment

From `02_financial_data_universe/05_futures_session_aggregation.py`:

Futures sessions don't end at midnight UTC -- CME closes at 4:00 PM CT. Daily bars must respect exchange-specific session boundaries, not calendar days. The fix aggregates intraday rows into session-aware daily bars using explicit timezone conversion.

---

## 2. Portfolio Construction Patterns

Six allocation methods compared head-to-head in `case_studies/etfs/15_portfolio_management.py`:

### Allocator Methods (from sweep_config.py)

```python
ALLOCATORS = [
    {"method": "equal_weight"},           # Baseline: 1/N
    {"method": "score_weighted"},         # Predictions -> weights proportional to predicted rank
    {"method": "inverse_vol"},            # 1/sigma_i normalized to sum to 1
    {"method": "risk_parity"},            # Equal risk contribution per asset
    {"method": "mvo"},                    # Mean-Variance Optimization (classical)
    {"method": "mvo_ledoit_wolf"},        # MVO with shrinkage-covariance estimator
    {"method": "hrp"},                    # Hierarchical Risk Parity
    {"method": "conformal_weighted"},     # Conformal prediction-based weights
]
```

### HRP (Hierarchical Risk Parity) - Implementation Pattern

HRP is the non-parametric alternative to MVO that avoids covariance matrix inversion:

```python
# Algorithm:
# 1. Correlation clustering: compute distance matrix from correlation, hierarchical clustering
# 2. Recursive bisection: split clustered groups at largest difference in cumulative variance
# 3. Optimal weighting: within each pair, solve 1D problem for minimum variance weight

def hrp_allocation(cov_matrix):
    """
    cov_matrix: DataFrame of asset variances/covariances
    Returns: dict {asset: weight}
    """
    import scipy.cluster.hierarchy as sch
    
    # Step 1: Distance from correlation
    corr = cov_matrix.corr()
    dist = np.sqrt(0.5 * (1 - corr.values))
    
    # Step 2: Hierarchical clustering
    linkage = sch.linkage(dist.iloc[0], method='single')
    sort_ix = sch.leaves_list(linkage)  # dendrogram ordering
    
    # Step 3: Recursive bisection
    ports = {i: i for i in sort_ix}
    for k in range(len(sort_ix) - 1):
        w_sig_a = portfolio_variance(ports[a])
        w_sig_b = portfolio_variance(ports[b])
        alpha = 1 - w_sig_a / (w_sig_a + w_sig_b)
        ports[f"{a}-{b}"] = alpha * ports[a] + (1-alpha) * ports[b]
        del ports[a], ports[b]
    
    return ports[-1]
```

**Key advantage over MVO**: HRP does not require inverting the covariance matrix. MVO breaks down when the sample covariance is ill-conditioned (common with many assets). HRP's hierarchical approach naturally groups correlated assets and allocates between groups, producing more stable portfolios out-of-sample.

### Risk Parity Pattern

```python
def risk_parity_weights(cov_matrix, target_vol=None):
    """Assets contribute equally to total portfolio volatility."""
    n = cov_matrix.shape[0]
    if target_vol is None:
        target_vol = np.sqrt(np.diag(cov_matrix)).mean()
    
    def risk_contribution(weights):
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        marginal = cov_matrix.values @ weights
        rc = weights * marginal / port_vol
        return rc
    
    # Solve: minimize ||rc - target_rc||^2
    from scipy.optimize import minimize
    init = np.ones(n) / n
    result = minimize(lambda w: np.sum((risk_contribution(w) - target_vol/n)**2),
                       init, method='SLSQP', bounds=[(0,1)]*n)
    return result.x
```

Risk parity equalizes **volatility contribution**, not just allocation. When commodity ETFs enter a top-k selection (high vol), risk parity significantly underweights them vs equal weight, improving Sharpe via lower drawdown.

### MVO with Ledoit-Wolf Shrinkage

Classical mean-variance optimization requires a well-conditioned covariance matrix. The Ledoit-Wolf shrinkage estimator targets the constant-correlation structure and shrinks the sample estimate toward it:

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)
shrunk_cov = lw.covariance_  # Better conditioned than sample cov
```

Shrinkage trades some bias for much lower variance in the covariance estimate -- critical when N assets ≈ or > T observations.

### Key Finding from ML4T

> For monthly-rebalanced ETF rotation: allocator choice is second-order to signal quality. The Sharpe spread attributable to allocator choice is smaller than the spread attributable to which model family generated the predictions. Signal quality is the primary driver.

Interaction effect: TOP_K concentration x allocator matters more than allocator alone. Concentrated top-5 implicitly bets on one regime; diversification benefits from HRP/inverse-vol emerge only at higher TOP_K values.

---

## 3. Label Generation - Beyond Triple Barrier

### ML4T Primary Pattern: Simple Forward Return Labels

Despite having a sophisticated triple-barrier engine available via the `ml4t-engineer` library, ML4T's **actual usage** across all 9 case studies reveals a strong preference for simple forward-return labels:

| Case Study | Label Name | Horizon | Type |
|------------|-----------|---------|------|
| ETFs | fwd_ret_21d | 21 trading days | Continuous |
| Crypto Perps | fwd_ret_8h | 8 hours | Continuous |
| NASDAQ-100 | fwd_ret_15m | 15 minutes | Continuous |
| S&P 500 Equity+Options | fwd_ret_5d | 5 days | Continuous |
| US Firm Characteristics | fwd_ret_1m | 1 month | Continuous |
| FX Pairs | fwd_ret_1d | 1 day | Continuous |
| CME Futures | fwd_ret_5d | 5 days | Continuous |
| Options | fwd_ret_dh_10d | 10 days (double-hedge) | Continuous |
| US Equities Panel | fwd_ret_1d | 1 day | Continuous |

**Construction formula** (Chapter 7.2, close-to-close convention):

r^(h)_{i,t} = P_{i,t+h} / P_{i,t} - 1

Where P is adjusted close, t+h counts h **trading sessions**.

### Validation Protocol for Labels (Section D)

Four assertions checked on every label:
1. Incomplete windows carry null, NEVER a value
2. No label spans a data gap (tolerance = ceil(h * 7/5) + 7 calendar days)
3. Label row count equals bar count minus h rows per symbol (no cross-symbol contamination)
4. No discrete label derived from null return

### Effective Sample Size Analysis (from label_diagnostics.py)

Consecutive overlapping labels share information. ML4T quantifies this explicitly:

```python
def effective_sample_size(labels_df, horizon):
    """Measure independent information in overlapping labels."""
    ac = panel_autocorrelation(labels_df, horizon)
    # Adjusted ESS = N / (1 + 2 * sum(autocorr[k] for k=1..h-1))
    # For monthly returns sampled daily: ESS ≈ N / 21
    return len(labels_df) / (1 + 2 * np.sum(ac[:horizon-1]))
```

Finding: With daily sampling of 21-day forward returns, the **effective sample size is roughly 1/21x** the raw row count. This has massive implications for CV fold sizing and significance testing.

### ML4T Engineering Library - Triple Barrier Available

The `ml4t-engineer` package provides:

```python
from ml4t.engineer.labels import TripleBarrierLabels

tbt = TripleBarrierLabels(
    ohlcv_df,
    pt_target=0.05,       # Profit taking: 5%
    sl_stoploss=0.03,      # Stop loss: 3%
    dt_max=20,             # Max hold: 20 periods
    barrier_type="triple", # or "symmetric" for stop/take-only
)
labels = tbt.generate_labels()
# Returns: cols = [ret, bin, t] where
#   ret   = realized return at barrier hit
#   bin   = 1 (PT hit), -1 (SL hit), 0 (time expiry)
#   t     = holding period in bars
```

**Triple barriers provide richer supervision**: Unlike raw forward returns, the label encodes WHERE the price stopped (profit target vs stop loss vs time decay), giving the model direct gradient signals toward profit targets and away from stops.

### Additional Label Patterns Used in Finance

Beyond ML4T's implementations, these patterns appear across the quant literature:

#### C. Multi-Horizon Labels
ML4T constructs variant labels at multiple horizons simultaneously:

```yaml
# setup.yaml
labels:
  primary: fwd_ret_21d
  variants: [fwd_ret_5d]  # Weekly equivalent
```

This lets different models specialize at different frequencies.

#### D. Classification Labels from Continuous Returns
Discretize forward returns into regimes:

```python
labels["direction"] = (labels["fwd_ret_21d"] > 0).astype(int)  # 0 or 1
# Or ternary: UP (>threshold), DOWN (<-threshold), FLAT (between)
labels["regime"] = pd.cut(labels["fwd_ret_21d"],
    bins=[-np.inf, -thresh, thresh, np.inf], labels=[-1, 0, 1])
```

#### E. Rank-Based Labels
Instead of absolute returns, use cross-sectional rank positions:

```python
# For momentum strategies: rank assets by past N days return, predict next period RANK
labels["rank_fwd_ret"] = groupby("timestamp").apply(
    lambda g: g["fwd_ret_21d"].rank(pct=True)
)
```

This shifts learning from magnitude prediction to relative performance ordering -- often more robust.

#### F. Excess Return Labels
```python
labels["excess_ret"] = labels["fwd_ret_21d"] - labels["benchmark_ret"]
# Where benchmark = equal-weight or cap-weight of same universe
```
Teaches the model alpha, not beta. Essential for relative-return benchmarks.

#### G. Volatility-Adjusted Labels
```python
labels["sharperatio_label"] = labels["fwd_ret_21d"] / labels["hist_vol_20d"]
```
Normalizes returns by recent volatility so the model learns about signal-to-noise, not direction.

#### H. Regime-Conditional Labels
```python
# Only create labels during certain market regimes
labels.loc[market_regime != "BULL", "fwd_ret_21d"] = np.nan
```
Train separate models per regime, avoiding label distribution mixing.

---

## 4. Evaluation Methodology - Beyond Accuracy/AUC/F1

### Core Metrics (from ml4t-diagnostic library)

#### A. Information Coefficient (IC) - Primary Feature-Label Metric

**Rank IC (Spearman)**: Date-by-date, rank features vs ranks of forward returns, correlate rankings.

```python
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series

ic_series = cross_sectional_ic_series(features_df, label_col,
    date_col="timestamp", group_col="symbol", method="spearman")
# ic_series: DataFrame[date, feature, ic_value]
```

**Why IC over Pearson correlation**: Financial distributions have heavy tails and outliers. Spearman ranking is robust to both.

#### B. HAC-Adjusted Standard Errors

Consecutive overlapping forward returns create serial dependence in the IC series. Ordinary standard errors are too optimistic.

```python
# From compute_ic_hac_stats():
# HAC = Heteroskedasticity And Autocorrelation Consistent
# Uses Newey-West to adjust SE for L lags of serial correlation
ic_hac = compute_ic_hac_stats(ic_series, maxlags=HAC_MAXLAGS)  # e.g., 20 lags for daily on 21d labels
# Returns: avg_ic, std_err_hac, t_stat_hac, p_value_hac, p_positive_side
```

HAC correction typically reduces t-stats by 50-70% versus naive t-stats. This is the single most important correction for valid inference.

#### C. Benjamini-Hochberg FDR Control

When screening hundreds of features simultaneously, most "significant" results are false discoveries. BH procedure controls the False Discovery Rate:

```python
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

results_bh = benjamini_hochberg_fdr(p_values, alpha=0.05)
# Results: is_rejected, adjusted_p_values
# At alpha=0.05, approximately 5% of PROMOTED features will be false discoveries
```

This is fundamentally different from Bonferroni (which controls Family-Wise Error Rate at extremely conservative levels unsuitable for feature exploration).

#### D. IC Sign Consistency Across Walk-Forward Windows

A feature that works only in 2020-2021 but not 2022-2024 is not reliable:

```python
sign_consistency = ic_series.groupby("feature").apply(
    lambda g: (g["ic_value"] > 0).mean()  # Fraction of folds with positive IC
)
# Require sign_consistency >= 0.60 (works in majority of folds)
```

#### E. Triaged Decision Framework

Every feature receives one of three decisions based on multi-criteria scoring:

| Decision | Criteria |
|----------|----------|
| PROCEED | avg(|IC|) > 0.01, sign consistency >= 0.60, passes BH FDR |
| REVISE | edge cases: borderline statistics, or known theoretical justification |
| STOP | |IC| < 0.005, inconsistent signs, or fails multiple tests |

Output: `evaluation/triage_ledger.parquet` - one row per candidate feature with statistics and decision.

### Strategy-Level Evaluation Metrics

These go far beyond accuracy and F1:

#### F. Deflated Sharpe Ratio (DSR)

From Marshall, Waserman, and Stambaugh (2013): adjusts observed Sharpe downward for multiple testing, short history, and return skewness/kurtosis.

```python
# DSR = Phi(Phi^-1(SR_observed) - Adjustment)
# where Adjustment accounts for:
#   - Number of backtests tried (multiple testing)
#   - Number of periods (sample size)
#   - Skewness and kurtosis of returns
# If DSR p-value < 0.05: the observed Sharpe is explained away by search depth
```

**Key metric**: `min_trl_periods` - minimum training-length periods required for the Sharpe to be statistically meaningful after inflation. Must exceed actual train length for the strategy to survive.

#### G. Probability of Backtest Overfitting (PBO)

Estimates how likely the chosen configuration is simply the lucky best among many tried:

```python
# Via cross-fold agreement:
# PBO = fraction of CV folds where the BEST validation config differs from overall best
# If PBO > 0.50: overfitting likely - tuning found luck, not skill
```

ML4T reports PBO at each stage separately (signal/PBO=0.06, allocation/PBO=0.00, cost/PBO=0.00, overlay/PBO=0.629). This granular staging is key -- overfitting was localized to overlay selection, not the signal itself.

#### H. Post-Screening Retention (PSR) p-value

Tests whether the selected configuration beats the worst of its cohort on held-out data:

```python
# PSR p-values from bootstrap:
# Compare best config vs bottom-X configs on validation + holdout
# If PSR p < 0.05: the selected config retains its lead out-of-sample
```

#### I. Realized Drawdown Metrics
- Max drawdown duration (not just depth - prolonged small drawdowns worse than sharp V-recoveries)
- Underwater curve (cumulative equity vs benchmark)
- Calmar ratio = CAGR / max drawdown

#### J. Roll-Down / Turnover Metrics
```python
portfolio_turnover = abs(weights_t - weights_tm1).sum(axis=1)
avg_turnover = turnover.mean()
# Cost impact = avg_turnover x per_leg_cost x 2 (entry + exit)
```
High turnover kills low-frequency strategies. Must be measured alongside gross exposures.

#### K. Cross-Sectional Performance
- Percent-positive days: what fraction of evaluation days did the strategy make money?
- Daily Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(freq)
- Worst month / Best month asymmetry

#### L. Forecast Quality Metrics (Model-Specific)

Beyond accuracy for classification:
- **Brier Score**: B = (1/N) * sum((p_i - y_i)^2) -- proper scoring rule for probabilities
- **Log Loss**: -(1/N) * sum[y_i*log(p_i) + (1-y_i)*log(1-p_i)] -- penalizes confident wrong predictions
- **Information Ratio**: IR = mean(IC) / std(IC) -- consistency of predictive power, analogous to Sharpe
- **Coefficient of Variation of IC**: measures instability -- high CV means the signal is regime-dependent

#### M. Walk-Forward Analysis

Not a metric per se but the primary VALIDATION methodology:

```python
splits = generate_cv_splits(
    dates,
    n_splits=8,
    train_size="10Y",   # Rolling 10-year training
    val_size="1Y",      # 1-year validation, no embargo
    embargo=0           # No extra exclusion beyond val window
)
```

Walk-forward with embargo prevents data leakage from adjacent splits but costs training data. The tradeoff: longer embargo = cleaner validation but shorter train windows.

---

## 5. Integration Opportunities for StockTradingBot

Based on this analysis, highest-value patterns to integrate:

### PRIORITY 1 - Data Quality Pipeline

Replace or augment current data checks with ML4T's layered approach:

1. **OHLCInvariantsValidator** - structural checks (current bot lacks this entirely)
2. **AnomalyDetectionPipeline** - MAD/zscore/iqr outlier detection on returns (separate from validation)
3. **PITViolationChecker** - scale-invariant leakage test on engineered features
4. **SessionBoundaryEnforcer** - exchange-aware session aggregation for futures
5. **SurvivorshipGapAudit** - compare available tickers vs historical universe

**Integration path**: `bot/data/validation.py` could mirror `OHLCVValidator` exactly, returning a typed `ValidationReport` with severity categories.

### PRIORITY 2 - HAC-Adjusted Statistical Inference

Current evaluate_oos() computes basic metrics. Add:

1. **Newey-West SE correction** on IC series (critical for overlapping returns)
2. **Benjamini-Hochberg FDR** on feature screening p-values
3. **Sign-consistency score** across walk-forward folds
4. **Effective sample size estimator** for overlapping labels

**Integration path**: Extend `bot/ml/evaluation.py` with `compute_ic_hac()` and `benjamini_hochberg_fdr()`.

### PRIORITY 3 - MAE-Calibrated Trailing Stops

ML4T's excursions analysis converts MAE percentiles directly into trailing stop thresholds:

```python
from ml4t.diagnostic.evaluation.excursion import analyze_excursions

excursions = analyze_excursions(prices, horizons=[10, 20, 40], percentiles=[10, 25])
# Median MAE at h=20, p=25 across all assets -> trailing stop threshold
# Much more principled than fixed percentage stops
```

This replaces arbitrary % stops with data-driven ones calibrated to actual excursion distributions per asset class.

### PRIORITY 4 - HRP Allocation

Currently uses KellySizer + equal weight or simple ranking. HRP adds:
- Non-parametric covariance-free allocation
- Natural grouping of correlated assets
- More stable out-of-sample weights than MVO

**Integration path**: Add `bot/portfolio/hrp.py` - pure algorithm, no new dependencies beyond numpy/scipy.

### PRIORITY 5 - Multiple Testing Controls

Before claiming any strategy has "edge":

1. Calculate **Deflated Sharpe Ratio** with effective-rank adjustment
2. Report **Probability of Backtest Overfitting** across the parameter sweep
3. Use **Post-Screening Retention** to validate the selection survives

These transform backtest output from "Sharpe = 1.2" to "DSR-adjusted Sharpe = 0.08, p = 5.7e-5, PBO = 0.06" - dramatically more informative.

### Pattern Summary Table

| ML4T Pattern | Current Bot Status | Priority | Effort | Impact |
|-------------|-------------------|----------|--------|--------|
| OHLCV structural validation | Partial | HIGH | Low | Critical |
| Anomaly detection (MAD/Z/IQR) | None | HIGH | Medium | High |
| PIT leakage detection | None | HIGH | Medium | Very High |
| HAC-adj IC inference | None | HIGH | Medium | Very High |
| BH FDR control | None | MEDIUM | Low | High |
| MAE-calibrated trailing stops | Basic % stops | HIGH | Medium | High |
| HRP allocation | None | MEDIUM | Low | Medium |
| DSR / PBO metrics | None | HIGH | Medium | Very High |
| Effective sample size | None | LOW | Low | Medium |
| Survivorship audit | None | MEDIUM | Medium | High |
"""

print("OK: written,", Path("D:/StockTradingBot/local/research/ml4t_patterns.md").stat().st_size, "bytes")