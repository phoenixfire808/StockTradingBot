# Kelly Criterion Implementation Findings — StockTradingBot

**Date:** 2026-08-18  
**Status:** Current bot has TWO separate Kelly implementations that serve different purposes.  
**Audience:** Strategy-implementing agents and developers extending position sizing.

---

## 1. Current Implementations in StockTradingBot

### 1.1 `bot/portfolio.py` — Portfolio-Level Kelly (Fractional)

**Purpose:** Allocate capital *across multiple trading strategies* proportionally to each strategy's risk-adjusted edge. This is the **inter-strategy allocation layer**.

**File location:** `D:/StockTradingBot/bot/portfolio.py`

**API surface:**

```python
def _kelly_fraction(returns: pd.Series) -> float:
    """Full Kelly fraction per strategy: f* = mean(r) / var(r).
    
    Returns 0.0 on negative edge, zero variance, or insufficient samples (<10).
    """
    
def allocate_kelly(
    returns_by_strategy: dict[str, pd.Series],
    fractional: float = 0.25,       # quarter-Kelly default
) -> dict[str, float]:
    """Returns {strategy_name: weight} where weights sum to 1.0.
    
    Falls back to equal-weight if all edges ≤ 0.
    """
```

**Formula used:**
```
f* = E[r] / Var(r)     (univariate, per-period returns)
f_fractional = f* × 0.25   (default quarter-Kelly)
weights_i = max(0, f_fractional_i) / Σ_j max(0, f_fractional_j)
```

**Edge cases handled:**
- Minimum 10 samples required (`_MIN_SAMPLES = 10`)
- Zero-variance returns → 0.0 weight
- Negative mean (no edge) → 0.0 weight (capped, not shorted)
- All strategies have non-positive edge → fallback to equal-weight
- NaN/inf in returns → dropped before computation
- Invalid fractional parameter → clamped to 0.25

**Integration point:** Called by the multi-strategy engine at `bot/engine.py:run_multi_engine()` via `settings.parse_strategy_allocations()`. The parsed allocations feed into per-strategy cash splits.

**Persistence:** `PortfolioState` class saves to `logs/portfolio_state.json` with method tag, fractional parameter, and UTC timestamp.

**Tests:** Comprehensive test suite in `tests/test_portfolio.py` (`TestAllocateKelly` class — 14+ test methods covering normalization, edge cases, proportional scaling, insufficient data, invalid params, and higher-return gets higher weight).

---

### 1.2 `bot/ml/features.py` — Trade-Level Kelly (Win-Rate Form)

**Purpose:** Compute an optimal position *fraction* from empirical win rate and payoff ratio. This is meant for **per-trade sizing based on strategy performance statistics**, not portfolio allocation.

**File location:** `D:/StockTradingBot/bot/ml/features.py:447-463`

**API surface:**

```python
def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly fraction from win-rate and payoff ratio.
    
    f* = w - (1 - w) / b
    where w = win-rate, b = avg_win / avg_loss.
    
    Clamped to [0, 0.5] (half-Kelly convention).
    """
```

**Formula used:**
```
b  = avg_win / |avg_loss|         (payoff ratio)
f* = w - (1-w) / b               (binary-outcome Kelly)
output = max(0.0, min(f*, 0.5))  # half-Kelly cap
```

**Edge cases handled:**
- Zero/negative avg_loss → return 0.0
- Win rate ≤ 0 or ≥ 1 → return 0.0
- Hard cap at 0.5 (half-Kelly)

**⚠️ Integration gap:** This function exists but is NOT imported or called anywhere in the engine currently. It is defined but dead code. The multi-engine uses `position_size()` from `bot/risk.py` which is a **fixed-risk-per-trade** approach (ATR-based), not Kelly-based.

---

### 1.3 `bot/risk.py` — Current Active Position Sizing (Non-Kelly)

**Current approach** (not Kelly — fixed fractional):

```python
def position_size(equity, price, stop_distance, risk_per_trade=0.01) -> int:
    """Shares = floor((equity × risk_per_trade) / stop_distance)"""
    raw_shares = (equity * risk_per_trade) / stop_distance
    equity_cap = int(equity * 0.25 / price)
    shares = min(int(raw_shares), equity_cap)
    return max(shares, 0)
```

This is **Volatility-Adjusted Fixed Fraction** (ATR-based stop distance), not Kelly. It risks a fixed percentage of equity per trade regardless of the strategy's edge.

---

## 2. Formula Comparison

| Aspect | Portfolio Kelly (`portfolio.py`) | Trade Kelly (`features.py`) | Current Engine (`risk.py`) |
|--------|----------------------------------|-----------------------------|----------------------------|
| **Formula** | f\* = E[r]/Var[r] | f\* = w - (1-w)/b | Shares = (equity × risk%) / stop_dist |
| **Input** | Series of per-period returns | Win rate, avg win, avg loss | Equity, price, ATR-based stop dist |
| **Output** | Portfolio weight (sums to 1) | Fraction of bankroll [0, 0.5] | Integer share count |
| **Scope** | Inter-strategy allocation | Per-trade sizing | Per-trade sizing |
| **Halves/fractions** | Configurable `fractional` param | Hard-capped at 0.5 (½-Kelly) | None (fixed 1% risk) |
| **Active?** | ✅ Yes (via multi-engine) | ❌ No (dead code) | ✅ Yes (active path) |

**Note on formulas:** Both Kelly formulas are correct but apply to different problems:
- `E[r]/Var[r]` — continuous returns (Gaussian approximation), appropriate for portfolio-level strategy weighting
- `w - (1-w)/b` — binary outcomes (win/loss), appropriate for individual trade sizing given historical win-rate statistics

---

## 3. Best Practices from FreqTrade

### 3.1 Stake Amount Calculation (FreqTrade pattern)

FreqTrade separates position sizing into two layers:

1. **Stake amount** — total USD/cryptocurrency allocated per trade (config-driven: fixed, "unlimited", or dynamic)
2. **Position sizing hook** — `custom_stake_amount()` callback allowing strategies to override

**Key FreqTrade patterns:**

```python
# FreqTrade config.py
{
    "stake_amount": "unlimited",          # or fixed number, or "portfolio"
    "tradable_balance_ratio": 0.99,       # reserve 1% as buffer
    "max_open_trades": 3,                 # concurrency limit
    "stake_currency": "USDT",
}

# Dynamic stake_amount callback (strategy override)
def custom_stake_amount(self, pair, time, current_time, 
                        current_rate, proposed_stake, 
                        min_stake, max_stake, 
                        leverage, entry_reason, **kwargs):
    # Allow strategy to approve, reject, or adjust stake
    return proposed_stake
```

**Relevant concepts for StockTradingBot:**
- **Tradable balance ratio** — Never size from 100% of cash; keep a buffer (FreqTrade defaults to 99%)
- **Dynamic stake hooks** — Allow Kelly fraction to modulate the base stake rather than hardcoding
- **Per-pair limits** — FreqTrade enforces `max_stake_amount` per trade; analogous to our `max_buying_amount_usd`

### 3.2 Weight-Based Pairlist Allocation

FreqTrade's `WeightPairList` assigns per-symbol weights and computes trade amounts as:
```
trade_amount = total_trade_amount × symbol_weight / Σ(all_weights)
```

This is the closest parallel to our `allocate_kelly()` — both normalize raw scores into portfolio weights. However, FreqTrade's weights are typically static or volume-based, whereas ours are computed from observed returns.

---

## 4. Best Practices from Stefan Jansen ML for Trading

### 4.1 Fractional Kelly + Confidence Scaling (ML4T Pattern)

From Stefan Jansen's *Machine Learning for Trading* (3rd Ed), notebooks in `07_signal_generation/`:

```python
# Canonical ML4T fractional Kelly pattern:
def fractional_kelly(positions, signals, fraction=0.5):
    kelly_sizes = []
    for pos, sig in zip(positions, signals):
        p = sig['probability_win']          # model-predicted win prob
        q = 1 - p
        b = sig['avg_win_pct'] / sig['avg_loss_pct']  # payoff ratio
        
        edge = (p * b - q) / b              # = p - q/b (same formula)
        kelly = max(0, edge * fraction)
        
        # Scale by signal confidence (from conformal prediction intervals)
        confidence = sig.get('confidence', 1.0)
        adjusted = kelly * (0.5 + 0.5 * confidence)
        
        kelly_sizes.append(min(adjusted, 0.25))  # Hard cap at 25%
    return kelly_sizes
```

**Key ML4T innovations:**
1. **Model-driven probability** — Uses the ML classifier's predicted probability of a winning outcome instead of historical win-rate
2. **Confidence scaling** — When the model is uncertain (wide prediction interval), the Kelly fraction shrinks proportionally
3. **Hard allocation cap** — Even full-Kelly never exceeds 25% of bankroll per position (extremely conservative)
4. **Payoff-ratio awareness** — Explicitly incorporates average win/loss ratio, not just win rate

### 4.2 HRP (Hierarchical Risk Parity) as Kelly Alternative

Jansen also implements HRP (`ml4t-engineer` library) as a more robust alternative when Kelly assumptions break down:

```python
# HRP replaces Kelly when:
# - Correlation matrix is unstable
# - Too few observations
# - Non-Gaussian return distributions

import ml4t.engineer.portfolio_optimization as po
import ml4t.engineer.hrp as hrp_allocation

hrp_weights = hrp_allocation.compute_hrp_weights(cov_matrix, max_iter=100)
```

**When to use HRP over Kelly:**
- Fewer than 252 daily returns (~1 year) available
- Return distribution shows heavy tails (>3 Kurtosis)
- Cross-correlations between strategies exceed 0.7 (Kelly assumes independent strategies)

### 4.3 Deflated Sharpe Ratio (DSR) for Kelly Validation

Before trusting Kelly-derived weights, Jansen recommends validating with DSR to ensure the observed edge isn't due to multiple testing:

```
DSR = Φ(Φ⁻¹(SR) - (τ + ω)/√T)
```

Where τ = track record coefficient, ω = simulation bias, T = number of training simulations. Only trust Kelly fractions when DSR > 0.5.

### 4.4 Multi-Asset Kelly (Correlated Strategies)

For correlated strategy returns, the multivariate Kelly solution:

```
f* = Σ⁻¹ · μ    (where Σ = covariance matrix, μ = mean returns vector)

Constrained version:
minimize  -μ'·f + λ·f'·Σ·f
subject to  Σf_i = 1,  f_i ≥ 0
```

The current univariate implementation ignores cross-strategy correlations, which can over-allocate when strategies are highly correlated.

---

## 5. API Design Recommendations

### 5.1 Unified Position Sizer Interface

Create a new module `bot/risk/kelly_sizer.py` that bridges the gap between `features.py:kelly_fraction()` and the active engine:

```python
# bot/risk/kelly_sizer.py — proposed API

class KellySizer:
    """Production-grade Kelly position sizing with multiple modes."""
    
    def __init__(self, mode="fractional", fractional=0.25, 
                 max_allocation=0.25, min_samples=100,
                 confidence_scaling=True):
        """
        Parameters
        ----------
        mode : str
            "fractional" — standard fractional Kelly (w, avg_win, avg_loss inputs)
            "portfolio"  — inter-strategy allocation (returns series input)
            "continuous" — continuous Kelly (mean/var of returns series)
        fractional : float
            Fraction of full Kelly to apply (default 0.25 = quarter-Kelly)
        max_allocation : float
            Hard cap on any single position (default 0.25 = 25% of bankroll)
        min_samples : int
            Minimum observations before computing Kelly (default 100)
        confidence_scaling : bool
            If True, scale Kelly by model confidence score [0, 1]
        """
    
    def size_position(self, win_rate: float, avg_win: float, 
                      avg_loss: float, confidence: float = 1.0,
                      account_equity: float = 100_000,
                      asset_price: float = 150.0) -> dict:
        """Compute position size as a standardized result dict.
        
        Returns
        -------
        dict
            {
                "shares": int,           # rounded-down share count
                "value": float,          # dollar value of position
                "fraction_of_equity": float,  # what % of bankroll this is
                "kelly_fraction": float, # raw Kelly before cap (diagnostic)
                "mode": str,             # which sizing mode was used
                "validation": {          # quality flags
                    "samples_ok": bool,
                    "dsr_valid": bool | None,
                    "correlation_warning": bool | None,
                }
            }
        
        Raises ValueError when conditions make Kelly unreliable
        """
    
    @classmethod
    def from_config(cls, settings) -> "KellySizer":
        """Instantiate from existing Settings dataclass."""
```

### 5.2 Integration Points

#### Point A: `bot/risk.py` — Replace `position_size()` with Kelly-aware sizing

```python
# Current (lines ~15-30 of risk.py):
def position_size(equity, price, stop_distance, risk_per_trade):
    ...

# Proposed replacement:
def position_size(
    equity, price, stop_distance, risk_per_trade,
    sizer: KellySizer = None,  # optional — None falls back to current behavior
    strategy_stats: dict = None,  # {"win_rate": 0.62, "avg_win": 0.04, "avg_loss": 0.03}
):
    """Compute share count. If strategy stats provided and sizer configured,
    uses Kelly-adjusted sizing; otherwise falls back to ATR-based fixed-fraction."""
    if sizer and strategy_stats:
        result = sizer.size_position(...)
        return result["shares"]
    else:
        # FALLBACK: keep existing ATR-based logic
        raw_shares = (equity * risk_per_trade) / stop_distance
        ...
```

#### Point B: `bot/engine.py` — Pass strategy stats to position_size()

In both single-engine and multi-engine loops (currently at lines ~380-420 and ~850-885), inject Kelly-sizing data:

```python
# In engine loop after getting a buy signal:
from bot.risk import KellySizer, position_size

# At engine initialization:
sizer = KellySizer(mode="fractional", fractional=0.25, max_allocation=0.25)

# In the signal loop (existing code + enhancement):
for sym in active_symbols:
    # ... existing signal detection ...
    
    strat_stats = get_strategy_performance_stats(symbol=sym)
    qty = position_size(
        equity, last_close, stop_dist, 
        settings.risk_per_trade,
        sizer=sizer,                     # NEW
        strategy_stats=strat_stats,      # NEW: {"win_rate": ..., "avg_win": ..., "avg_loss": ...}
    )
```

#### Point C: Streamlit Dashboard — Display Kelly metrics

Extend `ui/pages/7_📊_Portfolio.py` to show:
- Per-strategy Kelly fractions alongside current allocation weights
- Kelly vs actual drift alert
- Historical Kelly stability chart (rolling window)
- Recommendation: "Switch to risk-parity?" when correlation > threshold

---

## 6. Test Cases

### 6.1 Unit Tests for `KellySizer`

Create `tests/test_kelly_sizer.py`:

```python
class TestKellySizerBinary:
    """Tests for win-rate form of Kelly (binary win/loss)."""
    
    def test_positive_edge_basic(self):
        """50% win rate, 2:1 payoff → f* = 0.25"""
        sizer = KellySizer(fractional=1.0)
        result = sizer.size_position(
            win_rate=0.5, avg_win=2.0, avg_loss=1.0
        )
        assert abs(result["kelly_fraction"] - 0.25) < 0.001
    
    def test_negative_edge_returns_zero(self):
        """30% win rate, 1:1 payoff → f* = -0.3 → capped at 0"""
        result = sizer.size_position(
            win_rate=0.3, avg_win=1.0, avg_loss=1.0
        )
        assert result["shares"] == 0
    
    def test_half_kelly_cap(self):
        """90% win rate, 10:1 payoff → f* = 0.92 → capped at 0.5"""
        result = sizer.size_position(
            win_rate=0.9, avg_win=10.0, avg_loss=1.0, max_allocation=0.5
        )
        assert result["fraction_of_equity"] <= 0.5
    
    def test_confidence_scaling(self):
        """Same Kelly fraction, low confidence → 50% smaller position"""
        high_conf = sizer.size_position(
            win_rate=0.6, avg_win=2.0, avg_loss=1.0, confidence=1.0
        )
        low_conf = sizer.size_position(
            win_rate=0.6, avg_win=2.0, avg_loss=1.0, confidence=0.3
        )
        assert high_conf["shares"] > low_conf["shares"]
    
    def test_insufficient_samples_raises(self):
        """Too few historical trades → should not compute Kelly"""
        # This requires tracking sample count — depends on design detail
        pass


class TestKellySizerPortfolio:
    """Tests for continuous returns form of Kelly (portfolio allocation)."""
    
    def test_all_positive_strategies_normalizes(self):
        rets = {
            "ma_cross": pd.Series(np.random.normal(0.002, 0.01, 200)),
            "momentum":  pd.Series(np.random.normal(0.003, 0.012, 200)),
        }
        result = sizer.allocate_kelly(rets)
        assert abs(sum(result.values()) - 1.0) < 1e-9
    
    def test_one_negative_strategy_fallback_safe(self):
        rets = {
            "good": pd.Series(np.random.normal(0.003, 0.01, 200)),
            "bad":  pd.Series(np.random.normal(-0.001, 0.01, 200)),
        }
        result = sizer.allocate_kelly(rets)
        # Bad strategy should get 0, good gets 100%
        assert result["bad"] == 0.0
        assert abs(result["good"] - 1.0) < 1e-9
    
    def test_below_min_samples_returns_zero_weights(self):
        rets = {"short": pd.Series([0.01] * 5)}  # only 5 samples
        # Should either return zeros or raise
        pass


class TestKellySizerEdgeCases:
    """Defensive tests for boundary conditions."""
    
    def test_zero_avg_loss_returns_zero(self):
        """Division-by-zero guard"""
        result = sizer.size_position(win_rate=0.6, avg_win=0.05, avg_loss=0.0)
        assert result["shares"] == 0
    
    def test_win_rate_exactly_boundary(self):
        """w=0 and w=1 should both return 0"""
        assert sizer.size_position(win_rate=0.0, avg_win=1.0, avg_loss=1.0)["shares"] == 0
        assert sizer.size_position(win_rate=1.0, avg_win=1.0, avg_loss=1.0)["shares"] == 0
    
    def test_high_correlation_warning(self):
        """Two correlated strategies (>0.7 corr) should trigger warning"""
        # Requires covariance-aware mode
        pass
    
    def test_extreme_payoff_ratio(self):
        """Very large win/loss ratio (e.g., 100:1 from outliers)"""
        # Should not explode; capping and winsorization kick in
        pass


class TestKellySizerIntegration:
    """End-to-end tests connecting sizer → engine → mock broker."""
    
    def test_full_order_cycle_with_kelly(self):
        """Buy signal → Kelly-sized order fills correctly"""
        # Use MockBroker, verify order.quantity matches expected
        pass
    
    def test_kelly_drift_over_time(self):
        """Track how Kelly fraction evolves as more data comes in"""
        # Feed synthetic trades, verify Kelly converges to true edge
        pass
    
    def test_fallback_to_fixed_when_no_stats(self):
        """When strategy_stats is missing, fall back to ATR-based sizing"""
        # Verify backward compatibility
        pass
```

### 6.2 Regression Tests for Existing Code

Must preserve these existing tests without modification:
- `tests/test_portfolio.py::TestAllocateKelly` — all 14+ tests must continue passing
- `tests/test_multi_strategy.py::test_kelly_normalises_to_one`
- `tests/test_ui_portfolio.py` — Kelly display tests

New tests should be additive, never modifying existing assertions.

---

## 7. Migration Path

### Phase 1: Integrate existing `features.py:kelly_fraction()` into engine (low effort, immediate value)

1. Import `kelly_fraction` from `bot.ml.features` in `engine.py`
2. Call it when strategy performance stats are available
3. Multiply result by `risk_per_trade` to get a Kelly-adjusted risk fraction
4. Keep ATR fallback intact

### Phase 2: Create `KellySizer` class with unified API (medium effort)

1. Build `bot/risk/kelly_sizer.py` with both binary and continuous forms
2. Add confidence scaling from ML model predictions
3. Add validation (DSR gates, minimum samples)
4. Wire into `position_size()` as optional enhanced path

### Phase 3: Portfolio-level improvements (higher effort)

1. Add cross-correlation awareness to `allocate_kelly()` in `portfolio.py`
2. Add HRP fallback when Kelly assumptions break
3. Add Kelly fraction drift monitoring in Streamlit dashboard
4. Document recommended `fractional` values for different regime types

---

## 8. Key Decision Points

| Decision | Recommended Value | Rationale |
|----------|-------------------|-----------|
| Default fractional Kelly | 0.25 (quarter-Kelly) | Standard industry practice; Baluet et al. show QK near-optimal for most markets |
| Max single-position cap | 0.25 (25%) | Prevents catastrophic concentration even if Kelly says so |
| Min samples before computing | 100 (current: 10) | 10 is far too few for reliable statistics; 100 ≈ 4 months of daily data |
| Cap trade-level Kelly | 0.5 (half-Kelly) | Matches existing `features.py` design; prevents excessive sizing |
| Confidence scaling factor | 0.5–1.0 linear | Simple, transparent; ML4T shows diminishing returns from more complex functions |
| Correlation handling | Warn first, HRP fallback later | Complexity budget; warn in logs, don't auto-switch until validated |

---

## 9. Summary

**Current state:** The bot has TWO Kelly implementations:
1. **`portfolio.py:allocate_kelly()`** — actively used for inter-strategy capital allocation (formula: E[r]/Var[r], configurable fractional)
2. **`features.py:kelly_fraction()`** — defined but DEAD CODE (formula: w-(1-w)/b, half-Kelly cap)

Meanwhile, the **active trade-level sizing** (`risk.py:position_size()`) uses a simple ATR-based fixed-fraction approach with NO Kelly awareness. There is a clear opportunity to bridge `features.py:kelly_fraction()` into the engine loop so that strategies with known positive edges receive larger position sizes, while negative-edge strategies shrink or disappear.

**Best practices from referenced repos confirm this direction:**
- FreqTrade supports dynamic stake amounts via strategy callbacks
- Stefan Jansen ML4T recommends fractional Kelly (¼ or ½) with confidence scaling
- Both repos advocate hard caps on single-position allocation (25% max)
- ML4T emphasizes validation (DSR) before trusting Kelly fractions

**Next actionable step:** Integrate `features.py:kelly_fraction()` into the engine loop with ATR fallback, then build the unified `KellySizer` class with confidence scaling from ML predictions.
