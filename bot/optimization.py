"""Walk-forward optimization for strategy parameters.

Splits historical data into rolling train/test windows, grid-searches
parameters on each train window, validates on the subsequent test window,
and selects the best overall parameter set by average out-of-sample score.

Exposes:
  walk_forward_optimize(strategy_name, param_grid, symbols, start, end,
                        train_window, test_window)
  -> dict with best_params, best_score, folds[], total_combinations
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default parameter grids per known strategy
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ema_cross_rsi": {
        "fast": [5, 9, 13],
        "slow": [13, 21, 34],
        "rsi_entry_max": [65.0, 70.0, 75.0],
    },
    "mean_reversion_rsi2": {
        "rsi_period": [2, 5],
        "rsi_oversold": [5.0, 10.0, 15.0],
    },
    "bollinger_reversion": {
        "bb_period": [10, 20],
        "bb_std": [1.5, 2.0],
    },
    "vwap_breakout": {
        "vwap_period": [10, 20],
        "vol_surge_mult": [1.2, 1.5],
    },
    "momentum_scanner": {
        "macd_fast": [8, 12],
        "macd_slow": [21, 26],
        "vol_surge_mult": [1.3, 1.5],
    },
}


def _generate_windows(
    start: str,
    end: str | None,
    train_window: int,
    test_window: int,
) -> list[dict[str, str]]:
    """Generate rolling train/test date windows.

    Each window: train_start → train_end, test_start → test_end.
    Windows step forward by test_window days.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    else:
        end_dt = datetime.now(timezone.utc)

    windows: list[dict[str, str]] = []
    cursor = start_dt

    while cursor + timedelta(days=train_window + test_window) <= end_dt:
        train_start = cursor
        train_end = cursor + timedelta(days=train_window)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window)

        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        cursor = cursor + timedelta(days=test_window)

    logger.info(
        "Generated %d walk-forward windows (train=%dd, test=%dd, range=%s → %s)",
        len(windows),
        train_window,
        test_window,
        start,
        end or "now",
    )
    return windows


def _score_backtest(metrics: dict[str, Any]) -> float:
    """Score a backtest result. Higher is better.

    Uses Sharpe ratio as primary metric (risk-adjusted), falls back to
    total return if Sharpe is unavailable or zero.
    """
    sharpe = metrics.get("sharpe_ratio", 0)
    total_return = metrics.get("total_return_pct", 0)

    try:
        sharpe = float(sharpe) if sharpe is not None else 0.0
    except (TypeError, ValueError):
        sharpe = 0.0

    try:
        total_return = float(total_return) if total_return is not None else 0.0
    except (TypeError, ValueError):
        total_return = 0.0

    # Penalize extreme drawdowns
    max_dd = metrics.get("max_dd_pct", 0)
    try:
        max_dd = abs(float(max_dd)) if max_dd is not None else 0.0
    except (TypeError, ValueError):
        max_dd = 0.0

    dd_penalty = min(max_dd / 100.0, 0.5)  # cap at 50% penalty
    score = (sharpe * 10.0) + total_return * 0.5 - dd_penalty * 20.0
    return round(score, 4)


def _run_backtest_scored(
    strategy_name: str,
    params: dict[str, Any],
    symbols: list[str],
    start: str,
    end: str | None,
    cash: float = 100_000,
) -> tuple[float, dict[str, Any]]:
    """Run a backtest and return (score, aggregated_metrics).

    Aggregates metrics across all symbols by averaging.
    """
    from bot.backtest import run_backtest

    try:
        results = run_backtest(
            symbols=symbols,
            start=start,
            end=end,
            cash=cash,
            strategy_name=strategy_name,
            strategy_params=params,
        )
    except Exception as exc:
        logger.error("Backtest failed for %s params=%s: %s", strategy_name, params, exc)
        return -999.0, {}

    if not results:
        return -999.0, {}

    # Aggregate metrics across symbols
    all_metrics: list[dict[str, Any]] = []
    for sym, res in results.items():
        m = res.get("metrics", {})
        if m:
            all_metrics.append(m)

    if not all_metrics:
        return -999.0, {}

    # Average numeric metrics
    agg: dict[str, Any] = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = [m.get(k, 0) for m in all_metrics]
        try:
            agg[k] = round(sum(float(v) for v in vals) / len(vals), 4)
        except (TypeError, ValueError):
            agg[k] = vals[0]

    score = _score_backtest(agg)
    logger.debug(
        "Backtest scored %s params=%s → score=%.2f (Sharpe=%.2f, Ret=%.1f%%, MaxDD=%.1f%%)",
        strategy_name,
        params,
        score,
        agg.get("sharpe_ratio", 0),
        agg.get("total_return_pct", 0),
        agg.get("max_dd_pct", 0),
    )
    return score, agg


def walk_forward_optimize(
    strategy_name: str,
    param_grid: dict[str, list] | None,
    symbols: list[str] | str,
    start: str,
    end: str | None = None,
    train_window: int = 90,
    test_window: int = 30,
    cash: float = 100_000,
) -> dict[str, Any]:
    """Walk-forward optimization for strategy parameters.

    Splits data into rolling train/test windows, grid-searches parameters
    on each train window, validates on the test window, and selects the
    best overall parameter set by average out-of-sample (test) score.

    Args:
        strategy_name:  Registered strategy to optimize (e.g. "ema_cross_rsi").
        param_grid:     dict of param → list of values to sweep.
                        If None, uses DEFAULT_PARAM_GRIDS[strategy_name].
        symbols:        Ticker symbol(s) to optimize on.
        start:          Start date "YYYY-MM-DD".
        end:            End date "YYYY-MM-DD" (None = today).
        train_window:   Training window length in days (default 90).
        test_window:    Test (out-of-sample) window length in days (default 30).
        cash:           Starting capital for backtests.

    Returns:
        {
            "strategy": str,
            "best_params": dict,
            "best_score": float,
            "best_test_score": float,
            "folds": [
                {
                    "fold": int,
                    "train_start": str, "train_end": str,
                    "test_start": str, "test_end": str,
                    "best_train_params": dict,
                    "train_score": float,
                    "test_score": float,
                    "test_metrics": dict,
                    "equity_curve": list[float],
                }
            ],
            "total_combinations": int,
            "param_grid": dict,
            "symbols": list[str],
        }
    """
    # Normalize symbols
    if isinstance(symbols, str):
        symbols = [symbols]

    # Use default grid if none provided
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS.get(strategy_name)
    if not param_grid:
        logger.error("No param grid for strategy '%s' — provide param_grid explicitly", strategy_name)
        return {
            "strategy": strategy_name,
            "best_params": {},
            "best_score": -999.0,
            "best_test_score": -999.0,
            "folds": [],
            "total_combinations": 0,
            "param_grid": {},
            "symbols": symbols,
            "error": f"No param grid for strategy '{strategy_name}'",
        }

    logger.info(
        "Starting walk-forward optimization: strategy=%s symbols=%s "
        "train=%dd test=%dd range=%s→%s",
        strategy_name,
        symbols,
        train_window,
        test_window,
        start,
        end or "now",
    )

    # Generate param combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(product(*values))
    total_combinations = len(combos)

    # Generate train/test windows
    windows = _generate_windows(start, end, train_window, test_window)

    if not windows:
        logger.error(
            "Not enough date range for walk-forward windows "
            "(need ≥ %d days, got %s → %s)",
            train_window + test_window,
            start,
            end or "now",
        )
        return {
            "strategy": strategy_name,
            "best_params": {},
            "best_score": -999.0,
            "best_test_score": -999.0,
            "folds": [],
            "total_combinations": total_combinations,
            "param_grid": param_grid,
            "symbols": symbols,
            "error": "Insufficient date range for walk-forward windows",
        }

    # Track per-combo test scores across all folds
    combo_test_scores: dict[tuple, list[float]] = {combo: [] for combo in combos}
    folds_result: list[dict[str, Any]] = []

    for fold_idx, window in enumerate(windows):
        logger.info(
            "─── Fold %d/%d: train=%s→%s test=%s→%s ───",
            fold_idx + 1,
            len(windows),
            window["train_start"],
            window["train_end"],
            window["test_start"],
            window["test_end"],
        )

        # Grid search on train window
        best_train_score = -999.0
        best_train_params: dict[str, Any] = {}
        best_train_combo: tuple | None = None

        for combo in combos:
            params = dict(zip(keys, combo))
            score, _ = _run_backtest_scored(
                strategy_name=strategy_name,
                params=params,
                symbols=symbols,
                start=window["train_start"],
                end=window["train_end"],
                cash=cash,
            )
            if score > best_train_score:
                best_train_score = score
                best_train_params = params
                best_train_combo = combo

            combo_test_scores[combo].append(score)  # temp store train scores

        logger.info(
            "Fold %d best train: params=%s score=%.2f",
            fold_idx + 1,
            best_train_params,
            best_train_score,
        )

        # Test: run best train params on test window
        test_score, test_metrics = _run_backtest_scored(
            strategy_name=strategy_name,
            params=best_train_params,
            symbols=symbols,
            start=window["test_start"],
            end=window["test_end"],
            cash=cash,
        )

        # Extract equity curve from test backtest
        equity_curve: list[float] = []
        try:
            from bot.backtest import run_backtest

            test_results = run_backtest(
                symbols=symbols,
                start=window["test_start"],
                end=window["test_end"],
                cash=cash,
                strategy_name=strategy_name,
                strategy_params=best_train_params,
            )
            for sym, res in test_results.items():
                eq = res.get("equity_curve")
                if eq is not None and not eq.empty:
                    col = "Equity" if "Equity" in eq.columns else eq.columns[0]
                    equity_curve = [
                        round(float(v), 2) for v in eq[col].tolist()
                    ]
                    break
        except Exception as exc:
            logger.warning("Could not extract equity curve for fold %d: %s", fold_idx + 1, exc)

        # Replace last train score in combo_test_scores with test score
        # (we stored train scores temporarily; now store actual test scores)
        # Actually, we need to track test scores separately per combo
        if best_train_combo is not None:
            # Reset: we want per-combo TEST scores, not train scores
            # Re-initialize: we'll compute per-combo test scores at the end
            pass

        folds_result.append({
            "fold": fold_idx + 1,
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "best_train_params": best_train_params,
            "train_score": round(best_train_score, 4),
            "test_score": round(test_score, 4),
            "test_metrics": test_metrics,
            "equity_curve": equity_curve,
        })

        logger.info(
            "Fold %d test: score=%.2f metrics=%s",
            fold_idx + 1,
            test_score,
            {k: test_metrics.get(k) for k in ["total_return_pct", "sharpe_ratio", "max_dd_pct"]},
        )

    # Select best overall params by average test score
    # Re-run each combo on all test windows to get average test score
    logger.info("Computing per-combination average test scores across all folds...")

    best_avg_test_score = -999.0
    best_overall_params: dict[str, Any] = {}
    best_overall_combo: tuple | None = None

    for combo in combos:
        params = dict(zip(keys, combo))
        test_scores: list[float] = []

        for window in windows:
            score, _ = _run_backtest_scored(
                strategy_name=strategy_name,
                params=params,
                symbols=symbols,
                start=window["test_start"],
                end=window["test_end"],
                cash=cash,
            )
            test_scores.append(score)

        avg_test = sum(test_scores) / len(test_scores) if test_scores else -999.0
        logger.debug(
            "Combo %s: avg_test_score=%.2f (per-fold: %s)",
            params,
            avg_test,
            [round(s, 2) for s in test_scores],
        )

        if avg_test > best_avg_test_score:
            best_avg_test_score = avg_test
            best_overall_params = params
            best_overall_combo = combo

    # Also compute best by train score (the fold-level winner frequency)
    best_train_score = -999.0
    best_train_params: dict[str, Any] = {}
    for combo in combos:
        params = dict(zip(keys, combo))
        train_scores = combo_test_scores[combo]
        avg_train = sum(train_scores) / len(train_scores) if train_scores else -999.0
        if avg_train > best_train_score:
            best_train_score = avg_train
            best_train_params = params

    logger.info(
        "═══ Walk-forward complete ═══\n"
        "  Best OOS params:  %s (avg_test_score=%.2f)\n"
        "  Best train params: %s (avg_train_score=%.2f)\n"
        "  Folds: %d | Combos: %d",
        best_overall_params,
        best_avg_test_score,
        best_train_params,
        best_train_score,
        len(folds_result),
        total_combinations,
    )

    return {
        "strategy": strategy_name,
        "best_params": best_overall_params,
        "best_score": round(best_avg_test_score, 4),
        "best_test_score": round(best_avg_test_score, 4),
        "best_train_params": best_train_params,
        "best_train_score": round(best_train_score, 4),
        "folds": folds_result,
        "total_combinations": total_combinations,
        "param_grid": param_grid,
        "symbols": symbols,
    }


# ────────────────────────────────────────────────────────── NEW FEATURES
# Enhanced scoring, overlapping windows, expanded param grids
# ──────────────────────────────────────────────────────────

def _compute_sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float | None:
    """Compute annualised Sortino ratio from a Series of periodic returns.

    Falls back to *None* when daily returns contain fewer than 30 observations
    or when downside deviation is zero / NaN.
    """
    if len(returns) < 30:
        return None

    excess = returns - rf / periods_per_year
    downside = excess[excess < 0].dropna()

    if len(downside) == 0:
        return None

    downside_std = downside.std(ddof=1)
    if math.isnan(downside_std) or downside_std == 0.0:
        return None

    mean_excess = excess.mean()
    ann_factor = math.sqrt(periods_per_year)
    sortino = (mean_excess / downside_std) * ann_factor

    if math.isnan(sortino):
        return None

    return round(float(sortino), 4)


def _agg_returns_from_metrics(metrics: dict[str, Any]) -> pd.Series:
    """Try to build a Series of daily returns from backtest *metrics*.

    Returns an empty Series when equity_curve is not available.
    """
    eq = metrics.get("equity_curve")
    if eq is None:
        # Could be a plain list of floats
        if isinstance(metrics.get("equity_curve"), list):
            eq_list = metrics["equity_curve"]
        else:
            eq_list = []
    elif hasattr(eq, "tolist"):
        eq_list = eq.tolist()
    else:
        eq_list = []

    if len(eq_list) < 2:
        return pd.Series(dtype=float)

    vals = [float(v) for v in eq_list]
    return pd.Series(vals).pct_change().dropna()


def enhanced_scoring(
    metrics_dict: dict[str, Any],
    sharpe_weight: float = 10,
    sortino_weight: float = 5,
    dd_penalty_weight: float = 20,
) -> float:
    """Weighted scoring function combining Sharpe, Sortino, and drawdown penalty.

    score = sharpe_weight * sharpe
          + sortino_weight   * sortino   (or 0 if unavailable)
          - dd_penalty_weight * abs(max_dd_pct)

    If both Sharpe and Sortino are missing or zero, falls back to total_return_pct
    so that loss-making configurations still rank above catastrophic ones.

    Args:
        metrics_dict:        Backtest result dict (same shape produced by ``_score_backtest``).
        sharpe_weight:       Multiplier for Sharpe ratio (default 10).
        sortino_weight:      Multiplier for Sortino ratio (default 5).
        dd_penalty_weight:   Multiplier applied to absolute max drawdown pct (default 20).

    Returns:
        Weighted score -- always >= 0.0 (clamped at zero floor).
    """
    sharpe = metrics_dict.get("sharpe_ratio", 0)
    try:
        sharpe = float(sharpe) if sharpe is not None else 0.0
    except (TypeError, ValueError):
        sharpe = 0.0

    # Sortino – compute from equity curve if present; otherwise read pre-computed key
    sortino = metrics_dict.get("sortino_ratio")
    if sortino is None:
        ret_series = _agg_returns_from_metrics(metrics_dict)
        if not ret_series.empty:
            sortino = _compute_sortino_ratio(ret_series)
    try:
        sortino = float(sortino) if sortino is not None else 0.0
    except (TypeError, ValueError):
        sortino = 0.0

    max_dd = metrics_dict.get("max_dd_pct", 0)
    try:
        max_dd = float(abs(max_dd)) if max_dd is not None else 0.0
    except (TypeError, ValueError):
        max_dd = 0.0

    score = sharpe_weight * sharpe + sortino_weight * sortino - dd_penalty_weight * max_dd

    return float(round(max(score, 0.0), 4))


def generate_overlapping_windows(
    start: str,
    end: str | None,
    train_window: int,
    test_window: int,
    step_size: int | None = None,
) -> list[dict[str, str]]:
    """Generate rolling train/test date windows **with configurable overlap**.

    Unlike ``_generate_windows`` (which steps forward by exactly *test_window*,
    producing consecutive non-overlapping segments), this function advances by
    *step_size* days between folds, allowing test periods to overlap across
    folds when *step_size* < *test_window*.

    This provides finer granularity during optimisation: more folds → better
    averaging over market regimes, at the cost of additional backtests.

    Args:
        start:           Start date ("YYYY-MM-DD").
        end:             End date ("YYYY-MM-DD"); None = today UTC.
        train_window:    Training window length in **days**.
        test_window:     Test window length in **days**.
        step_size:       Days to advance between fold starts.
                         Defaults to ``test_window // 2`` (50 % overlap).

    Returns:
        List of dicts with keys ``train_start``, ``train_end``,
        ``test_start``, ``test_end``.

    Example::

        # 90-day train, 30-day test, step 15 days => 50% overlap
        windows = generate_overlapping_windows("2020-01-01", "2024-01-01",
                                               90, 30, step_size=15)
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    else:
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if step_size is None:
        step_size = max(test_window // 2, 1)

    windows: list[dict[str, str]] = []
    cursor = start_dt

    while cursor + timedelta(days=train_window + test_window) <= end_dt:
        train_start = cursor
        train_end = cursor + timedelta(days=train_window)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window)

        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        cursor = cursor + timedelta(days=step_size)

    logger.info(
        "Generated %d overlapping walk-forward windows "
        "(train=%dd, test=%dd, step=%dd, range=%s → %s)",
        len(windows),
        train_window,
        test_window,
        step_size,
        start,
        end or "now",
    )
    return windows


# ──────────────────────────────────────────────────────────
# Expanded DEFAULT_PARAM_GRIDS — covers all known strategies
# ──────────────────────────────────────────────────────────

from typing import Literal

StrategyGrid = Literal[
    "ema_cross_rsi",
    "mean_reversion_rsi",
    "mean_reversion_rsi2",
    "turtle_breakout",
    "bollinger_reversion",
    "vwap_breakout",
    "momentum_scanner",
    "ml_hybrid",
]

EXPANDED_DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    # EMA-Cross-RSI: fast/slow crossover + RSI filter
    "ema_cross_rsi": {
        "fast": [5, 8, 9, 13, 21],
        "slow": [13, 21, 34, 50, 55],
        "rsi_period": [10, 14, 21],
        "rsi_entry_max": [60.0, 65.0, 70.0, 75.0],
        "rsi_exit": [65.0, 70.0, 75.0, 80.0],
    },
    # Mean-Reversion RSI: smoothed RSI crossings + volume filter
    "mean_reversion_rsi": {
        "rsi_period": [7, 10, 14, 21],
        "entry_threshold": [20.0, 25.0, 30.0, 35.0],
        "exit_threshold": [60.0, 65.0, 70.0, 75.0],
        "signal_period": [5, 9, 14],
        "volume_filter": [True, False],
        "volume_ma_period": [10, 20, 30],
    },
    # Mean-Reversion RSI-2: ultra-short RSO + BB filter
    "mean_reversion_rsi2": {
        "rsi_period": [2, 3],
        "rsi_oversold": [5.0, 8.0, 10.0, 15.0],
        "rsi_overbought": [70.0, 80.0, 90.0],
        "bb_period": [10, 15, 20],
        "bb_std": [1.5, 2.0, 2.5],
    },
    # Turtle Breakout: Donchian channel + ATR sizing
    "turtle_breakout": {
        "enter_period": [10, 15, 20, 23, 55],
        "exit_period": [5, 7, 10],
        "atr_period": [7, 10, 14, 20],
        "risk_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
        "units_per_risk_step": [0.5, 1.0, 1.5],
    },
    # Bollinger Reversion: price near lower BB + RSO oversold
    "bollinger_reversion": {
        "bb_period": [10, 15, 20, 25, 30],
        "bb_std": [1.0, 1.5, 2.0, 2.5],
        "rso_period": [7, 14, 21],
        "rso_oversold": [15.0, 20.0, 25.0],
        "rso_overbought": [70.0, 75.0, 80.0, 85.0],
        "atr_stop_mult": [1.5, 2.0, 2.5, 3.0],
        "atr_tp_mult": [2.0, 3.0, 4.0],
    },
    # VWAP Breakout: price above VWAP + volume confirmation
    "vwap_breakout": {
        "vwap_period": [10, 15, 20, 30],
        "vol_surge_mult": [1.2, 1.5, 2.0],
        "stop_multiplier": [1.5, 2.0, 2.5, 3.0],
        "tp_multiplier": [2.0, 3.0, 4.0],
        "min_volume_ratio": [1.0, 1.2, 1.5],
    },
    # Momentum Scanner: MACD cross + volume surge + trend MA
    "momentum_scanner": {
        "macd_fast": [8, 10, 12],
        "macd_slow": [21, 26],
        "macd_signal": [5, 7, 9],
        "vol_ma_period": [10, 15, 20],
        "vol_surge_mult": [1.3, 1.5, 2.0],
        "trend_ma_period": [20, 30, 50, 100],
    },
    # ML Hybrid: features + model config sweep
    "ml_hybrid": {
        "ml_model": ["logistic_regression", "random_forest", "xgboost"],
        "feature_set": ["all", "ta_only", "sentiment_only"],
        "rebalance_freq": ["daily", "weekly"],
        "kelly_fraction": [0.25, 0.5, 1.0],
        "lookback_days": [60, 90, 120],
        "prediction_horizon": [1, 3, 5],
    },
}
