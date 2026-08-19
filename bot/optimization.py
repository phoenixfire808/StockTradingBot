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
