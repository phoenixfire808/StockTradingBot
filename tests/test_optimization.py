"""Tests for bot.optimization — walk-forward optimization."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from bot.optimization import (
    walk_forward_optimize,
    _generate_windows,
    _score_backtest,
    DEFAULT_PARAM_GRIDS,
)


class TestWindowGeneration:
    def test_generates_correct_count(self):
        # 365 days, 90 train + 30 test = 120 day cycle
        # → 365 / 30 = ~12 windows (stepping by test_window)
        start = "2023-01-01"
        end = "2024-01-01"
        windows = _generate_windows(start, end, train_window=90, test_window=30)
        assert len(windows) > 0
        # Each window has required keys
        w = windows[0]
        assert "train_start" in w
        assert "train_end" in w
        assert "test_start" in w
        assert "test_end" in w

    def test_windows_dont_overlap_test_periods(self):
        windows = _generate_windows("2023-01-01", "2024-01-01", 90, 30)
        for i in range(1, len(windows)):
            # test_start of window i should be after test_start of window i-1
            prev = windows[i - 1]
            curr = windows[i]
            assert curr["test_start"] > prev["test_start"]

    def test_train_precedes_test(self):
        windows = _generate_windows("2023-01-01", "2024-01-01", 90, 30)
        for w in windows:
            assert w["train_start"] < w["train_end"]
            assert w["train_end"] == w["test_start"]
            assert w["test_start"] < w["test_end"]

    def test_insufficient_range_returns_empty(self):
        # Only 30 days, need 120 → no windows
        windows = _generate_windows("2023-01-01", "2023-01-31", 90, 30)
        assert len(windows) == 0


class TestScoringFunction:
    def test_positive_sharpe_positive_return(self):
        metrics = {"sharpe_ratio": 1.5, "total_return_pct": 20.0, "max_dd_pct": 5.0}
        score = _score_backtest(metrics)
        assert score > 0

    def test_negative_return_negative_score(self):
        metrics = {"sharpe_ratio": -0.5, "total_return_pct": -15.0, "max_dd_pct": 10.0}
        score = _score_backtest(metrics)
        assert score < 0

    def test_high_drawdown_penalized(self):
        good = {"sharpe_ratio": 1.0, "total_return_pct": 10.0, "max_dd_pct": 2.0}
        bad = {"sharpe_ratio": 1.0, "total_return_pct": 10.0, "max_dd_pct": 40.0}
        assert _score_backtest(good) > _score_backtest(bad)

    def test_missing_metrics_handles_gracefully(self):
        score = _score_backtest({})
        assert score == 0.0

    def test_none_values_handled(self):
        metrics = {"sharpe_ratio": None, "total_return_pct": None, "max_dd_pct": None}
        score = _score_backtest(metrics)
        assert score == 0.0


class TestDefaultParamGrids:
    def test_ema_cross_rsi_grid(self):
        grid = DEFAULT_PARAM_GRIDS["ema_cross_rsi"]
        assert "fast" in grid
        assert "slow" in grid
        assert len(grid["fast"]) >= 2

    def test_bollinger_grid(self):
        grid = DEFAULT_PARAM_GRIDS["bollinger_reversion"]
        assert "bb_period" in grid
        assert "bb_std" in grid


class TestWalkForwardOptimize:
    def test_insufficient_date_range(self):
        """When date range is too short, should return error dict."""
        result = walk_forward_optimize(
            strategy_name="ema_cross_rsi",
            param_grid={"fast": [5, 9], "slow": [21]},
            symbols="AAPL",
            start="2023-01-01",
            end="2023-01-15",  # only 14 days
            train_window=90,
            test_window=30,
        )
        assert "error" in result
        assert result["best_params"] == {}
        assert result["total_combinations"] == 2

    def test_no_param_grid_unknown_strategy(self):
        """Unknown strategy with no grid → error."""
        result = walk_forward_optimize(
            strategy_name="unknown_strategy",
            param_grid=None,
            symbols="AAPL",
            start="2023-01-01",
            end="2024-01-01",
        )
        assert "error" in result
        assert result["total_combinations"] == 0

    def test_mocked_full_run(self):
        """Mock run_backtest to simulate a full optimization run."""
        def mock_scored(*args, **kwargs):
            return 1.0, {"sharpe_ratio": 1.0, "total_return_pct": 10.0, "max_dd_pct": 5.0}

        with patch("bot.optimization._run_backtest_scored", side_effect=mock_scored):
            with patch("bot.backtest.run_backtest") as mock_bt:
                # Equity curve mock
                mock_bt.return_value = {
                    "AAPL": {
                        "metrics": {"sharpe_ratio": 1.0, "total_return_pct": 10.0, "max_dd_pct": 5.0},
                        "equity_curve": pd.DataFrame({"Equity": [100000, 105000, 110000]}),
                    }
                }
                result = walk_forward_optimize(
                    strategy_name="ema_cross_rsi",
                    param_grid={"fast": [5, 9], "slow": [21]},
                    symbols="AAPL",
                    start="2023-01-01",
                    end="2023-12-31",
                    train_window=60,
                    test_window=30,
                )

        assert result["strategy"] == "ema_cross_rsi"
        assert result["total_combinations"] == 2
        assert len(result["folds"]) > 0
        assert "best_params" in result
        assert "best_score" in result

    def test_returns_correct_structure(self):
        """Even with errors, structure should be consistent."""
        with patch("bot.optimization._run_backtest_scored", return_value=(-999.0, {})):
            result = walk_forward_optimize(
                strategy_name="ema_cross_rsi",
                param_grid={"fast": [5], "slow": [21]},
                symbols="AAPL",
                start="2023-01-01",
                end="2023-12-31",
                train_window=60,
                test_window=30,
            )
        # Structure
        expected_keys = {"strategy", "best_params", "best_score", "folds",
                         "total_combinations", "param_grid", "symbols"}
        assert expected_keys.issubset(result.keys())
        assert result["strategy"] == "ema_cross_rsi"
        assert result["total_combinations"] == 1
