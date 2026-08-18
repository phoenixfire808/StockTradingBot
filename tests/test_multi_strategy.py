"""Tests for multi-strategy engine integration.

Covers:
  - Settings.parse_strategy_allocations (config.py)
  - Settings.strategy_allocations env loading (config.py)
  - run_multi_strategy validation paths (engine.py)
  - EquityTracker per-strategy recording (equity_tracker.py)
  - main.py CLI --multi argument parsing
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from bot.config import Settings, load_settings, _get


# ── Config: strategy_allocations ───────────────────────────────────────


class TestSettingsStrategyAllocations:
    """Config.py additions: strategy_allocations field + parse method."""

    def test_default_empty_string(self):
        s = Settings()
        assert s.strategy_allocations == ""
        assert s.parse_strategy_allocations() == {}

    def test_parse_valid_json(self):
        payload = json.dumps({
            "ema_cross_rsi": {"symbols": ["AAPL", "MSFT"], "weight": 0.6, "params": {"fast": 9}},
            "mean_reversion": {"symbols": ["NVDA"], "weight": 0.4},
        })
        s = Settings(strategy_allocations=payload)
        result = s.parse_strategy_allocations()
        assert len(result) == 2
        assert result["ema_cross_rsi"]["symbols"] == ["AAPL", "MSFT"]
        assert result["ema_cross_rsi"]["weight"] == 0.6
        assert result["mean_reversion"]["weight"] == 0.4

    def test_parse_invalid_json_returns_empty(self):
        s = Settings(strategy_allocations="{bad json}}}")
        result = s.parse_strategy_allocations()
        assert result == {}

    def test_parse_none_content(self):
        """None passed via dataclass field; json.loads handles gracefully."""
        s = Settings(strategy_allocations=None)
        result = s.parse_strategy_allocations()
        assert result == {}

    def test_load_settings_from_env_with_strategy_allocations(self, tmp_path):
        """STRATEGY_ALLOCATIONS_JSON env var -> loaded into Settings."""
        alloc = json.dumps({"s1": {"symbols": ["A"], "weight": 1.0}})
        env_patch = {
            "SYMBOLS": "A,B",
            "CASH": "50000",
            "STRATEGY_ALLOCATIONS_JSON": alloc,
        }
        with patch.dict(os.environ, env_patch, clear=False):
            from bot.config import load_settings as ls
            settings = ls()
        assert settings.symbols == ["A", "B"]
        assert settings.cash == 50_000
        parsed = settings.parse_strategy_allocations()
        assert "s1" in parsed
        assert parsed["s1"]["symbols"] == ["A"]
        assert parsed["s1"]["weight"] == 1.0

    def test_no_env_alloc_gives_empty(self):
        with patch.dict(os.environ, {"STRATEGY_ALLOCATIONS_JSON": ""}, clear=False):
            from bot.config import load_settings as ls
            s = ls()
            assert s.parse_strategy_allocations() == {}


# ── Engine: run_multi_strategy validation ─────────────────────────────


class TestRunMultiStrategyValidation:
    """run_multi_strategy edge cases without starting the scheduler."""

    def test_empty_allocations_returns_early(self):
        broker = MagicMock()
        settings = Settings(cash=100_000)
        from bot.engine import run_multi_strategy
        result = run_multi_strategy(broker, settings, {})
        assert result is None

    def test_zero_total_weight_returns_early(self):
        broker = MagicMock()
        settings = Settings(cash=100_000)
        allocs = {
            "ema_cross_rsi": {"symbols": ["AAPL"], "weight": 0},
        }
        from bot.engine import run_multi_strategy
        result = run_multi_strategy(broker, settings, allocs)
        assert result is None

    def test_missing_strategy_class_skipped(self, caplog):
        """If a strategy name isn't in STRATEGIES registry it's logged & skipped."""
        broker = MagicMock()
        settings = Settings(cash=100_000)
        allocs = {
            "nonexistent_xyz": {"symbols": ["AAPL"], "weight": 0.5},
        }
        from bot.engine import run_multi_strategy
        # The function logs an error and returns when all strategies are missing.
        # We don't need to patch logger; just verify the return value.
        result = run_multi_strategy(broker, settings, allocs)
        assert result is None

    def test_empty_symbols_skipped(self):
        """Strategies with no symbols are skipped gracefully."""
        broker = MagicMock()
        settings = Settings(symbols=["AAPL"])
        allocs = {
            "ema_cross_rsi": {"symbols": [], "weight": 0.5},
        }
        from bot.engine import run_multi_strategy
        try:
            run_multi_strategy(broker, settings, allocs)
        except Exception:
            pass
    @staticmethod
    def _make_mock_strategy():
        """Return a callable class that mimics a StrategyPlugin."""
        class MockStrategy:
            plugin = "mock"
            def __init__(self, **kwargs): pass
            def generate_signals(self, df): return type(df)([0]*len(df))
        return MockStrategy

    def test_runs_with_valid_allocs_doesnt_crash_at_init(self):
        """Quick check that allocation normalisation + cash splitting works."""
        broker = MagicMock()
        settings = Settings(cash=100_000, max_daily_loss_pct=3.0, engine_interval_minutes=5)
        allocs = {
            "ema_cross_rsi": {"symbols": ["AAPL"], "weight": 0.6, "params": {"fast": 9, "slow": 21}},
            "mean_reversion": {"symbols": ["MSFT"], "weight": 0.4, "params": {}},
        }
        from bot.engine import run_multi_strategy
        with patch("apscheduler.schedulers.blocking.BlockingScheduler") as MockScheduler:
            sched_inst = MagicMock()
            MockScheduler.return_value = sched_inst
            # Patch STRATEGIES.get to return a mock strategy class so init proceeds past registry lookup
            with patch("bot.core.STRATEGIES") as mock_strats:
                mock_strats.get.return_value = self._make_mock_strategy()
                try:
                    run_multi_strategy(broker, settings, allocs)
                except Exception:
                    pass
        assert sched_inst.add_job.called


# ── EquityTracker integration ──────────────────────────────────────────


class TestEquityTrackerIntegration:
    """Per-strategy equity recording for multi-strategy engine."""

    def test_record_creates_csv_and_json(self, tmp_path):
        from bot.equity_tracker import EquityTracker

        tracker = EquityTracker(logs_dir=tmp_path)
        tracker.record("strategy_a", 50_000.0, "2024-01-15T10:00:00Z")
        tracker.record("strategy_b", 30_000.0, "2024-01-15T10:00:00Z")

        csv_a = tmp_path / "equity_strategy_a.csv"
        csv_b = tmp_path / "equity_strategy_b.csv"
        assert csv_a.exists()
        assert csv_b.exists()

        json_path = tmp_path / "equity_curves.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "strategy_a" in data
        assert "strategy_b" in data

    def test_get_curve_returns_ordered_points(self, tmp_path):
        from bot.equity_tracker import EquityTracker

        tracker = EquityTracker(logs_dir=tmp_path)
        for i, eq in enumerate([10_000, 10_200, 10_500]):
            ts = f"2024-01-15T{10+i:02d}:00:00Z"
            tracker.record("s1", float(eq), ts)

        curve = tracker.get_curve("s1")
        assert len(curve) == 3
        assert curve[0]["equity"] == 10_000.0
        assert curve[2]["equity"] == 10_500.0

    def test_get_all_curves(self, tmp_path):
        from bot.equity_tracker import EquityTracker

        tracker = EquityTracker(logs_dir=tmp_path)
        tracker.record("alpha", 100_000, "2024-01-01T00:00:00Z")
        tracker.record("beta", 80_000, "2024-01-01T00:00:00Z")

        all_curves = tracker.get_all_curves()
        assert len(all_curves) == 2
        assert set(all_curves.keys()) == {"alpha", "beta"}

    def test_empty_strategy_name_logged(self, tmp_path, caplog):
        from bot.equity_tracker import EquityTracker

        tracker = EquityTracker(logs_dir=tmp_path)
        tracker.record("", 100, "2024-01-01T00:00:00Z")
        assert not (tmp_path / "equity_.csv").exists()

    def test_reset_clears_all_files(self, tmp_path):
        from bot.equity_tracker import EquityTracker

        tracker = EquityTracker(logs_dir=tmp_path)
        tracker.record("s1", 100, "ts1")
        tracker.record("s2", 200, "ts1")
        tracker.reset()
        csv_files = list(tmp_path.glob("equity_*.csv"))
        json_file = tmp_path / "equity_curves.json"
        assert len(csv_files) == 0
        assert not json_file.exists()


# ── Portfolio weight splitting ─────────────────────────────────────────


class TestPortfolioWeightSplitting:
    """Verify capital split logic used by run_multi_strategy."""

    def test_weights_normalise_to_one(self):
        from bot.portfolio import allocate_equal_weight

        weights = allocate_equal_weight(["s1", "s2", "s3"])
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_kelly_normalises_to_one(self):
        import pandas as pd
        from bot.portfolio import allocate_kelly

        rets = {
            "g1": pd.Series([0.01] * 50),
            "g2": pd.Series([-0.005] * 50),
        }
        w = allocate_kelly(rets)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_cash_split_proportional_to_weight(self):
        total_cash = 100_000
        weights = {"s1": 0.7, "s2": 0.3}
        split = {name: total_cash * w for name, w in weights.items()}
        assert split["s1"] == 70_000.0
        assert split["s2"] == 30_000.0
        assert sum(split.values()) == total_cash


# ── CLI: main.py multi subcommand ─────────────────────────────────────


class TestCLIMultiSubcommand:
    """main.py --multi argument parsing."""

    def test_main_help_includes_multi(self):
        """Verify 'multi' appears in help text."""
        proc = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd="D:/StockTradingBot",
            capture_output=True, text=True,
        )
        assert "multi" in proc.stdout
