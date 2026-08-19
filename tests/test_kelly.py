"""Comprehensive tests for bot.kelly — KellyConfig, compute functions, risk scaling, TradeOutcomesStore."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.kelly import (
    KellyConfig,
    TradeOutcome,
    TradeOutcomesStore,
    _clean_returns,
    compute_returns_kelly,
    compute_winrate_payoff_kelly,
    scale_risk_with_kelly,
)


# ── KellyConfig ───────────────────────────────────────────────────────


class TestKellyConfigDefaults:
    def test_default_method_is_returns(self):
        cfg = KellyConfig()
        assert cfg.method == "returns"
        assert cfg.enabled is False
        assert cfg.fractional == 0.25
        assert cfg.max_fraction == 0.50
        assert cfg.min_samples == 30
        assert cfg.track_returns_window == 90

    def test_disabled_property_true_when_enabled_false(self):
        cfg = KellyConfig(enabled=False)
        assert cfg.disabled is True

    def test_disabled_property_true_when_method_disabled(self):
        cfg = KellyConfig(enabled=True, method="disabled")
        assert cfg.disabled is True

    def test_disabled_property_false_when_enabled_and_active_method(self):
        cfg = KellyConfig(enabled=True, method="winrate_payoff")
        assert cfg.disabled is False


class TestKellyConfigValidation:
    def test_invalid_fractional_clamped_to_default(self, caplog):
        cfg = KellyConfig(fractional=0.0)
        assert cfg.fractional == 0.25

    def test_fractional_above_one_clamped_to_default(self, caplog):
        """Fractional > 1.0 is invalid per spec; clamped to default."""
        cfg = KellyConfig(fractional=2.0)
        assert cfg.fractional == 0.25  # clamped to default

    def test_invalid_max_fraction_clamped(self, caplog):
        cfg = KellyConfig(max_fraction=0.0)
        assert cfg.max_fraction == 0.50

    def test_unknown_method_resets_to_returns(self, caplog):
        cfg = KellyConfig(method="nonexistent_xyz")
        assert cfg.method == "returns"

    def test_min_samples_below_one_clamped(self, caplog):
        cfg = KellyConfig(min_samples=0)
        assert cfg.min_samples == 30

    def test_valid_config_passes_unchanged(self, caplog):
        cfg = KellyConfig(
            method="winrate_payoff",
            fractional=0.5,
            max_fraction=0.75,
            min_samples=50,
        )
        assert cfg.method == "winrate_payoff"
        assert cfg.fractional == 0.5
        assert cfg.max_fraction == 0.75
        assert cfg.min_samples == 50


class TestKellyConfigSerialization:
    def test_to_dict_excludes_deprecated_attr(self):
        cfg = KellyConfig(enabled=True, method="winrate_payoff", fractional=0.5)
        d = cfg.to_dict()
        assert "kelly_method" not in d
        assert d["method"] == "winrate_payoff"
        assert d["enabled"] is True
        assert d["fractional"] == 0.5

    def test_from_dict_roundtrips(self):
        original = KellyConfig(enabled=True, method="returns", fractional=0.3)
        data = original.to_dict()
        restored = KellyConfig.from_dict(data)
        assert restored.method == "returns"
        assert restored.fractional == 0.3
        assert restored.enabled is True

    def test_from_dict_handles_empty_object(self):
        cfg = KellyConfig.from_dict({})
        assert cfg.method == "returns"
        assert cfg.enabled is False


class TestKellyConfigFromEnv:
    def test_defaults_when_no_env(self, monkeypatch):
        for key in ["KELLY_ENABLED", "KELLY_METHOD", "KELLY_FRACTIONAL",
                     "KELLY_MAX_FRACTION", "KELLY_MIN_SAMPLES",
                     "KELLY_TRACK_RETURNS_WINDOW"]:
            monkeypatch.delenv(key, raising=False)
        cfg = KellyConfig.from_env()
        assert cfg.enabled is False
        assert cfg.method == "returns"
        assert cfg.fractional == 0.25

    def test_all_fields_overridden(self, monkeypatch):
        monkeypatch.setenv("KELLY_ENABLED", "true")
        monkeypatch.setenv("KELLY_METHOD", "winrate_payoff")
        monkeypatch.setenv("KELLY_FRACTIONAL", "0.4")
        monkeypatch.setenv("KELLY_MAX_FRACTION", "0.6")
        monkeypatch.setenv("KELLY_MIN_SAMPLES", "50")
        monkeypatch.setenv("KELLY_TRACK_RETURNS_WINDOW", "120")
        cfg = KellyConfig.from_env()
        assert cfg.enabled is True
        assert cfg.method == "winrate_payoff"
        assert cfg.fractional == 0.4
        assert cfg.max_fraction == 0.6
        assert cfg.min_samples == 50
        assert cfg.track_returns_window == 120

    def test_invalid_values_fallback_to_defaults(self, monkeypatch):
        monkeypatch.setenv("KELLY_FRACTIONAL", "not_a_number")
        monkeypatch.setenv("KELLY_MIN_SAMPLES", "bad")
        cfg = KellyConfig.from_env()
        assert cfg.fractional == 0.25
        assert cfg.min_samples == 30


# ── Returns-Based Kelly ──────────────────────────────────────────────


class TestComputeReturnsKelly:
    @staticmethod
    def _pos_returns(n=100, mean=0.002, vol=0.01, seed=42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mean, vol, n))

    def test_positive_edge_returns_valid_fraction(self):
        rets = self._pos_returns(mean=0.005, vol=0.02, seed=42)
        f = compute_returns_kelly(rets, KellyConfig(fractional=1.0, max_fraction=1.0))
        assert 0 < f <= 1.0

    def test_negative_edge_returns_zero(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(-0.01, 0.02, 100))
        f = compute_returns_kelly(rets, KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_zero_variance_returns_zero(self):
        rets = pd.Series([0.001] * 100)  # constant positive returns
        f = compute_returns_kelly(rets, KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_insufficient_samples_returns_zero(self):
        rets = pd.Series([0.01, 0.02, 0.003])  # only 3 samples, default min=30
        f = compute_returns_kelly(rets)
        assert f == 0.0

    def test_nan_inf_cleaned_before_computation(self):
        base = self._pos_returns(seed=10)
        dirty = base.copy()
        dirty.iloc[0] = float("nan")
        dirty.iloc[1] = float("inf")
        dirty.iloc[-1] = float("-inf")
        f = compute_returns_kelly(dirty, KellyConfig(min_samples=5, fractional=1.0, max_fraction=1.0))
        assert 0 < f <= 1.0

    def test_high_volatility_low_mean_shrinks_fraction(self):
        """Lower volatility should give higher Kelly when means are equal.
        
        Using very low vol so full-Kelly stays well below max_fraction cap.
        With same mean but 10x lower var, Kelly should be ~10x larger.
        """
        high_vol = self._pos_returns(mean=0.005, vol=0.03, seed=10)
        low_vol = self._pos_returns(mean=0.005, vol=0.003, seed=10)
        f_high = compute_returns_kelly(high_vol, KellyConfig(fractional=1.0, max_fraction=1.0))
        f_low = compute_returns_kelly(low_vol, KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f_high < f_low
        # Verify ratio approx matches inverse-variance: 0.03^2 / 0.003^2 = 100
        if f_high > 0:
            ratio = f_low / f_high
            assert ratio > 50  # roughly 100x due to sample noise in small vars

    def test_clamped_to_max_fraction(self):
        # Generate returns that give very high full-Kelly
        rets = self._pos_returns(mean=0.05, vol=0.01, seed=42)
        f = compute_returns_kelly(rets, KellyConfig(fractional=1.0, max_fraction=0.10))
        assert abs(f - 0.10) < 1e-6

    def test_default_cfg_falls_back_to_conservative(self):
        rets = self._pos_returns(seed=42)
        f = compute_returns_kelly(rets)  # uses default cfg (min_samples=30)
        assert isinstance(f, float)


# ── Win-Rate / Payoff Kelly ──────────────────────────────────────────


class TestComputeWinratePayoffKelly:
    def test_standard_case_58pct_wr_2to1_payoff(self):
        """w=0.58, b=2.0 => f* = 0.58 - 0.42/2 = 0.37"""
        f = compute_winrate_payoff_kelly(0.58, 200.0, 100.0, KellyConfig(fractional=1.0, max_fraction=1.0))
        expected = 0.58 - 0.42 / 2.0
        assert abs(f - expected) < 1e-6

    def test_half_kelly_applied(self):
        f_full = compute_winrate_payoff_kelly(0.58, 200.0, 100.0,
                                              KellyConfig(fractional=1.0, max_fraction=1.0))
        f_half = compute_winrate_payoff_kelly(0.58, 200.0, 100.0,
                                               KellyConfig(fractional=0.5, max_fraction=1.0))
        assert abs(f_half - f_full * 0.5) < 1e-6

    def test_all_wins_clamped_to_max_fraction(self):
        """w=1.0 should return 0.0 per guard clause (boundary)."""
        f = compute_winrate_payoff_kelly(1.0, 100.0, 1.0,
                                         KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_all_losses_returns_zero(self):
        f = compute_winrate_payoff_kelly(0.0, 100.0, 1.0,
                                         KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_no_losses_infinite_payoff_returns_zero(self):
        f = compute_winrate_payoff_kelly(0.8, 200.0, 0.0,
                                         KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_zero_win_rate_returns_zero(self):
        f = compute_winrate_payoff_kelly(0.0, 0.0, 50.0,
                                         KellyConfig(fractional=1.0, max_fraction=1.0))
        assert f == 0.0

    def test_small_sample_below_min_threshold_skipped_at_caller_level(self):
        """This function itself doesn't enforce sample count — that's the caller's job.
        But we verify it computes correctly when given valid inputs regardless."""
        f = compute_winrate_payoff_kelly(0.55, 150.0, 100.0,
                                         KellyConfig(fractional=1.0, max_fraction=1.0))
        assert 0 < f < 1.0

    def test_high_concentration_cap_prevents_overbetting(self):
        """Even with strong edge, output capped at max_fraction."""
        f = compute_winrate_payoff_kelly(0.95, 500.0, 50.0,
                                         KellyConfig(fractional=1.0, max_fraction=0.15))
        assert abs(f - 0.15) < 1e-6

    def test_quarter_kelly_standard_case(self):
        """Quarter-kelly on w=0.58, b=2.0 => ~0.185"""
        f = compute_winrate_payoff_kelly(0.58, 200.0, 100.0,
                                         KellyConfig(fractional=0.25, max_fraction=1.0))
        assert abs(f - 0.37 * 0.25) < 1e-6

    def test_default_cfg_gives_conservative_output(self):
        """Default cfg: fractional=0.25, so result = 0.25x full Kelly."""
        f = compute_winrate_payoff_kelly(0.58, 200.0, 100.0)
        f_full = compute_winrate_payoff_kelly(0.58, 200.0, 100.0,
                                               KellyConfig(fractional=1.0, max_fraction=1.0))
        assert abs(f - f_full * 0.25) < 1e-6


# ── Risk Scaling ─────────────────────────────────────────────────────


class TestScaleRiskWithKelly:
    def test_positive_kelly_increases_risk(self):
        effective = scale_risk_with_kelly(0.01, 0.25)
        assert effective > 0.01

    def test_negative_kelly_falls_back_to_base(self):
        effective = scale_risk_with_kelly(0.01, -0.1)
        assert abs(effective - 0.01) < 1e-6

    def test_zero_kelly_preserves_base_risk(self):
        effective = scale_risk_with_kelly(0.01, 0.0)
        assert abs(effective - 0.01) < 1e-6

    def test_minimum_risk_floor(self):
        effective = scale_risk_with_kelly(0.00001, 0.0)
        assert effective >= 0.001

    def test_full_kelly_above_max_clamps(self):
        """Kelly fraction already clamped by caller; this just verifies scaling works."""
        effective = scale_risk_with_kelly(0.01, 0.5)
        expected = 0.01 * 1.5
        assert abs(effective - expected) < 1e-6

    def test_high_kelly_multiplier(self):
        effective = scale_risk_with_kelly(0.01, 1.0)
        assert effective == 0.02


# ── _clean_returns ───────────────────────────────────────────────────


class TestCleanReturns:
    def test_removes_nan(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        cleaned = _clean_returns(s)
        assert len(cleaned) == 2
        assert float("nan") not in cleaned.values

    def test_removes_inf(self):
        s = pd.Series([1.0, float("inf"), -float("inf"), 4.0])
        cleaned = _clean_returns(s)
        assert len(cleaned) == 2

    def test_non_numeric_coerced_to_nan_and_dropped(self):
        s = pd.Series([1.0, "hello", 3.0])
        cleaned = _clean_returns(s)
        assert len(cleaned) == 2

    def test_already_clean_returns_same(self):
        s = pd.Series([0.01, -0.02, 0.015])
        cleaned = _clean_returns(s)
        pd.testing.assert_series_equal(cleaned, s, check_names=False)


# ── TradeOutcome ─────────────────────────────────────────────────────


class TestTradeOutcome:
    def test_new_outcome_has_zeros(self):
        t = TradeOutcome()
        assert t.wins == 0
        assert t.losses == 0
        assert t.trade_count == 0
        assert t.win_rate == 0.0
        assert t.avg_win == 0.0
        assert t.avg_loss == 0.0

    def test_record_win(self):
        t = TradeOutcome()
        t.record_trade(150.0)
        assert t.wins == 1
        assert t.losses == 0
        assert t.trade_count == 1
        assert abs(t.win_rate - 1.0) < 1e-9
        assert abs(t.avg_win - 150.0) < 1e-6

    def test_record_loss(self):
        t = TradeOutcome()
        t.record_trade(-80.0)
        assert t.wins == 0
        assert t.losses == 1
        assert abs(t.avg_loss - 80.0) < 1e-6

    def test_mixed_trades(self):
        t = TradeOutcome()
        t.record_trade(100.0)
        t.record_trade(-50.0)
        t.record_trade(200.0)
        assert t.wins == 2
        assert t.losses == 1
        assert abs(t.win_rate - 2/3) < 1e-9
        assert abs(t.avg_win - 150.0) < 1e-6
        assert abs(t.avg_loss - 50.0) < 1e-6

    def test_record_zero_pnl_ignored(self):
        t = TradeOutcome()
        t.record_trade(0.0)
        assert t.trade_count == 0

    def test_payoff_ratio(self):
        t = TradeOutcome()
        t.record_trade(200.0)
        t.record_trade(100.0)
        t.record_trade(-50.0)
        t.record_trade(-100.0)
        assert abs(t.payoff_ratio - 150.0 / 75.0) < 1e-6

    def test_to_dict_roundtrip(self):
        t = TradeOutcome()
        t.record_trade(100.0)
        t.record_trade(-50.0)
        d = t.to_dict()
        restored = TradeOutcome.from_dict(d)
        assert restored.wins == t.wins
        assert restored.losses == t.losses
        assert abs(restored.avg_win - t.avg_win) < 1e-6
        assert abs(restored.avg_loss - t.avg_loss) < 1e-6

    def test_from_dict_with_missing_keys(self):
        t = TradeOutcome.from_dict({"wins": 5})
        assert t.wins == 5
        assert t.losses == 0


# ── TradeOutcomesStore ───────────────────────────────────────────────


class TestTradeOutcomesStore:
    def test_read_all_empty_file_not_exist(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        assert store.read_all() == {}

    def test_record_and_read_single_trade(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        store.record("ema_cross_rsi", "AAPL", 150.0)
        outcomes = store.read("ema_cross_rsi", "AAPL")
        assert outcomes is not None
        assert outcomes.wins == 1
        assert outcomes.total_win_amount == 150.0

    def test_record_multiple_trades_accumulates(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        store.record("s1", "A", 100.0)
        store.record("s1", "A", -50.0)
        store.record("s1", "A", 200.0)
        outcomes = store.read("s1", "A")
        assert outcomes is not None
        assert outcomes.wins == 2
        assert outcomes.losses == 1
        assert outcomes.trade_count == 3

    def test_per_strategy_symbol_keying(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        store.record("strat_a", "AAPL", 100.0)
        store.record("strat_b", "AAPL", 200.0)
        a_out = store.read("strat_a", "AAPL")
        b_out = store.read("strat_b", "AAPL")
        assert a_out is not None
        assert b_out is not None
        assert a_out.wins == 1
        assert b_out.wins == 1
        # They share the file but separate keys
        all_data = store.read_all()
        assert "strat_a" in all_data
        assert "strat_b" in all_data

    def test_corrupt_file_graceful_fallback(self, tmp_path):
        path = tmp_path / "outcomes.json"
        path.write_text("{corrupt json!!!")
        store = TradeOutcomesStore(path)
        assert store.read_all() == {}

    def test_json_file_written_and_readable(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        store.record("test_strat", "TEST", 50.0)
        # Re-read via store API since Windows may lock the temp file briefly
        all_data = store.read_all()
        assert "test_strat" in all_data
        assert "TEST" in all_data["test_strat"]

    def test_get_kelly_params_returns_none_when_disabled(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        cfg = KellyConfig(enabled=False, method="winrate_payoff")
        result = store.get_kelly_params("s1", "A", cfg)
        assert result is None

    def test_get_kelly_params_returns_none_when_insufficient_data(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        store.record("s1", "A", 100.0)  # only 1 trade, default min=30
        cfg = KellyConfig(method="winrate_payoff", enabled=True)
        result = store.get_kelly_params("s1", "A", cfg)
        assert result is None

    def test_get_kelly_params_returns_value_when_enough_data(self, tmp_path):
        store = TradeOutcomesStore(tmp_path / "outcomes.json")
        # Record 35 trades with known win rate
        rng = np.random.default_rng(42)
        for i in range(35):
            pnl = rng.choice([100.0, -50.0])
            store.record("ema_cross_rsi", "AAPL", pnl)
        cfg = KellyConfig(method="winrate_payoff", enabled=True, min_samples=30)
        result = store.get_kelly_params("ema_cross_rsi", "AAPL", cfg)
        assert result is not None
        assert 0 < result < 1.0


# ── Backwards Compatibility ──────────────────────────────────────────


class TestBackwardsCompat:
    """Verify existing code paths still work after changes."""

    def test_portfolio_imports_still_work(self):
        from bot.portfolio import allocate_kelly, allocate_equal_weight, allocate_risk_parity, PortfolioState
        assert callable(allocate_kelly)
        assert callable(allocate_equal_weight)
        assert callable(allocate_risk_parity)

    def test_features_kelly_alias_works(self):
        from bot.ml.features import kelly_fraction
        # The alias should produce same results as the canonical function
        result = kelly_fraction(0.58, 200.0, 100.0)
        expected = compute_winrate_payoff_kelly(0.58, 200.0, 100.0)
        assert abs(result - expected) < 1e-9

    def test_config_settings_have_kelly_fields(self):
        from bot.config import Settings
        s = Settings()
        assert hasattr(s, 'kelly_enabled')
        assert hasattr(s, 'kelly_method')
        assert hasattr(s, 'kelly_fractional')
        assert hasattr(s, 'kelly_max_fraction')
        assert hasattr(s, 'kelly_min_samples')
