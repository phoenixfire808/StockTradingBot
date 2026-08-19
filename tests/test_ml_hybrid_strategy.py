"""Tests for bot.plugins.strategies.ml_hybrid — base + ML filter hybrid strategy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bot.ml.model import GradientBoostedSignal
from bot.plugins.strategies.ml_hybrid import MLHybridStrategy, plugin
from bot.plugins.strategies.ml_signal_filter import MLSignalFilterStrategy
from bot.plugins.strategies.ema_cross_rsi import EmaCrossRsi
from bot.strategy import Strategy


# ── Fixtures ─────────────────────────────────────────────────────────
def _make_ohlcv(n: int = 250, seed: int = 42, trend: float = 0.0006) -> pd.DataFrame:
    """Build a deterministic OHLCV frame with a mild random-walk drift."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.02, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0, 0.012, n))
    low = close * (1 - rng.uniform(0, 0.012, n))
    opn = close * (1 + rng.normal(0, 0.004, n))
    vol = rng.integers(1000, 50000, n).astype(float)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _fit_external_model(df: pd.DataFrame) -> GradientBoostedSignal:
    """Fit a small GradientBoostedSignal on df's features."""
    from bot.ml.features import build_feature_frame

    feats = build_feature_frame(df, sentiment_score=0.0).dropna()
    close = df["Close"]
    fwd_ret = close.pct_change().shift(-1)
    y = (fwd_ret.reindex(feats.index) > 0).astype(int)
    if y.nunique() < 2:
        y = (fwd_ret.reindex(feats.index).fillna(0) > 0).astype(int)
    feats = feats.loc[y.index]
    model = GradientBoostedSignal()
    model.fit(feats, y)
    return model


def _valid_feature_index(df: pd.DataFrame) -> pd.Index:
    """Return the index of df rows that have valid (non-NaN) features."""
    from bot.ml.features import build_feature_frame

    feats = build_feature_frame(df, sentiment_score=0.0)
    return feats.dropna().index


# ── Plugin contract ──────────────────────────────────────────────────
class TestPluginContract:
    def test_plugin_handle_exists(self):
        assert isinstance(plugin, MLHybridStrategy)
        assert plugin.name == "ml_hybrid"

    def test_is_strategy_subclass(self):
        assert issubclass(MLHybridStrategy, Strategy)

    def test_default_params(self):
        strat = MLHybridStrategy()
        assert strat.params["base_strategy_name"] == "ema_cross_rsi"
        assert strat.params["ml_long_threshold"] == 0.55
        assert strat.params["ml_bearish_threshold"] == 0.40
        assert strat.params["min_history"] == 60
        assert strat.params["fit_on_init"] is True

    def test_custom_params(self):
        strat = MLHybridStrategy(
            base_strategy_name="ema_cross_rsi",
            ml_long_threshold=0.7,
            ml_bearish_threshold=0.3,
            min_history=80,
            fit_on_init=False,
        )
        assert strat.params["ml_long_threshold"] == 0.7
        assert strat.params["ml_bearish_threshold"] == 0.3
        assert strat.params["min_history"] == 80
        assert strat.params["fit_on_init"] is False

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            MLHybridStrategy(ml_long_threshold=1.5)
        with pytest.raises(ValueError):
            MLHybridStrategy(ml_bearish_threshold=-0.1)
        with pytest.raises(ValueError):
            MLHybridStrategy(ml_long_threshold=0.3, ml_bearish_threshold=0.5)


# ── Base strategy loading ───────────────────────────────────────────
class TestBaseStrategyLoading:
    def test_uses_ema_cross_rsi_by_default(self):
        strat = MLHybridStrategy()
        base = strat._get_base_strategy()
        assert isinstance(base, EmaCrossRsi)

    def test_base_params_forwarded(self):
        strat = MLHybridStrategy(fast=5, slow=15, rsi_period=10, rsi_entry_max=65.0, rsi_exit=80.0)
        base = strat._get_base_strategy()
        assert base.fast == 5
        assert base.slow == 15
        assert base.rsi_period == 10
        assert base.rsi_entry_max == 65.0
        assert base.rsi_exit == 80.0

    def test_base_strategy_cached_after_first_load(self):
        strat = MLHybridStrategy()
        b1 = strat._get_base_strategy()
        b2 = strat._get_base_strategy()
        assert b1 is b2


# ── Signal generation behavior ──────────────────────────────────────
class TestSignalGeneration:
    def test_insufficient_history_passes_through(self):
        df = _make_ohlcv(40)
        strat = MLHybridStrategy(min_history=60)
        base = strat._get_base_strategy()
        sig = strat.generate_signals(df)
        base_sig = base.generate_signals(df)
        pd.testing.assert_series_equal(
            sig.astype("int8"), base_sig.astype("int8"), check_names=False
        )
        assert strat.model is None

    def test_fit_on_init_false_no_model_passes_through(self):
        """When fit_on_init=False and no pre-fitted model, base signals pass through unchanged."""
        df = _make_ohlcv(200)
        strat = MLHybridStrategy(fit_on_init=False, min_history=60)
        sig = strat.generate_signals(df)
        # Model never gets created
        assert strat.model is None
        # Signals should match base strategy output
        base = strat._get_base_strategy()
        base_sig = base.generate_signals(df)
        pd.testing.assert_series_equal(
            sig.astype("int8"), base_sig.astype("int8"), check_names=False
        )

    def test_exits_preserved_regardless_of_ml(self):
        """Base exits (-1) must always pass through, even when ML is bearish."""
        df = _make_ohlcv(250)
        strat = MLHybridStrategy(fit_on_init=True, min_history=60)
        # Inject an all-bearish mock so any base long is vetoed
        from bot.ml.features import build_feature_frame

        feats = build_feature_frame(df, sentiment_score=0.0)

        class _BearishModel:
            is_fitted_ = True
            feature_names_ = list(feats.columns)

            def predict_proba(self, _df):
                return np.full(len(_df), 0.05)

        strat.model = _BearishModel()  # type: ignore[assignment]
        with patch.object(MLHybridStrategy, "_ensure_model", return_value=True):
            sig = strat.generate_signals(df)
        base_sig = strat._get_base_strategy().generate_signals(df)
        base_exits = int((base_sig == -1).sum())
        result_exits = int((sig == -1).sum())
        assert result_exits == base_exits

    def test_ml_vetoes_base_longs(self):
        """When ML is strongly bearish, base longs (in the ML-scored region) should be flattened."""
        df = _make_ohlcv(250)
        strat = MLHybridStrategy(
            ml_long_threshold=0.55,
            ml_bearish_threshold=0.40,
            min_history=60,
            fit_on_init=True,
        )
        from bot.ml.features import build_feature_frame

        feats = build_feature_frame(df, sentiment_score=0.0)

        class _BearishModel:
            is_fitted_ = True
            feature_names_ = list(feats.columns)

            def predict_proba(self, _df):
                return np.full(len(_df), 0.05)  # well below bearish threshold

        strat.model = _BearishModel()  # type: ignore[assignment]
        with patch.object(MLHybridStrategy, "_ensure_model", return_value=True):
            sig = strat.generate_signals(df)
        base_sig = strat._get_base_strategy().generate_signals(df)

        # Only check the valid-feature region (where ML scoring actually happened)
        valid_idx = _valid_feature_index(df)
        base_longs_in_valid = int((base_sig.loc[valid_idx] == 1).sum())
        sig_longs_in_valid = int((sig.loc[valid_idx] == 1).sum())
        # All base longs in the valid region should be vetoed since ML proba < bearish threshold
        assert sig_longs_in_valid == 0
        # Sanity check: the test setup actually has base longs to veto
        assert base_longs_in_valid > 0

    def test_ml_keeps_base_longs_when_bullish(self):
        """When ML is bullish, base longs should pass through."""
        df = _make_ohlcv(250)
        strat = MLHybridStrategy(
            ml_long_threshold=0.55,
            ml_bearish_threshold=0.40,
            min_history=60,
        )
        from bot.ml.features import build_feature_frame

        feats = build_feature_frame(df, sentiment_score=0.0)

        class _BullishModel:
            is_fitted_ = True
            feature_names_ = list(feats.columns)

            def predict_proba(self, _df):
                return np.full(len(_df), 0.95)  # well above long threshold

        strat.model = _BullishModel()  # type: ignore[assignment]
        with patch.object(MLHybridStrategy, "_ensure_model", return_value=True):
            sig = strat.generate_signals(df)
        base_sig = strat._get_base_strategy().generate_signals(df)
        # Compare counts of longs in valid region
        valid_idx = _valid_feature_index(df)
        assert (
            int((sig.loc[valid_idx] == 1).sum())
            == int((base_sig.loc[valid_idx] == 1).sum())
        )

    def test_pre_fitted_model_used(self):
        df = _make_ohlcv(200)
        model = _fit_external_model(df)
        strat = MLHybridStrategy(model=model, fit_on_init=False, min_history=60)
        sig = strat.generate_signals(df)
        assert strat.model is model
        assert isinstance(sig, pd.Series)
        assert sig.dtype == np.int8


# ── Auto-fit integration ────────────────────────────────────────────
class TestAutoFitIntegration:
    def test_auto_fit_then_filter(self):
        df = _make_ohlcv(300)
        strat = MLHybridStrategy(min_history=60)
        assert strat.model is None
        sig = strat.generate_signals(df)
        assert strat.model is not None
        assert strat.model.is_fitted_
        assert set(sig.unique().tolist()).issubset({-1, 0, 1})

    def test_long_count_no_greater_than_base(self):
        """Hybrid should never emit more longs than the base strategy alone."""
        df = _make_ohlcv(300)
        strat = MLHybridStrategy(min_history=60)
        sig = strat.generate_signals(df)
        base = strat._get_base_strategy().generate_signals(df)
        assert (sig == 1).sum() <= (base == 1).sum()


# ── Signal shape & determinism ──────────────────────────────────────
class TestSignalShape:
    def test_index_aligned(self):
        df = _make_ohlcv(200)
        strat = MLHybridStrategy(min_history=60)
        sig = strat.generate_signals(df)
        assert sig.index.equals(df.index)
        assert len(sig) == len(df)

    def test_signals_are_int8(self):
        df = _make_ohlcv(200)
        strat = MLHybridStrategy(min_history=60)
        sig = strat.generate_signals(df)
        assert sig.dtype == np.int8

    def test_repeated_calls_stable(self):
        df = _make_ohlcv(200)
        strat = MLHybridStrategy(min_history=60)
        sig1 = strat.generate_signals(df)
        sig2 = strat.generate_signals(df)
        pd.testing.assert_series_equal(
            sig1.astype("int8"), sig2.astype("int8"), check_names=False
        )


# ── Auto-discovery ──────────────────────────────────────────────────
class TestAutoDiscovery:
    def test_registered_in_registry(self):
        from bot.core import STRATEGIES
        from bot.core.plugins import discover_all

        discover_all()
        # Registry uses .get() / __contains__-style access; verify lookups work
        reg_plugin = STRATEGIES.get("ml_hybrid")
        assert reg_plugin is not None
        # Verify it points at our MLHybridStrategy class instance
        assert isinstance(reg_plugin, MLHybridStrategy)

    def test_ml_signal_filter_also_registered(self):
        from bot.core import STRATEGIES
        from bot.core.plugins import discover_all

        discover_all()
        reg_plugin = STRATEGIES.get("ml_signal_filter")
        assert reg_plugin is not None
        assert isinstance(reg_plugin, MLSignalFilterStrategy)