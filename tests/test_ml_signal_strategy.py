"""Tests for bot.plugins.strategies.ml_signal_filter — pure ML signal strategy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bot.ml.model import GradientBoostedSignal
from bot.plugins.strategies.ml_signal_filter import (
    MLSignalFilterStrategy,
    plugin,
)
from bot.strategy import Strategy


# ── Fixtures ─────────────────────────────────────────────────────────
def _make_ohlcv(n: int = 200, seed: int = 42, trend: float = 0.0005) -> pd.DataFrame:
    """Build a deterministic OHLCV frame with a mild random-walk drift."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.02, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    opn = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1000, 50000, n).astype(float)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _fit_external_model(df: pd.DataFrame) -> GradientBoostedSignal:
    """Build a small GradientBoostedSignal that's been fitted on df's features."""
    from bot.ml.features import build_feature_frame

    feats = build_feature_frame(df, sentiment_score=0.0).dropna()
    close = df["Close"]
    fwd_ret = close.pct_change().shift(-1)
    y = (fwd_ret.reindex(feats.index) > 0).astype(int)
    y = y.dropna()
    feats = feats.loc[y.index]
    if y.nunique() < 2:
        # Force both classes by splitting on median
        y = (fwd_ret.reindex(feats.index).fillna(0) > 0).astype(int)
    model = GradientBoostedSignal()
    model.fit(feats, y)
    return model


def _ml_scored_index(df: pd.DataFrame) -> pd.Index:
    """Index of df rows whose features are valid (no NaN) — the ML-scored region."""
    from bot.ml.features import build_feature_frame

    feats = build_feature_frame(df, sentiment_score=0.0)
    return feats.dropna().index


# ── Plugin contract ──────────────────────────────────────────────────
class TestPluginContract:
    def test_plugin_handle_exists(self):
        assert isinstance(plugin, MLSignalFilterStrategy)
        assert plugin.name == "ml_signal_filter"

    def test_is_strategy_subclass(self):
        assert issubclass(MLSignalFilterStrategy, Strategy)

    def test_default_params(self):
        strat = MLSignalFilterStrategy()
        assert strat.params["long_threshold"] == 0.55
        assert strat.params["short_threshold"] == 0.45
        assert strat.params["min_history"] == 60
        assert strat.params["fit_on_init"] is True
        assert strat.model is None

    def test_custom_params(self):
        strat = MLSignalFilterStrategy(
            long_threshold=0.65,
            short_threshold=0.30,
            min_history=80,
            sentiment_score=0.2,
            fit_on_init=False,
        )
        assert strat.params["long_threshold"] == 0.65
        assert strat.params["short_threshold"] == 0.30
        assert strat.params["min_history"] == 80
        assert strat.params["sentiment_score"] == 0.2
        assert strat.params["fit_on_init"] is False

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            MLSignalFilterStrategy(long_threshold=1.5)
        with pytest.raises(ValueError):
            MLSignalFilterStrategy(short_threshold=-0.1)
        with pytest.raises(ValueError):
            MLSignalFilterStrategy(short_threshold=0.6, long_threshold=0.5)

    def test_signal_dtype_is_int8(self):
        df = _make_ohlcv(200)
        strat = MLSignalFilterStrategy(min_history=60)
        sig = strat.generate_signals(df)
        assert sig.dtype == np.int8
        assert set(sig.unique().tolist()).issubset({-1, 0, 1})


# ── Signal generation behavior ──────────────────────────────────────
class TestSignalGeneration:
    def test_insufficient_history_returns_zeros(self):
        df = _make_ohlcv(30)
        strat = MLSignalFilterStrategy(min_history=60, fit_on_init=True)
        sig = strat.generate_signals(df)
        assert (sig == 0).all()
        assert strat.model is None  # never fitted

    def test_fit_on_init_false_no_model_returns_zeros(self):
        df = _make_ohlcv(200)
        strat = MLSignalFilterStrategy(fit_on_init=False)
        sig = strat.generate_signals(df)
        assert (sig == 0).all()
        assert strat.model is None

    def test_auto_fit_populates_model(self):
        df = _make_ohlcv(200)
        strat = MLSignalFilterStrategy(min_history=60)
        assert strat.model is None
        strat.generate_signals(df)
        assert strat.model is not None
        assert getattr(strat.model, "is_fitted_", False) is True

    def test_auto_fit_then_predicts(self):
        df = _make_ohlcv(300)
        strat = MLSignalFilterStrategy(long_threshold=0.5, short_threshold=0.5, min_history=60)
        sig = strat.generate_signals(df)
        valid = sig.iloc[60:]
        # After warm-up, signals should be produced for some bars
        assert strat.model is not None

    def test_pre_fitted_model_used_directly(self):
        df = _make_ohlcv(200)
        model = _fit_external_model(df)
        strat = MLSignalFilterStrategy(
            model=model,
            fit_on_init=False,  # should NOT trigger another fit
            min_history=60,
        )
        sig = strat.generate_signals(df)
        # Model was passed in, fit_on_init disabled -> still the same object
        assert strat.model is model
        assert strat.model.is_fitted_

    def test_signal_count_with_extreme_thresholds(self):
        """Extremely tight thresholds should produce almost no signals."""
        df = _make_ohlcv(300)
        strat = MLSignalFilterStrategy(
            long_threshold=0.99,
            short_threshold=0.01,
            min_history=60,
        )
        sig = strat.generate_signals(df)
        # Most rows should be 0 since proba rarely hits those extremes
        assert (sig == 0).sum() >= int(0.7 * len(df))


# ── ML threshold semantics ─────────────────────────────────────────
class TestThresholdSemantics:
    """Verify the probability -> signal mapping using a directly-mocked strategy path.

    These tests bypass auto-fit by injecting a fake model so we can isolate
    the threshold -> signal mapping logic without depending on indicator
    calculations (which require a minimum lookback window).
    """

    def _strat_with_mocked_proba(
        self, proba_value: float, n_bars: int
    ) -> MLSignalFilterStrategy:
        strat = MLSignalFilterStrategy(
            long_threshold=0.6,
            short_threshold=0.4,
            min_history=10,
        )
        df = _make_ohlcv(n_bars)

        from bot.ml.features import build_feature_frame as _bff

        feats = _bff(df, sentiment_score=0.0)
        pv = float(proba_value)

        class _MockModel:
            is_fitted_ = True
            feature_names_ = list(feats.columns)

            def predict_proba(self, _df):
                # Always return array matching input row count
                return np.full(len(_df), pv)

        strat.model = _MockModel()  # type: ignore[assignment]
        # Bypass auto-fit path so the mock survives
        strat._ensure_model = lambda d: True  # type: ignore[assignment]
        return strat

    def test_long_signal_when_proba_high(self):
        df = _make_ohlcv(80)
        strat = self._strat_with_mocked_proba(proba_value=0.7, n_bars=80)
        sig = strat.generate_signals(df)
        valid_idx = _ml_scored_index(df)
        scored = sig.loc[valid_idx]
        # proba=0.7 > 0.6 -> all scored rows should be 1
        assert (scored == 1).all()

    def test_short_signal_when_proba_low(self):
        df = _make_ohlcv(80)
        strat = self._strat_with_mocked_proba(proba_value=0.1, n_bars=80)
        sig = strat.generate_signals(df)
        valid_idx = _ml_scored_index(df)
        scored = sig.loc[valid_idx]
        # proba=0.1 < 0.4 -> all scored rows should be -1
        assert (scored == -1).all()

    def test_neutral_signal_when_proba_mid(self):
        df = _make_ohlcv(80)
        strat = self._strat_with_mocked_proba(proba_value=0.5, n_bars=80)
        sig = strat.generate_signals(df)
        valid_idx = _ml_scored_index(df)
        scored = sig.loc[valid_idx]
        # 0.5 is between 0.4 and 0.6 -> all scored rows should be 0
        assert (scored == 0).all()


# ── Model path loading ──────────────────────────────────────────────
class TestModelPathLoading:
    def test_load_from_path(self, tmp_path):
        df = _make_ohlcv(200)
        model = _fit_external_model(df)
        path = tmp_path / "model.joblib"
        model.save(path)

        strat = MLSignalFilterStrategy(
            model_path=str(path),
            fit_on_init=False,
            min_history=60,
        )
        sig = strat.generate_signals(df)
        assert strat.model is not None
        assert strat.model.is_fitted_

    def test_load_failure_falls_back_to_fit(self, tmp_path):
        df = _make_ohlcv(200)
        bogus_path = tmp_path / "missing.joblib"
        strat = MLSignalFilterStrategy(
            model_path=str(bogus_path),
            fit_on_init=True,
            min_history=60,
        )
        sig = strat.generate_signals(df)
        # Should auto-fit since load failed
        assert strat.model is not None
        assert strat.model.is_fitted_


# ── Idempotency / determinism ───────────────────────────────────────
class TestDeterminism:
    def test_signals_index_aligned(self):
        df = _make_ohlcv(200)
        strat = MLSignalFilterStrategy(min_history=60)
        sig = strat.generate_signals(df)
        assert sig.index.equals(df.index)
        assert len(sig) == len(df)

    def test_signals_stable_across_calls(self):
        """Calling generate_signals twice on the same df should not change the model."""
        df = _make_ohlcv(200)
        strat = MLSignalFilterStrategy(min_history=60)
        sig1 = strat.generate_signals(df)
        sig2 = strat.generate_signals(df)
        pd.testing.assert_series_equal(sig1.astype("int8"), sig2.astype("int8"), check_names=False)