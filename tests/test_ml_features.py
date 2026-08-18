"""Tests for bot.ml.features — feature frame construction."""

import numpy as np
import pandas as pd
import pytest

from bot.ml.features import build_feature_frame, FEATURE_COLUMNS


def _make_ohlcv(n=120, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0
    # Random-walk close with mild upward drift
    rets = rng.normal(0.0005, 0.02, n)
    close = base * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(1000, 50000, n).astype(float)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


class TestBuildFeatureFrame:
    def test_returns_dataframe_with_canonical_columns(self):
        df = _make_ohlcv()
        feats = build_feature_frame(df)
        assert isinstance(feats, pd.DataFrame)
        assert list(feats.columns) == FEATURE_COLUMNS
        assert len(feats) == len(df)

    def test_has_at_least_12_features(self):
        assert len(FEATURE_COLUMNS) >= 12

    def test_index_aligned_with_input(self):
        df = _make_ohlcv(80)
        feats = build_feature_frame(df)
        assert feats.index.equals(df.index)

    def test_sentiment_scalar_broadcast(self):
        df = _make_ohlcv(60)
        feats = build_feature_frame(df, sentiment_score=0.42)
        assert (feats["sentiment"] == 0.42).all()

    def test_sentiment_series_overrides_scalar(self):
        df = _make_ohlcv(60)
        sent = pd.Series(np.linspace(-1, 1, 60), index=df.index)
        feats = build_feature_frame(df, sentiment_score=99.0, sentiment_series=sent)
        assert feats["sentiment"].tolist() == pytest.approx(sent.tolist())

    def test_rsi_in_valid_range(self):
        df = _make_ohlcv(200)
        feats = build_feature_frame(df)
        valid = feats["rsi_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_returns_columns_near_zero_mean(self):
        df = _make_ohlcv(300)
        feats = build_feature_frame(df)
        assert feats["ret_1"].dropna().abs().mean() < 0.10

    def test_calendar_features_populated(self):
        df = _make_ohlcv(60)
        feats = build_feature_frame(df)
        # day_of_week on business days should be 0-4
        assert feats["day_of_week"].between(0, 4).all()
        assert feats["day_of_month"].between(1, 31).all()

    def test_works_with_lowercase_columns(self):
        df = _make_ohlcv(80)
        df.columns = [c.lower() for c in df.columns]
        feats = build_feature_frame(df)
        assert not feats["rsi_14"].dropna().empty

    def test_nans_present_for_initial_lookback(self):
        """Early rows should have NaN due to rolling windows."""
        df = _make_ohlcv(50)
        feats = build_feature_frame(df)
        assert feats.iloc[0].isna().any(), "First row should contain NaN from lookback windows"

    def test_volume_z_bounded_typically(self):
        df = _make_ohlcv(100)
        feats = build_feature_frame(df)
        vz = feats["volume_z"].dropna()
        assert vz.abs().max() < 10  # z-scores should be reasonable
