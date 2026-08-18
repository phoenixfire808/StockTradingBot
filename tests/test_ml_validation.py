"""Tests for bot.ml.validation — walk-forward splits and OOS evaluation."""

import numpy as np
import pandas as pd
import pytest

from bot.ml.validation import walk_forward_split, evaluate_oos


def _make_frame(n=200, n_features=5, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    y = pd.Series((df["f0"] > 0).astype(int), name="y")
    return df, y


class TestWalkForwardSplit:
    def test_yields_correct_number_of_folds(self):
        df, _ = _make_frame(n=100)
        folds = list(walk_forward_split(df, train_size=40, step_size=20))
        # 40+20=60 per block, 100 rows → folds: [0:60], [20:80], [40:100] → 3
        assert len(folds) == 3

    def test_train_test_sizes_correct(self):
        df, _ = _make_frame(n=100)
        for train_df, test_df in walk_forward_split(df, train_size=30, step_size=20):
            assert len(train_df) == 30
            assert len(test_df) == 20

    def test_no_overlap_between_train_and_test(self):
        df, _ = _make_frame(n=100)
        for train_df, test_df in walk_forward_split(df, train_size=40, step_size=20):
            train_idx = set(train_df.index)
            test_idx = set(test_df.index)
            assert train_idx.isdisjoint(test_idx), "Train/test must not overlap"

    def test_test_window_immediately_follows_train(self):
        df, _ = _make_frame(n=100)
        for train_df, test_df in walk_forward_split(df, train_size=40, step_size=20):
            assert test_df.index[0] == train_df.index[-1] + 1 or \
                   test_df.index[0] > train_df.index[-1]

    def test_too_small_raises(self):
        df, _ = _make_frame(n=30)
        with pytest.raises(ValueError, match="too small"):
            list(walk_forward_split(df, train_size=40, step_size=10))

    def test_invalid_sizes_raise(self):
        df, _ = _make_frame(n=100)
        with pytest.raises(ValueError, match="must be > 0"):
            list(walk_forward_split(df, train_size=0, step_size=10))
        with pytest.raises(ValueError, match="must be > 0"):
            list(walk_forward_split(df, train_size=10, step_size=0))

    def test_expanding_stride(self):
        """Each successive fold's test block should shift forward by step_size."""
        df, _ = _make_frame(n=120)
        starts = []
        for train_df, test_df in walk_forward_split(df, train_size=40, step_size=20):
            starts.append(test_df.index[0])
        # n=120, train=40, step=20 → folds: [0:60],[20:80],[40:100],[60:120] → 4 folds
        assert starts == [40, 60, 80, 100]


class TestEvaluateOos:
    def test_returns_all_metrics(self):
        df, y = _make_frame(n=200)
        # Use a dummy model with predict_proba
        from bot.ml.model import GradientBoostedSignal
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(df.iloc[:150], y.iloc[:150])
        metrics = evaluate_oos(model, df.iloc[150:], y.iloc[150:])
        expected_keys = {
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "avg_precision", "n_samples",
            "positive_rate", "pred_mean",
        }
        assert expected_keys.issubset(metrics.keys())
        assert metrics["n_samples"] == 50

    def test_accuracy_in_valid_range(self):
        df, y = _make_frame(n=200)
        from bot.ml.model import GradientBoostedSignal
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(df.iloc[:150], y.iloc[:150])
        metrics = evaluate_oos(model, df.iloc[150:], y.iloc[150:])
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_single_class_y_returns_nan_auc(self):
        df, y = _make_frame(n=200)
        from bot.ml.model import GradientBoostedSignal
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(df.iloc[:150], y.iloc[:150])
        # All-zero test target
        y_single = pd.Series(np.zeros(50, dtype=int))
        metrics = evaluate_oos(model, df.iloc[150:], y_single)
        assert np.isnan(metrics["roc_auc"])
        assert np.isnan(metrics["avg_precision"])

    def test_walk_forward_pipeline_integration(self):
        """End-to-end: walk-forward split → fit → evaluate over folds."""
        from bot.ml.features import build_feature_frame, FEATURE_COLUMNS
        from bot.ml.model import GradientBoostedSignal

        # Build OHLCV → features → label
        rng = np.random.default_rng(99)
        n = 300
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
        ohlcv = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1000, 50000, n).astype(float),
        }, index=pd.bdate_range("2024-01-01", periods=n))
        feats = build_feature_frame(ohlcv)
        # Label: 1 if next-bar return > 0
        y = (pd.Series(close, index=ohlcv.index).pct_change(1).shift(-1) > 0).astype(int)

        feats = feats.dropna()
        y = y.reindex(feats.index).dropna()
        common = feats.index.intersection(y.index)
        feats, y = feats.loc[common], y.loc[common]

        aucs = []
        for train_df, test_df in walk_forward_split(feats, train_size=100, step_size=50):
            y_train = y.loc[train_df.index]
            y_test = y.loc[test_df.index]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            model = GradientBoostedSignal(backend="sklearn")
            model.fit(train_df, y_train)
            metrics = evaluate_oos(model, test_df, y_test)
            aucs.append(metrics["roc_auc"])

        assert len(aucs) > 0, "Should produce at least one valid fold"
        # AUC should be finite (not perfect, but not NaN)
        assert all(not np.isnan(a) for a in aucs)
