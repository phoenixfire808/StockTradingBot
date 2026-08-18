"""Tests for bot.ml.model — GradientBoostedSignal fit/predict/persist."""

import numpy as np
import pandas as pd
import pytest

from bot.ml.model import GradientBoostedSignal


def _make_classification_data(n=400, n_features=18, seed=42):
    """Build a separable-ish binary classification frame."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    # Linear separation: positive class when weighted sum exceeds threshold
    weights = rng.normal(0, 1, n_features)
    logits = X @ weights + rng.normal(0, 0.3, n)
    y = (logits > np.median(logits)).astype(int)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


class TestGradientBoostedSignal:
    def test_fit_and_predict_proba(self):
        X, y = _make_classification_data()
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(y)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_fit_returns_self(self):
        X, y = _make_classification_data(n=200)
        model = GradientBoostedSignal(backend="sklearn")
        ret = model.fit(X, y)
        assert ret is model

    def test_predict_before_fit_raises(self):
        X, _ = _make_classification_data(n=100)
        model = GradientBoostedSignal(backend="sklearn")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)

    def test_single_class_target_raises(self):
        X, _ = _make_classification_data(n=100)
        y = pd.Series(np.zeros(100, dtype=int))
        model = GradientBoostedSignal(backend="sklearn")
        with pytest.raises(ValueError, match="at least 2 classes"):
            model.fit(X, y)

    def test_feature_name_alignment_on_predict(self):
        X, y = _make_classification_data(n=300)
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(X, y)
        # Reorder columns — should still work because names are stored
        X_shuffled = X[list(reversed(X.columns))]
        proba = model.predict_proba(X_shuffled)
        assert len(proba) == len(y)

    def test_missing_column_raises(self):
        X, y = _make_classification_data(n=200)
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(X, y)
        X_missing = X.drop(columns=["f0"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            model.predict_proba(X_missing)

    def test_save_and_load_roundtrip(self, tmp_path):
        X, y = _make_classification_data(n=300)
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(X, y)
        path = tmp_path / "gb.joblib"
        model.save(path)
        loaded = GradientBoostedSignal.load(path)
        assert loaded.backend == model.backend
        assert loaded.is_fitted_ is True
        # Predictions should match
        p1 = model.predict_proba(X)
        p2 = loaded.predict_proba(X)
        np.testing.assert_allclose(p1, p2, atol=1e-10)

    def test_learns_better_than_random(self):
        """On a separable problem the model should beat ~50% accuracy."""
        X, y = _make_classification_data(n=500)
        model = GradientBoostedSignal(backend="sklearn")
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        acc = (preds == y.values).mean()
        assert acc > 0.60, f"Model should beat random; got acc={acc:.3f}"

    @pytest.mark.parametrize("backend", ["xgboost", "lightgbm", "sklearn"])
    def test_all_backends_fit_predict(self, backend):
        """Verify every available backend can fit and predict."""
        X, y = _make_classification_data(n=200)
        model = GradientBoostedSignal(backend=backend)
        # If backend unavailable, the class auto-selects; just verify it runs
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(y)
        assert proba.min() >= 0
        assert proba.max() <= 1
