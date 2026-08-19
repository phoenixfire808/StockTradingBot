"""Gradient-boosted signal classifier with graceful backend fallback.

Tries xgboost → lightgbm → sklearn GradientBoostingClassifier so the model
works whether or not the high-performance backends are installed.  Persists
to ``models/gradient_boosted.joblib`` via joblib.

Public surface:
    GradientBoostedSignal.fit(X, y)
    GradientBoostedSignal.predict_proba(df)
    GradientBoostedSignal.save(path)
    GradientBoostedSignal.load(path)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Backend selection ────────────────────────────────────────────────
_HAS_XGB = False
_HAS_LGBM = False
_HAS_SKL = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    pass

try:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_SKL = True
except ImportError:
    pass

if not (_HAS_XGB or _HAS_LGBM or _HAS_SKL):
    raise ImportError(
        "No gradient-boosting backend found — install xgboost, lightgbm, or scikit-learn."
    )

logger.info(
    "ML backends: xgboost=%s lightgbm=%s sklearn=%s",
    _HAS_XGB, _HAS_LGBM, _HAS_SKL,
)

DEFAULT_MODEL_PATH = Path("models/gradient_boosted.joblib")


class GradientBoostedSignal:
    """Gradient-boosted binary classifier producing bullish probability.

    Parameters
    ----------
    backend : str, optional
        Force ``"xgboost"``, ``"lightgbm"``, or ``"sklearn"``.  When
        ``None`` (default) the best available backend is chosen
        automatically.
    **kwargs
        Passed to the underlying estimator constructor.  Sensible
        defaults are set per-backend when the key is absent.
    """

    def __init__(self, backend: str | None = None, **kwargs):
        self.backend = self._resolve_backend(backend)
        self.model = self._instantiate(**kwargs)
        self.feature_names_: list[str] | None = None
        self.is_fitted_ = False
        logger.info("GradientBoostedSignal initialised backend=%s", self.backend)

    # ── Backend resolution ────────────────────────────────────────────
    @staticmethod
    def _resolve_backend(backend: str | None) -> str:
        if backend is not None:
            if backend == "xgboost" and _HAS_XGB:
                return backend
            if backend == "lightgbm" and _HAS_LGBM:
                return backend
            if backend == "sklearn" and _HAS_SKL:
                return backend
            logger.warning("Requested backend '%s' unavailable — auto-selecting.", backend)
        if _HAS_XGB:
            return "xgboost"
        if _HAS_LGBM:
            return "lightgbm"
        return "sklearn"

    def _instantiate(self, **kwargs):
        if self.backend == "xgboost":
            defaults = dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
        elif self.backend == "lightgbm":
            defaults = dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )
        else:  # sklearn
            defaults = dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
            )
        defaults.update(kwargs)
        logger.debug("Instantiating %s with params: %s", self.backend, defaults)
        if self.backend == "xgboost":
            return XGBClassifier(**defaults)
        if self.backend == "lightgbm":
            return LGBMClassifier(**defaults)
        return GradientBoostingClassifier(**defaults)

    # ── Fit ───────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Fit the gradient-boosted model.

        Parameters
        ----------
        X : DataFrame or array
            Feature matrix.  When a DataFrame is passed, column names
            are stored and enforced at predict time.
        y : Series or array
            Binary target (0/1).
        """
        # Pass a DataFrame (with feature columns) to the underlying estimator so
        # sklearn/lgbm/xgb all share consistent feature names; this avoids the
        # "X does not have valid feature names" warning from lightgbm at predict.
        X_df, y_arr = self._prepare_xy(X, y)

        logger.info(
            "Fitting %s on %d samples × %d features (positive rate=%.3f)",
            self.backend, X_df.shape[0], X_df.shape[1], float(np.mean(y_arr)),
        )

        self.model.fit(X_df, y_arr)

        self.is_fitted_ = True
        logger.info("Model fit complete. backend=%s", self.backend)
        return self

    # ── Predict ───────────────────────────────────────────────────────
    def predict_proba(self, df) -> np.ndarray:
        """Return probability of the positive class for each row.

        Accepts a DataFrame (columns reordered to match training) or a
        numpy array (assumed already ordered).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted — call fit() first.")
        # Pass a DataFrame (with named columns) to keep feature names consistent
        # with training; otherwise lightgbm/sklearn warn on predict.
        X_df = self._prepare_x(df)
        proba = self.model.predict_proba(X_df)
        # Column 1 = P(class==1)
        col_idx = 1 if proba.shape[1] > 1 else 0
        result = proba[:, col_idx]
        logger.debug(
            "predict_proba: %d predictions | mean=%.4f min=%.4f max=%.4f",
            len(result), float(np.mean(result)), float(np.min(result)), float(np.max(result)),
        )
        return result

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, path: str | Path | None = None) -> Path:
        """Persist model (including backend + feature names) to joblib."""
        path = Path(path) if path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "model": self.model,
            "feature_names_": self.feature_names_,
            "is_fitted_": self.is_fitted_,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s (backend=%s)", path, self.backend)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GradientBoostedSignal":
        """Load a persisted model from joblib."""
        path = Path(path) if path else DEFAULT_MODEL_PATH
        payload = joblib.load(path)
        inst = cls.__new__(cls)
        inst.backend = payload["backend"]
        inst.model = payload["model"]
        inst.feature_names_ = payload.get("feature_names_")
        inst.is_fitted_ = payload.get("is_fitted_", True)
        logger.info("Model loaded from %s (backend=%s fitted=%s)", path, inst.backend, inst.is_fitted_)
        return inst

    # ── Internal helpers ──────────────────────────────────────────────
    def _prepare_xy(self, X, y):
        """Validate X/y and return a DataFrame with named feature columns.

        Returning a DataFrame (not a numpy array) lets every backend —
        lightgbm, xgboost, sklearn — share consistent feature names, so
        ``feature_names_in_`` matches between fit and predict and we avoid
        the lightgbm "X does not have valid feature names" warning.
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_ = list(X_df.columns)
        else:
            arr = self._to_array(X)
            cols = [f"f{i}" for i in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=cols)
            self.feature_names_ = cols
        y_arr = np.asarray(y).ravel().astype(int)
        if X_df.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X/y row mismatch: X={X_df.shape[0]} y={y_arr.shape[0]}"
            )
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Target y must contain at least 2 classes for classification.")
        return X_df, y_arr

    def _prepare_x(self, df) -> pd.DataFrame:
        """Validate and reorder predict input to a named-column DataFrame.

        Aligns columns with ``feature_names_`` captured at fit time so the
        underlying estimator's ``feature_names_in_`` matches and no warning
        is emitted.
        """
        if isinstance(df, pd.DataFrame):
            if self.feature_names_ is not None:
                missing = [c for c in self.feature_names_ if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing feature columns at predict: {missing}")
                df = df[self.feature_names_]
            logger.debug("Predict input DataFrame %d×%d", df.shape[0], df.shape[1])
            return df
        arr = self._to_array(df)
        cols = self.feature_names_ if self.feature_names_ is not None else [f"f{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)


    @staticmethod
    def _to_array(df_or_arr) -> np.ndarray:
        if isinstance(df_or_arr, pd.DataFrame) or isinstance(df_or_arr, pd.Series):
            return df_or_arr.to_numpy(dtype=np.float64)
        arr = np.asarray(df_or_arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
