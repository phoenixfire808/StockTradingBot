"""ML Signal-Filter Strategy Plugin — emits buy/sell signals directly from a
gradient-boosted ML model trained on engineered OHLCV features.

The strategy builds a feature frame from the input bars using
``bot.ml.features.build_feature_frame``, fits (or reuses) a
``GradientBoostedSignal`` classifier, and converts the per-bar bullish
probability into discrete signals:

    P(bullish) >= long_threshold  →  1  (long entry)
    P(bullish) <= short_threshold → -1  (exit / short)
    otherwise                     →  0  (flat)

If no pre-trained model is supplied, the strategy fits one on the first
``generate_signals`` call using a synthetic target derived from the
forward return of the supplied history (``ret_1 > 0``).  This lets the
plugin be used out-of-the-box while still being override-friendly — pass
a fitted ``GradientBoostedSignal`` (or a ``model_path``) for production
use.

Designed to follow the same auto-discovery contract as the other
``bot.plugins.strategies`` plugins (subclass of ``Strategy`` with a
module-level ``plugin`` attribute).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from bot.ml.features import build_feature_frame
from bot.ml.model import GradientBoostedSignal
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MLSignalFilterStrategy(Strategy):
    """Pure ML signal strategy — emits long/exit signals from model probability.

    Parameters
    ----------
    long_threshold : float
        Minimum bullish probability to emit a ``1`` (long entry). Default 0.55.
    short_threshold : float
        Maximum bullish probability to emit a ``-1`` (exit/short). Default 0.45.
    min_history : int
        Minimum bars required before any signal is emitted (lookback for
        feature windows plus label construction). Default 60.
    sentiment_score : float
        Default sentiment broadcast into the feature frame when no
        per-bar series is supplied. Default 0.0 (neutral).
    fit_on_init : bool
        When True and no model is supplied, fit a fresh model on the
        first ``generate_signals`` call using ``df`` itself as training
        data with synthetic labels (``ret_1 > 0``).  When False and no
        model is supplied, all signals are returned as ``0`` until a
        model is provided. Default True.
    model : GradientBoostedSignal, optional
        Pre-trained model instance. Takes precedence over ``model_path``.
    model_path : str, optional
        Filesystem path to a joblib-saved model. Loaded lazily on the
        first ``generate_signals`` call when supplied.
    backend : str, optional
        Backend hint (``"xgboost"``, ``"lightgbm"``, ``"sklearn"``) used
        only when the strategy has to instantiate its own model.
    """

    name = "ml_signal_filter"
    params: dict[str, Any] = {}

    def __init__(
        self,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        min_history: int = 60,
        sentiment_score: float = 0.0,
        fit_on_init: bool = True,
        model: Optional[GradientBoostedSignal] = None,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        if not 0.0 <= short_threshold <= 1.0:
            raise ValueError(f"short_threshold must be in [0, 1], got {short_threshold}")
        if not 0.0 <= long_threshold <= 1.0:
            raise ValueError(f"long_threshold must be in [0, 1], got {long_threshold}")
        if short_threshold > long_threshold:
            raise ValueError(
                f"short_threshold ({short_threshold}) must be <= long_threshold ({long_threshold})"
            )

        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.min_history = int(min_history)
        self.sentiment_score = float(sentiment_score)
        self.fit_on_init = bool(fit_on_init)
        self.model: Optional[GradientBoostedSignal] = model
        self.model_path = model_path
        self.backend = backend

        self.params = {
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "min_history": self.min_history,
            "sentiment_score": self.sentiment_score,
            "fit_on_init": self.fit_on_init,
            "model_path": self.model_path,
            "backend": self.backend,
        }

        logger.info(
            "MLSignalFilterStrategy init: long>=%.2f short<=%.2f min_history=%d fit_on_init=%s",
            self.long_threshold,
            self.short_threshold,
            self.min_history,
            self.fit_on_init,
        )

    # ── Model bootstrap helpers ──────────────────────────────────────
    def _ensure_model(self, df: pd.DataFrame) -> bool:
        """Make sure ``self.model`` is fitted and ready.

        Returns True if a fitted model is available, False if signals
        should remain flat (insufficient history or fitting disabled).
        """
        if self.model is not None and getattr(self.model, "is_fitted_", False):
            return True

        # Try loading from disk first
        if self.model is None and self.model_path is not None:
            try:
                loaded = GradientBoostedSignal.load(self.model_path)
                if getattr(loaded, "is_fitted_", False):
                    self.model = loaded
                    logger.info("Loaded ML model from %s", self.model_path)
                    return True
            except Exception as exc:
                logger.warning("Could not load model from %s: %s", self.model_path, exc)

        # Auto-fit if requested
        if not self.fit_on_init:
            logger.debug("No model and fit_on_init=False — flat signals.")
            return False

        return self._fit_on_df(df)

    def _fit_on_df(self, df: pd.DataFrame) -> bool:
        """Fit a fresh model on ``df`` using synthetic labels (ret_1 > 0).

        Returns True on success, False if insufficient data / single class.
        """
        if len(df) < self.min_history:
            logger.warning(
                "MLSignalFilter: insufficient history (%d < %d) — flat signals",
                len(df),
                self.min_history,
            )
            return False

        try:
            feats = build_feature_frame(df, sentiment_score=self.sentiment_score)
        except Exception as exc:
            logger.error("MLSignalFilter: feature build failed: %s", exc)
            return False

        # Drop rows with NaN (lookback windows haven't warmed up yet)
        valid = feats.dropna()
        if len(valid) < max(50, self.min_history // 2):
            logger.warning(
                "MLSignalFilter: only %d valid rows after NaN drop — cannot fit",
                len(valid),
            )
            return False

        # Synthetic target: next-bar return > 0 → 1, else 0
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 4]
        fwd_ret = close.pct_change().shift(-1)
        y = (fwd_ret.reindex(valid.index) > 0).astype(int)

        # Drop the very last row (no forward return available)
        if y.isna().any():
            valid = valid.loc[~y.isna()]
            y = y.loc[~y.isna()]

        if len(valid) < 50 or y.nunique() < 2:
            logger.warning(
                "MLSignalFilter: insufficient labelled rows (%d) or single class — flat signals",
                len(valid),
            )
            return False

        try:
            kwargs: dict[str, Any] = {}
            if self.backend is not None:
                kwargs["backend"] = self.backend
            self.model = GradientBoostedSignal(**kwargs)
            self.model.fit(valid, y)
            logger.info(
                "MLSignalFilter: fitted fresh model on %d rows × %d cols (backend=%s)",
                len(valid),
                valid.shape[1],
                self.model.backend,
            )
            return True
        except Exception as exc:
            logger.error("MLSignalFilter: fit failed: %s", exc)
            return False

    # ── Signal generation ────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return an int8 Series of model-derived signals aligned to ``df``.

        Returns zeros for the warm-up window where features contain NaN.
        """
        signal = pd.Series(0, index=df.index, dtype="int8")

        if len(df) < self.min_history:
            logger.debug(
                "MLSignalFilter: df has %d rows (< min_history=%d) — flat",
                len(df),
                self.min_history,
            )
            return signal

        if not self._ensure_model(df):
            return signal

        try:
            feats = build_feature_frame(df, sentiment_score=self.sentiment_score)
        except Exception as exc:
            logger.error("MLSignalFilter: feature build failed at predict: %s", exc)
            return signal

        # Predict only over rows with valid features (drop NaN windows)
        valid_mask = ~feats.isna().any(axis=1)
        if not valid_mask.any():
            logger.debug("MLSignalFilter: no valid feature rows — flat")
            return signal

        try:
            proba = self.model.predict_proba(feats.loc[valid_mask])
        except Exception as exc:
            logger.error("MLSignalFilter: predict_proba failed: %s", exc)
            return signal

        # Map probabilities → discrete signals
        valid_signals = np.where(
            proba >= self.long_threshold, 1,
            np.where(proba <= self.short_threshold, -1, 0),
        ).astype("int8")

        signal.loc[valid_mask] = valid_signals

        logger.debug(
            "MLSignalFilter: emitted %d longs, %d shorts across %d bars",
            int((signal == 1).sum()),
            int((signal == -1).sum()),
            len(df),
        )
        return signal.astype("int8")


# Module-level plugin handle for auto-discovery
plugin = MLSignalFilterStrategy()