"""ML Hybrid Strategy Plugin — combines a base technical strategy with an
ML probability filter.

The hybrid runs a wrapped base strategy (default ``ema_cross_rsi``) to
produce candidate signals, then re-scores them with an ML model:

* Long entries (1) from the base strategy are kept only when the ML
  bullish probability exceeds ``ml_long_threshold``.
* Exits (-1) from the base strategy always pass through — risk exits
  are never overridden by ML.
* When the ML model flags strong bearish (proba < ``ml_bearish_threshold``)
  AND the base strategy emits a long, that long is converted to flat.

Follows the same auto-discovery contract as ``sentiment_filtered``:
subclass of ``Strategy`` with a module-level ``plugin`` attribute.
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


class MLHybridStrategy(Strategy):
    """Hybrid strategy combining a base technical strategy with ML filtering.

    Parameters
    ----------
    base_strategy_name : str
        Name of the wrapped base strategy. Default ``"ema_cross_rsi"``.
    fast, slow, rsi_period, rsi_entry_max, rsi_exit
        Parameters forwarded to ``EmaCrossRsi`` when the base strategy
        cannot be resolved from the registry.
    ml_long_threshold : float
        Minimum bullish probability required to allow a base long
        entry through. Default 0.55.
    ml_bearish_threshold : float
        If the ML probability is below this AND the base emits a long,
        the long is converted to flat (acts as a veto). Default 0.40.
    min_history : int
        Minimum bars needed before any ML-filtered signal is emitted.
        Default 60.
    sentiment_score : float
        Default sentiment broadcast into the feature frame. Default 0.0.
    fit_on_init : bool
        When True and no pre-fitted model is supplied, fit a fresh model
        on the first ``generate_signals`` call. Default True.
    model : GradientBoostedSignal, optional
        Pre-fitted model instance; takes precedence over ``model_path``.
    model_path : str, optional
        Filesystem path to a joblib-saved model. Loaded lazily on the
        first ``generate_signals`` call.
    backend : str, optional
        Backend hint used only when the strategy must instantiate its
        own model.
    """

    name = "ml_hybrid"
    params: dict[str, Any] = {}

    def __init__(
        self,
        base_strategy_name: str = "ema_cross_rsi",
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
        ml_long_threshold: float = 0.55,
        ml_bearish_threshold: float = 0.40,
        min_history: int = 60,
        sentiment_score: float = 0.0,
        fit_on_init: bool = True,
        model: Optional[GradientBoostedSignal] = None,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        if not 0.0 <= ml_bearish_threshold <= 1.0:
            raise ValueError(f"ml_bearish_threshold must be in [0, 1], got {ml_bearish_threshold}")
        if not 0.0 <= ml_long_threshold <= 1.0:
            raise ValueError(f"ml_long_threshold must be in [0, 1], got {ml_long_threshold}")
        if ml_bearish_threshold > ml_long_threshold:
            raise ValueError(
                f"ml_bearish_threshold ({ml_bearish_threshold}) must be <= "
                f"ml_long_threshold ({ml_long_threshold})"
            )

        self.base_strategy_name = base_strategy_name
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_entry_max = rsi_entry_max
        self.rsi_exit = rsi_exit
        self.ml_long_threshold = float(ml_long_threshold)
        self.ml_bearish_threshold = float(ml_bearish_threshold)
        self.min_history = int(min_history)
        self.sentiment_score = float(sentiment_score)
        self.fit_on_init = bool(fit_on_init)
        self.model: Optional[GradientBoostedSignal] = model
        self.model_path = model_path
        self.backend = backend

        self.params = {
            "base_strategy_name": base_strategy_name,
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
            "ml_long_threshold": self.ml_long_threshold,
            "ml_bearish_threshold": self.ml_bearish_threshold,
            "min_history": self.min_history,
            "sentiment_score": self.sentiment_score,
            "fit_on_init": self.fit_on_init,
            "model_path": self.model_path,
            "backend": self.backend,
        }

        self._base_strategy: Optional[Strategy] = None

        logger.info(
            "MLHybridStrategy init: base=%s ml_long>=%.2f ml_bearish<%.2f min_history=%d fit_on_init=%s",
            base_strategy_name,
            self.ml_long_threshold,
            self.ml_bearish_threshold,
            self.min_history,
            self.fit_on_init,
        )

    # ── Base strategy loader ─────────────────────────────────────────
    def _get_base_strategy(self) -> Strategy:
        """Lazily fetch the wrapped base strategy (registry → direct import)."""
        if self._base_strategy is not None:
            return self._base_strategy

        try:
            from bot.core import STRATEGIES
            from bot.core.plugins import discover_all

            discover_all()
            base = STRATEGIES.get(self.base_strategy_name)
            base_params = {
                "fast": self.fast,
                "slow": self.slow,
                "rsi_period": self.rsi_period,
                "rsi_entry_max": self.rsi_entry_max,
                "rsi_exit": self.rsi_exit,
            }
            self._base_strategy = type(base)(**base_params)
            logger.debug("MLHybrid: loaded base strategy from registry: %s", self.base_strategy_name)
        except Exception as exc:
            logger.warning(
                "MLHybrid: registry lookup for '%s' failed (%s) — instantiating EmaCrossRsi directly",
                self.base_strategy_name,
                exc,
            )
            from bot.plugins.strategies.ema_cross_rsi import EmaCrossRsi

            self._base_strategy = EmaCrossRsi(
                fast=self.fast,
                slow=self.slow,
                rsi_period=self.rsi_period,
                rsi_entry_max=self.rsi_entry_max,
                rsi_exit=self.rsi_exit,
            )
        return self._base_strategy

    # ── Model bootstrap ──────────────────────────────────────────────
    def _ensure_model(self, df: pd.DataFrame) -> bool:
        """Make sure ``self.model`` is fitted; True when ready, False otherwise."""
        if self.model is not None and getattr(self.model, "is_fitted_", False):
            return True

        if self.model is None and self.model_path is not None:
            try:
                loaded = GradientBoostedSignal.load(self.model_path)
                if getattr(loaded, "is_fitted_", False):
                    self.model = loaded
                    logger.info("MLHybrid: loaded model from %s", self.model_path)
                    return True
            except Exception as exc:
                logger.warning("MLHybrid: model load failed (%s)", exc)

        if not self.fit_on_init:
            return False

        if len(df) < self.min_history:
            logger.warning(
                "MLHybrid: insufficient history (%d < %d) — base signals pass through unmodified",
                len(df),
                self.min_history,
            )
            return False

        try:
            feats = build_feature_frame(df, sentiment_score=self.sentiment_score)
        except Exception as exc:
            logger.error("MLHybrid: feature build failed: %s", exc)
            return False

        valid = feats.dropna()
        if len(valid) < max(50, self.min_history // 2):
            logger.warning("MLHybrid: only %d valid feature rows — cannot fit", len(valid))
            return False

        close = df["Close"] if "Close" in df.columns else df.iloc[:, 4]
        fwd_ret = close.pct_change().shift(-1)
        y = (fwd_ret.reindex(valid.index) > 0).astype(int)
        if y.isna().any():
            valid = valid.loc[~y.isna()]
            y = y.loc[~y.isna()]

        if len(valid) < 50 or y.nunique() < 2:
            logger.warning("MLHybrid: insufficient labelled rows or single class — cannot fit")
            return False

        try:
            kwargs: dict[str, Any] = {}
            if self.backend is not None:
                kwargs["backend"] = self.backend
            self.model = GradientBoostedSignal(**kwargs)
            self.model.fit(valid, y)
            logger.info(
                "MLHybrid: fitted fresh model on %d rows × %d cols (backend=%s)",
                len(valid),
                valid.shape[1],
                self.model.backend,
            )
            return True
        except Exception as exc:
            logger.error("MLHybrid: fit failed: %s", exc)
            return False

    # ── Signal generation ────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate hybrid signals: base + ML filter.

        Base exits (-1) always pass through. Base longs (1) require
        ML bullish probability ≥ ``ml_long_threshold``. Strongly bearish
        ML (proba < ``ml_bearish_threshold``) vetoes base longs.
        """
        base = self._get_base_strategy()
        base_signals = base.generate_signals(df)
        result = base_signals.astype("int8").copy()

        if len(df) < self.min_history:
            logger.debug(
                "MLHybrid: df has %d rows (< min_history=%d) — base signals pass through",
                len(df),
                self.min_history,
            )
            return result

        if not self._ensure_model(df):
            return result

        try:
            feats = build_feature_frame(df, sentiment_score=self.sentiment_score)
        except Exception as exc:
            logger.error("MLHybrid: feature build failed at predict: %s", exc)
            return result

        valid_mask = ~feats.isna().any(axis=1)
        if not valid_mask.any():
            return result

        try:
            proba = self.model.predict_proba(feats.loc[valid_mask])
        except Exception as exc:
            logger.error("MLHybrid: predict_proba failed: %s", exc)
            return result

        # Convert base longs → flat unless ML agrees strongly bullish
        long_mask_in_valid = (base_signals.loc[valid_mask] == 1).to_numpy()
        veto_mask_in_valid = long_mask_in_valid & (proba < self.ml_bearish_threshold)
        allow_mask_in_valid = long_mask_in_valid & (proba >= self.ml_long_threshold)

        veto_indices = feats.index[valid_mask][veto_mask_in_valid]
        allowed_indices = feats.index[valid_mask][allow_mask_in_valid]

        vetoed = len(veto_indices)
        kept = len(allowed_indices)

        # Reset all base longs in valid region to 0 first; then re-allow the keepers
        long_indices_in_valid = feats.index[valid_mask][long_mask_in_valid]
        if len(long_indices_in_valid) > 0:
            result.loc[long_indices_in_valid] = 0
        if kept > 0:
            result.loc[allowed_indices] = 1

        logger.info(
            "MLHybrid: base longs=%d, kept_by_ml=%d, vetoed_by_ml=%d, exits_preserved=%d",
            int(long_mask_in_valid.sum()),
            kept,
            vetoed,
            int((base_signals == -1).sum()),
        )
        return result.astype("int8")


# Module-level plugin handle for auto-discovery
plugin = MLHybridStrategy()