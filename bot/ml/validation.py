"""Walk-forward validation for ML signal models.

Provides:
    walk_forward_split — time-respecting expanding/rolling window generator
    evaluate_oos       — classification metrics on held-out test fold
    PurgedKFoldSplit   — cross-validation with label purge & embargo zones
    LookaheadBiasDetector — detects features leaking future information
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PurgedKFoldSplit
# ──────────────────────────────────────────────

class PurgedKFoldSplit(KFold):
    """Time-series-aware cross-validation with label purging and embargo.

    Extends :class:`sklearn.model_selection.KFold` by removing from each
    test-fold any training samples whose labels overlap in time with the
    test labels (the *purge* step), and then excising a configurable gap
    (the *embargo*) immediately after every test sample to prevent
    look-ahead leakage from label windows that extend past the boundary.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    train_size : int | None
        Absolute number of rows to use for training per split.  When
        ``None`` the full non-test pool is used.
    test_size : int
        Number of contiguous test rows per fold.
    pct_embargo : float
        Fraction of *test_size* placed behind the embargo zone (0–1;
        default 0.05).  The absolute embargo count is
        ``max(1, int(test_size * pct_embargo))``.
    gap : int
        Extra integer padding between test end and embargo start
        (useful when labels span multiple rows).  Default 0.

    Attributes
    ----------
    splits_ : list[tuple[np.ndarray, np.ndarray]]
        Filled after :meth:`split`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        pct_embargo: float = 0.05,
        gap: int = 0,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.train_size = train_size
        self.test_size = test_size or test_size
        self.pct_embargo = pct_embargo
        self.gap = gap
        self.splits_: list[tuple[np.ndarray, np.ndarray]] = []

    # -- public API --------------------------------------------------

    def get_n_splits(
        self, X: pd.DataFrame | None = None, y=None, groups=None
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` tuples with purge + embargo."""
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to split.")

        # Determine how many rows belong in each test fold.
        test_rows = self.test_size or max(1, n_samples // (self.n_splits + 1))
        total_test = test_rows * self.n_splits
        embargo_rows = max(1, int(test_rows * self.pct_embargo))

        # Build contiguous, non-overlapping test blocks spread across the data.
        step = n_samples // self.n_splits
        for i in range(self.n_splits):
            test_start = i * step
            test_end = min(test_start + test_rows, n_samples)
            if test_end > n_samples:
                break

            test_idx = np.arange(test_start, test_end)

            # All other indices are the candidate training pool.
            pool = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])

            if self.train_size is not None:
                pool = pool[: self.train_size]

            # --- Purge: remove any training rows whose labels overlap
            # the test label window.  We assume labels span exactly the
            # same rows as the test set (standard triple-barrier usage).
            # Rows touching or within gap rows after the test set get removed.
            embargo_start = test_end
            embargo_end = min(test_end + embargo_rows + self.gap, n_samples)
            embargo_mask = np.isin(pool, np.arange(embargo_start, embargo_end))
            train_idx = pool[~embargo_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning("PurgedKFoldSplit fold %d: empty train or test", i)
                continue

            self.splits_.append((train_idx, test_idx))
            yield train_idx, test_idx

    @property
    def n_splits_(self) -> int:
        return len(self.splits_)


# ──────────────────────────────────────────────
# LookaheadBiasDetector
# ──────────────────────────────────────────────

class LookaheadBiasDetector:
    """Detect features that contain information about future outcomes.

    For every column in *X*, the detector computes the correlation between
    the *current* feature values and the *next-k* target values (label
    horizon *k*).  Columns whose absolute correlation exceeds *threshold*
    are flagged as leaky — they would cause look-ahead bias if trained on
    them without appropriate lagging.

    Parameters
    ----------
    lookback : int
        How many steps into the future to check (default 5).
    threshold : float
        Absolute Pearson-correlation cutoff above which a column is
        flagged (default 0.1).

    Attributes
    ----------
    correlation_matrix_ : pd.DataFrame
        Correlations between every feature and the forward-shifted target.
    flagged_columns_ : list[str]
        Column names exceeding the threshold.

    Examples
    --------
    >>> detector = LookaheadBiasDetector(lookback=10, threshold=0.05)
    >>> detector.fit(X, y)
    >>> detector.flagged_columns_
    ['momentum_20d', 'volume_spread']
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.1) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self.correlation_matrix_: pd.DataFrame | None = None
        self.flagged_columns_: list[str] = []

    # -- public API --------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> LookaheadBiasDetector:
        """Compute correlations and flag leaky columns.

        Parameters
        ----------
        X : DataFrame
            Feature matrix, chronologically ordered.
        y : Series or DataFrame
            Target labels / outcomes.

        Returns
        -------
        self
        """
        feature_cols = list(X.columns)
        y_flat = np.asarray(y, dtype=float).ravel()

        n_lookahead = min(self.lookback, len(y_flat) - 1)
        if n_lookahead <= 0:
            self.correlation_matrix_ = pd.DataFrame(
                np.eye(len(feature_cols)),
                index=feature_cols,
                columns=feature_cols,
            )
            self.flagged_columns_ = []
            return self

        # Build correlation matrix: rows = horizons (1..n), cols = features
        corr_rows: dict[str, list[float]] = {}
        for col in feature_cols:
            corr_rows[col] = [np.nan] * n_lookahead

        # Correlate CURRENT feature values with FORWARD-shifted targets.
        # For horizon k: corr(X[i], Y[i+k]) => y shifts forward, x stays aligned.
        for k in range(1, n_lookahead + 1):
            yf = y_flat[k:]  # y shifted forward k steps: y[k], y[k+1], ...
            for feat_idx, feat in enumerate(feature_cols):
                x_vals = X[feat].values.astype(float)[:len(yf)]  # truncate to match yf length
                mask = np.isfinite(x_vals) & np.isfinite(yf)
                xa, yf_m = x_vals[mask], yf[mask]
                if len(xa) < 3:
                    continue
                c = np.corrcoef(xa, yf_m)[0, 1]
                corr_rows[feat][feat_idx] = float(c) if np.isfinite(c) else np.nan

        # Transpose so rows=horizons, cols=features for final DataFrame
        transposed: dict[str, list[float]] = {}
        for h in range(1, n_lookahead + 1):
            label = str(h)
            transposed[label] = [corr_rows[feat][h - 1] for feat in feature_cols]

        self.correlation_matrix_ = pd.DataFrame(
            transposed,
            index=feature_cols,
            columns=[str(k) for k in range(1, n_lookahead + 1)],
        ).T

        # Flag any column that has abs(corr) > threshold in ANY horizon.
        self.flagged_columns_ = [
            col
            for col in feature_cols
            if self.correlation_matrix_[col].abs().max() > self.threshold
        ]

        return self

    def detect(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
        """One-shot convenience: fit + return summary dict."""
        self.fit(X, y)
        return {
            "flagged_columns": self.flagged_columns_,
            "correlation_matrix": self.correlation_matrix_,
            "n_flagged": len(self.flagged_columns_),
            "total_features": len(self.correlation_matrix_.columns) if self.correlation_matrix_ is not None else 0,
            "threshold": self.threshold,
            "lookback": self.lookback,
        }


# ──────────────────────────────────────────────
# Walk-forward helpers (unchanged core logic)
# ──────────────────────────────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    step_size: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, test) DataFrame pairs via walk-forward windows.

    Walks forward through *df* (assumed chronologically ordered) emitting
    contiguous train windows of ``train_size`` rows followed by test
    windows of ``step_size`` rows.  After each yield the train window
    advances by ``step_size`` rows.

    Parameters
    ----------
    df : DataFrame
        Chronologically ordered data (features or full frame).
    train_size : int
        Number of rows in each training fold.
    step_size : int
        Number of rows in each test fold (and the stride).

    Yields
    ------
    (train_df, test_df) : tuple[DataFrame, DataFrame]

    Raises
    ------
    ValueError
        If sizes are invalid or the frame is too small for even one fold.
    """
    n = len(df)
    if train_size <= 0 or step_size <= 0:
        raise ValueError(f"train_size and step_size must be > 0, got {train_size}/{step_size}")
    if n < train_size + step_size:
        raise ValueError(
            f"DataFrame too small ({n} rows) for train_size={train_size} + step_size={step_size}"
        )

    start = 0
    fold_idx = 0
    while start + train_size + step_size <= n:
        train_df = df.iloc[start : start + train_size]
        test_df = df.iloc[start + train_size : start + train_size + step_size]
        logger.info(
            "walk_forward fold %d: train=[%d:%d] test=[%d:%d]",
            fold_idx, start, start + train_size,
            start + train_size, start + train_size + step_size,
        )
        yield train_df, test_df
        start += step_size
        fold_idx += 1

    logger.info("walk_forward_split: emitted %d folds from %d rows", fold_idx, n)


# ──────────────────────────────────────────────
# evaluate_oos — enhanced
# ──────────────────────────────────────────────

def evaluate_oos(model, X_test: pd.DataFrame, y_test) -> dict:
    """Evaluate a fitted model's out-of-sample classification performance.

    Parameters
    ----------
    model : object
        Anything with a ``predict_proba`` method returning P(class==1).
    X_test : DataFrame
        Test features.
    y_test : Series or array
        Binary truth labels.

    Returns
    -------
    dict
        Keys include:

        * **Standard** – ``accuracy``, ``precision``, ``recall``, ``f1``,
          ``roc_auc``, ``avg_precision``, ``n_samples``,
          ``positive_rate``, ``pred_mean``.
        * **Per-class** – ``precision_pos``, ``precision_neg``,
          ``recall_pos``, ``recall_neg``, ``f1_pos``, ``f1_neg``.
        * **Confusion matrix** – ``confusion_matrix`` (4-element list
          ``[[TN, FP], [FN, TP]]``).
        * **SHAP-like importance** – ``feature_importance`` dict mapping
          each feature name to the mean absolute difference in its value
          between positive and negative label groups (higher = more
          discriminative).
    """
    y_arr = np.asarray(y_test).ravel().astype(int)
    proba = model.predict_proba(X_test)
    y_pred = (proba[:, 1] >= 0.5).astype(int)

    classes = np.unique(y_arr)
    pos_class = 1
    neg_class = 0

    metrics: dict = {
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "n_samples": int(len(y_arr)),
        "positive_rate": float(np.mean(y_arr)),
        "pred_mean": float(np.mean(proba[:, 1])),
    }

    # ROC-AUC / PR-AUC require both classes present
    if len(classes) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_arr, proba[:, 1]))
        metrics["avg_precision"] = float(average_precision_score(y_arr, proba[:, 1]))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
        logger.warning("y_test has single class — ROC-AUC/avg_precision set to NaN")

    # --- Per-class precision, recall, F1 ---
    if len(classes) > 1:
        p_per = precision_score(y_arr, y_pred, labels=[pos_class, neg_class], zero_division=0, average=None)
        r_per = recall_score(y_arr, y_pred, labels=[pos_class, neg_class], zero_division=0, average=None)
        f1_per = f1_score(y_arr, y_pred, labels=[pos_class, neg_class], zero_division=0, average=None)
        metrics["precision_pos"] = float(p_per[0])
        metrics["precision_neg"] = float(p_per[1]) if len(p_per) > 1 else 0.0
        metrics["recall_pos"] = float(r_per[0])
        metrics["recall_neg"] = float(r_per[1]) if len(r_per) > 1 else 0.0
        metrics["f1_pos"] = float(f1_per[0])
        metrics["f1_neg"] = float(f1_per[1]) if len(f1_per) > 1 else 0.0
    else:
        for suffix in ("pos", "neg"):
            metrics[f"precision_{suffix}"] = 0.0
            metrics[f"recall_{suffix}"] = 0.0
            metrics[f"f1_{suffix}"] = 0.0

    # --- Confusion matrix [[TN, FP], [FN, TP]] ---
    cm = confusion_matrix(y_arr, y_pred, labels=[neg_class, pos_class])
    metrics["confusion_matrix"] = cm.tolist()

    # --- SHAP-like feature importance ---
    # Mean absolute difference in feature value distributions between
    # positive and negative label subsets of *y_test*.
    feat_imp: dict[str, float] = {}
    try:
        X_arr = np.asarray(X_test)
        cols = list(X_test.columns)
        mask_pos = y_arr == 1
        mask_neg = y_arr == 0
        if mask_pos.sum() > 0 and mask_neg.sum() > 0:
            for i, col_name in enumerate(cols):
                mean_pos = np.nanmean(X_arr[mask_pos, i])
                mean_neg = np.nanmean(X_arr[mask_neg, i])
                feat_imp[str(col_name)] = float(abs(mean_pos - mean_neg))
        else:
            logger.warning("evaluate_oos: cannot compute feature importance — insufficient class balance")
    except Exception:
        logger.exception("evaluate_oos: feature importance computation failed; skipping")
        feat_imp = {}
    metrics["feature_importance"] = feat_imp

    logger.info(
        "OOS eval: n=%d acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f ap=%.4f",
        metrics["n_samples"], metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        metrics.get("roc_auc", float("nan")),
        metrics.get("avg_precision", float("nan")),
    )
    return metrics
