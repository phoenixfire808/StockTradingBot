"""Walk-forward validation for ML signal models.

Provides:
    walk_forward_split — time-respecting expanding/rolling window generator
    evaluate_oos       — classification metrics on held-out test fold
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


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
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, ``avg_precision``, ``n_samples``,
        ``positive_rate``, ``pred_mean``.
    """
    y_arr = np.asarray(y_test).ravel().astype(int)
    proba = model.predict_proba(X_test)
    y_pred = (proba >= 0.5).astype(int)

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "n_samples": int(len(y_arr)),
        "positive_rate": float(np.mean(y_arr)),
        "pred_mean": float(np.mean(proba)),
    }

    # ROC-AUC / PR-AUC require both classes present
    if len(np.unique(y_arr)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_arr, proba))
        metrics["avg_precision"] = float(average_precision_score(y_arr, proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
        logger.warning("y_test has single class — ROC-AUC/avg_precision set to NaN")

    logger.info(
        "OOS eval: n=%d acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f ap=%.4f",
        metrics["n_samples"], metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        metrics["roc_auc"], metrics["avg_precision"],
    )
    return metrics
