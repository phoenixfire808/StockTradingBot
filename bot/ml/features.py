"""Feature engineering for ML signal model.

Builds a per-bar feature frame combining technical indicators, price-derived
quantities, calendar effects, and an external sentiment score.  The returned
DataFrame is index-aligned with the input OHLCV frame so labels and predictions
stay in lockstep.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from bot.indicators import atr, bollinger, ema, rsi

logger = logging.getLogger(__name__)

# ── Canonical feature column order ────────────────────────────────────
# Keep this explicit so train/persist/predict always use the same columns
# in the same order.  Tests and callers may reference FEATURE_COLUMNS.
FEATURE_COLUMNS: list[str] = [
    # returns
    "ret_1",
    "ret_2",
    "ret_5",
    # volatility
    "vol_10",
    "vol_20",
    # EMA distances (price vs EMA, normalised)
    "dist_ema9",
    "dist_ema21",
    "dist_ema50",
    # momentum / oscillators
    "rsi_14",
    "macd_hist",
    # range / volatility
    "atr_14",
    "atr_pct",
    # Bollinger position
    "bb_pos",
    # volume
    "volume_z",
    # sentiment
    "sentiment",
    # calendar
    "day_of_week",
    "day_of_month",
    "is_month_end",
]


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line − signal line)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _normalize_col(df: pd.DataFrame, upper: str, lower: str, pos: int) -> pd.Series:
    """Resolve a column by preferred upper/lower name, else positional."""
    if upper in df.columns:
        return df[upper]
    if lower in df.columns:
        return df[lower]
    return df.iloc[:, pos]


def build_feature_frame(
    df: pd.DataFrame,
    sentiment_score: float = 0.0,
    sentiment_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build a feature DataFrame from OHLCV bars.

    Parameters
    ----------
    df : DataFrame
        OHLCV bars.  Recognises both ``Close`` and ``close`` conventions;
        falls back to positional columns (0=Open, 1=High, 2=Low,
        3=Volume, 4=Close) when names are absent.
    sentiment_score : float
        Scalar sentiment value broadcast to every row.  Ignored when
        ``sentiment_series`` is supplied.
    sentiment_series : Series, optional
        Per-bar sentiment aligned to ``df.index``.  Takes precedence over
        ``sentiment_score``.

    Returns
    -------
    DataFrame
        Feature frame indexed like *df* with columns in
        ``FEATURE_COLUMNS`` order.  Rows with insufficient lookback are
        ``NaN``-containing but preserved so callers can drop or
        forward-fill as needed.
    """
    logger.info("Building feature frame from %d bars", len(df))

    close = _normalize_col(df, "Close", "close", 4)
    high = _normalize_col(df, "High", "high", 1)
    low = _normalize_col(df, "Low", "low", 2)
    volume = _normalize_col(df, "Volume", "volume", 3)

    feats = pd.DataFrame(index=df.index)

    # ── Returns ──────────────────────────────────────────────────────
    feats["ret_1"] = close.pct_change(1)
    feats["ret_2"] = close.pct_change(2)
    feats["ret_5"] = close.pct_change(5)

    # ── Volatility (rolling std of returns) ──────────────────────────
    feats["vol_10"] = feats["ret_1"].rolling(10).std()
    feats["vol_20"] = feats["ret_1"].rolling(20).std()

    # ── EMA distances (normalised by price) ──────────────────────────
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    feats["dist_ema9"] = (close - ema9) / close
    feats["dist_ema21"] = (close - ema21) / close
    feats["dist_ema50"] = (close - ema50) / close

    # ── Momentum / oscillators ───────────────────────────────────────
    feats["rsi_14"] = rsi(close, 14)
    feats["macd_hist"] = _macd(close)

    # ── ATR (absolute and percentage) ────────────────────────────────
    atr_14 = atr(df, 14)
    feats["atr_14"] = atr_14
    feats["atr_pct"] = atr_14 / close

    # ── Bollinger position: 0=lower band, 1=upper band ───────────────
    bands = bollinger(close, period=20, num_std=2.0)
    bandwidth = bands["upper"] - bands["lower"]
    feats["bb_pos"] = (close - bands["lower"]) / bandwidth.replace(0, np.nan)

    # ── Volume z-score ───────────────────────────────────────────────
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    feats["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # ── Sentiment ────────────────────────────────────────────────────
    if sentiment_series is not None:
        sentiment = sentiment_series.reindex(df.index)
    else:
        sentiment = pd.Series(sentiment_score, index=df.index, dtype=float)
    feats["sentiment"] = sentiment

    # ── Calendar features ────────────────────────────────────────────
    # Work on the index if it's datetime-like; fall back to a RangeIndex
    # so synthetic/non-datetime inputs don't crash.
    if isinstance(df.index, pd.DatetimeIndex):
        feats["day_of_week"] = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
        feats["day_of_month"] = pd.Series(df.index.day, index=df.index, dtype=float)
        feats["is_month_end"] = pd.Series(df.index.is_month_end, index=df.index, dtype=float)
    else:
        try:
            dt_index = pd.to_datetime(df.index)
            feats["day_of_week"] = pd.Series(dt_index.dayofweek, index=df.index, dtype=float)
            feats["day_of_month"] = pd.Series(dt_index.day, index=df.index, dtype=float)
            feats["is_month_end"] = pd.Series(dt_index.is_month_end, index=df.index, dtype=float)
        except Exception:
            logger.warning("Index not datetime-parseable — calendar features zeroed.")
            feats["day_of_week"] = 0.0
            feats["day_of_month"] = 0.0
            feats["is_month_end"] = 0.0

    # Re-order to canonical column list
    feats = feats[FEATURE_COLUMNS]

    logger.info(
        "Feature frame built: %d rows × %d cols | NaN rows: %d",
        len(feats), feats.shape[1], int(feats.isna().any(axis=1).sum()),
    )
    return feats
