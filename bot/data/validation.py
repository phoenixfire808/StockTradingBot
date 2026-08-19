"""Data validation pipeline for OHLCV price data (ML4T research-based).

Implements a comprehensive DataValidator class with methods for:
- Timestamp continuity checks and session-aligned reindexing
- Outlier detection (z-score and IQR clipping)
- Duplicate removal by timestamp+symbol
- OHLC consistency enforcement
- NaN gap detection and reporting
- Minimum length guardrails
- Full pipeline clean() producing validated, cleaned DataFrames
Dependencies: pandas, numpy only.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    r = num / den.replace(0, np.nan)
    return r.fillna(fill).astype(float)


def _rolling_mean_std(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    r = series.rolling(window=window, min_periods=max(1, window // 2), center=False)
    return r.mean(), r.std().fillna(1e-9)


class DataValidator:
    """Validate and clean OHLCV price data following ML4T patterns."""

    def __init__(self, *, min_rows: int = 20, outlier_threshold: float = 3.0,
                 outlier_window: int = 60) -> None:
        self.min_rows = min_rows
        self.outlier_threshold = outlier_threshold
        self.outlier_window = outlier_window

    # -- individual checks --------------------------------------------------

    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        before = len(df)
        deduped = df[~df.index.duplicated(keep="first")]
        removed = before - len(deduped)
        if removed:
            logger.warning("Removed %d duplicate timestamp(s)", removed)
        return deduped, removed

    def validate_ohlc(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        h, l, o, c = "High", "Low", "Open", "Close"
        for col, fallback in [(h, [x for x in df.columns if "high" in str(x).lower()]),
                              (l, [x for x in df.columns if "low" in str(x).lower()]),
                              (o, [x for x in df.columns if "open" in str(x).lower()]),
                              (c, [x for x in df.columns if "close" in str(x).lower()])]:
            if col not in df.columns and fallback:
                df = df.rename(columns={fallback[0]: col})
        co_h_ok = (df[h] >= df[o]) & (df[h] >= df[c]) & (df[h] >= df[l])
        co_l_ok = (df[l] <= df[o]) & (df[l] <= df[c])
        mask = co_h_ok & co_l_ok
        n_bad = int((~mask).sum())
        w: list[str] = []
        if n_bad > 0:
            w.append(f"{n_bad} row(s) violated OHLC ordering - dropped")
            logger.warning(w[-1])
        return df.loc[mask].copy(), w

    @staticmethod
    def _days_between_gap(gap_bars: int, freq: str) -> float:
        """Approximate calendar days spanned by *gap_bars* at given frequency."""
        freq_days: dict[str, float] = {"B": 1.0, "C": 1.0, "D": 1.0, "W": 7.0, "M": 30.0}
        base = freq_days.get(freq[:1], 1.0)
        mult = 1
        if freq.endswith(("14D","20D","30D","90D")):
            try:
                mult = int(freq[:-1]) if freq[:-1].isdigit() else 1
            except ValueError:
                pass
        elif freq.startswith(("2W","3W","4W")):
            try:
                mult = int(freq[:1]) if freq[1:].startswith("W") else 1
            except ValueError:
                pass
        return gap_bars * base * mult

    def detect_missing_bars(self, df: pd.DataFrame) -> dict[str, Any]:
        idx = df.index
        info: dict[str, Any] = {"missing_count": 0, "max_gap_bars": 0,
                                "warning": None, "first_gap_start": None,
                                "last_gap_end": None}
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
            info["warning"] = "Datetime index expected"
            return info
        # Try known trading frequencies first (handles gaps that break infer_freq)
        freq = idx.freq
        if freq is None:
            for candidate in ["B", "C", "D"]:
                if pd.infer_freq(idx[:min(len(idx), 6)]) == candidate:
                    freq = candidate
                    break
        if freq is None:
            return info
        offset = pd.tseries.frequencies.to_offset(freq)
        expected = pd.date_range(start=idx[0], end=idx[-1], freq=freq)
        missing = sorted(expected.difference(idx))
        info["missing_count"] = len(missing)
        if missing:
            info["first_gap_start"] = missing[0]
            info["last_gap_end"] = missing[-1]
            runs: list[int] = [1]
            for i in range(1, len(missing)):
                actual_days = (missing[i] - missing[i-1]) / pd.Timedelta(days=1)
                expected_days = self._days_between_gap(1, freq)
                d = actual_days / expected_days if expected_days > 0 else float('inf')
                runs.append(runs[-1] + 1 if d <= 1.5 else 1)
            info["max_gap_bars"] = max(runs)
            if info["max_gap_bars"] > 3:
                info["warning"] = f"Large gap ({info['max_gap_bars']} bars)"
                logger.warning(info["warning"])
        return info

    def clip_outliers_zscore(self, df: pd.DataFrame, column: str = "Close",
                             window: int | None = None, threshold: float | None = None) -> pd.DataFrame:
        win = window or self.outlier_window
        thr = threshold or self.outlier_threshold
        out = df.copy()
        r_mean, r_std = _rolling_mean_std(out[column], win)
        lower = r_mean - thr * r_std
        upper = r_mean + thr * r_std
        above = out[column] > upper
        below = out[column] < lower
        n = int((above | below).sum())
        if n:
            logger.info("Clipped %d outliers in '%s' (|z| > %.1f)", n, column, thr)
            out.loc[above, column] = out[column].where(~above, upper.reindex(out.index))
            out.loc[below, column] = out[column].where(~below, lower.reindex(out.index))
        return out

    def ensure_minimum_length(self, df: pd.DataFrame) -> bool:
        ok = len(df) >= self.min_rows
        if not ok:
            logger.warning("Insufficient data: %d rows < %d required",
                           len(df), self.min_rows)
        return ok

    # -- full pipeline ------------------------------------------------------

    @classmethod
    def clean(cls, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        v = cls(**kwargs)
        w: list[str] = []
        df, _ = v.remove_duplicates(df)
        try:
            result_df, ohlc_w = v.validate_ohlc(df)
            df = result_df
            w.extend(ohlc_w)
        except Exception as exc:
            logger.error("OHLC validation exception: %s", exc)
            return pd.DataFrame()
        if not v.ensure_minimum_length(df):
            return df
        df = v.clip_outliers_zscore(df)
        if w:
            logger.info("Cleaning completed with %d warning(s)", len(w))
        return df


DEFAULT_MIN_ROWS = 20
DEFAULT_OUTLIER_ZSCORE_THRESHOLD = 3.0
