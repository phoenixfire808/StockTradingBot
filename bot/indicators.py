"""Technical indicators. Tries pandas-ta first; falls back to pure-pandas formulas."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)

# ── Indicator backend ────────────────────────────────────────────────
try:
    import pandas_ta  # noqa: F401
    _USE_PANDAS_TA = True
except ImportError:
    _USE_PANDAS_TA = False
    logger.warning(
        "pandas-ta not available — falling back to manual indicator calculations."
    )


def _normalize_col(df: pd.DataFrame, prefer_upper: str, prefer_lower: str, pos: int) -> pd.Series:
    """Get a column by preferred upper-case name, lower-case name, or position."""
    if prefer_upper in df.columns:
        return df[prefer_upper]
    if prefer_lower in df.columns:
        return df[prefer_lower]
    return df.iloc[:, pos]


def ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    if _USE_PANDAS_TA:
        result = pandas_ta.rsi(series, length=period)
        if result is not None:
            return result
        raise ValueError("pandas_ta.rsi returned None")

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. Accepts both High/Low/Close and high/low/close columns."""
    h = _normalize_col(df, "High", "high", 1)
    l = _normalize_col(df, "Low", "low", 2)
    c = _normalize_col(df, "Close", "close", 4)
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, pd.Series]:
    """Bollinger Bands — returns dict with 'mid', 'upper', 'lower' keys."""
    if _USE_PANDAS_TA:
        bbands = pandas_ta.bbands(series, length=period, std=num_std)
        if bbands is not None:
            # pandas_ta names columns like BBM_20, BBU_20, BBL_20 (or BBM_20_2.0 with extra suffix)
            col_map = {
                "middle": [c for c in bbands.columns if "BBM" in c],
                "upper": [c for c in bbands.columns if "BBU" in c],
                "lower": [c for c in bbands.columns if "BBL" in c],
            }
            for k, cols in col_map.items():
                if not cols:
                    break
                bbands[k] = bbands[cols[0]]  # pick first matching column, rename to canonical
            if "middle" in bbands.columns and "upper" in bbands.columns and "lower" in bbands.columns:
                return {"mid": bbands["middle"], "upper": bbands["upper"], "lower": bbands["lower"]}
            else:
                logger.warning("Could not extract BB columns from pandas_ta output")

    upper = series.rolling(period).mean() + num_std * series.rolling(period).std()
    lower = series.rolling(period).mean() - num_std * series.rolling(period).std()
    mid = series.rolling(period).mean()
    return {"upper": upper, "mid": mid, "lower": lower}
