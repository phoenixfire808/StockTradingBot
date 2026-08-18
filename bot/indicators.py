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


def ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Exponential Moving Average."""
    if _USE_PANDAS_TA:
        return series.ewm(span=period, adjust=False).mean()
    # Manual fallback via pandas_ta API compatibility would go here
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
    """Average True Range."""
    if _USE_PANDAS_TA:
        high = df.get("High", df["high"]) if "High" in df else df.iloc[:, 1]
        low = df.get("Low", df["low"]) if "Low" in df else df.iloc[:, 2]
        close = df.get("Close", df["close"]) if "Close" in df else df.iloc[:, 4]
        tr = pandas_ta.true_range(high, low, close)
        return tr.rolling(window=period).mean()
    high = df["High"] if "High" in df else df.iloc[:, 1]
    low = df["Low"] if "Low" in df else df.iloc[:, 2]
    prev_close = df["Close"].shift(1) if "Close" in df else df.iloc[:, 4].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, pd.Series]:
    """Bollinger Bands — returns dict with 'mid', 'upper', 'lower' keys."""
    if _USE_PANDAS_TA:
        bbands = pandas_ta.bbands(series, length=period, std=num_std)
        if bbands is not None:
            return {
                "mid": bbands[f"BBL_{period}_{num_std}"],
                "upper": bbands[f"BBU_{period}_{num_std}"],
                "lower": bbands[f"BBM_{period}_{num_std}"],
            }
    upper = series.rolling(period).mean() + num_std * series.rolling(period).std()
    lower = series.rolling(period).mean() - num_std * series.rolling(period).std()
    mid = series.rolling(period).mean()
    return {"upper": upper, "mid": mid, "lower": lower}
