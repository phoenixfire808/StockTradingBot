"""Momentum Scanner Strategy — MACD + Volume surge + Trend filter.

Buys when MACD crosses above signal AND volume surges above average AND
price is above key moving averages (confirmed trend).
Exits when MACD crosses below signal or volume dries up.
Designed to capture momentum breakouts with confirmation.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MomentumScanner(Strategy):
    """Momentum breakout strategy: MACD crossover + volume confirmation + trend filter."""

    name = "momentum_scanner"
    params: dict[str, Any] = {}

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vol_ma_period: int = 20,
        vol_surge_mult: float = 1.5,
        trend_ma_period: int = 50,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.vol_ma_period = vol_ma_period
        self.vol_surge_mult = vol_surge_mult
        self.trend_ma_period = trend_ma_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.params = {
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "vol_ma_period": vol_ma_period,
            "vol_surge_mult": vol_surge_mult,
            "trend_ma_period": trend_ma_period,
            "atr_stop_multiplier": atr_stop_mult,
            "atr_tp_multiplier": atr_tp_mult,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"] if "Close" in df else df.iloc[:, 4]
        volume = df["Volume"] if "Volume" in df else df.iloc[:, 4]

        # MACD
        ema_fast = close.ewm(span=self.macd_fast).mean()
        ema_slow = close.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line

        # MACD cross: MACD line crosses above signal line
        macd_cross_up = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )

        # Volume surge: volume > X% of moving average
        vol_avg = volume.rolling(window=self.vol_ma_period).mean()
        vol_surge = volume > vol_avg * self.vol_surge_mult

        # Trend filter: price above long-term MA
        trend_ma = close.ewm(span=self.trend_ma_period, adjust=False).mean()
        above_trend = close > trend_ma

        # Exit: MACD crosses below signal
        macd_cross_down = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )

        # Only enter when ALL three confirm: MACD cross + volume surge + trend aligned
        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[macd_cross_up & vol_surge & above_trend] = 1
        signal[macd_cross_down] = -1

        return signal


# Plugin handle for auto-discovery
plugin = MomentumScanner()
