"""Mean Reversion Strategy — RSI-2 with Bollinger Bands filter.

Buys when price dips sharply (RSO-2 < 10) near lower BB band.
Exits when price rebounds (RSI > 50) or hits take-profit.
Designed for sideways/choppy markets where mean reversion works best.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MeanReversionRSI2(Strategy):
    """Mean reversion using ultra-short RSO-2 + Bollinger Band filter."""

    name = "mean_reversion_rsi2"
    params: dict[str, Any] = {}

    def __init__(
        self,
        rsi_period: int = 2,
        rsi_oversold: float = 10.0,
        rsi_overbought: float = 80.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_tp_mult = atr_tp_multiplier
        self.params = {
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "atr_stop_multiplier": atr_stop_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"] if "Close" in df else df.iloc[:, 4]

        from bot.indicators import rsi as _rsi, bollinger as _bbands

        # Ultra-short RSO for quick reversals
        rso_vals = _rsi(close, self.rsi_period)

        # Bollinger Band positions
        bands = _bbands(close, period=self.bb_period, num_std=self.bb_std)

        # Entry: RSO deeply oversold AND price touches/near lower band
        rso_deep_oversold = rso_vals < self.rsi_oversold
        near_lower_band = close <= bands["lower"] * 1.01

        # Exit: RSO rebounds significantly
        rso_rebound = rso_vals > 50.0

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[rso_deep_oversold & near_lower_band] = 1  # Buy dip
        signal[rso_rebound] = -1  # Exit on recovery

        return signal


# Plugin handle for auto-discovery
plugin = MeanReversionRSI2()
