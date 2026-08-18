"""Bollinger Reversion Strategy — Mean reversion using Bollinger Bands (20, 2) + RSO filter.

Buys when price touches/nears lower BB band with RSO deeply oversold (< 20).
Exits when price rebounds to mid-band or RSO rises above 70.
Designed for sideways/choppy markets where mean reversion works best.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class BollingerReversion(Strategy):
    """Mean reversion using Bollinger Bands + RSO-2 filter."""

    name = "bollinger_reversion"
    params: dict[str, Any] = {}

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rso_period: int = 14,
        rso_oversold: float = 20.0,
        rso_overbought: float = 80.0,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rso_period = rso_period
        self.rso_oversold = rso_oversold
        self.rso_overbought = rso_overbought
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.params = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "rso_period": rso_period,
            "rso_oversold": rso_oversold,
            "rso_overbought": rso_overbought,
            "atr_stop_mult": atr_stop_mult,
            "atr_tp_mult": atr_tp_mult,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"] if "Close" in df else df.iloc[:, 4]

        # Bollinger Bands
        from bot.indicators import bollinger as _bbands
        bands = _bbands(close, period=self.bb_period, num_std=self.bb_std)

        # RSO
        from bot.indicators import rsi as _rsi
        rso_vals = _rsi(close, self.rso_period)

        # Entry: price near/at lower BB AND RSO deeply oversold
        touch_lower = close <= bands["lower"] * 1.01
        near_lower = close.between(bands["lower"], bands["mid"], inclusive="both")
        rso_deep_oversold = rso_vals < self.rso_oversold
        entry_signal = (touch_lower | near_lower) & rso_deep_oversold

        # Exit: RSO rebounds above threshold OR price touches mid band
        rso_rebound = rso_vals > self.rso_overbought
        touch_mid = close >= bands["mid"] * 0.99
        exit_signal = rso_rebound | touch_mid

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[entry_signal.astype(bool)] = 1
        signal[exit_signal.astype(bool)] = -1

        return signal


# Plugin handle for auto-discovery
plugin = BollingerReversion()
