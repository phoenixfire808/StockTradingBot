"""EMA(9/21) crossover with RSI filter strategy plugin."""

import logging
import pandas as pd
from typing import Any
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class EmaCrossRsi(Strategy):
    """EMA fast/slow crossover filtered by RSI.

    Long: EMA-fast crosses above EMA-slow AND RSI < entry_max.
    Exit: EMA-fast crosses below EMA-slow OR RSI > exit_level.
    """

    name = "ema_cross_rsi"
    params: dict[str, Any] = {}

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_entry_max = rsi_entry_max
        self.rsi_exit = rsi_exit
        self.params = {
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"] if "Close" in df else df.iloc[:, 4]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        from bot.indicators import rsi as _rsi
        rsi_vals = _rsi(close, self.rsi_period)

        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        rsi_overbought = rsi_vals > self.rsi_exit
        rsi_oversold_ok = rsi_vals < self.rsi_entry_max

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[cross_up & rsi_oversold_ok] = 1
        signal[cross_down | rsi_overbought] = -1
        prev_was_exit = (signal.shift(1).fillna(1) == -1).astype(object)
        mask = ((signal == 1) & ~prev_was_exit).astype(object)
        signal[mask] = 1

        return signal


# Plugin handle for auto-discovery
plugin = EmaCrossRsi()
