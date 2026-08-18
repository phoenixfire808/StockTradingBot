"""VWAP Breakout Strategy — Volume-weighted Average Price breakout detection.

Buys when price breaks above VWAP with volume confirmation.
Exits on volume-diminished pullback or stop-loss hit.
Designed for intraday momentum breakouts.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class VWAPBreakout(Strategy):
    """VWAP breakout with volume confirmation filter."""

    name = "vwap_breakout"
    params: dict[str, Any] = {}

    def __init__(
        self,
        vwap_period: int = 20,
        vol_surge_mult: float = 1.5,
        stop_multiplier: float = 2.0,
        tp_multiplier: float = 3.0,
        min_volume_ratio: float = 1.2,
    ) -> None:
        self.vwap_period = vwap_period
        self.vol_surge_mult = vol_surge_mult
        self.stop_mult = stop_multiplier
        self.tp_mult = tp_multiplier
        self.min_volume_ratio = min_volume_ratio
        self.params = {
            "vwap_period": vwap_period,
            "vol_surge_mult": vol_surge_mult,
            "stop_multiplier": stop_multiplier,
            "tp_multiplier": tp_multiplier,
            "min_volume_ratio": self.min_volume_ratio,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"] if "Close" in df else df.iloc[:, 4]
        high = df["High"] if "High" in df else df.iloc[:, 2]
        low = df["Low"] if "Low" in df else df.iloc[:, 3]
        volume = df["Volume"] if "Volume" in df else df.iloc[:, 4]

        # VWAP calculation (typical price weighted by volume)
        typical_price = (high + low + close) / 3.0
        cumulative_tp_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        vwap = cumulative_tp_volume / cumulative_volume.replace(0, 1)

        # Volume moving average
        vol_ma = volume.rolling(window=self.vwap_period).mean()

        # Volume surge: current volume > X% of average
        vol_surge = volume > vol_ma * self.vol_surge_mult

        # VWAP breakout: price crosses above VWAP AND volume surges
        vwap_break_above = (close > vwap) & (close.shift(1) <= vwap.shift(1))

        # Entry: breakout + volume confirmation
        entry_signal = vwap_break_above & vol_surge

        # Exit: price closes back below VWAP with declining volume
        vwap_pullback = (close < vwap) & (volume < vol_ma * self.min_vol_ratio)
        exit_signal = vwap_pullback

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[entry_signal] = 1
        signal[exit_signal] = -1

        return signal


# Plugin handle for auto-discovery
plugin = VWAPBreakout()
