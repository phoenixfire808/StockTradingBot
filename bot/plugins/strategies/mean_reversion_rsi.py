"""Mean Reversion Strategy — RSI with EMA smoothing and volume filter.

Buys when smoothed RSI crosses above an oversold threshold (entry).
Sells when smoothed RSI crosses below an overbought threshold (exit).

Designed for sideways/choppy markets where prices tend to revert to their mean.
The optional volume filter requires signals to occur on above-average volume,
reducing false entries during low-liquidity periods.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MeanReversionRsi(Strategy):
    """Mean-reversion strategy using smoothed RSI crossings with volume filter.

    Entry signal: smoothed RSI crosses *above* the oversold level (default 30)
                   from below, confirming a reversal to the upside.
    Exit signal:  smoothed RSI crosses *below* the overbought level (default 70)
                   from above, signalling momentum has faded.

    When volume_filter is enabled both entry and exit require the current bar's
    volume to exceed its N-period simple moving average, filtering out signals
    that occur during thin trading sessions.
    """

    name = "mean_reversion_rsi"

    params: dict[str, Any] = {}

    def __init__(
        self,
        rsi_period: int = 14,
        entry_threshold: float = 30.0,
        exit_threshold: float = 70.0,
        signal_period: int = 9,
        volume_filter: bool = True,
        volume_ma_period: int = 20,
    ) -> None:
        self.rsi_period = rsi_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.signal_period = signal_period
        self.volume_filter = volume_filter
        self.volume_ma_period = volume_ma_period
        self.params = {
            "rsi_period": rsi_period,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "signal_period": signal_period,
            "volume_filter": volume_filter,
            "volume_ma_period": volume_ma_period,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return an int8 Series aligned to *df*: 1=long, -1=exit, 0=flat."""
        close = df["Close"] if "Close" in df else df.iloc[:, 4]

        from bot.indicators import rsi as _rsi

        # 1) Raw RSI
        rsi_raw = _rsi(close, self.rsi_period)

        # 2) EMA-smoothed RSI signal line
        rsi_signal = rsi_raw.ewm(span=self.signal_period, adjust=False).mean()

        # 3) Volume filter (optional)
        if self.volume_filter and "Volume" in df.columns:
            vol_ma = df["Volume"].rolling(window=self.volume_ma_period).mean()
            high_volume = df["Volume"] > vol_ma
        else:
            high_volume = pd.Series(True, index=df.index)

        # 4) Cross detection
        #    Entry: previous RSI < entry_threshold, current RSI >= entry_threshold
        cross_above_entry = (
            (rsi_signal.shift(1) < self.entry_threshold)
            & (rsi_signal >= self.entry_threshold)
        )
        #    Exit: previous RSI > exit_threshold, current RSI <= exit_threshold
        cross_below_exit = (
            (rsi_signal.shift(1) > self.exit_threshold)
            & (rsi_signal <= self.exit_threshold)
        )

        # 5) Apply filters, produce int8 signal
        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[cross_above_entry & high_volume] = 1
        signal[cross_below_exit & high_volume] = -1

        return signal


# Plugin handle for auto-discovery
plugin = MeanReversionRsi()
