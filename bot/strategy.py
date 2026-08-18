"""Strategy base class + starter EMA-cross-with-RSI plugin."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Base strategy — every plugin must subclass this and expose `plugin`."""

    name: str = "base"
    params: dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return an int8 Series aligned to *df*: 1=long, -1=exit, 0=flat."""
        ...

    def to_backtesting_strategy(self) -> type:
        """Convert this strategy into a backtesting.py compatible subclass."""
        signals_df = self.generate_signals(pd.DataFrame())

        class _BTStrategy:
            name = self.name

            def __init__(self, engine) -> None:
                # Pre-compute signals over the full bar dataframe
                full_df = pd.DataFrame(
                    {c: self.data[c] for c in ("Open", "High", "Low", "Close", "Volume")}
                    if hasattr(engine, "_symbol") else {}
                )
                full_df.index = range(len(self.Close))
                self._sig = self.generate_signals(full_df).astype(int)

            @property
            def Close(self):
                return self.data["Close"]

            @property
            def High(self):
                return self.data["High"]

            @property
            def Low(self):
                return self.data["Low"]

            @property
            def Open(self):
                return self.data["Open"]

            @property
            def Volume(self):
                return self.data["Volume"]

            @property
            def position(self):
                return self.position

            def next(self) -> None:
                idx = len(self) - 1
                sig = self._sig.iloc[idx] if idx < len(self._sig) else 0
                if sig == 1 and not self.position:
                    self.buy()
                elif sig == -1 and self.position:
                    self.close()

        return _BTStrategy


# ── Starter Plugin ───────────────────────────────────────────────────

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

        # Prevent immediate re-entry on same bar as exit
        signal[signal == 1] &= ~signal.shift(1).fillna(1).eq(-1)

        return signal


# Module-level plugin handle for auto-discovery
plugin = EmaCrossRsi()
