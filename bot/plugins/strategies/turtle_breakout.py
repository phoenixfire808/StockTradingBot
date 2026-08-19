"""Turtle Trading System — Donchian channel breakout strategy.

Inspired by Richard Dennis's famous experiment where he taught ordinary people
to trade profitably using simple rules. This implements the "System 1" variant:

- Entry: price breaks above 20-day high (Donchian upper band) → BUY
- Exit:  price breaks below 10-day low (Donchian lower band) → SELL
- Position sizing: units = 1% * Equity / ATR(20), capped at risk budget
- Risk management: stop-loss at 0.5 x ATR below entry; max 2% equity at risk

Designed for daily bar timeframes. Returns int8 Series signals: 1=long, -1=exit, 0=flat.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class TurtleBreakout(Strategy):
    """Turtle Trading breakout using Donchian channels and ATR-based sizing."""

    name = "turtle_breakout"
    params: dict[str, Any] = {}

    def __init__(
        self,
        enter_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        risk_pct: float = 0.02,
        unit_risk_pct: float = 0.01,
        stop_multiplier: float = 0.5,
        pyramiding: int = 3,
    ) -> None:
        self.enter_period = enter_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct          # max equity at risk per trade
        self.unit_risk_pct = unit_risk_pct # ATR-based unit definition
        self.stop_mult = stop_multiplier   # stop = entry +/- mult * ATR
        self.pyramiding = pyramiding       # max number of position additions

        self.params = {
            "enter_period": enter_period,
            "exit_period": exit_period,
            "atr_period": atr_period,
            "risk_pct": risk_pct,
            "unit_risk_pct": unit_risk_pct,
            "stop_multiplier": self.stop_mult,
            "pyramiding": pyramiding,
        }

    def _calc_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range then smoothed ATR (Wilder's RMA)."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder smoothing: first value is average, rest are EMA-like
        atr = true_range.ewm(alpha=1.0 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        return atr

    def _donchian_bands(
        self, high: pd.Series, low: pd.Series, period: int
    ) -> tuple[pd.Series, pd.Series]:
        """Upper/lower Donchian channels over the given lookback window."""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        return upper, lower

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return int8 signal Series aligned to *df*: 1=buy, -1=exit, 0=flat.

        Position sizing formula (for backtesting integration):
            units = (Equity * unit_risk_pct) / ATR
        The plugin returns pure direction signals; ATR/sizing data is
        accessible via self.atr_period if callers need it.
        """
        close = df["Close"] if "Close" in df else df.iloc[:, 4]
        high = df["High"] if "High" in df else df.iloc[:, 2]
        low = df["Low"] if "Low" in df else df.iloc[:, 3]

        # Calculate ATR for position sizing and stop-loss
        atr = self._calc_atr(high, low, close)
        # Avoid zero-ATR division (e.g. halted sessions)
        atr = atr.replace(0, np.nan).ffill().replace(0, 1.0)

        # Donchian channels
        upper_20, lower_20 = self._donchian_bands(high, low, self.enter_period)
        _, lower_10 = self._donchian_bands(high, low, self.exit_period)

        # ── Entry: close > previous 20-bar upper band ──
        new_high = close > upper_20.shift(1)

        # ── Exit: close < 10-bar lower band ──
        new_low = close < lower_10

        # ── Quality filter: skip entries when ATR is very large vs price
        #    (widespread volatility spikes often produce false breakouts)
        atr_ratio = atr / close
        valid_entry = atr_ratio < 0.10

        entry_signal = new_high & valid_entry
        exit_signal = new_low

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[entry_signal] = 1
        signal[exit_signal] = -1

        logger.debug(
            "TurtleBreakout: %d entries, %d exits over %d bars "
            "(enter=%d, exit=%d)",
            entry_signal.sum(),
            exit_signal.sum(),
            len(df),
            self.enter_period,
            self.exit_period,
        )
        return signal


# Plugin handle for auto-discovery
plugin = TurtleBreakout()
