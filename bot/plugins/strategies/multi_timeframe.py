"""Multi-Timeframe Strategy Plugin — composes a base strategy across multiple
timeframes (1m/5m/15m/1h/1d) and aggregates signals.

Delegates to bot/multi_timeframe.py MultiTimeframeComposer for the heavy
lifting (data fetching, resampling, aggregation).
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MultiTimeframeStrategy(Strategy):
    """Composes a base strategy across multiple timeframes.

    Args:
        base_strategy_name: Registered strategy to run on each timeframe.
        timeframes:         List of intervals to fetch (e.g. ["5m","1h","1d"]).
        aggregation:        How to combine: majority_vote, weighted, unanimous, any_long.
    """

    name = "multi_timeframe"
    params: dict[str, Any] = {}

    def __init__(
        self,
        base_strategy_name: str = "ema_cross_rsi",
        timeframes: list[str] | None = None,
        aggregation: str = "majority_vote",
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
    ) -> None:
        self.base_strategy_name = base_strategy_name
        self.timeframes = timeframes or ["5m", "1h", "1d"]
        self.aggregation = aggregation
        self.params = {
            "base_strategy_name": base_strategy_name,
            "timeframes": self.timeframes,
            "aggregation": aggregation,
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
        }
        self._base_strategy = None
        self._composer = None
        logger.info(
            "MultiTimeframeStrategy init: base=%s timeframes=%s aggregation=%s",
            base_strategy_name,
            self.timeframes,
            aggregation,
        )

    def _get_base_strategy(self) -> Strategy:
        """Lazily fetch the wrapped base strategy."""
        if self._base_strategy is not None:
            return self._base_strategy
        try:
            from bot.core import STRATEGIES
            from bot.core.plugins import discover_all

            discover_all()
            base = STRATEGIES.get(self.base_strategy_name)
            base_params = {
                "fast": self.params.get("fast", 9),
                "slow": self.params.get("slow", 21),
                "rsi_period": self.params.get("rsi_period", 14),
                "rsi_entry_max": self.params.get("rsi_entry_max", 70.0),
                "rsi_exit": self.params.get("rsi_exit", 75.0),
            }
            self._base_strategy = type(base)(**base_params)
        except Exception as exc:
            logger.warning(
                "Failed to load base '%s': %s — using EmaCrossRsi",
                self.base_strategy_name,
                exc,
            )
            from bot.plugins.strategies.ema_cross_rsi import EmaCrossRsi

            self._base_strategy = EmaCrossRsi(
                fast=self.params.get("fast", 9),
                slow=self.params.get("slow", 21),
                rsi_period=self.params.get("rsi_period", 14),
                rsi_entry_max=self.params.get("rsi_entry_max", 70.0),
                rsi_exit=self.params.get("rsi_exit", 75.0),
            )
        return self._base_strategy

    def _get_composer(self):
        """Lazily create the MultiTimeframeComposer."""
        if self._composer is not None:
            return self._composer
        from bot.multi_timeframe import MultiTimeframeComposer

        base = self._get_base_strategy()
        self._composer = MultiTimeframeComposer(
            base_strategy=base,
            timeframes=self.timeframes,
            aggregation=self.aggregation,
        )
        return self._composer

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using multi-timeframe composition.

        If df.attrs contains a "symbol" key, fetches additional intraday
        timeframes and aggregates. Otherwise, runs the base strategy on
        the provided daily DataFrame only.
        """
        symbol = getattr(df, "attrs", {}).get("symbol", None)
        composer = self._get_composer()

        try:
            if symbol is not None:
                logger.info("Multi-timeframe composition with symbol=%s", symbol)
                return composer.generate_from_daily(df, symbol=symbol)
            else:
                logger.debug("No symbol context — using base strategy on daily df only")
                return composer.generate_from_daily(df, symbol=None)
        except Exception as exc:
            logger.error(
                "Multi-timeframe composition failed: %s — falling back to base strategy",
                exc,
            )
            base = self._get_base_strategy()
            return base.generate_signals(df)


# Plugin handle for auto-discovery
plugin = MultiTimeframeStrategy()
