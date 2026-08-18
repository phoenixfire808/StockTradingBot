"""Sector Rotation Strategy Plugin — uses sector ETF momentum to bias signals.

Delegates to bot/sector_rotation.py SectorRotationModel for momentum
computation and signal biasing.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class SectorRotationStrategy(Strategy):
    """Wraps a base strategy and biases signals using sector momentum.

    Args:
        base_strategy_name:  Registered strategy to wrap.
        lookback_days:        Momentum lookback period for sector ETFs.
        momentum_threshold:   Minimum |RoC| to classify sector as bullish/bearish.
        block_threshold:      Bias level below which longs are blocked.
        boost_threshold:      Bias level above which longs are maintained.
    """

    name = "sector_rotation"
    params: dict[str, Any] = {}

    def __init__(
        self,
        base_strategy_name: str = "ema_cross_rsi",
        lookback_days: int = 20,
        momentum_threshold: float = 0.02,
        block_threshold: float = -0.3,
        boost_threshold: float = 0.3,
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
    ) -> None:
        self.base_strategy_name = base_strategy_name
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.block_threshold = block_threshold
        self.boost_threshold = boost_threshold
        self.params = {
            "base_strategy_name": base_strategy_name,
            "lookback_days": lookback_days,
            "momentum_threshold": momentum_threshold,
            "block_threshold": block_threshold,
            "boost_threshold": boost_threshold,
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
        }
        self._base_strategy = None
        self._rotation_model = None
        logger.info(
            "SectorRotationStrategy init: base=%s lookback=%d threshold=%.2f%% "
            "block=%.2f boost=%.2f",
            base_strategy_name,
            lookback_days,
            momentum_threshold * 100,
            block_threshold,
            boost_threshold,
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

    def _get_rotation_model(self):
        """Lazily create the SectorRotationModel."""
        if self._rotation_model is not None:
            return self._rotation_model
        from bot.sector_rotation import SectorRotationModel

        self._rotation_model = SectorRotationModel(
            lookback_days=self.lookback_days,
            momentum_threshold=self.momentum_threshold,
        )
        return self._rotation_model

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate base signals then apply sector rotation bias.

        - Computes sector momentum for the symbol's sector ETF.
        - Blocks longs when sector is in downtrend (bias < block_threshold).
        - Maintains longs when sector is in uptrend (bias > boost_threshold).
        - Exits always pass through.
        """
        base = self._get_base_strategy()
        signals = base.generate_signals(df)

        symbol = getattr(df, "attrs", {}).get("symbol", None)
        if symbol is None:
            logger.debug("No symbol context — passing base signals through")
            return signals

        model = self._get_rotation_model()

        try:
            bias = model.get_sector_bias(symbol)
        except Exception as exc:
            logger.warning(
                "Sector bias computation failed for %s: %s — passing through",
                symbol,
                exc,
            )
            return signals

        from bot.sector_rotation import bias_signals

        result = bias_signals(
            signals,
            bias,
            block_threshold=self.block_threshold,
            boost_threshold=self.boost_threshold,
        )

        logger.info(
            "SectorRotation %s: bias=%.2f → %d long, %d exit, %d flat "
            "(base: %d long, %d exit)",
            symbol,
            bias,
            int((result == 1).sum()),
            int((result == -1).sum()),
            int((result == 0).sum()),
            int((signals == 1).sum()),
            int((signals == -1).sum()),
        )
        return result


# Plugin handle for auto-discovery
plugin = SectorRotationStrategy()
