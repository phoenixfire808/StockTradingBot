"""Sentiment-Filtered Strategy Plugin — wraps a base strategy and uses VADER
sentiment to modify/block signals.

When sentiment is strongly bearish (net_score below -threshold), new long
entries are blocked. When sentiment is strongly bullish, existing long
signals are preserved/amplified. Exit signals always pass through.

Designed to layer sentiment awareness on top of any technical strategy.
"""

import logging
from typing import Any

import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class SentimentFilteredStrategy(Strategy):
    """Wraps a base strategy and filters signals using VADER sentiment.

    Args:
        base_strategy_name: Name of registered strategy to wrap (default "ema_cross_rsi").
        sentiment_threshold: Minimum |net_score| to act on sentiment (default 0.15).
        block_bearish: If True, block longs when net_score < -threshold.
        boost_bullish: If True, maintain longs when net_score > threshold.
    """

    name = "sentiment_filtered"
    params: dict[str, Any] = {}

    def __init__(
        self,
        base_strategy_name: str = "ema_cross_rsi",
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
        sentiment_threshold: float = 0.15,
        block_bearish: bool = True,
        boost_bullish: bool = True,
    ) -> None:
        self.base_strategy_name = base_strategy_name
        self.sentiment_threshold = sentiment_threshold
        self.block_bearish = block_bearish
        self.boost_bullish = boost_bullish
        self.params = {
            "base_strategy_name": base_strategy_name,
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
            "sentiment_threshold": sentiment_threshold,
            "block_bearish": block_bearish,
            "boost_bullish": boost_bullish,
        }
        self._base_strategy = None
        self._current_symbol: str | None = None
        self._cached_sentiment: float | None = None
        logger.info(
            "SentimentFilteredStrategy init: base=%s threshold=%.2f block_bearish=%s boost_bullish=%s",
            base_strategy_name,
            sentiment_threshold,
            block_bearish,
            boost_bullish,
        )

    def _get_base_strategy(self) -> Strategy:
        """Lazily fetch the wrapped base strategy from the registry."""
        if self._base_strategy is not None:
            return self._base_strategy
        try:
            from bot.core import STRATEGIES
            from bot.core.plugins import discover_all

            discover_all()
            base = STRATEGIES.get(self.base_strategy_name)
            # Create fresh instance with matching params
            base_params = {
                "fast": self.params.get("fast", 9),
                "slow": self.params.get("slow", 21),
                "rsi_period": self.params.get("rsi_period", 14),
                "rsi_entry_max": self.params.get("rsi_entry_max", 70.0),
                "rsi_exit": self.params.get("rsi_exit", 75.0),
            }
            self._base_strategy = type(base)(**base_params)
            logger.debug("Loaded base strategy: %s", self.base_strategy_name)
        except Exception as exc:
            logger.warning(
                "Failed to load base strategy '%s': %s — using EmaCrossRsi directly",
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

    def _fetch_sentiment(self, symbol: str) -> float:
        """Fetch current sentiment net_score for symbol.

        Returns a float in [-1, 1]. Returns 0.0 if sentiment engine
        unavailable or no data found (neutral → pass-through).
        """
        if self._cached_sentiment is not None and self._current_symbol == symbol:
            return self._cached_sentiment

        try:
            from bot.sentiment import SentimentEngine

            engine = SentimentEngine()
            score = engine.score(symbol, hours=24)
            net = score.net_score
            self._cached_sentiment = net
            self._current_symbol = symbol
            logger.info(
                "Sentiment for %s: net=%.3f bullish=%d bearish=%d neutral=%d mentions=%d",
                symbol,
                net,
                score.bullish,
                score.bearish,
                score.neutral,
                score.mentions,
            )
            return net
        except Exception as exc:
            logger.warning(
                "Sentiment fetch failed for %s: %s — treating as neutral (0.0)",
                symbol,
                exc,
            )
            self._cached_sentiment = 0.0
            self._current_symbol = symbol
            return 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate base signals then apply sentiment filter.

        - Block new longs (1→0) when net_score < -sentiment_threshold and block_bearish.
        - Preserve exits (-1) always — sentiment doesn't override risk exits.
        - When bullish (net > threshold) and boost_bullish: keep longs as-is.
        """
        base = self._get_base_strategy()
        signals = base.generate_signals(df)

        # Attempt to infer symbol from context; default to neutral if unknown
        symbol = getattr(df, "attrs", {}).get("symbol", None)
        if symbol is None:
            # No symbol context → pass through unchanged (neutral)
            logger.debug("No symbol context in df.attrs — passing signals through unchanged")
            return signals

        sentiment = self._fetch_sentiment(symbol)

        result = signals.copy()

        if self.block_bearish and sentiment < -self.sentiment_threshold:
            # Block new long entries
            blocked_mask = result == 1
            blocked_count = int(blocked_mask.sum())
            result[blocked_mask] = 0
            logger.info(
                "Sentiment FILTER: %s net=%.3f < -%.2f → blocked %d long entries",
                symbol,
                sentiment,
                self.sentiment_threshold,
                blocked_count,
            )
        elif self.boost_bullish and sentiment > self.sentiment_threshold:
            # Bullish sentiment: maintain existing longs, no blocking
            logger.info(
                "Sentiment BOOST: %s net=%.3f > %.2f → maintaining %d long signals",
                symbol,
                sentiment,
                self.sentiment_threshold,
                int((result == 1).sum()),
            )

        # Exit signals always preserved
        return result.astype("int8")


# Plugin handle for auto-discovery
plugin = SentimentFilteredStrategy()
