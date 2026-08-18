"""Tests for bot.plugins.strategies.sentiment_filtered — VADER sentiment filtering."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bot.plugins.strategies.sentiment_filtered import SentimentFilteredStrategy
from bot.strategy import Strategy


def _make_trend_df(direction="up", length=200):
    """Create synthetic OHLCV DataFrame with a clear trend."""
    base = 100.0
    if direction == "up":
        trend = [base - i * 0.5 for i in range(25)] + \
                [base - 12.0 + (i - 25) * 0.6 for i in range(25, length)]
    elif direction == "down":
        trend = [base - i * 0.5 for i in range(length)]
    else:
        trend = [base + (i % 10) * 0.1 for i in range(length)]
    import random
    random.seed(42)
    closes = [t + random.uniform(-2, 2) for t in trend]
    opens = [c + random.uniform(-1, 1) for c in closes]
    highs = [max(o, c) + random.uniform(0, 2) for o, c in zip(opens, closes)]
    lows = [min(o, c) - random.uniform(0, 2) for o, c in zip(opens, closes)]
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1000] * length})
    return df


class TestSentimentFilteredStrategyBasics:
    def test_plugin_handle_exists(self):
        from bot.plugins.strategies.sentiment_filtered import plugin
        assert isinstance(plugin, SentimentFilteredStrategy)
        assert plugin.name == "sentiment_filtered"

    def test_is_strategy_subclass(self):
        assert issubclass(SentimentFilteredStrategy, Strategy)

    def test_default_params(self):
        strat = SentimentFilteredStrategy()
        assert strat.params["base_strategy_name"] == "ema_cross_rsi"
        assert strat.params["sentiment_threshold"] == 0.15
        assert strat.params["block_bearish"] is True
        assert strat.params["boost_bullish"] is True

    def test_custom_params(self):
        strat = SentimentFilteredStrategy(
            base_strategy_name="momentum_scanner",
            sentiment_threshold=0.3,
            block_bearish=False,
            boost_bullish=False,
        )
        assert strat.params["base_strategy_name"] == "momentum_scanner"
        assert strat.params["sentiment_threshold"] == 0.3
        assert strat.params["block_bearish"] is False


class TestSignalGenerationWithoutSymbol:
    """When df has no symbol attrs, signals pass through unchanged."""

    def test_no_symbol_passes_through(self):
        strat = SentimentFilteredStrategy()
        df = _make_trend_df("up", 200)
        # No df.attrs["symbol"] set
        signals = strat.generate_signals(df)
        # Should still produce valid signals (from base strategy)
        unique_vals = set(signals.tolist())
        assert unique_vals <= {0, 1, -1}, f"Unexpected signal values: {unique_vals}"


class TestSentimentBlocking:
    """When sentiment is bearish, long entries should be blocked."""

    def test_bearish_sentiment_blocks_longs(self):
        strat = SentimentFilteredStrategy(
            sentiment_threshold=0.15,
            block_bearish=True,
            boost_bullish=False,
        )
        df = _make_trend_df("up", 200)
        df.attrs["symbol"] = "AAPL"

        # Mock sentiment to be strongly bearish
        with patch.object(strat, "_fetch_sentiment", return_value=-0.5):
            signals = strat.generate_signals(df)

        # When bearish, all 1s should be blocked → 0
        assert (signals == 1).sum() == 0, "Bearish sentiment should block all long entries"
        # Exits (-1) should still be present if the base strategy generated any
        # (or 0 if base strategy had none on this data)

    def test_bullish_sentiment_preserves_longs(self):
        strat = SentimentFilteredStrategy(
            sentiment_threshold=0.15,
            block_bearish=False,
            boost_bullish=True,
        )
        df = _make_trend_df("up", 200)
        df.attrs["symbol"] = "AAPL"

        # Get base signals first (no symbol → pass-through)
        base_signals = strat.generate_signals(df.copy())
        base_longs = int((base_signals == 1).sum())

        with patch.object(strat, "_fetch_sentiment", return_value=0.5):
            signals = strat.generate_signals(df)

        # Bullish sentiment should preserve longs
        assert (signals == 1).sum() == base_longs, "Bullish sentiment should preserve long signals"

    def test_neutral_sentiment_passes_through(self):
        strat = SentimentFilteredStrategy(
            sentiment_threshold=0.15,
            block_bearish=True,
            boost_bullish=True,
        )
        df = _make_trend_df("up", 200)
        df.attrs["symbol"] = "AAPL"

        with patch.object(strat, "_fetch_sentiment", return_value=0.0):
            signals = strat.generate_signals(df)

        # Neutral → pass through unchanged
        base_signals = strat._get_base_strategy().generate_signals(df)
        pd.testing.assert_series_equal(
            signals.astype("int8"), base_signals.astype("int8"),
            check_names=False,
        )

    def test_exits_always_preserved(self):
        strat = SentimentFilteredStrategy(
            sentiment_threshold=0.15,
            block_bearish=True,
        )
        df = _make_trend_df("down", 200)
        df.attrs["symbol"] = "AAPL"

        with patch.object(strat, "_fetch_sentiment", return_value=-0.5):
            signals = strat.generate_signals(df)

        # Even with bearish sentiment, if base produces exits (-1), they should pass
        # Exits should not be blocked
        base_signals = strat._get_base_strategy().generate_signals(df)
        base_exits = int((base_signals == -1).sum())
        result_exits = int((signals == -1).sum())
        assert result_exits == base_exits, "Exit signals must be preserved regardless of sentiment"


class TestSentimentFetchFallback:
    def test_sentiment_fetch_failure_returns_neutral(self):
        """When sentiment fetch fails, should default to 0.0 (neutral)."""
        strat = SentimentFilteredStrategy()
        # Simulate failed fetch
        with patch("bot.sentiment.SentimentEngine") as mock_engine:
            mock_engine.return_value.score.side_effect = Exception("network error")
            score = strat._fetch_sentiment("FAKE")
        assert score == 0.0
