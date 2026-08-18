"""Tests for bot.multi_timeframe + bot.plugins.strategies.multi_timeframe."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bot.multi_timeframe import (
    MultiTimeframeComposer,
    TIMEFRAMES,
    compose_multi_timeframe,
    aggregate_signals,
)
from bot.plugins.strategies.multi_timeframe import MultiTimeframeStrategy
from bot.strategy import Strategy


def _make_ohlcv(length=100, start_price=100.0, trend="up"):
    """Create synthetic OHLCV DataFrame."""
    if trend == "up":
        closes = [start_price + i * 0.5 for i in range(length)]
    elif trend == "down":
        closes = [start_price - i * 0.5 for i in range(length)]
    else:
        closes = [start_price + (i % 10) * 0.1 for i in range(length)]
    import random
    random.seed(42)
    opens = [c + random.uniform(-1, 1) for c in closes]
    highs = [max(o, c) + random.uniform(0, 2) for o, c in zip(opens, closes)]
    lows = [min(o, c) - random.uniform(0, 2) for o, c in zip(opens, closes)]
    dates = pd.date_range("2023-01-01", periods=length, freq="D")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1000] * length},
        index=dates,
    )


class TestMultiTimeframeComposer:
    def test_init_defaults(self):
        composer = MultiTimeframeComposer(base_strategy=MagicMock())
        assert composer.timeframes == ["5m", "1h", "1d"]
        assert composer.aggregation == "majority_vote"

    def test_init_custom(self):
        composer = MultiTimeframeComposer(
            base_strategy=MagicMock(),
            timeframes=["1h", "1d"],
            aggregation="weighted",
        )
        assert composer.timeframes == ["1h", "1d"]
        assert composer.aggregation == "weighted"

    def test_resample_to_daily_intraday(self):
        # Create intraday signals (hourly)
        idx = pd.date_range("2023-01-01", periods=48, freq="h")
        signals = pd.Series([1 if i < 24 else -1 for i in range(48)], index=idx, dtype="int8")
        daily = MultiTimeframeComposer._resample_to_daily(signals)
        # Should be daily frequency
        assert len(daily) == 2  # 2 days
        # Last value of day 1 (first 24h) should be 1
        assert daily.iloc[0] == 1
        # Last value of day 2 should be -1
        assert daily.iloc[1] == -1

    def test_resample_to_daily_already_daily(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="D")
        signals = pd.Series([1, -1, 0, 1, 0, -1, 1, 1, 0, 1], index=idx, dtype="int8")
        daily = MultiTimeframeComposer._resample_to_daily(signals)
        assert len(daily) == 10


class TestAggregationMethods:
    def _make_signals_dict(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="D")
        return {
            "5m": pd.Series([1, 1, -1, 0, 1], index=idx, dtype="int8"),
            "1h": pd.Series([1, -1, -1, 1, 1], index=idx, dtype="int8"),
            "1d": pd.Series([1, 1, 1, 1, 1], index=idx, dtype="int8"),
        }
    def test_majority_vote(self):
        sigs = self._make_signals_dict()
        result = aggregate_signals(sigs, "majority_vote")
        # Weights: 5m=0.5, 1h=0.9, 1d=1.0
        # Day 0: 0.5+0.9+1.0=2.4 → long(1)
        # Day 1: 0.5-0.9+1.0=0.6 → long(1) [weighted sum positive]
        # Day 2: -0.5-0.9+1.0=-0.4 → exit(-1) [weighted sum negative]
        # Day 3: 0+0.9+1.0=1.9 → long(1)
        # Day 4: 0.5+0.9+1.0=2.4 → long(1)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 1  # weighted sum 0.6 > 0
        assert result.iloc[2] == -1  # weighted sum -0.4 < 0
        assert result.iloc[3] == 1
        assert result.iloc[4] == 1
        assert result.iloc[3] == 1  # 2 longs, 1 exit → weighted long
        assert result.iloc[4] == 1  # 3 longs

    def test_unanimous(self):
        sigs = self._make_signals_dict()
        result = aggregate_signals(sigs, "unanimous")
        # Only day 1 and 5 are unanimously long
        assert result.iloc[0] == 1
        assert result.iloc[4] == 1
        # Day 3 is not unanimous (2 exit, 1 long)
        assert result.iloc[2] == 0

    def test_any_long(self):
        sigs = self._make_signals_dict()
        result = aggregate_signals(sigs, "any_long")
        # Any long → 1, unless there's an exit (exits override)
        # Day 1: all long → 1; Day 2: has exit → -1; Day 3: has exit → -1
        assert result.iloc[0] == 1
        assert result.iloc[1] == -1  # exit overrides
        assert result.iloc[2] == -1


class TestPluginHandle:
    def test_plugin_exists(self):
        from bot.plugins.strategies.multi_timeframe import plugin
        assert isinstance(plugin, MultiTimeframeStrategy)
        assert plugin.name == "multi_timeframe"

    def test_is_strategy_subclass(self):
        assert issubclass(MultiTimeframeStrategy, Strategy)

    def test_params(self):
        strat = MultiTimeframeStrategy()
        assert strat.params["base_strategy_name"] == "ema_cross_rsi"
        assert strat.params["timeframes"] == ["5m", "1h", "1d"]
        assert strat.params["aggregation"] == "majority_vote"


class TestGenerateFromDailyNoSymbol:
    def test_no_symbol_uses_base_only(self):
        """Without a symbol, should use base strategy on the daily df."""
        strat = MultiTimeframeStrategy(timeframes=["1d"])
        df = _make_ohlcv(100, trend="up")
        signals = strat.generate_signals(df)
        unique_vals = set(signals.tolist())
        assert unique_vals <= {0, 1, -1}

    def test_with_symbol_mocks_fetch(self):
        """With symbol, should attempt fetch and aggregate."""
        strat = MultiTimeframeStrategy(timeframes=["5m", "1d"], aggregation="majority_vote")
        df = _make_ohlcv(100, trend="up")
        df.attrs["symbol"] = "AAPL"

        # Mock fetch to return None for intraday → falls back to daily only
        with patch.object(MultiTimeframeComposer, "_fetch_timeframe_data", return_value=None):
            signals = strat.generate_signals(df)
        unique_vals = set(signals.tolist())
        assert unique_vals <= {0, 1, -1}
