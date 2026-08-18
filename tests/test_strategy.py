"""Tests for bot.strategy — Strategy ABC + EmaCrossRsi."""

import pandas as pd
import pytest
from bot.strategy import Strategy, EmaCrossRsi


class TestStrategyABC:
    def test_can_subclass(self):
        class MyStrat(Strategy):
            name = "test"
            params = {}
            def generate_signals(self, df):
                return pd.Series([0] * len(df), dtype="int8", index=df.index)
        assert issubclass(MyStrat, Strategy)


class TestEmaCrossRsi:
    def _make_trend_df(self, direction="up", length=200):
        """Create synthetic OHLCV DataFrame with trend and noise."""
        base = 100.0
        if direction == "up":
            # Initial decline → strong uptrend → produces clear EMA cross-up with low RSO at crossover point
            trend = [base - i * 0.5 for i in range(25)] + \
                    [base - 12.0 + (i - 25) * 0.6 for i in range(25, length)]
        elif direction == "down":
            trend = [base - i * 0.5 for i in range(length)]
        else:
            trend = [base + (i % 10) * 0.1 for i in range(length)]  # sideways
        # Add some volatility
        import random
        random.seed(42)
        closes = [t + random.uniform(-2, 2) for t in trend]
        opens = [closes[i] + random.uniform(-1, 1) for i, c in enumerate(closes)]
        highs = [max(o, c) + random.uniform(0, 2) for o, c in zip(opens, closes)]
        lows = [min(o, c) - random.uniform(0, 2) for o, c in zip(opens, closes)]
        return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1000] * length})

    def test_strategy_generates_signals_on_up_trend_with_cross(self):
        """Signal should flip from 0→1 after EMA cross-up in uptrend."""
        strategy = EmaCrossRsi(fast=9, slow=21, rsi_period=14)
        df = self._make_trend_df("up", 200)
        signals = strategy.generate_signals(df)
        unique_vals = set(signals.tolist())
        assert unique_vals <= {0, 1, -1}, f"Unexpected signal values: {unique_vals}"
        assert any(v == 1 for v in signals), f"Uptrend should produce entry signals (1). Got: {sorted(unique_vals)}"

    def test_rsi_overbought_forces_exit(self):
        """When RSI > exit threshold, signal should be -1 regardless of EMA."""
        # Create series where price jumps up sharply so RSO spikes
        close_prices = [100.0] * 50 + [100.0 + i * 5 for i in range(30)]
        df = pd.DataFrame({
            "Close": close_prices,
            "Open": close_prices,
            "High": close_prices,
            "Low": close_prices,
            "Volume": [1000] * len(close_prices),
        })
        strategy = EmaCrossRsi(rsi_entry_max=70, rsi_exit=75)
        signals = strategy.generate_signals(df)
        # After the price surge (indices 50+), RSO will be high → signals should include -1
        has_exit = -1 in signals.values
        assert has_exit, f"RSI overbought should trigger exit (-1), got signals: {signals.iloc[-5:].tolist()}"

    def test_crossover_detection(self):
        """Test that EMA crossover generates a 1 signal."""
        # Price starts low, then crosses up dramatically
        values = list(range(1, 60)) + list(range(60, 0, -1)) + list(range(1, 100))
        df = pd.DataFrame({
            "Close": values, "Open": values, "High": [v+1 for v in values],
            "Low": [v-1 for v in values], "Volume": [1000]*len(values)
        })
        strategy = EmaCrossRsi(fast=5, slow=15, rsi_entry_max=90, rsi_exit=95)
        signals = strategy.generate_signals(df)
        assert 1 in signals.values, "Upward move after down-move should produce entry signal"

    def test_to_backtesting_strategy_returns_class(self):
        """Converting to backtesting format returns a Strategy subclass."""
        strategy = EmaCrossRsi()
        bt_class = strategy.to_backtesting_strategy()
        assert hasattr(bt_class, 'init'), "backtesting.Strategy needs init method"
        assert hasattr(bt_class, 'next'), "backtesting.Strategy needs next method"
        assert bt_class.name == "ema_cross_rsi_BT"
