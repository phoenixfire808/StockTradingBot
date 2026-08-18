"""Tests for bot.sector_rotation + bot.plugins.strategies.sector_rotation."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bot.sector_rotation import (
    SectorRotationModel,
    SECTOR_ETFS,
    SYMBOL_SECTOR_MAP,
    bias_signals,
    compute_sector_momentum,
)
from bot.plugins.strategies.sector_rotation import SectorRotationStrategy
from bot.strategy import Strategy


def _make_ohlcv(length=100, start_price=100.0, trend="up"):
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


class TestSectorMapping:
    def test_get_sector_etf_known(self):
        model = SectorRotationModel()
        assert model.get_sector_etf("AAPL") == "XLK"
        assert model.get_sector_etf("JPM") == "XLF"
        assert model.get_sector_etf("XOM") == "XLE"

    def test_get_sector_etf_case_insensitive(self):
        model = SectorRotationModel()
        assert model.get_sector_etf("aapl") == "XLK"

    def test_get_sector_etf_unknown(self):
        model = SectorRotationModel()
        assert model.get_sector_etf("UNKNOWN") is None


class TestMomentumComputation:
    def test_compute_momentum_with_mock_data(self):
        """Mock fetch to return controlled data → verify RoC calculation."""
        model = SectorRotationModel(lookback_days=5, momentum_threshold=0.01)


        # Mock fetch: ETF closes: 99, 100, 101, 102, 103, 106 → RoC = (106-100)/100 = 0.06
        mock_df = pd.DataFrame({
            "Close": [99, 100, 101, 102, 103, 106],
        })
        with patch.object(model, "_fetch_etf_data", return_value=mock_df):
            result = model.compute_sector_momentum(etfs=["XLK"])

        assert "XLK" in result
        # RoC over 5 days: (106 - 100) / 100 = 0.06 (last vs 5 bars back)
        assert abs(result["XLK"] - 0.06) < 0.001

    def test_momentum_cache(self):
        """Second call within cache window should not refetch."""
        model = SectorRotationModel(lookback_days=5)
        mock_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105, 106]})
        call_count = [0]

        def mock_fetch(*args, **kwargs):
            call_count[0] += 1
            return mock_df

        with patch.object(model, "_fetch_etf_data", side_effect=mock_fetch):
            r1 = model.compute_sector_momentum(etfs=["XLK"])
            r2 = model.compute_sector_momentum(etfs=["XLK"])

        assert call_count[0] == 1, "Second call should use cache"
        assert r1 == r2


class TestBiasScoring:
    def test_bullish_bias(self):
        model = SectorRotationModel(lookback_days=5, momentum_threshold=0.01)
        # 10% gain → strong bullish
        mock_df = pd.DataFrame({"Close": [100, 102, 104, 106, 108, 110, 112]})
        with patch.object(model, "_fetch_etf_data", return_value=mock_df):
            bias = model.get_sector_bias("AAPL")  # AAPL → XLK
        assert bias > 0, "Strong positive momentum → positive bias"
        assert bias <= 1.0

    def test_bearish_bias(self):
        model = SectorRotationModel(lookback_days=5, momentum_threshold=0.01)
        # 10% loss → strong bearish
        mock_df = pd.DataFrame({"Close": [110, 108, 106, 104, 102, 100, 98]})
        with patch.object(model, "_fetch_etf_data", return_value=mock_df):
            bias = model.get_sector_bias("AAPL")
        assert bias < 0, "Strong negative momentum → negative bias"
        assert bias >= -1.0

    def test_neutral_bias(self):
        model = SectorRotationModel(lookback_days=5, momentum_threshold=0.05)
        # Small change within threshold → neutral
        mock_df = pd.DataFrame({"Close": [100, 100.2, 100.5, 100.3, 100.8, 100.5, 101]})
        with patch.object(model, "_fetch_etf_data", return_value=mock_df):
            bias = model.get_sector_bias("AAPL")
        assert bias == 0.0, "Small RoC within threshold → neutral bias"

    def test_unmapped_symbol_neutral(self):
        model = SectorRotationModel()
        bias = model.get_sector_bias("UNKNOWN")
        assert bias == 0.0


class TestBiasSignals:
    def test_block_longs_when_bearish(self):
        signals = pd.Series([1, 1, 0, -1, 1], dtype="int8")
        result = bias_signals(signals, bias=-0.5, block_threshold=-0.3)
        # Longs blocked, exit preserved
        assert (result == 1).sum() == 0
        assert (result == -1).sum() == 1  # exit preserved

    def test_preserve_longs_when_bullish(self):
        signals = pd.Series([1, 0, 0, -1, 1], dtype="int8")
        result = bias_signals(signals, bias=0.5, boost_threshold=0.3)
        # Longs preserved
        assert (result == 1).sum() == 2
        assert (result == -1).sum() == 1

    def test_neutral_passes_through(self):
        signals = pd.Series([1, 0, -1, 1, 0], dtype="int8")
        result = bias_signals(signals, bias=0.0)
        pd.testing.assert_series_equal(result, signals.astype("int8"))


class TestPluginHandle:
    def test_plugin_exists(self):
        from bot.plugins.strategies.sector_rotation import plugin
        assert isinstance(plugin, SectorRotationStrategy)
        assert plugin.name == "sector_rotation"

    def test_is_strategy_subclass(self):
        assert issubclass(SectorRotationStrategy, Strategy)

    def test_params(self):
        strat = SectorRotationStrategy()
        assert strat.params["base_strategy_name"] == "ema_cross_rsi"
        assert strat.params["lookback_days"] == 20
        assert strat.params["momentum_threshold"] == 0.02


class TestPluginGenerateSignals:
    def test_no_symbol_passes_through(self):
        strat = SectorRotationStrategy()
        df = _make_ohlcv(100, trend="up")
        signals = strat.generate_signals(df)
        unique_vals = set(signals.tolist())
        assert unique_vals <= {0, 1, -1}

    def test_with_symbol_blocks_longs(self):
        strat = SectorRotationStrategy(lookback_days=5, momentum_threshold=0.01, block_threshold=-0.3)
        df = _make_ohlcv(100, trend="up")
        df.attrs["symbol"] = "AAPL"

        # Mock rotation model to return bearish bias
        with patch.object(SectorRotationModel, "get_sector_bias", return_value=-0.5):
            signals = strat.generate_signals(df)
        # All longs should be blocked
        assert (signals == 1).sum() == 0
