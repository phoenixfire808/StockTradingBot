"""Tests for bot.indicators — EMA, RSI, ATR, Bollinger."""

import pandas as pd
import pytest
from bot.indicators import ema, rsi, atr, bollinger


class TestEMA:
    def test_ema_simple(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(series, period=3)
        assert len(result) == len(series)
        assert result.iloc[-1] > result.iloc[0], "EMA should rise with rising data"

    def test_ema_constant(self):
        series = pd.Series([5.0] * 10)
        result = ema(series, period=3)
        # EMA of constant is near-constant
        diffs = result.diff().dropna()
        assert diffs.abs().max() < 1e-10


class TestRSI:
    def test_rsi_all_rising(self):
        series = pd.Series(range(1, 30))
        result = rsi(series, period=14)
        # When price keeps rising, RSI should be high (>70)
        last_valid = result.dropna().iloc[-1]
        assert last_valid > 70, f"RSI of all-rising series should be >70, got {last_valid}"

    def test_rsi_all_falling(self):
        series = pd.Series(range(30, 0, -1))
        result = rsi(series, period=14)
        last_valid = result.dropna().iloc[-1]
        assert last_valid < 30, f"RSI of all-falling series should be <30, got {last_valid}"

    def test_rsi_range(self):
        series = pd.DataFrame({"Close": [100 + i * (1 if i % 3 else -1) for i in range(100)]})["Close"]
        result = rsi(series, period=14)
        valid = result.dropna()
        assert valid.min() >= 0 and valid.max() <= 100


class TestATR:
    def test_atr_basic(self):
        df = pd.DataFrame({
            "high": [10, 11, 12, 13, 14],
            "low": [8, 9, 10, 11, 12],
            "close": [9, 10, 11, 12, 13],
        })
        result = atr(df, period=3)
        assert len(result.dropna()) > 0
        assert result.dropna().min() > 0


class TestBollinger:
    def test_bollinger_mid_equals_ma(self):
        series = pd.Series(range(1, 31), dtype=float)
        bands = bollinger(series, period=5, num_std=2)
        mid = bands["mid"]
        expected_mean = series.rolling(5).mean()
        pd.testing.assert_series_equal(mid, expected_mean, check_names=False)

    def test_bollinger_upper_below_lower(self):
        series = pd.Series(range(1, 101), dtype=float)
        bands = bollinger(series, period=10)
        valid = bands["upper"].dropna()
        assert (bands["upper"].dropna() > bands["lower"].dropna()).all()
