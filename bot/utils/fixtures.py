"""Test fixtures for StockTradingBot unit and integration tests.

Factory functions return valid objects out of the box \u2014 pass keyword overrides to
tune behaviour without rebuilding every fixture in every test.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mock_ohlcv_df(
    n_bars: int = 200,
    start_price: float = 150.0,
    volatility: float = 0.02,
    drift: float = 0.0002,
    seed: Optional[int] = None,
    freq: str = "1D",
    index_start: dt.datetime | None = None,
) -> pd.DataFrame:
    """Build a realistic OHLCV DataFrame.

    Uses Geometric Brownian Motion to generate close prices then derives
    high/low/open from the close series so patterns are internally consistent.

    Parameters
    ----------
    n_bars: number of bars (default 200).
    start_price: first bar close (default 150).
    volatility: daily vol sigma (default 0.02 ~ 2%).
    drift: per-bar mean return (default 0.0002).
    seed: random seed for reproducibility.
    freq: pandas timedelta alias passed to ``pd.date_range`` (default ``"1D"``).
    index_start: start for DatetimeIndex; defaults to ``"2025-01-01"`` (n_bars days backwards).

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        end=index_start or pd.Timestamp('2025-01-01'), periods=n_bars, freq=freq, tz=None,
    )
    # GBM log returns
    log_returns = (drift - 0.5 * volatility**2) + volatility * rng.standard_normal(n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))

    intraday_range = volatility * start_price * 0.5  # H-L spread scale
    open_prices = close + rng.uniform(-intraday_range, intraday_range, n_bars)
    highs = np.maximum(close, open_prices) + abs(rng.uniform(0, intraday_range * 0.7, n_bars))
    lows = np.minimum(close, open_prices) - abs(rng.uniform(0, intraday_range * 0.7, n_bars))

    volume = rng.integers(1_000_000, 10_000_000, size=n_bars).astype(float)
    volume = volume * (1 + 0.5 * abs(log_returns))  # volume follows volatility

    df = pd.DataFrame({
        "Open": open_prices,
        "High": highs,
        "Low": lows,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


class TrackingMockBroker:
    """Wrap MockBroker and record every order/trade for test assertions."""

    def __init__(self, starting_equity: float = 100_000) -> None:
        from bot.broker import MockBroker

        self._inner = MockBroker(starting_equity=starting_equity)
        self.trades: list[dict] = []       # every submit_order call
        self.cancels: list[str] = []       # cancel_all invocations
        self.option_trades: list[dict] = []
        self.crypto_trades: list[dict] = []

    async def get_equity(self) -> float:
        return await self._inner.get_equity()

    async def get_positions(self) -> dict[str, int]:
        return await self._inner.get_positions()

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop: Optional[float] = None,
        target: Optional[float] = None,
    ) -> str:
        order_id = await self._inner.submit_order(symbol, qty, side, stop, target)
        self.trades.append({
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "stop": stop,
            "target": target,
        })
        return order_id

    async def cancel_all(self) -> None:
        self.cancels.append(f"cancel_all@{len(self.cancels)}")
        await self._inner.cancel_all()

    def is_market_open(self) -> bool:
        return self._inner.is_market_open()

    async def test_connection(self) -> bool:
        return await self._inner.test_connection()

    async def submit_option_order(self, order) -> str:
        order_id = await self._inner.submit_option_order(order)
        self.option_trades.append({
            "order_id": order_id,
            "symbol": getattr(order, "symbol", "?"),
            "side": getattr(order, "side", "?"),
            "quantity": getattr(order, "quantity", 0),
        })
        return order_id

    async def get_option_chain(self, symbol, expiry=None, option_type=None):
        return await self._inner.get_option_chain(symbol, expiry, option_type)

    async def cancel_all_options(self) -> None:
        await self._inner.cancel_all_options()

    async def submit_crypto_order(self, order) -> str:
        order_id = await self._inner.submit_crypto_order(order)
        self.crypto_trades.append({
            "order_id": order_id,
            "symbol": getattr(order, "symbol", "?"),
            "side": getattr(order, "side", "?"),
            "quantity": getattr(order, "quantity", 0),
        })
        return order_id

    async def get_crypto_quotes(self, symbols):
        return await self._inner.get_crypto_quotes(symbols)


def mock_broker(starting_equity: float = 100_000) -> TrackingMockBroker:
    """Return a fresh TrackingMockBroker ready for dry-run trading tests."""
    return TrackingMockBroker(starting_equity=starting_equity)


def settings_factory(overrides: Optional[dict] = None) -> "bot.config.Settings":
    """Create a bot.config.Settings instance with controlled defaults.

    Parameters
    ----------
    overrides: optional dict mapping Settings attribute names ~ desired values.

    Returns
    -------
    Fully populated bot.config.Settings object.
    """
    from bot.config import Settings

    base: dict = {
        "symbols": ["AAPL", "MSFT", "NVDA"],
        "cash": 100_000.0,
        "risk_per_trade": 0.01,
        "max_daily_loss_pct": 3.0,
        "engine_interval_minutes": 5,
        "sentiment_lookback_hours": 24,
        "log_level": "DEBUG",
        "strategy_allocations": "",
        "min_buying_amount_usd": 1.0,
        "max_buying_amount_usd": 10_000.0,
        "min_selling_amount_usd": 1.0,
        "max_selling_amount_usd": 50_000.0,
        "symbol_exclusions": [],
    }
    if overrides:
        base.update(overrides)

    return Settings(**base)


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_scale: float = 0.75,
    sl_scale: float = 0.5,
    time_limit: int = 5,
    atr_col: str = "atr_14",
) -> pd.Series:
    """Shorthand for ``build_triple_barrier_labels`` with sensible test defaults.

    Wraps ``bot.ml.features.build_triple_barrier_labels`` so callers don't need to
    import the module under test directly \u2014 keeps fixture boundaries clean.

    Parameters
    ----------
    df: OHLCV DataFrame (must have High/Low/Close columns).
    pt_scale: take-profit multiplier on ATR (default 0.75).
    sl_scale: stop-loss multiplier on ATR (default 0.5).
    time_limit: bars after which to expire position (default 5).
    atr_col: column name pre-computed ATR (default ``"atr_14"``); when absent
             the function computes an ATR fallback internally.

    Returns
    -------
    pd.Series of int8 labels: 1 (TP hit), -1 (SL hit), 0 (time expired).
    """
    from bot.ml.features import build_triple_barrier_labels

    return build_triple_barrier_labels(
        df, pt_scale=pt_scale, sl_scale=sl_scale, time_limit=time_limit, atr_col=atr_col,
    )


def labelled_dataset(
    n_bars: int = 300,
    ohlcv_kwargs: Optional[dict] = None,
    feature_kwargs: Optional[dict] = None,
    label_kwargs: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """End-to-end helper: generate OHLCV -> features -> labels, return aligned frames.

    Convenience fixture that lets a single call produce a ready-to-train
    (X, y) split. All three stages are exercised so edge-cases (NaN wicking,
    short tails) are caught together.

    Parameters
    ----------
    n_bars: bars for OHLCV generation.
    ohlcv_kwargs: passed to ``mock_ohlcv_df``.
    feature_kwargs: passed to ``bot.ml.features.build_feature_frame``.
    label_kwargs: passed to ``triple_barrier_labels`` above.

    Returns
    -------
    (ohlcxv, features, labels) - feature frame has NaN-wrapped index;
    labels is a pd.Series aligned to ``features``.
    """
    _ohlcv_kwargs: dict = ohlcv_kwargs or {}
    _feature_kwargs: dict = feature_kwargs or {}
    _label_kwargs: dict = label_kwargs or {}

    ohlcv = mock_ohlcv_df(n_bars=n_bars, **_ohlcv_kwargs)

    from bot.ml.features import build_feature_frame

    features = build_feature_frame(ohlcv, **_feature_kwargs)

    # Pad OHLCV with enough rows so label computation has tail room
    extended = mock_ohlcv_df(n_bars=n_bars + 10, **_ohlcv_kwargs)

    labels = triple_barrier_labels(extended, **_label_kwargs)

    # Align: keep only indices present in both features and non-null labels,
    # then further restrict to rows where every feature value is non-null.
    common_idx = features.index.intersection(labels.dropna().index)
    valid_features = features.loc[common_idx].dropna(how="any")
    final_idx = valid_features.index.intersection(labels.dropna().index)
    return ohlcv, features.loc[final_idx], labels.loc[final_idx]


__all__ = [
    "mock_ohlcv_df",
    "mock_broker",
    "TrackingMockBroker",
    "settings_factory",
    "triple_barrier_labels",
    "labelled_dataset",
]
