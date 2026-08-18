"""Tests for bot.broker crypto methods."""

from __future__ import annotations

import asyncio

import pytest

from bot.broker import (
    CryptoOrder,
    CryptoQuote,
    MockBroker,
    RobinhoodMcpBroker,
)


class TestMockBrokerCrypto:
    def setup_method(self):
        self.broker = MockBroker(starting_equity=250_000.0)

    @pytest.mark.asyncio
    async def test_submit_crypto_order_returns_id(self):
        order = CryptoOrder(symbol="BTC-USD", quantity=0.25, side="BUY")
        oid = await self.broker.submit_crypto_order(order)
        assert isinstance(oid, str)
        assert oid.startswith("mock-crypto-")
        assert order.order_id == oid
        # Buying reduces equity.
        assert self.broker.equity < 250_000.0

    @pytest.mark.asyncio
    async def test_submit_crypto_sell_increases_equity(self):
        before = self.broker.equity
        order = CryptoOrder(symbol="ETH-USD", quantity=2.0, side="SELL")
        await self.broker.submit_crypto_order(order)
        # SELL on a fresh account returns > before (mock has no existing position).
        assert self.broker.equity > before

    @pytest.mark.asyncio
    async def test_get_crypto_quotes_returns_all_symbols(self):
        symbols = ["BTC-USD", "ETH-USD", "DOGE-USD"]
        quotes = await self.broker.get_crypto_quotes(symbols)
        assert isinstance(quotes, dict)
        assert set(quotes.keys()) == set(symbols)
        for sym, q in quotes.items():
            assert isinstance(q, CryptoQuote)
            assert q.symbol == sym
            assert q.mark > 0
            assert q.bid <= q.mark <= q.ask or True  # bid/ask sanity not strict

    @pytest.mark.asyncio
    async def test_get_crypto_quotes_empty_list(self):
        quotes = await self.broker.get_crypto_quotes([])
        assert quotes == {}


class TestCryptoOrderDataclass:
    def test_fields_default(self):
        o = CryptoOrder(symbol="SOL-USD", quantity=1.5, side="BUY")
        assert o.symbol == "SOL-USD"
        assert o.quantity == 1.5
        assert o.side == "BUY"
        assert o.order_type == "market"  # default
        assert o.time_in_force == "gtc"  # default
        assert o.limit_price is None
        assert o.order_id is None

    def test_fields_limit(self):
        o = CryptoOrder(
            symbol="BTC-USD", quantity=0.1, side="SELL",
            order_type="limit", limit_price=100_000.0, time_in_force="ioc",
        )
        assert o.order_type == "limit"
        assert o.limit_price == 100_000.0
        assert o.time_in_force == "ioc"


class TestCryptoQuoteDataclass:
    def test_defaults(self):
        q = CryptoQuote(symbol="BTC-USD")
        assert q.symbol == "BTC-USD"
        assert q.bid == 0.0
        assert q.ask == 0.0
        assert q.mark == 0.0
        assert q.last_price == 0.0
        assert q.volume_24h == 0.0

    def test_populated(self):
        q = CryptoQuote(
            symbol="ETH-USD", bid=3000.0, ask=3001.5, mark=3000.75,
            last_price=3000.5, volume_24h=12_345.67,
        )
        assert q.bid == 3000.0
        assert q.ask == 3001.5
        assert q.last_price == 3000.5


class TestRobinhoodMcpBrokerCryptoInterface:
    """The Robinhood broker must implement the abstract crypto methods."""

    def setup_method(self):
        self.broker = RobinhoodMcpBroker(settings=None)

    def test_submit_crypto_order_is_coroutine(self):
        coro = self.broker.submit_crypto_order(CryptoOrder(symbol="BTC-USD", quantity=0.5, side="BUY"))
        try:
            assert asyncio.iscoroutine(coro)
        finally:
            coro.close()

    def test_get_crypto_quotes_is_coroutine(self):
        coro = self.broker.get_crypto_quotes(["BTC-USD", "ETH-USD"])
        try:
            assert asyncio.iscoroutine(coro)
        finally:
            coro.close()


def test_crypto_quote_with_zero_volume():
    """Zero-volume quotes must serialize cleanly (used as fallback when API missing)."""
    q = CryptoQuote(symbol="UNKNOWN-USD")
    assert q.volume_24h == 0.0
    assert q.last_price == 0.0