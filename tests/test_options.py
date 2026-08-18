"""Tests for bot.broker option methods + bot.options re-exports."""

from __future__ import annotations

import asyncio

import pytest

from bot.broker import (
    Broker,
    BrokerError,
    CryptoOrder,
    CryptoQuote,
    MockBroker,
    OptionChain,
    OptionOrder,
    RobinhoodMcpBroker,
)
from bot.options import CryptoOrder as OptCryptoOrder
from bot.options import CryptoQuote as OptCryptoQuote
from bot.options import OptionChain as OptOptionChain
from bot.options import OptionOrder as OptOptionOrder


class TestOptionsReexport:
    """bot.options should re-export the dataclasses defined in bot.broker."""

    def test_option_order_is_same_class(self):
        assert OptOptionOrder is OptionOrder

    def test_crypto_order_is_same_class(self):
        assert OptCryptoOrder is CryptoOrder

    def test_option_chain_is_same_class(self):
        assert OptOptionChain is OptionChain

    def test_crypto_quote_is_same_class(self):
        assert OptCryptoQuote is CryptoQuote


class TestMockBrokerOptions:
    """MockBroker must implement every option-related abstract method."""

    def setup_method(self):
        self.broker = MockBroker(starting_equity=100_000)

    def test_is_broker_subclass(self):
        assert isinstance(self.broker, Broker)

    @pytest.mark.asyncio
    async def test_submit_option_order_returns_id(self):
        order = OptionOrder(
            symbol="AAPL", quantity=2, side="BUY", position_effect="open",
            legs=[{"type": "call", "strike": 150.0, "expiry": "2026-09-18", "side": "buy"}],
            order_type="market",
        )
        oid = await self.broker.submit_option_order(order)
        assert isinstance(oid, str)
        assert oid.startswith("mock-option-")
        assert order.order_id == oid

    @pytest.mark.asyncio
    async def test_get_option_chain_filters_by_type(self):
        chain = await self.broker.get_option_chain("AAPL", option_type="call")
        assert isinstance(chain, list)
        assert len(chain) > 0
        for c in chain:
            assert isinstance(c, OptionChain)
            assert c.option_type == "call"
            assert c.symbol == "AAPL"
            assert c.bid is not None and c.bid > 0
            assert c.ask is not None and c.ask > 0

    @pytest.mark.asyncio
    async def test_get_option_chain_returns_all_when_no_filter(self):
        chain = await self.broker.get_option_chain("AAPL")
        types = {c.option_type for c in chain}
        assert {"call", "put"} <= types

    @pytest.mark.asyncio
    async def test_cancel_all_options_runs_clean(self):
        # No exceptions is success.
        await self.broker.cancel_all_options()


class TestRobinhoodMcpBrokerOptionsInterface:
    """RobinhoodMcpBroker must implement the abstract methods (without making real calls)."""

    def setup_method(self):
        self.broker = RobinhoodMcpBroker(settings=None)

    def test_submit_option_order_is_coroutine(self):
        order = OptionOrder(
            symbol="AAPL", quantity=1, side="BUY", position_effect="open",
            legs=[{"type": "put", "strike": 145.0, "expiry": "2026-09-18", "side": "buy"}],
        )
        coro = self.broker.submit_option_order(order)
        try:
            assert asyncio.iscoroutine(coro)
        finally:
            coro.close()

    def test_cancel_all_options_is_coroutine(self):
        coro = self.broker.cancel_all_options()
        try:
            assert asyncio.iscoroutine(coro)
        finally:
            coro.close()

    def test_get_option_chain_is_coroutine(self):
        coro = self.broker.get_option_chain("AAPL")
        try:
            assert asyncio.iscoroutine(coro)
        finally:
            coro.close()


def test_broker_error_is_exception():
    """BrokerError must subclass Exception so engine can catch it."""
    assert issubclass(BrokerError, Exception)
    err = BrokerError("test")
    assert str(err) == "test"


def test_option_order_dataclass_fields():
    """OptionOrder must keep the documented field names so strategies can rely on them."""
    o = OptionOrder(
        symbol="SPY", quantity=1, side="SELL", position_effect="close",
        legs=[{"type": "call", "strike": 450.0, "expiry": "2026-12-31", "side": "sell"}],
        order_type="limit", limit_price=2.50, stop_price=None, time_in_force="gtc",
    )
    assert o.symbol == "SPY"
    assert o.quantity == 1
    assert o.side == "SELL"
    assert o.position_effect == "close"
    assert o.order_type == "limit"
    assert o.limit_price == 2.50
    assert o.order_id is None  # filled on submit


def test_option_chain_dataclass_fields():
    c = OptionChain(
        symbol="AAPL", strike=150.0, expiry="2026-09-18", option_type="call",
        bid=3.10, ask=3.20, mark=3.15, volume=100, open_interest=500,
    )
    assert c.symbol == "AAPL"
    assert c.strike == 150.0
    assert c.option_type == "call"
    assert c.delta is None  # optional fields default to None