"""Broker abstraction — Robinhood MCP implementation + MockBroker for dry-run."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.config import Settings

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised when a broker operation fails. Engine catches this and applies kill-switch."""


@dataclass
class OptionOrder:
    """Represents an options order submitted to a broker.

    Fields map to Robinhood MCP ``place_option_order`` parameters.
    """
    symbol: str              # underlying symbol, e.g. "AAPL"
    quantity: int            # number of contracts
    side: str               # "BUY" | "SELL"
    position_effect: str    # "open" | "close"
    legs: list[dict] = field(default_factory=list)  # [{type:"call"|"put", strike:float, expiry:"YYYY-MM-DD", side:"buy"|"sell"}]
    order_type: str = "market"          # "market" | "limit" | "stop" | "stop_limit"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "gtc"          # "gtc" | "gfd" | "ioc" | "opg"
    order_id: str | None = None         # filled on submit

@dataclass
class CryptoOrder:
    """Represents a crypto order submitted to a broker.

    Fields map to Robinhood MCP crypto order parameters.
    """
    symbol: str              # crypto pair, e.g. "BTC-USD"
    quantity: float          # fractional quantity in base currency
    side: str               # "BUY" | "SELL"
    order_type: str = "market"          # "market" | "limit"
    limit_price: float | None = None
    time_in_force: str = "gtc"          # "gtc" | "gfd" | "ioc"
    order_id: str | None = None         # filled on submit

@dataclass
class OptionChain:
    """Represents an option chain entry for a given expiry/strike/type."""
    symbol: str
    strike: float
    expiry: str            # "YYYY-MM-DD"
    option_type: str       # "call" | "put"
    bid: float = 0.0
    ask: float = 0.0
    mark: float = 0.0
    volume: int = 0
    open_interest: int = 0
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

@dataclass
class CryptoQuote:
    """Represents a real-time crypto quote."""
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    mark: float = 0.0
    last_price: float = 0.0
    volume_24h: float = 0.0


class Broker(ABC):
    """Abstract broker interface. All real/mock brokers subclass this."""

    @abstractmethod
    async def get_equity(self) -> float:
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, int]:
        """Return {symbol: qty}."""
        ...

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # "BUY" | "SELL"
        stop: float | None = None,
        target: float | None = None,
    ) -> str:
        """Place an order. Returns order ID string."""
        ...

    @abstractmethod
    async def cancel_all(self) -> None:
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Verify connectivity; returns True/False."""
        ...
    @abstractmethod
    async def submit_option_order(self, order: OptionOrder) -> str:
        """Submit an options order. Returns order ID string."""
        ...

    @abstractmethod
    async def get_option_chain(
        self,
        symbol: str,
        expiry: str | None = None,
        option_type: str | None = None,
    ) -> list[OptionChain]:
        """Get option chain for *symbol* filtered by expiry/type."""
        ...

    @abstractmethod
    async def cancel_all_options(self) -> None:
        """Cancel all open options orders."""
        ...

    @abstractmethod
    async def submit_crypto_order(self, order: CryptoOrder) -> str:
        """Submit a crypto order. Returns order ID string."""
        ...

    @abstractmethod
    async def get_crypto_quotes(self, symbols: list[str]) -> dict[str, CryptoQuote]:
        """Get real-time crypto quotes for up to N symbols."""
        ...


# ── Mock Broker (dry-run / testing) ──────────────────────────────────

class MockBroker(Broker):
    """In-memory broker that logs [MOCK] and mutates state. Never touches real money."""

    def __init__(self, starting_equity: float = 100_000) -> None:
        self.equity = starting_equity
        self.positions: dict[str, int] = {}
        self._order_id_counter = 0

    async def get_equity(self) -> float:
        return self.equity

    async def get_positions(self) -> dict[str, int]:
        return dict(self.positions)

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop: float | None = None,
        target: float | None = None,
    ) -> str:
        self._order_id_counter += 1
        price = 100.0 + (hash(symbol) % 200)  # deterministic fake price
        self.equity -= qty * price if side == "BUY" else -(qty * price)
        if side == "BUY":
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
        else:
            self.positions[symbol] = max(0, self.positions.get(symbol, 0) - qty)
            if self.positions.get(symbol, 0) <= 0:
                self.positions.pop(symbol, None)
        msg = f"[MOCK] {side} {symbol} {qty}@{price:.2f} → EQ={self.equity:.2f}"
        logger.info(msg)
        return f"mock-order-{self._order_id_counter}"

    async def cancel_all(self) -> None:
        logger.info("[MOCK] cancel_all called")

    def is_market_open(self) -> bool:
        return True  # mock always open

    async def test_connection(self) -> bool:
        return True
    async def submit_option_order(self, order: OptionOrder) -> str:
        self._order_id_counter += 1
        mock_price = 2.50 + (hash(order.symbol) % 100) / 10.0
        cost = order.quantity * mock_price * 100  # 1 contract = 100 shares
        self.equity -= cost if order.side.upper() == "BUY" else -cost
        order.order_id = f"mock-option-{self._order_id_counter}"
        logger.info(
            "[MOCK] OPTION %s %d %s contracts @ %.2f → EQ=%.2f",
            order.side, order.quantity, order.symbol, mock_price, self.equity,
        )
        return order.order_id

    async def get_option_chain(
        self,
        symbol: str,
        expiry: str | None = None,
        option_type: str | None = None,
    ) -> list[OptionChain]:
        import datetime as _dt
        # Generate a synthetic mock chain
        base_strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        if expiry is None:
            expiry = (_dt.date.today() + _dt.timedelta(days=30)).isoformat()
        types = [option_type] if option_type else ["call", "put"]
        chain = []
        for t in types:
            for strike in base_strikes:
                chain.append(OptionChain(
                    symbol=symbol, strike=strike, expiry=expiry, option_type=t,
                    bid=max(0.50, 5.0 - abs(strike - 100) * 0.3),
                    ask=max(0.55, 5.2 - abs(strike - 100) * 0.3),
                    mark=max(0.52, 5.1 - abs(strike - 100) * 0.3),
                    volume=1000, open_interest=500,
                ))
        logger.info("[MOCK] OPTION CHAIN %s: %d contracts", symbol, len(chain))
        return chain

    async def cancel_all_options(self) -> None:
        logger.info("[MOCK] cancel_all_options called")

    async def submit_crypto_order(self, order: CryptoOrder) -> str:
        self._order_id_counter += 1
        mock_price = 50000.0 + (hash(order.symbol) % 10000)
        cost = order.quantity * mock_price
        self.equity -= cost if order.side.upper() == "BUY" else -cost
        order.order_id = f"mock-crypto-{self._order_id_counter}"
        logger.info(
            "[MOCK] CRYPTO %s %.6f %s @ %.2f → EQ=%.2f",
            order.side, order.quantity, order.symbol, mock_price, self.equity,
        )
        return order.order_id

    async def get_crypto_quotes(self, symbols: list[str]) -> dict[str, CryptoQuote]:
        quotes = {}
        for sym in symbols:
            price = 50000.0 + (hash(sym) % 10000)
            quotes[sym] = CryptoQuote(
                symbol=sym, bid=price - 5.0, ask=price + 5.0,
                mark=price, last_price=price, volume_24h=1_000_000.0,
            )
        logger.info("[MOCK] CRYPTO QUOTES: %d symbols", len(quotes))
        return quotes


# ── Robinhood MCP Broker ─────────────────────────────────────────────

class RobinhoodMcpBroker(Broker):
    """Real broker using the Robinhood Agentic Trading MCP server.

    Connects via SSE HTTP or local stdio proxy based on settings.
    Every tool call is logged (never logging secrets).
    Errors are raised as BrokerError so the engine can handle them.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._session = None
        self._initialized = False

    async def _ensure_session(self) -> None:
        """Lazily create and initialize the MCP client session."""
        if self._initialized and self._session is not None:
            return

        try:
            from mcp import ClientSession, StdioServerParameters, SseServerParams
        except ImportError:
            raise BrokerError("mcp package not installed. Run: pip install mcp httpx[sse]")

        try:
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
        except ImportError:
            raise BrokerError("MCP client modules not available.")

        if self._settings and self._settings.robinhood_mcp_command:
            # Local stdio proxy mode
            args = json.loads(self._settings.robinhood_mcp_args) if self._settings.robinhood_mcp_args else []
            params = StdioServerParameters(
                command=self._settings.robinhood_mcp_command,
                args=args,
            )
            stdio_ctx = stdio_client(params)
            stdio, write = await stdio_ctx.__aenter__()
            self._session = ClientSession(stdio, write)
        else:
            # Remote SSE mode
            url = self._settings.robinhood_mcp_url if self._settings else "https://agent.robinhood.com/mcp/trading"
            headers = {}
            if self._settings and self._settings.robinhood_mcp_auth_header:
                headers["Authorization"] = self._settings.robinhood_mcp_auth_header
            sse_ctx = sse_client(url, headers=headers if headers else None)
            read, write = await sse_ctx.__aenter__()
            self._session = ClientSession(read, write)

        await self._session.initialize()
        self._initialized = True
        logger.info("RobinhoodMcpBroker session established (%s)", "sse" if not (self._settings and self._settings.robinhood_mcp_command) else "stdio")

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Execute an MCP tool call. Logs name + arg keys (not values — never log secrets)."""
        await self._ensure_session()
        try:
            result = await self._session.call_tool(tool_name, arguments or {})
            # Parse response content (MCP returns list of content blocks)
            content_parts = []
            if hasattr(result, 'content'):
                for block in result.content:
                    if hasattr(block, 'text'):
                        content_parts.append(block.text)
                    elif isinstance(block, dict) and 'text' in block:
                        content_parts.append(block['text'])
                    elif isinstance(block, str):
                        content_parts.append(block)
            text_result = "\n".join(content_parts)
            # Try to parse as JSON
            try:
                return json.loads(text_result)
            except (json.JSONDecodeError, ValueError):
                return {"raw": text_result}
        except Exception as exc:
            logger.error(f"MCP tool '{tool_name}' failed: {exc}")
            raise BrokerError(f"Tool call failed: {tool_name}: {exc}") from exc

    async def get_equity(self) -> float:
        """Get portfolio equity and buying power."""
        data = await self._call_tool("get_portfolio")
        if isinstance(data, dict):
            total_value = data.get("equity", data.get("totalValue", data.get("portfolio_value", 0)))
            if isinstance(total_value, (int, float)):
                return float(total_value)
        return 0.0

    async def get_positions(self) -> dict[str, int]:
        """Get all equity positions."""
        data = await self._call_tool("get_equity_positions")
        positions = {}
        if isinstance(data, dict):
            pos_list = data.get("positions", data.get("items", []))
            if isinstance(pos_list, list):
                for pos in pos_list:
                    symbol = pos.get("symbol", pos.get("ticker", ""))
                    qty = int(pos.get("quantity", pos.get("qty", 0)))
                    if qty != 0:
                        positions[symbol] = qty
        elif isinstance(data, list):
            for pos in data:
                symbol = pos.get("symbol", pos.get("ticker", ""))
                qty = int(pos.get("quantity", pos.get("qty", 0)))
                if qty != 0:
                    positions[symbol] = qty
        return positions

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop: float | None = None,
        target: float | None = None,
    ) -> str:
        """Place an equity order via Robinhood MCP."""
        tool_name = "place_equity_order"
        arguments = {
            "symbol": symbol,
            "qty": qty,
            "side": side.lower(),  # robinhood expects lowercase
            "type": "market",
        }
        if stop:
            arguments["stop_price"] = round(stop, 2)
        if target:
            arguments["take_profit_price"] = round(target, 2)

        logger.info(f"Order: {side} {qty} {symbol} at market{' w/ stops' if stop else ''}")
        result = await self._call_tool(tool_name, arguments)
        order_id = result.get("id", result.get("orderId", result.get("order_id", "unknown")))
        return str(order_id)

    async def cancel_all(self) -> None:
        """Cancel all open orders."""
        try:
            data = await self._call_tool("get_equity_orders")
            open_orders = []
            if isinstance(data, dict):
                open_orders = data.get("orders", data.get("items", []))
            elif isinstance(data, list):
                open_orders = data

            for order in open_orders:
                order_id = order.get("id", order.get("orderId", ""))
                if order_id:
                    logger.info(f"Cancelling order {order_id}")
                    await self._call_tool("cancel_equity_order", {"order_id": order_id})
        except Exception as exc:
            logger.warning(f"Error during cancel_all: {exc}")

    def is_market_open(self) -> bool:
        """Check NYSE hours (9:30–16:00 ET weekdays)."""
        import datetime
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))  # ET
        if now.weekday() >= 5:  # Saturday, Sunday
            return False
        hour = now.hour
        minute = now.minute
        time_val = hour + minute / 60.0
        return 9.5 <= time_val < 16.0

    async def test_connection(self) -> bool:
        """Quick connection test via get_accounts tool."""
        try:
            result = await self._call_tool("get_accounts")
            return True
        except Exception as exc:
            logger.warning(f"Connection test failed: {exc}")
            return False

    async def review_order(self, symbol: str, qty: int, side: str) -> dict:
        """Preview/simulate an order before placing (returns warnings)."""
        try:
            args = {"symbol": symbol, "qty": qty, "side": side.lower()}
            return await self._call_tool("review_equity_order", args)
        except Exception as exc:
            logger.warning(f"Order review failed: {exc}")
            return {"warnings": [], "errors": []}

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get real-time quotes for up to 20 symbols."""
        try:
            args = {"symbols": symbols}
            result = await self._call_tool("get_equity_quotes", args)
            if isinstance(result, dict):
                return result.get("quotes", result)
            return {s: result if isinstance(result, dict) else {} for s in symbols}
        except Exception as exc:
            logger.warning(f"Quotes fetch failed: {exc}")
            return {s: {} for s in symbols}
    # ── Options & Crypto (delegated to Robinhood MCP tools) ────────────

    async def submit_option_order(self, order: "OptionOrder") -> str:
        """Submit a single- or multi-leg options order via ``place_option_order``.

        Returns the broker order id. Raises BrokerError on transport / parse failure.
        """
        args: dict[str, Any] = {
            "symbol": order.symbol,
            "quantity": order.quantity,
            "side": order.side.lower(),
            "position_effect": order.position_effect,
            "legs": order.legs,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.limit_price is not None:
            args["limit_price"] = round(order.limit_price, 2)
        if order.stop_price is not None:
            args["stop_price"] = round(order.stop_price, 2)
        logger.info(
            "Submitting options order: %s %d %s (%s) %s",
            order.side, order.quantity, order.symbol, order.order_type,
            [f"{l.get('type')}@{l.get('strike')}" for l in order.legs],
        )
        result = await self._call_tool("place_option_order", args)
        order_id = str(result.get("id") or result.get("orderId") or result.get("order_id") or "unknown")
        order.order_id = order_id
        return order_id

    async def get_option_chain(
        self,
        symbol: str,
        expiry: str | None = None,
        option_type: str | None = None,
    ) -> list["OptionChain"]:
        """Fetch the option chain for *symbol*. Robinhood returns one chain entry per
        (expiry, strike, type); we normalize each into ``OptionChain`` dataclass.
        """
        args: dict[str, Any] = {"symbol": symbol}
        if expiry is not None:
            args["expiry"] = expiry
        if option_type is not None:
            args["type"] = option_type
        data = await self._call_tool("get_option_chain", args)
        items: list[dict] = []
        if isinstance(data, dict):
            items = data.get("chain", data.get("options", data.get("items", [])))
        elif isinstance(data, list):
            items = data
        chain: list[OptionChain] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            try:
                chain.append(OptionChain(
                    symbol=raw.get("symbol", symbol),
                    strike=float(raw.get("strike", 0.0)),
                    expiry=str(raw.get("expiry") or raw.get("expiration_date") or ""),
                    option_type=str(raw.get("type") or raw.get("option_type") or ""),
                    bid=_safe_float(raw.get("bid")),
                    ask=_safe_float(raw.get("ask")),
                    mark=_safe_float(raw.get("mark") or raw.get("last_price")),
                    volume=_safe_int(raw.get("volume")),
                    open_interest=_safe_int(raw.get("open_interest")),
                    delta=_safe_float(raw.get("delta")),
                    gamma=_safe_float(raw.get("gamma")),
                    theta=_safe_float(raw.get("theta")),
                    vega=_safe_float(raw.get("vega")),
                ))
            except (TypeError, ValueError) as exc:
                logger.debug("Skipping malformed chain entry for %s: %s", symbol, exc)
        logger.info("Option chain %s: %d contracts", symbol, len(chain))
        return chain

    async def cancel_all_options(self) -> None:
        """Cancel every open options order via ``get_option_orders`` + ``cancel_option_order``."""
        try:
            data = await self._call_tool("get_option_orders", {})
            open_orders: list[dict] = []
            if isinstance(data, dict):
                open_orders = data.get("orders", data.get("items", []))
            elif isinstance(data, list):
                open_orders = data
            for o in open_orders:
                oid = o.get("id") or o.get("orderId")
                if oid:
                    logger.info("Cancelling option order %s", oid)
                    await self._call_tool("cancel_option_order", {"order_id": oid})
        except BrokerError:
            raise
        except Exception as exc:
            logger.warning("cancel_all_options failed: %s", exc)
            raise BrokerError(f"cancel_all_options failed: {exc}") from exc

    async def submit_crypto_order(self, order: "CryptoOrder") -> str:
        """Submit a crypto order via ``place_crypto_order``."""
        args: dict[str, Any] = {
            "symbol": order.symbol,
            "quantity": order.quantity,
            "side": order.side.lower(),
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.limit_price is not None:
            args["limit_price"] = round(order.limit_price, 2)
        logger.info("Submitting crypto order: %s %.6f %s", order.side, order.quantity, order.symbol)
        result = await self._call_tool("place_crypto_order", args)
        order_id = str(result.get("id") or result.get("orderId") or result.get("order_id") or "unknown")
        order.order_id = order_id
        return order_id

    async def get_crypto_quotes(self, symbols: list[str]) -> dict[str, "CryptoQuote"]:
        """Fetch real-time crypto quotes via ``get_crypto_quotes``."""
        try:
            data = await self._call_tool("get_crypto_quotes", {"symbols": symbols})
            out: dict[str, CryptoQuote] = {}
            items: list[dict] | dict[str, dict] = []
            if isinstance(data, dict):
                items = data.get("quotes", data.get("items", {}))
            elif isinstance(data, list):
                items = data
            if isinstance(items, dict):
                iterable = items.items()
            else:
                iterable = [(q.get("symbol") or q.get("asset"), q) for q in items if isinstance(q, dict)]
            for sym, raw in iterable:
                if not sym or not isinstance(raw, dict):
                    continue
                out[sym] = CryptoQuote(
                    symbol=sym,
                    bid=_safe_float(raw.get("bid")),
                    ask=_safe_float(raw.get("ask")),
                    mark=_safe_float(raw.get("mark") or raw.get("last_price")),
                    last_price=_safe_float(raw.get("last_price") or raw.get("mark")),
                    volume_24h=_safe_float(raw.get("volume_24h") or raw.get("volume")),
                )
            # Ensure every requested symbol is present (fill missing with zero quote).
            for sym in symbols:
                out.setdefault(sym, CryptoQuote(symbol=sym))
            logger.info("Crypto quotes: %d symbols", len(out))
            return out
        except Exception as exc:
            logger.warning("get_crypto_quotes failed: %s", exc)
            return {s: CryptoQuote(symbol=s) for s in symbols}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

