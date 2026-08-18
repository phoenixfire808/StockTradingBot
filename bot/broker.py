"""Broker abstraction — Robinhood MCP implementation + MockBroker for dry-run."""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.config import Settings

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised on API / tool failures."""
    pass


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
        logger.info("RobinhoodMcpBroker session established (%s)", "sse" if not self._settings?.robinhood_mcp_command else "stdio")

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
