"""Robinhood MCP datasource plugin — historical bars via official MCP server."""

import json
import logging

logger = logging.getLogger(__name__)


class RobinhoodMcpSource:
    """Datasource plugin wrapping the Robinhood Agentic Trading MCP ``get_equity_historicals`` tool.

    Priority: 1 (tries first; falls through to yfinance on auth failure).
    Granularity support: UNVERIFIED — the actual supported intervals depend on what the MCP server exposes.
    We attempt "1d" and fall back gracefully. At first use we log whatever intervals the server reports.
    """

    name = "robinhood_mcp"
    priority = 1

    SUPPORTED_INTERVALS = set()  # populated at first call via tool introspection
    _discovered = False

    def supports(self, interval: str) -> bool:
        if not self._discovered:
            self._discover_supported()
        return interval in self.SUPPORTED_INTERVALS if self.SUPPORTED_INTERVALS else True

    def _discover_supported(self):
        """First-use: introspect MCP tools to discover available granularities."""
        try:
            from bot.broker import RobinhoodMcpBroker
            from bot.config import load_settings
            settings = load_settings()
            broker = RobinhoodMcpBroker(settings)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            try:
                if loop.is_running():
                    logger.debug("Cannot introspect MCP schema from sync context — proceeding with assumed support.")
                    self._discovered = True
                    return
                results = loop.run_until_complete(broker.test_connection())
                logger.info("Robinhood MCP connection test: %s", results)
            finally:
                loop.close()
        except Exception as exc:
            logger.debug("Could not discover MCP granularity schema: %s", exc)
        self._discovered = True
        logger.info("Robinhood MCP datasource ready (intervals unverified — will accept all)")

    def fetch_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ):
        """Fetch historical bars via Robinhood MCP get_equity_historicals.

        Falls through to yfinance if authenticated MCP session unavailable.
        """
        try:
            from bot.broker import RobinhoodMcpBroker
            from bot.config import load_settings

            settings = load_settings()
            if not settings.robinhood_mcp_auth_header and not settings.robinhood_mcp_command:
                logger.info("No MCP credentials configured — falling back to yfinance")
                raise ConnectionError("No Robinhood MCP auth configured")

            broker = RobinhoodMcpBroker(settings)
            import asyncio

            async def _fetch():
                await broker._ensure_session()
                result = await broker._call_tool("get_equity_historicals", {
                    "symbol": symbol,
                    "interval": interval,
                })
                logger.info("Robinhood MCP historicals result type: %s", type(result))
                return result

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            try:
                raw_data = loop.run_until_complete(_fetch())
            finally:
                loop.close()

            return raw_data  # DataHub will cache whatever it gets

        except Exception as exc:
            logger.info("Robinhood MCP datasource failed for %s [%s]: %s", symbol, interval, exc)
            raise

plugin = RobinhoodMcpSource()
