"""Application settings and logging bootstrap."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from logging import handlers
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """All configuration from .env with sane defaults."""
    # Robinhood MCP
    robinhood_mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    robinhood_mcp_auth_header: str = ""
    robinhood_mcp_command: str = ""
    robinhood_mcp_args: str = ""

    # Alternative market-data providers (Polygon.io, Databento).
    # Empty / unset = datasource disabled; reads from .env at startup.
    polygon_api_key: str = ""
    databento_api_key: str = ""

    # Trading scope
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    cash: float = 100_000
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 3.0
    engine_interval_minutes: int = 5
    sentiment_lookback_hours: int = 24
    log_level: str = "DEBUG"

    # Multi-strategy allocation JSON string
    strategy_allocations: str = ""

    # Trade amount guidelines (siropkin/robinhood-ai-trading-bot pattern)
    min_buying_amount_usd: float = 1.0
    max_buying_amount_usd: float = 10_000.0
    min_selling_amount_usd: float = 1.0
    max_selling_amount_usd: float = 50_000.0

    # Symbol exclusions - signals/trades filtered for these symbols
    symbol_exclusions: list[str] = field(default_factory=list)

    # Blacklist/whitelist pair management (-inspired)
    symbol_whitelist: list[str] = field(default_factory=list)  # Empty = trade anything; non-empty = only these symbols
    blacklisted_symbols: list[str] = field(default_factory=lambda: _parse_list(_get("BLACKLISTED_SYMBOLS", "GOOGL,TSLA")))

    # Stop-loss / take-profit management (-inspired patterns)
    stop_loss_pct: float = 0.05           # Default fixed stop-loss % below entry (5%)
    trailing_stop_enabled: bool = True     # Enable/disable trailing stops
    trailing_stop_pct: float = 0.03        # Trailing distance % below highest price
    trailing_stop_positive_offset: float = 0.05  # Profit threshold before trailing kicks in
    max_hold_minutes: int = 1440           # Max hold time (24h default), 0 = unlimited
    roi_exit_enabled: bool = True          # Enable ROI-based tiered exits
    roi_exit_table: str = ''               # -style ROI table JSON: {"10": 0, "5": 60, "0": 1440}

    # Kelly Criterion parameters (feature-flagged off-by-default)
    kelly_enabled: bool = False
    kelly_method: str = "returns"        # "returns" | "winrate_payoff" | "disabled"
    kelly_fractional: float = 0.25       # Quarter-Kelly default (industry standard)
    kelly_max_fraction: float = 0.50     # Hard cap prevents overbetting
    kelly_min_samples: int = 30          # Min trades before using Kelly estimate
    kelly_track_returns_window: int = 90 # Rolling window length for returns-based Kelly

    @property
    def auth_mode(self) -> str | None:
        if self.robinhood_mcp_command:
            return "stdio"
        elif self.robinhood_mcp_auth_header:
            return "sse"
        else:
            return None

    def __repr__(self) -> str:
        return f"Settings(symbols={self.symbols}, mode={self.auth_mode or 'none'})"

    def parse_strategy_allocations(self) -> dict[str, dict]:
        """Parse STRATEGY_ALLOCATIONS_JSON into {name: {symbols, weight}}."""
        if not self.strategy_allocations:
            return {}
        try:
            data = json.loads(self.strategy_allocations)
            logger.info("Parsed strategy allocations: %d strategies", len(data))
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse STRATEGY_ALLOCATIONS_JSON: %s", exc)
            return {}

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Return True if *symbol* passes whitelist and blacklist checks."""
        sym = symbol.strip().upper()
        # Blacklist always wins
        if sym in [s.upper() for s in self.blacklisted_symbols]:
            return False
        # Whitelist gate - empty means open market
        if self.symbol_whitelist and sym not in [s.upper() for s in self.symbol_whitelist]:
            return False
        return True


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _parse_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()] if value else []


def load_settings() -> Settings:
    """Load Settings from environment with sensible defaults."""
    symbols_str = _get("SYMBOLS", "AAPL,MSFT,NVDA")
    symbols = _parse_list(symbols_str)

    auth_header = _get("ROBINHOOD_MCP_AUTH_HEADER", "")

    return Settings(
        robinhood_mcp_url=_get("ROBINHOOD_MCP_URL"),
        robinhood_mcp_auth_header=auth_header,
        robinhood_mcp_command=_get("ROBINHOOD_MCP_COMMAND"),
        robinhood_mcp_args=_get("ROBINHOOD_MCP_ARGS"),
        polygon_api_key=_get("POLYGON_API_KEY"),
        databento_api_key=_get("DATABENTO_API_KEY"),
        symbols=symbols or ["AAPL", "MSFT", "NVDA"],
        cash=float(_get("CASH", "100000")),
        risk_per_trade=float(_get("RISK_PER_TRADE", "0.01")),
        max_daily_loss_pct=float(_get("MAX_DAILY_LOSS_PCT", "3.0")),
        engine_interval_minutes=int(_get("ENGINE_INTERVAL_MINUTES", "5")),
        sentiment_lookback_hours=int(_get("SENTIMENT_LOOKBACK_HOURS", "24")),
        log_level=_get("LOG_LEVEL", "DEBUG"),
        strategy_allocations=_get("STRATEGY_ALLOCATIONS_JSON", ""),
        min_buying_amount_usd=float(_get("MIN_BUYING_AMOUNT_USD", "1.0")),
        max_buying_amount_usd=float(_get("MAX_BUYING_AMOUNT_USD", "10000.0")),
        min_selling_amount_usd=float(_get("MIN_SELLING_AMOUNT_USD", "1.0")),
        max_selling_amount_usd=float(_get("MAX_SELLING_AMOUNT_USD", "50000.0")),
        symbol_exclusions=[s.strip() for s in _get("SYMBOL_EXCLUSIONS", "").split(",") if s.strip()],
        symbol_whitelist=_parse_list(_get("SYMBOL_WHITELIST", "")),
        blacklisted_symbols=[s.strip() for s in _get("BLACKLISTED_SYMBOLS", "GOOGL,TSLA").split(",") if s.strip()],
        kelly_enabled=_get("KELLY_ENABLED", "false").lower() == "true",
        kelly_method=_get("KELLY_METHOD", "returns"),
        kelly_fractional=float(_get("KELLY_FRACTIONAL", "0.25")),
        kelly_max_fraction=float(_get("KELLY_MAX_FRACTION", "0.50")),
        kelly_min_samples=int(_get("KELLY_MIN_SAMPLES", "30")),
        kelly_track_returns_window=int(_get("KELLY_TRACK_RETURNS_WINDOW", "90")),
    )


def setup_logging(level: str = "DEBUG") -> None:
    """Configure logging: rotating file (DEBUG) + console (INFO)."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    fh = handlers.RotatingFileHandler(
        logs_dir / "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.addHandler(fh)
    root.addHandler(ch)