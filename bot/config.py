"""Application settings and logging bootstrap."""

import logging
import os
import sys
from dataclasses import dataclass, field
from logging import handlers
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """All configuration from .env with sane defaults."""
    # Robinhood MCP
    robinhood_mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    robinhood_mcp_auth_header: str = ""
    robinhood_mcp_command: str = ""
    robinhood_mcp_args: str = ""

    # Trading scope
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    cash: float = 100_000
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 3.0
    engine_interval_minutes: int = 5
    sentiment_lookback_hours: int = 24
    log_level: str = "DEBUG"

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
        symbols=symbols or ["AAPL", "MSFT", "NVDA"],
        cash=float(_get("CASH", "100000")),
        risk_per_trade=float(_get("RISK_PER_TRADE", "0.01")),
        max_daily_loss_pct=float(_get("MAX_DAILY_LOSS_PCT", "3.0")),
        engine_interval_minutes=int(_get("ENGINE_INTERVAL_MINUTES", "5")),
        sentiment_lookback_hours=int(_get("SENTIMENT_LOOKBACK_HOURS", "24")),
        log_level=_get("LOG_LEVEL", "DEBUG"),
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
