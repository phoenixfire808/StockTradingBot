from bot.core.registry import Registry
from bot.core.plugins import discover_all

STRATEGIES = Registry("strategy")
DATASOURCES = Registry("datasource")
SENTIMENT_SOURCES = Registry("sentiment_source")

__all__ = ["STRATEGIES", "DATASOURCES", "SENTIMENT_SOURCES", "discover_all"]
