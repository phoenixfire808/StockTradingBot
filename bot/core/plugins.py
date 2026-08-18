"""Auto-discovery: scan bot/plugins/{kind}/*.py and register each module's `plugin` attr."""

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGINS_DIR = Path(__file__).parent.parent / "plugins"
KINDS = ("strategies", "datasources", "sentiment_sources")


def _scan_kind(kind: str, registry) -> int:
    """Import every .py in bot/plugins/{kind}/ and register module.plugin."""
    dirpath = PLUGINS_DIR / kind
    if not dirpath.is_dir():
        logger.info("No plugins directory for %s", kind)
        return 0

    count = 0
    for mod_info in pkgutil.iter_modules([str(dirpath)]):
        if mod_info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"bot.plugins.{kind}.{mod_info.name}")
        except Exception:
            logger.exception("Failed to load plugin module bot.plugins.%s.%s", kind, mod_info.name)
            continue

        plugin = getattr(mod, "plugin", None)
        if plugin is None:
            logger.warning("Module bot.plugins.%s.%s has no 'plugin' attribute — skipping", kind, mod_info.name)
            continue

        name = getattr(plugin, "name", None)
        if not name:
            logger.warning("Plugin in bot.plugins.%s.%s missing 'name' attr — skipping", kind, mod_info.name)
            continue

        try:
            registry.register(name, plugin)
            count += 1
        except Exception:
            logger.exception("Failed to register plugin %s from %s", name, mod_info.name)

    return count


def discover_all() -> dict[str, int]:
    """Discover plugins across all kinds. Returns {kind: count_registered}."""
    from bot.core import DATASOURCES, SENTIMENT_SOURCES, STRATEGIES

    results = {}
    for kind, reg in [("strategies", STRATEGIES), ("datasources", DATASOURCES), ("sentiment_sources", SENTIMENT_SOURCES)]:
        results[kind] = _scan_kind(kind, reg)

    total = sum(results.values())
    logger.info("Discovered %d plugins — %s", total, "; ".join(f"{k}={v}" for k, v in results.items()))
    return results
