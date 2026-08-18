"""Tests for bot.core.registry — register/get/KeyError behavior."""

import sys
import pytest
from bot.core.registry import Registry


class TestRegistry:
    def setup_method(self):
        self.reg = Registry("test")

    def test_register_and_get(self):
        self.reg.register("a", {"value": 1})
        self.reg.register("b", {"value": 2})
        assert self.reg.get("a") == {"value": 1}
        assert self.reg.get("b") == {"value": 2}

    def test_key_error_on_unknown(self):
        self.reg.register("only_one", "val")
        with pytest.raises(KeyError) as exc_info:
            self.reg.get("nonexistent")
        err_msg = str(exc_info.value)
        assert "nonexistent" in err_msg
        assert "only_one" in err_msg

    def test_names_returns_sorted_list(self):
        self.reg.register("zoo", 1)
        self.reg.register("apple", 2)
        self.reg.register("mango", 3)
        names = self.reg.names()
        assert names == ["apple", "mango", "zoo"]

    def test_all_returns_values_in_name_order(self):
        self.reg.register("b", 2)
        self.reg.register("a", 1)
        items = self.reg.items()
        assert items[0][0] == "a"
        assert items[0][1] == 1
        assert items[1][0] == "b"
        assert items[1][1] == 2

    def test_duplicate_registration_wins_last(self):
        self.reg.register("key", "first")
        self.reg.register("key", "second")
        assert self.reg.get("key") == "second"

    def test_repr_shows_counts(self):
        self.reg.register("one", 1)
        r = repr(self.reg)
        assert "test" in r
        assert "1" in r


def test_plugin_discovery_finds_ema_cross_rsi():
    """Verify auto-discovery finds our starter strategy plugin."""
    import importlib
    mod = importlib.import_module("bot.plugins.strategies.ema_cross_rsi")
    plugin = getattr(mod, "plugin", None)
    assert plugin is not None, "Plugin module must expose 'plugin' attribute"
    assert plugin.name == "ema_cross_rsi"


def test_datasource_plugins_exist():
    """Verify datasource plugins have proper contract."""
    from bot.core import DATASOURCES
    discover_result = __import__("bot.core.plugins", fromlist=["discover_all"]).discover_all()
    assert "yfinance" in DATASOURCES.names(), "YFinance source must be registered"
    yf_ds = DATASOURCES.get("yfinance")
    assert hasattr(yf_ds, "supports"), "Datasource must have supports() method"
    assert hasattr(yf_ds, "fetch_history"), "Datasource must have fetch_history()"


def test_sentiment_plugins_exist():
    """Verify sentiment plugins have proper contract."""
    from bot.core import SENTIMENT_SOURCES
    discover_result = __import__("bot.core.plugins", fromlist=["discover_all"]).discover_all()
    assert "stocktwits" in SENTIMENT_SOURCES.names(), "StockTwits source must be registered"
    st_plugin = SENTIMENT_SOURCES.get("stocktwits")
    assert hasattr(st_plugin, "name")
    assert hasattr(st_plugin, "fetch")
