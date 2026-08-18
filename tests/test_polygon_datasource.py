"""Tests for the Polygon.io datasource plugin."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.plugins.datasources.polygon import PolygonSource, _normalize_interval, _resolve_dates


class TestPolygonSourceContract:
    """Contract / structural tests that don't hit the network."""

    def test_plugin_attribute(self):
        from bot.plugins.datasources import polygon
        assert polygon.plugin is not None
        assert polygon.plugin.name == "polygon"
        assert polygon.plugin.priority == 5

    def test_supports_known_intervals(self):
        src = PolygonSource(api_key="dummy")
        for itv in ("1m", "5m", "15m", "1h", "1d", "1wk", "1mo"):
            assert src.supports(itv)

    def test_rejects_unknown_interval(self):
        src = PolygonSource(api_key="dummy")
        assert not src.supports("3mo")
        assert not src.supports("9m")

    def test_missing_api_key_raises(self):
        src = PolygonSource(api_key="")  # explicit empty
        # The constructor itself does NOT raise; fetch_history raises.
        with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="1d")

    def test_unsupported_interval_raises(self):
        src = PolygonSource(api_key="dummy")
        with pytest.raises(ValueError, match="Polygon does not support"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="3mo")


class TestPolygonHelpers:
    def test_normalize_interval_minutes(self):
        assert _normalize_interval("1m") == (1, "minute")
        assert _normalize_interval("5m") == (5, "minute")
        assert _normalize_interval("15m") == (15, "minute")

    def test_normalize_interval_daily_weekly(self):
        assert _normalize_interval("1d") == (1, "day")
        assert _normalize_interval("1wk") == (1, "week")
        assert _normalize_interval("1mo") == (1, "month")

    def test_normalize_interval_raises(self):
        with pytest.raises(ValueError):
            _normalize_interval("bogus")

    def test_resolve_dates_absolute(self):
        from_date, to_date = _resolve_dates("2026-01-01", "2026-02-01")
        assert from_date == "2026-01-01"
        assert to_date == "2026-02-01"

    def test_resolve_dates_relative(self):
        import datetime as _dt
        from_date, to_date = _resolve_dates("-7D", None)
        expected_start = (_dt.date.today() - _dt.timedelta(days=7)).isoformat()
        assert from_date == expected_start
        assert to_date == _dt.date.today().isoformat()

    def test_resolve_dates_none(self):
        import datetime as _dt
        from_date, to_date = _resolve_dates(None, None)
        expected_start = (_dt.date.today() - _dt.timedelta(days=30)).isoformat()
        assert from_date == expected_start
        assert to_date == _dt.date.today().isoformat()


class TestPolygonSourceNetworkFailure:
    """Validate graceful failure paths when the API is unreachable."""

    def test_auth_failure_raises(self, monkeypatch):
        import requests as _requests

        class _FakeResp:
            status_code = 401
            text = "Unauthorized"

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise _requests.exceptions.HTTPError("401")

        monkeypatch.setattr(
            _requests, "get",
            lambda url, params=None, timeout=None: _FakeResp(),
        )
        src = PolygonSource(api_key="bad-key")
        with pytest.raises(RuntimeError, match="auth failed"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="1d")


class TestPolygonDiscovered:
    def test_in_datasource_registry(self):
        from bot.core import DATASOURCES
        from bot.core.plugins import discover_all
        discover_all()
        assert "polygon" in DATASOURCES.names()
        ds = DATASOURCES.get("polygon")
        assert hasattr(ds, "fetch_history")
        assert hasattr(ds, "supports")