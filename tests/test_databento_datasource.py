"""Tests for the Databento datasource plugin."""

from __future__ import annotations

import pytest

from bot.plugins.datasources.databento import DatabentoSource, _resolve_dates


class TestDatabentoSourceContract:
    def test_plugin_attribute(self):
        from bot.plugins.datasources import databento
        assert databento.plugin is not None
        assert databento.plugin.name == "databento"
        assert databento.plugin.priority == 4

    def test_supports_known_intervals(self):
        src = DatabentoSource(api_key="dummy")
        for itv in ("1m", "5m", "15m", "1h", "1d", "1wk"):
            assert src.supports(itv)

    def test_rejects_unknown_interval(self):
        src = DatabentoSource(api_key="dummy")
        assert not src.supports("3mo")
        assert not src.supports("9m")

    def test_missing_api_key_raises(self):
        src = DatabentoSource(api_key="")
        with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="1d")

    def test_unsupported_interval_raises(self):
        src = DatabentoSource(api_key="dummy")
        with pytest.raises(ValueError, match="Databento does not support"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="bogus")


class TestDatabentoHelpers:
    def test_resolve_dates_absolute(self):
        from_date, to_date = _resolve_dates("2026-01-01", "2026-02-01")
        assert from_date == "2026-01-01"
        assert to_date == "2026-02-01"

    def test_resolve_dates_relative(self):
        import datetime as _dt
        from_date, to_date = _resolve_dates("-90D", None)
        expected = (_dt.date.today() - _dt.timedelta(days=90)).isoformat()
        assert from_date == expected
        assert to_date == _dt.date.today().isoformat()


class TestDatabentoRestFailure:
    def test_auth_failure_via_rest(self, monkeypatch):
        import requests as _requests

        class _FakeResp:
            status_code = 401
            text = "Unauthorized"

            def raise_for_status(self):
                raise _requests.exceptions.HTTPError("401")

        monkeypatch.setattr(
            _requests, "get",
            lambda url, params=None, timeout=None, auth=None: _FakeResp(),
        )
        src = DatabentoSource(api_key="bad-key")
        # Force the SDK to be unavailable so we exercise the REST path.
        src._sdk = None
        with pytest.raises(RuntimeError, match="auth failed"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="1d")

    def test_empty_records_raises(self, monkeypatch):
        import requests as _requests

        class _FakeResp:
            status_code = 200
            text = "{}"

            def json(self):
                return {"records": []}

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            _requests, "get",
            lambda url, params=None, timeout=None, auth=None: _FakeResp(),
        )
        src = DatabentoSource(api_key="dummy")
        src._sdk = None
        with pytest.raises(ValueError, match="No Databento data"):
            src.fetch_history("AAPL", start="-30D", end=None, interval="1d")


class TestDatabentoDiscovered:
    def test_in_datasource_registry(self):
        from bot.core import DATASOURCES
        from bot.core.plugins import discover_all
        discover_all()
        assert "databento" in DATASOURCES.names()
        ds = DATASOURCES.get("databento")
        assert hasattr(ds, "fetch_history")
        assert hasattr(ds, "supports")


class TestDatabentoRestSuccess:
    """Validate the REST parser produces a usable OHLCV DataFrame."""

    def test_records_parsed_to_dataframe(self, monkeypatch):
        import requests as _requests

        sample = [
            {"ts_event": 1700000000000000000, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1500},
            {"ts_event": 1700000060000000000, "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1700},
        ]

        class _FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"records": sample}

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            _requests, "get",
            lambda url, params=None, timeout=None, auth=None: _FakeResp(),
        )
        src = DatabentoSource(api_key="dummy")
        src._sdk = None
        df = src.fetch_history("AAPL", start="-1D", end=None, interval="1m")
        assert hasattr(df, "columns")
        assert "Open" in df.columns
        assert "High" in df.columns
        assert len(df) == 2