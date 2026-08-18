"""Databento datasource — institutional-grade historical tick & bar data.

API docs: https://docs.databento.com/

The Databento Python client (``databento``) is preferred when installed. When
unavailable (e.g. for offline tests / CI) we fall back to the REST historical
endpoint via plain ``requests``.

Free tier (signup bonus) covers ~$25 of historical data — enough for
backtesting a single ticker over a few months.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DatabentoSource:
    """Databento historical bars. Priority: 4 (after Robinhood MCP, before Polygon/yfinance)."""

    name = "databento"
    priority = 4

    # Databento schema names — ``ohlcv-1m`` is the most common granularity.
    SUPPORTED_INTERVALS = {
        "1m": "ohlcv-1m",
        "5m": "ohlcv-5m",
        "15m": "ohlcv-15m",
        "30m": "ohlcv-30m",
        "1h": "ohlcv-1h",
        "1d": "ohlcv-1d",
        "1wk": "ohlcv-1w",
    }

    BASE_URL = "https://hist.databento.com/v0"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or _resolve_api_key()
        # Lazy import: we don't require the official ``databento`` SDK at install time.
        self._sdk = None
        if self._api_key:
            try:
                import databento  # type: ignore
                self._sdk = databento.Historical(api_key=self._api_key)
            except ImportError:
                logger.debug("databento SDK not installed; using REST fallback")
            except Exception as exc:
                logger.debug("Databento SDK init failed (%s); using REST fallback", exc)

    @property
    def supports(self):
        return self.SUPPORTED_INTERVALS.__contains__

    def fetch_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        dataset: str = "XNAS.ITCH",
    ) -> "Any":
        """Fetch OHLCV bars from Databento.

        ``dataset`` defaults to ``XNAS.ITCH`` (Nasdaq TotalView-ITCH) which is
        a common equity dataset. Other useful values: ``GLBX.MDP3`` (futures),
        ``DBEQ.BASIC`` (basic US equities).
        """
        if not self._api_key:
            raise RuntimeError("DATABENTO_API_KEY not configured")

        schema = self.SUPPORTED_INTERVALS.get(interval)
        if schema is None:
            raise ValueError(f"Databento does not support interval '{interval}'")

        # Resolve dates like the polygon plugin does.
        from_str, to_str = _resolve_dates(start, end)

        if self._sdk is not None:
            return self._fetch_via_sdk(symbol, dataset, schema, from_str, to_str)
        return self._fetch_via_rest(symbol, dataset, schema, from_str, to_str)

    def _fetch_via_sdk(self, symbol: str, dataset: str, schema: str, from_str: str, to_str: str):
        import pandas as pd
        data = self._sdk.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol.upper()],
            schema=schema,
            start=from_str,
            end=to_str,
        )
        df = data.to_df()
        # Databento returns lowercase column names.
        rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        df = df.rename(columns=rename)
        keep = [c for c in ["Open", "High", "Low", "Close", "volume"] if c in df.columns]
        df = df[keep]
        logger.info("Databento SDK fetched %d rows for %s [%s]", len(df), symbol, schema)
        return df

    def _fetch_via_rest(self, symbol: str, dataset: str, schema: str, from_str: str, to_str: str):
        import requests as _requests
        import pandas as pd
        url = f"{self.BASE_URL}/timeseries.get_range"
        params = {
            "dataset": dataset,
            "symbols": symbol.upper(),
            "schema": schema,
            "start": from_str,
            "end": to_str,
            "format": "json",
        }
        try:
            resp = _requests.get(url, params=params, auth=(self._api_key, ""), timeout=30)
            if resp.status_code == 401:
                raise RuntimeError("Databento auth failed (401): check DATABENTO_API_KEY")
            if resp.status_code == 403:
                raise RuntimeError(f"Databento forbidden (403): {resp.text[:200]}")
            resp.raise_for_status()
            payload = resp.json()
        except _requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Databento HTTP error: {exc}") from exc

        # The REST endpoint returns either a list of records or a path to a
        # downloaded file depending on schema. Normalize to a DataFrame.
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            df = pd.DataFrame(payload.get("records", payload))
        else:
            raise RuntimeError(f"Unexpected Databento response: {type(payload)}")

        if df.empty:
            raise ValueError(f"No Databento data for {symbol} [{schema}]")

        rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        df = df.rename(columns=rename)
        keep = [c for c in ["Open", "High", "Low", "Close", "volume"] if c in df.columns]
        df = df[keep]
        logger.info("Databento REST fetched %d rows for %s [%s]", len(df), symbol, schema)
        return df


def _resolve_dates(start: str | None, end: str | None) -> tuple[str, str]:
    import re
    import datetime as _dt

    def _abs(s: str | None) -> str:
        if not s:
            return (_dt.date.today() - _dt.timedelta(days=30)).isoformat()
        m = re.match(r"-(\d+)D", s)
        if m:
            days = int(m.group(1))
            return (_dt.date.today() - _dt.timedelta(days=days)).isoformat()
        return s

    return _abs(start), _abs(end) if end else _dt.date.today().isoformat()


def _resolve_api_key() -> str:
    import os
    return os.getenv("DATABENTO_API_KEY", "")


plugin = DatabentoSource()