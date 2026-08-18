"""Polygon.io REST API datasource — paid market data with tick-level granularity.

Free tier (delayed quotes, end-of-day bars) works without a paid plan. Higher
resolution (1m, 5m, tick) and real-time require a paid subscription.

API docs: https://polygon.io/docs/stocks/getting-started
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class PolygonSource:
    """Polygon.io REST datasource. Priority: 5 (between Robinhood MCP and yfinance)."""

    name = "polygon"
    priority = 5

    # Multipliers + timespans supported by /v2/aggs/ticker/{ticker}/range/{mult}/{tspan}/{from}/{to}
    SUPPORTED_INTERVALS = {
        "1m", "5m", "15m", "30m", "60m", "1h",
        "1d", "1wk", "1mo",
    }

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import of requests is allowed; we still import at top for fast-fail.
        import requests as _requests  # noqa: F401
        self._requests = _requests
        self._api_key = api_key or _resolve_api_key()

    @property
    def supports(self):
        return self.SUPPORTED_INTERVALS.__contains__

    def fetch_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        limit: int = 5_000,
    ) -> "Any":
        """Fetch OHLCV bars from Polygon.io ``/v2/aggs/ticker/{ticker}/range/...``.

        Parameters mirror the yfinance/robinhood plugin contract. ``start``/``end``
        may be ISO-8601 dates or relative offsets like ``-30D`` (handled like yfinance).
        Returns a ``pandas.DataFrame`` with columns ``Open/High/Low/Close/Volume``.
        Raises ``RuntimeError`` on auth/transport failure (DataHub falls through).
        """
        if not self._api_key:
            raise RuntimeError("POLYGON_API_KEY not configured")

        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required")

        mult, tspan = _normalize_interval(interval)
        from_date, to_date = _resolve_dates(start, end)

        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol.upper()}/range/{mult}/{tspan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apiKey": self._api_key,
        }

        try:
            resp = self._requests.get(url, params=params, timeout=15)
            if resp.status_code == 401:
                raise RuntimeError(f"Polygon auth failed (401): check POLYGON_API_KEY")
            if resp.status_code == 429:
                raise RuntimeError(f"Polygon rate-limit hit (429): {resp.text[:200]}")
            resp.raise_for_status()
            payload = resp.json()
        except self._requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Polygon HTTP error: {exc}") from exc

        results = payload.get("results") or []
        if not results:
            logger.warning("Polygon returned empty results for %s [%s]", symbol, interval)
            raise ValueError(f"No Polygon data for {symbol} [{interval}]")

        df = pd.DataFrame(results)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
        })
        df = df.set_index("timestamp")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        logger.info("Polygon fetched %d bars for %s [%s]", len(df), symbol, interval)
        return df


def _normalize_interval(interval: str) -> tuple[int, str]:
    """Map unified ``"5m"`` style to Polygon ``(multiplier, timespan)``."""
    mapping = {
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "60m": (60, "minute"),
        "1h": (1, "hour"),
        "1d": (1, "day"),
        "1wk": (1, "week"),
        "1mo": (1, "month"),
    }
    if interval not in mapping:
        raise ValueError(f"Polygon does not support interval '{interval}'")
    return mapping[interval]


def _resolve_dates(start: str | None, end: str | None) -> tuple[str, str]:
    """Resolve ISO-8601 dates or relative offsets to ``YYYY-MM-DD``."""
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
    """Read API key from env (lazy)."""
    import os
    return os.getenv("POLYGON_API_KEY", "")


plugin = PolygonSource()