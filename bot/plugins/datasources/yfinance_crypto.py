"""YFinance crypto datasource — free 24/7 OHLCV for crypto pairs.

YFinance exposes crypto via the ``{symbol}-USD`` (or ``-EUR`` etc.) ticker
suffix, e.g. ``BTC-USD``, ``ETH-USD``. 24/7 coverage; granularity limited to
1m / 2m / 5m / 15m / 30m / 60m / 90m / 1h / 1d / 5d / 1wk / 1mo / 3mo.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class YFinanceCryptoSource:
    """YFinance-backed crypto datasource. Priority: 10 (fallback)."""

    name = "yfinance_crypto"
    priority = 10

    SUPPORTED_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"}

    @property
    def supports(self):
        return self.SUPPORTED_INTERVALS.__contains__

    @staticmethod
    def _normalize(symbol: str) -> str:
        """``BTC`` -> ``BTC-USD``; ``BTC/USD`` -> ``BTC-USD``; pass-through otherwise."""
        s = symbol.upper().replace("/", "-")
        if "-" not in s:
            s = f"{s}-USD"
        return s

    def fetch_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> "Any":
        """Fetch crypto OHLCV via yfinance.

        ``symbol`` may be passed as ``BTC``, ``BTC/USD``, or ``BTC-USD``; the
        underlying yfinance ticker always uses the dash form.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")

        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required")

        ticker = self._normalize(symbol)

        # Relative offsets like ``-7D`` are common across datasources.
        rel_match = re.match(r"-(\d+)D", str(start))
        if rel_match:
            days = int(rel_match.group(1))
            start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No crypto data for {symbol} [{interval}]")

        # Flatten MultiIndex columns (yfinance returns them when a single ticker has yfinance>=0.2.40)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        keep = [c for c in required if c in df.columns]
        if not keep:
            raise ValueError(f"yfinance returned no usable columns for {symbol}")

        logger.info("yfinance_crypto fetched %d bars for %s [%s]", len(df), symbol, interval)
        return df[keep]


plugin = YFinanceCryptoSource()