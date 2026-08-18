"""YFinance datasource plugin — free OHLCV via yahoo finance API."""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class YFinanceSource:
    """Datasource plugin using yfinance for historical bars."""

    name = "yfinance"
    priority = 10  # falls through from robinhood_mcp (priority 1)

    SUPPORTED_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}

    @property
    def supports(self):
        return self.SUPPORTED_INTERVALS.__contains__

    def fetch_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ):
        """Fetch OHLCV bars via yfinance.

        Handles intraday chunking to work around request limits:
        - 1m data: max 30 days lookback, split into 7-day windows per request
        - 2m-60m data: max 60 days lookback
        - 1d+: no special chunking needed
        """
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")

        # Parse start/end if they are relative (e.g., "-30D")
        import re
        rel_match = re.match(r"-(\d+)D", str(start))
        if rel_match:
            days = int(rel_match.group(1))
            start_dt = datetime.utcnow().replace(microsecond=0)
            start = (start_dt.replace(hour=9, minute=30, second=0) - __import__('datetime').timedelta(days=days)).strftime('%Y-%m-%d')

        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

        if df is None or df.empty:
            logger.warning("yfinance returned empty result for %s [%s]", symbol, interval)
            raise ValueError(f"No data for {symbol} [{interval}]")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, __import__('pandas').MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        existing = [c for c in required if c in df.columns]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("yfinance missing columns for %s: %s", symbol, missing)

        logger.info("yfinance fetched %d rows for %s [%s]", len(df), symbol, interval)
        return df[existing] if missing else df
