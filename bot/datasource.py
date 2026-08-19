"""Market data fetching via DataHub — priority-ordered datasource plugins + CSV cache."""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Raised when no datasource can provide requested bars."""
    pass


def _get_cache_path(symbol: str, interval: str) -> Path:
    return Path("data") / f"{symbol}_{interval}.csv"


def _read_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Serve cache if younger than 1 day
        if (datetime.now() - pd.Timestamp(df.index[-1])) < timedelta(days=1):
            logger.debug("Serving cached %s", path.name)
            return df
        logger.info("Cache expired for %s — will refresh", path.name)
    except Exception:
        logger.exception("Failed to read cache %s", path.name)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.debug("Saved cache %s (%d rows)", path.name, len(df))


def fetch_history(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV for *symbol* using registered datasources in priority order.
    Falls back through DATASOURCES registry; caches to CSV on first fetch.
    Raises DataError if every source fails.
    """
    from bot.core import DATASOURCES

    cache_path = _get_cache_path(symbol, interval)
    cached = _read_cached(cache_path)
    if cached is not None:
        return cached

    errors: list[tuple[str, Exception]] = []
    for ds_name, ds_plugin in DATASOURCES.items():
        if not ds_plugin.supports(interval):
            logger.debug("datasource '%s' does not support interval '%s'", ds_name, interval)
            continue
        try:
            logger.info("Fetching %s via datasource '%s'", symbol, ds_name)
            df = ds_plugin.fetch_history(symbol, start, end, interval)
            if df is None or df.empty:
                raise ValueError(f"Empty result from {ds_name}")
            _save_cache(df, cache_path)
            return df
        except Exception as exc:
            logger.warning("Datasource '%s' failed for %s: %s", ds_name, symbol, exc)
            errors.append((ds_name, exc))

    raise DataError(
        f"All datasource attempts failed for {symbol} [{interval}]: "
        f"{'; '.join(f'{n}: {e}' for n, e in errors)}"
    )


def fetch_latest_bars(
    symbol: str,
    lookback: int = 100,
    interval: str = "1d",
) -> pd.DataFrame:
    """Convenience: returns the last *lookback* rows from fetch_history."""
    df = fetch_history(symbol, start="-{}D".format(lookback), interval=interval)
    return df.tail(lookback).reset_index(drop=True)
