"""Sector rotation model — uses sector ETF momentum to bias trading signals.

Computes momentum (rate-of-change) for SPDR sector ETFs, maps individual
stocks to their sector, and produces a bias score that can filter or
amplify strategy signals.

Used by bot/plugins/strategies/sector_rotation.py plugin.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# SPDR sector ETFs (ticker → sector name)
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Communication Services",
}

# Reverse lookup: sector name → ETF ticker
SECTOR_TO_ETF: dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# Default symbol → sector ETF mapping for common large-cap stocks
SYMBOL_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLC",
    "GOOG": "XLC", "META": "XLC", "AVGO": "XLK", "ORCL": "XLK",
    "CRM": "XLK", "AMD": "XLK", "INTC": "XLK", "QCOM": "XLK",
    "ADBE": "XLK", "CSCO": "XLK", "TXN": "XLK", "IBM": "XLK",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
    "MS": "XLF", "C": "XLF", "BLK": "XLF", "AXP": "XLF",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE",
    "EOG": "XLE", "PSX": "XLE", "MPC": "XLE",
    # Health Care
    "JNJ": "XLV", "UNH": "XLV", "LLY": "XLV", "PFE": "XLV",
    "ABBV": "XLV", "MRK": "XLV", "TMO": "XLV", "ABT": "XLV",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
    "NKE": "XLY", "LOW": "XLY", "SBUX": "XLY",
    # Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
    "COST": "XLP", "MDLZ": "XLP", "CL": "XLP",
    # Industrials
    "BA": "XLI", "CAT": "XLI", "GE": "XLI", "HON": "XLI",
    "UPS": "XLI", "RTX": "XLI", "LMT": "XLI",
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "AEP": "XLU",
    # Real Estate
    "AMT": "XLRE", "PLD": "XLRE", "SPG": "XLRE", "EQIX": "XLRE",
    # Materials
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "ECL": "XLB",
    # Communication Services
    "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC", "T": "XLC",
    "VZ": "XLC",
}


class SectorRotationModel:
    """Compute sector momentum and produce bias scores for individual stocks.

    The bias score ranges from -1.0 (strongly bearish sector) to +1.0
    (strongly bullish sector), used to filter or amplify strategy signals.
    """

    def __init__(
        self,
        lookback_days: int = 20,
        momentum_threshold: float = 0.02,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            lookback_days:       Period for momentum (rate-of-change) calculation.
            momentum_threshold:  Minimum |RoC| to classify as bullish/bearish.
                                 RoC between ±threshold → neutral (0 bias).
            sector_map:          Override the default SYMBOL_SECTOR_MAP.
        """
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.sector_map = sector_map if sector_map is not None else SYMBOL_SECTOR_MAP.copy()
        self._momentum_cache: dict[str, float] = {}
        self._cache_ts: datetime | None = None
        logger.info(
            "SectorRotationModel init: lookback=%d days threshold=%.2f%% sectors=%d",
            lookback_days,
            momentum_threshold * 100,
            len(SECTOR_ETFS),
        )

    # ── sector lookup ───────────────────────────────────────────────

    def get_sector_etf(self, symbol: str) -> str | None:
        """Return the sector ETF ticker for a given stock symbol."""
        etf = self.sector_map.get(symbol.upper())
        if etf is None:
            logger.debug("No sector mapping for %s — treating as market-neutral", symbol)
        return etf

    # ── momentum computation ─────────────────────────────────────────

    def _fetch_etf_data(
        self, etf: str, lookback: int
    ) -> pd.DataFrame | None:
        """Fetch recent OHLCV for a sector ETF."""
        try:
            from bot.data import fetch_latest_bars

            df = fetch_latest_bars(etf, lookback=lookback + 10, interval="1d")
            if df is None or df.empty:
                logger.warning("No data for sector ETF %s", etf)
                return None
            return df
        except Exception as exc:
            logger.warning("Failed to fetch sector ETF %s: %s", etf, exc)
            return None

    def compute_sector_momentum(
        self,
        etfs: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute rate-of-change momentum for all (or specified) sector ETFs.

        Returns {etf_ticker: momentum_pct} where momentum_pct is the
        percentage change over lookback_days.
        """
        target_etfs = etfs or list(SECTOR_ETFS.keys())
        end_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Use cache if fresh (< 1 hour old)
        if (
            self._cache_ts is not None
            and (end_cutoff - self._cache_ts) < timedelta(hours=1)
            and self._momentum_cache
        ):
            logger.debug("Using cached sector momentum (%d ETFs)", len(self._momentum_cache))
            return {
                etf: self._momentum_cache.get(etf, 0.0)
                for etf in target_etfs
            }

        lookback = max(self.lookback_days, 5)
        results: dict[str, float] = {}

        for etf in target_etfs:
            df = self._fetch_etf_data(etf, lookback)
            if df is None or len(df) < lookback:
                logger.warning(
                    "Insufficient data for %s momentum (got %d rows, need %d)",
                    etf,
                    len(df) if df is not None else 0,
                    lookback,
                )
                results[etf] = 0.0
                continue

            close = df["Close"] if "Close" in df else df.iloc[:, 4]
            try:
                close = pd.Series(close).astype(float)
            except Exception:
                pass

            if len(close) < lookback:
                results[etf] = 0.0
                continue

            current = float(close.iloc[-1])
            past = float(close.iloc[-lookback])
            if past <= 0:
                results[etf] = 0.0
                continue

            roc = (current - past) / past
            results[etf] = round(roc, 6)
            logger.debug(
                "Sector %s: RoC %.2f%% over %d days (current=%.2f past=%.2f)",
                etf,
                roc * 100,
                lookback,
                current,
                past,
            )

        # Update cache
        self._momentum_cache = results.copy()
        self._cache_ts = datetime.now(timezone.utc)

        # Log sector ranking
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            "Sector momentum ranking: %s",
            ", ".join(f"{e}={v*100:.1f}%" for e, v in ranked),
        )

        return results

    # ── bias computation ────────────────────────────────────────────

    def get_sector_bias(self, symbol: str) -> float:
        """Compute sector rotation bias for a stock.

        Returns a float in [-1.0, +1.0]:
          - Positive → sector momentum bullish → favor longs.
          - Negative → sector momentum bearish → block longs.
          - 0.0     → neutral or unmapped symbol.
        """
        etf = self.get_sector_etf(symbol)
        if etf is None:
            return 0.0

        momentum = self.compute_sector_momentum(etfs=[etf])
        roc = momentum.get(etf, 0.0)

        if roc > self.momentum_threshold:
            # Scale: 0..5% maps to 0..1, capped at 1.0
            bias = min(roc / 0.05, 1.0)
        elif roc < -self.momentum_threshold:
            # Scale: 0..-5% maps to 0..-1, capped at -1.0
            bias = max(roc / 0.05, -1.0)
        else:
            bias = 0.0

        logger.info(
            "Sector bias for %s (sector=%s, RoC=%.2f%%): bias=%.2f",
            symbol,
            etf,
            roc * 100,
            bias,
        )
        return round(bias, 4)

    def get_market_breadth(self) -> float:
        """Compute overall market breadth from sector momentum.

        Returns average momentum across all sectors, scaled to [-1, 1].
        Positive → broad market uptrend; negative → broad decline.
        """
        all_momentum = self.compute_sector_momentum()
        if not all_momentum:
            return 0.0

        avg_roc = sum(all_momentum.values()) / len(all_momentum)
        # Scale: ±5% average maps to ±1.0
        breadth = max(-1.0, min(1.0, avg_roc / 0.05))
        logger.info(
            "Market breadth: avg_sector_roc=%.2f%% → breadth=%.2f",
            avg_roc * 100,
            breadth,
        )
        return round(breadth, 4)


def bias_signals(
    signals: pd.Series,
    bias: float,
    block_threshold: float = -0.3,
    boost_threshold: float = 0.3,
) -> pd.Series:
    """Modify a signal series based on a sector bias score.

    - bias < block_threshold → block new longs (set 1 → 0), keep exits.
    - bias > boost_threshold → keep longs, allow additional longs where
      base signal was 0 and short-term momentum is positive.
    - Otherwise: pass through unchanged.

    Exit signals (-1) are always preserved regardless of bias.
    """
    result = signals.copy()

    if bias <= block_threshold:
        # Block new longs in bearish sector
        blocked = (result == 1)
        result[blocked] = 0
        logger.info(
            "Sector bias %.2f ≤ %.2f → blocked %d long signals",
            bias,
            block_threshold,
            int(blocked.sum()),
        )
    elif bias >= boost_threshold:
        # Boost: keep existing longs (no change needed, they're already 1)
        logger.debug(
            "Sector bias %.2f ≥ %.2f → maintaining %d long signals",
            bias,
            boost_threshold,
            int((result == 1).sum()),
        )

    return result.astype("int8")


def compute_sector_momentum(
    lookback_days: int = 20,
    etfs: list[str] | None = None,
) -> dict[str, float]:
    """Convenience function: compute sector momentum without a class instance."""
    model = SectorRotationModel(lookback_days=lookback_days)
    return model.compute_sector_momentum(etfs=etfs)
