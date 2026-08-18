"""Multi-timeframe signal composition module.

Fetches data at multiple timeframes (1m/5m/15m/1h/1d), generates signals
on each using a base strategy, and aggregates into a single daily-aligned
signal series using voting / weighting / unanimous / any-long methods.

Used by bot/plugins/strategies/multi_timeframe.py plugin.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Standard timeframe definitions (yfinance-compatible intervals)
# weight = relative influence in weighted/majority aggregation
TIMEFRAMES: dict[str, dict[str, Any]] = {
    "1m":  {"weight": 0.5},
    "5m":  {"weight": 0.6},
    "15m": {"weight": 0.7},
    "30m": {"weight": 0.8},
    "1h":  {"weight": 0.9},
    "1d":  {"weight": 1.0},
}


class MultiTimeframeComposer:
    """Compose signals from a base strategy across multiple timeframes.

    Aggregation methods:
      - "majority_vote": weighted sum > 0 → long, < 0 → exit, else flat.
      - "weighted":      weighted sum must exceed threshold to trigger.
      - "unanimous":     all timeframes must agree to trigger.
      - "any_long":      any timeframe long → long; any exit → exit (exits win).
    """

    def __init__(
        self,
        base_strategy: Any,
        timeframes: list[str] | None = None,
        aggregation: str = "majority_vote",
    ) -> None:
        self.base_strategy = base_strategy
        self.timeframes = timeframes or ["5m", "1h", "1d"]
        self.aggregation = aggregation
        logger.info(
            "MultiTimeframeComposer init: strategy=%s timeframes=%s aggregation=%s",
            getattr(base_strategy, "name", "unknown"),
            self.timeframes,
            aggregation,
        )

    # ── data fetching ───────────────────────────────────────────────

    def _fetch_timeframe_data(
        self,
        symbol: str,
        start: str,
        end: str | None,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data for a specific timeframe via DataHub."""
        try:
            from bot.data import fetch_history

            df = fetch_history(symbol, start, end, interval=timeframe)
            if df is None or df.empty:
                logger.warning("No data for %s at %s timeframe", symbol, timeframe)
                return None
            logger.debug("Fetched %s %s: %d rows", symbol, timeframe, len(df))
            return df
        except Exception as exc:
            logger.warning("Failed to fetch %s %s: %s", symbol, timeframe, exc)
            return None

    @staticmethod
    def _extract_date_range(df: pd.DataFrame) -> tuple[str, str | None]:
        """Extract (start, end) date strings from a DataFrame index."""
        idx = df.index
        if not hasattr(idx, "min"):
            return "2020-01-01", None
        start_val = idx.min()
        end_val = idx.max()
        start_str = start_val.strftime("%Y-%m-%d") if hasattr(start_val, "strftime") else str(start_val)[:10]
        end_str = end_val.strftime("%Y-%m-%d") if hasattr(end_val, "strftime") else str(end_val)[:10]
        return start_str, end_str

    # ── signal resampling ───────────────────────────────────────────

    @staticmethod
    def _resample_to_daily(signals: pd.Series) -> pd.Series:
        """Resample intraday signals to daily by taking last value per day."""
        s = signals.copy()
        s.index = pd.to_datetime(s.index)
        # If already daily frequency, return as-is
        try:
            freq = s.index.freq
            if freq is not None and "D" in str(freq):
                return s.astype("int8")
        except Exception:
            pass
        daily = s.resample("D").last().fillna(0).astype("int8")
        return daily

    # ── aggregation ─────────────────────────────────────────────────

    def _aggregate(self, signals_dict: dict[str, pd.Series]) -> pd.Series:
        """Aggregate per-timeframe signals into a single daily series."""
        # Resample all to daily
        daily_signals: dict[str, pd.Series] = {}
        for tf, sig in signals_dict.items():
            daily_signals[tf] = self._resample_to_daily(sig)

        # Union of all daily indices
        all_indices = pd.DatetimeIndex(
            sorted(set().union(*(s.index for s in daily_signals.values())))
        )

        aligned = pd.DataFrame(index=all_indices)
        for tf, sig in daily_signals.items():
            aligned[tf] = sig.reindex(all_indices).fillna(0)

        weights = {
            tf: TIMEFRAMES.get(tf, {"weight": 1.0})["weight"]
            for tf in signals_dict
        }

        if self.aggregation == "majority_vote":
            weighted_sum = sum(aligned[tf] * weights[tf] for tf in signals_dict)
            result = weighted_sum.apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            ).astype("int8")

        elif self.aggregation == "weighted":
            weighted_sum = sum(aligned[tf] * weights[tf] for tf in signals_dict)
            threshold = 0.5 * sum(weights.values())
            result = weighted_sum.apply(
                lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
            ).astype("int8")

        elif self.aggregation == "unanimous":
            n_tfs = len(signals_dict)
            pos_count = (aligned > 0).sum(axis=1)
            neg_count = (aligned < 0).sum(axis=1)
            result = pd.Series(0, index=all_indices, dtype="int8")
            result[pos_count == n_tfs] = 1
            result[neg_count == n_tfs] = -1

        elif self.aggregation == "any_long":
            any_long = (aligned > 0).any(axis=1)
            any_exit = (aligned < 0).any(axis=1)
            result = pd.Series(0, index=all_indices, dtype="int8")
            result[any_long] = 1
            result[any_exit] = -1  # exits override longs

        else:
            logger.warning(
                "Unknown aggregation '%s' — defaulting to majority_vote",
                self.aggregation,
            )
            weighted_sum = sum(aligned[tf] * weights[tf] for tf in signals_dict)
            result = weighted_sum.apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            ).astype("int8")

        logger.info(
            "Aggregated %d timeframes (%s): %d long, %d exit, %d flat signals",
            len(signals_dict),
            self.aggregation,
            int((result == 1).sum()),
            int((result == -1).sum()),
            int((result == 0).sum()),
        )
        return result

    # ── public API ──────────────────────────────────────────────────

    def compose(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch data at each timeframe, generate signals, aggregate to daily.

        Returns a daily-frequency int8 Series (1=long, -1=exit, 0=flat).
        Returns empty Series if no timeframe data is available.
        """
        per_tf_signals: dict[str, pd.Series] = {}

        for tf in self.timeframes:
            df = self._fetch_timeframe_data(symbol, start, end, tf)
            if df is None:
                continue
            try:
                sig = self.base_strategy.generate_signals(df)
                per_tf_signals[tf] = sig
                logger.debug(
                    "Generated %s signals for %s %s: %d non-zero",
                    getattr(self.base_strategy, "name", "?"),
                    symbol,
                    tf,
                    int((sig != 0).sum()),
                )
            except Exception as exc:
                logger.warning(
                    "Signal generation failed for %s %s: %s",
                    symbol,
                    tf,
                    exc,
                )
                continue

        if not per_tf_signals:
            logger.error(
                "No timeframe data available for %s — returning flat signals",
                symbol,
            )
            return pd.Series(dtype="int8")

        return self._aggregate(per_tf_signals)

    def generate_from_daily(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
    ) -> pd.Series:
        """Generate aggregated signals from a daily DataFrame.

        - Always generates base signals on the provided daily df.
        - If symbol is provided, fetches additional intraday timeframes
          and aggregates using the configured method.
        - Result is reindexed to the input df's index for alignment.
        """
        # Generate base signals on the provided daily df
        base_signals = self.base_strategy.generate_signals(df)

        if symbol is None:
            logger.debug(
                "No symbol provided for multi-timeframe — using base daily signals only"
            )
            return base_signals

        # Determine date range from df
        start, end = self._extract_date_range(df)

        # Collect signals: daily from the passed df + intraday fetched
        per_tf: dict[str, pd.Series] = {"1d": base_signals}

        for tf in self.timeframes:
            if tf == "1d":
                continue
            tf_df = self._fetch_timeframe_data(symbol, start, end, tf)
            if tf_df is None:
                continue
            try:
                sig = self.base_strategy.generate_signals(tf_df)
                per_tf[tf] = sig
            except Exception as exc:
                logger.warning(
                    "Signal generation failed for %s %s: %s",
                    symbol,
                    tf,
                    exc,
                )

        if len(per_tf) == 1:
            # Only daily available — return base signals
            return base_signals

        # Aggregate
        aggregated = self._aggregate(per_tf)

        if aggregated.empty:
            return base_signals

        # Reindex aggregated daily signals to match input df index
        result = aggregated.reindex(df.index, method="ffill").fillna(0).astype("int8")

        logger.info(
            "Multi-timeframe generate_from_daily: %d long, %d exit, %d flat "
            "(base: %d long, %d exit)",
            int((result == 1).sum()),
            int((result == -1).sum()),
            int((result == 0).sum()),
            int((base_signals == 1).sum()),
            int((base_signals == -1).sum()),
        )
        return result


def compose_multi_timeframe(
    base_strategy: Any,
    symbol: str,
    timeframes: list[str] | None,
    start: str,
    end: str | None = None,
    aggregation: str = "majority_vote",
) -> pd.Series:
    """Convenience function: create composer and compose in one call."""
    composer = MultiTimeframeComposer(base_strategy, timeframes, aggregation)
    return composer.compose(symbol, start, end)


def aggregate_signals(
    signals_dict: dict[str, pd.Series],
    method: str = "majority_vote",
) -> pd.Series:
    """Standalone aggregation of pre-computed per-timeframe signals."""
    composer = MultiTimeframeComposer.__new__(MultiTimeframeComposer)
    composer.aggregation = method
    return composer._aggregate(signals_dict)
