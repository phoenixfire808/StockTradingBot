"""Multi-Timeframe Strategy Plugin — composes a base strategy across multiple
timeframes (1m/5m/15m/1h/1d) and aggregates signals.

Delegates to bot/multi_timeframe.py MultiTimeframeComposer for the heavy
lifting (data fetching, resampling, aggregation).
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class MultiTimeframeStrategy(Strategy):
    """Composes a base strategy across multiple timeframes.

    Args:
        base_strategy_name: Registered strategy to run on each timeframe.
        timeframes:         List of intervals to fetch (e.g. ["5m","1h","1d"]).
        aggregation:        How to combine: majority_vote, weighted, unanimous, any_long.
    """

    name = "multi_timeframe"
    params: dict[str, Any] = {}

    def __init__(
        self,
        base_strategy_name: str = "ema_cross_rsi",
        timeframes: list[str] | None = None,
        aggregation: str = "majority_vote",
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit: float = 75.0,
        multi_timeframe: bool = True,
        vol_adjusted: bool = False,
        vol_lookback: int = 20,
    ) -> None:
        self.base_strategy_name = base_strategy_name
        self.timeframes = timeframes or ["5m", "1h", "1d"]
        self.aggregation = aggregation
        self.params = {
            "base_strategy_name": base_strategy_name,
            "timeframes": self.timeframes,
            "aggregation": aggregation,
            "fast": fast,
            "slow": slow,
            "rsi_period": rsi_period,
            "rsi_entry_max": rsi_entry_max,
            "rsi_exit": rsi_exit,
            "multi_timeframe": multi_timeframe,
            "vol_adjusted": vol_adjusted,
            "vol_lookback": vol_lookback,
        }
        self._base_strategy = None
        self._composer = None
        self._vol_adjusted = vol_adjusted
        self._vol_lookback = vol_lookback
        logger.info(
            "MultiTimeframeStrategy init: base=%s timeframes=%s aggregation=%s",
            base_strategy_name,
            self.timeframes,
            aggregation,
        )


    def _compute_volatility_weights(
        self,
        timeframe_signals: dict[str, pd.Series],
        dataframe: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute volatility-adjusted weights for each timeframe.

        Lower-volatility timeframes receive higher signal weight (more reliable
        signals). Uses rolling volatility of Close resampled per timeframe.

        Args:
            timeframe_signals: Dict mapping timeframe names to signal Series.
            dataframe:         OHLCV DataFrame (must contain "Close" column).

        Returns:
            Dict mapping timeframe name to normalised weight.
        """
        close = dataframe["Close"]
        raw_weights: dict[str, float] = {}

        tf_to_bars: dict[str, int] = {
            "1m": 60 * 7,
            "5m": 12 * 7,
            "15m": 4 * 7,
            "30m": 2 * 7,
            "1h": 7,
            "240min": 7,
            "4h": 4,
            "1d": 20,
            "1w": 5,
            "1M": 3,
        }

        for tf_name in timeframe_signals:
            n_bars = tf_to_bars.get(tf_name, 20)
            try:
                # Map custom timeframe names to pandas-compatible offsets
                _tf_offset_map = {
                    "1m": "1min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "1h": "1h", "240min": "4h",
                    "4h": "4h", "1d": "1D", "1w": "1W", "1M": "ME",
                }
                _offset = _tf_offset_map.get(tf_name, None)
                if _offset is None:
                    raw_weights[tf_name] = 1.0
                    continue
                resampled = close.resample(_offset).agg(["last"])
                if len(resampled) < 2:
                    raw_weights[tf_name] = 1.0
                    continue
                returns = resampled["last"].pct_change().dropna()
                lookback = min(n_bars, max(2, len(returns) - 1))
                rolling_vol = returns.rolling(window=lookback, min_periods=2).std()
                latest_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else returns.std()
                if pd.isna(latest_vol) or latest_vol <= 0:
                    latest_vol = 1e-6
                raw_weights[tf_name] = 1.0 / latest_vol
            except Exception:
                raw_weights[tf_name] = 1.0

        total = sum(raw_weights.values())
        if total <= 0:
            return {k: 1.0 / len(raw_weights) for k in raw_weights}
        return {k: v / total for k, v in raw_weights.items()}


    def _get_base_strategy(self) -> Strategy:
        """Lazily fetch the wrapped base strategy."""
        if self._base_strategy is not None:
            return self._base_strategy
        try:
            from bot.core import STRATEGIES
            from bot.core.plugins import discover_all

            discover_all()
            base = STRATEGIES.get(self.base_strategy_name)
            base_params = {
                "fast": self.params.get("fast", 9),
                "slow": self.params.get("slow", 21),
                "rsi_period": self.params.get("rsi_period", 14),
                "rsi_entry_max": self.params.get("rsi_entry_max", 70.0),
                "rsi_exit": self.params.get("rsi_exit", 75.0),
            }
            self._base_strategy = type(base)(**base_params)
        except Exception as exc:
            logger.warning(
                "Failed to load base '%s': %s — using EmaCrossRsi",
                self.base_strategy_name,
                exc,
            )
            from bot.plugins.strategies.ema_cross_rsi import EmaCrossRsi

            self._base_strategy = EmaCrossRsi(
                fast=self.params.get("fast", 9),
                slow=self.params.get("slow", 21),
                rsi_period=self.params.get("rsi_period", 14),
                rsi_entry_max=self.params.get("rsi_entry_max", 70.0),
                rsi_exit=self.params.get("rsi_exit", 75.0),
            )
        return self._base_strategy

    def _get_composer(self):
        """Lazily create the MultiTimeframeComposer."""
        if self._composer is not None:
            return self._composer
        from bot.multi_timeframe import MultiTimeframeComposer

        base = self._get_base_strategy()
        self._composer = MultiTimeframeComposer(
            base_strategy=base,
            timeframes=self.timeframes,
            aggregation=self.aggregation,
        )
        return self._composer

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using multi-timeframe composition.

        If df.attrs contains a "symbol" key, fetches additional intraday
        timeframes and aggregates. Otherwise, runs the base strategy on
        the provided daily DataFrame only.

        When ``self._vol_adjusted`` is True, signal weights from the
        aggregated result are re-weighted inversely to each timeframe's
        realised volatility so that quieter charts dominate the consensus.
        """
        symbol = getattr(df, "attrs", {}).get("symbol", None)
        composer = self._get_composer()

        try:
            if symbol is not None:
                logger.info("Multi-timeframe composition with symbol=%s", symbol)
                result = composer.generate_from_daily(df, symbol=symbol)
            else:
                logger.debug("No symbol context — using base strategy on daily df only")
                result = composer.generate_from_daily(df, symbol=None)

            # Volatility-adjusted post-processing when enabled
            if self._vol_adjusted and symbol is not None:
                try:
                    start, end = composer._extract_date_range(df)
                    base = self._get_base_strategy()
                    per_tf_raw: dict[str, pd.Series] = {"1d": base.generate_signals(df)}
                    for tf in self.timeframes:
                        if tf == "1d":
                            continue
                        tf_df = composer._fetch_timeframe_data(symbol, start, end, tf)
                        if tf_df is not None:
                            per_tf_raw[tf] = base.generate_signals(tf_df)
                    if len(per_tf_raw) > 1:
                        vol_weights = self._compute_volatility_weights(per_tf_raw, df)
                        vol_sig = MultiTimeframeStrategy.aggregate_signals(
                            per_tf_raw,
                            weights=vol_weights,
                            default_aggregation=self.aggregation,
                        )
                        vol_sig = (
                            vol_sig.reindex(df.index, method="ffill")
                            .fillna(0)
                            .astype("int8")
                        )
                        logger.info(
                            "Volatility-adjusted multi-timeframe: weights=%s",
                            vol_weights,
                        )
                        return vol_sig
                except Exception as exc:
                    logger.warning(
                        "Volatility adjustment failed, using standard result: %s", exc
                    )

            return result
        except Exception as exc:
            logger.error(
                "Multi-timeframe composition failed: %s -- falling back to base strategy",
                exc,
            )
            base = self._get_base_strategy()
            return base.generate_signals(df)


    @staticmethod
    def aggregate_signals(
        timeframe_signals: dict[str, pd.Series],
        weights: dict[str, float] | None = None,
        default_aggregation: str = "majority_vote",
    ) -> pd.Series:
        """Aggregate pre-computed per-timeframe signals into a single series.

        Inspired by ML4T cross-frequency analysis and Microsoft Qlib signal
        fusion patterns: align all signals to the highest-frequency index,
        apply configurable weighting, then produce a consensus signal.

        Args:
            timeframe_signals:   Dict mapping timeframe name -> pd.Series[int8].
                                 Signal values: 1 (long), -1 (exit/short), 0 (flat).
            weights:             Optional explicit {timeframe_name: weight}.
                                 If omitted, higher timeframes (1d > 1h > 5m)
                                 get boosted via an internal priority scheme.
            default_aggregation: One of "majority_vote", "weighted", "unanimous",
                                 "any_long". Defaults to "majority_vote".

        Returns:
            pd.Series[int8] aligned to the union of all input indices.
        """
        if not timeframe_signals:
            return pd.Series(dtype="int8")

        # Timeframe priority for default weighting (higher = longer horizon)
        tf_priority: dict[str, int] = {
            "1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4,
            "240min": 5, "4h": 6, "1d": 7, "1w": 8, "1M": 9,
        }

        # Default weights: boost longer timeframes
        default_w: dict[str, float] = {}
        for tf_name in timeframe_signals:
            priority = tf_priority.get(tf_name, 5)
            default_w[tf_name] = 0.3 + 0.07 * priority

        use_weights = weights if weights is not None else default_w

        # Normalise weights to sum to 1
        w_total = sum(use_weights.values())
        if w_total <= 0:
            norm_weights = {k: 1.0 / len(use_weights) for k in use_weights}
        else:
            norm_weights = {k: v / w_total for k, v in use_weights.items()}

        # Union of all datetime indices
        all_indices: set[pd.Timestamp] = set()
        for sig in timeframe_signals.values():
            idx = pd.to_datetime(sig.index)
            all_indices.update(idx)
        unified_idx = pd.DatetimeIndex(sorted(all_indices)).tz_localize(None)

        # Build aligned DataFrame
        aligned = pd.DataFrame(index=unified_idx)
        for tf_name, sig in timeframe_signals.items():
            s = pd.to_numeric(pd.Series(sig, index=pd.to_datetime(sig.index)))
            aligned[tf_name] = s.reindex(unified_idx).fillna(0)

        weighted_sum = sum(
            aligned[col] * norm_weights.get(col, 0.0)
            for col in aligned.columns
        )

        if default_aggregation == "majority_vote":
            result = weighted_sum.apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            ).astype("int8")

        elif default_aggregation == "weighted":
            threshold = 0.5 * sum(norm_weights.values())
            result = weighted_sum.apply(
                lambda x: 1 if x > threshold
                else (-1 if x < -threshold else 0)
            ).astype("int8")

        elif default_aggregation == "unanimous":
            result = pd.Series(0, index=unified_idx, dtype="int8")
            pos_count = (aligned > 0).sum(axis=1)
            neg_count = (aligned < 0).sum(axis=1)
            n_tfs = len(aligned.columns)
            result[pos_count == n_tfs] = 1
            result[neg_count == n_tfs] = -1

        elif default_aggregation == "any_long":
            any_long = (aligned > 0).any(axis=1)
            any_exit = (aligned < 0).any(axis=1)
            result = pd.Series(0, index=unified_idx, dtype="int8")
            result[any_long] = 1
            result[any_exit] = -1

        else:
            result = weighted_sum.apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            ).astype("int8")

        logger.info(
            "Standalone aggregate_signals: %d timeframes (%s): "
            "%d long, %d exit, %d flat",
            len(timeframe_signals),
            default_aggregation,
            int((result == 1).sum()),
            int((result == -1).sum()),
            int((result == 0).sum()),
        )
        return result


# Plugin handle for auto-discovery
plugin = MultiTimeframeStrategy()
