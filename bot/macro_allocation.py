"""Macro regime-aware asset allocation framework.

Institutional-style overlay that classifies the current macro environment
and adjusts portfolio targets accordingly.  Inspired by ML4T regime
detection and Qlib expression-based factor thinking.

Regime classification uses three indicators derivable from standard OHLCV
data (no external API keys beyond price history):

    1.  VIX proxy       – annualised realised volatility from recent returns
    2.  Yield-curve     – short-term / long-term momentum ratio (trend slope)
    3.  Growth proxy    – multi-horizon return trend steepness

Output regimes: ``"risk-on"``, ``"neutral"``, ``"risk-off"``.

Weight maps differ per regime; rebalance signals fire when actual weights
drift past a configurable threshold.

No external dependencies beyond ``pandas`` / ``numpy``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Enums & Data Classes ────────────────────────────────────────────────


class Regime(str, Enum):
    RISK_ON = "risk-on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk-off"


@dataclass(frozen=True)
class _VolRegimeRules:
    """Thresholds for the volatility (VIX-proxy) component."""

    low_thresh: float = 0.18       # < 18% annual vol → calm
    high_thresh: float = 0.35      # >= 35% annual vol → stressed


@dataclass(frozen=True)
class _YieldCurveRules:
    """Thresholds for the yield-curve slope proxy."""

    bull_thresh: float = 0.97      # short > long * 0.97 → bullish slope
    bear_thresh: float = 0.92      # short < long * 0.92 → inverted / bearish


@dataclass(frozen=True)
class _GrowthRules:
    """Thresholds for the growth proxy."""

    strong_thresh: float = 0.06    # 60d return >= 6% → expansion
    weak_thresh: float = -0.06     # 60d return <= -6% → contraction


@dataclass
class MacroConfig:
    """Tunable parameters for the macro regime engine.

    All defaults are conservative (i.e. they tend toward ``"neutral"``).
    """

    vol_rules: _VolRegimeRules = field(default_factory=_VolRegimeRules)
    yield_curve_rules: _YieldCurveRules = field(
        default_factory=_YieldCurveRules,
    )
    growth_rules: _GrowthRules = field(default_factory=_GrowthRules)

    # Momentum look-back windows (trading days)
    short_window: int = 20
    mid_window: int = 60
    long_window: int = 126

    # Rebalance drift tolerance
    drift_tolerance: float = 0.05  # 5 % absolute drift → rebalance

    # Minimum data points before any classification is issued
    min_obs: int = 63  # at least ~3 months


# ── Indicator Builders ─────────────────────────────────────────────────


def _annualised_vol(returns: pd.Series) -> float:
    """Annualised realised volatility from daily return Series."""
    if len(returns) < 10:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(252))


def _momentum(close: pd.Series, window: int) -> float:
    """Close-to-close return over *window* trading days."""
    if len(close) < window + 1:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-(window + 1)] - 1)


def _yield_curve_slope(close: pd.Series) -> float:
    """Slope proxy: short-window momentum / long-window momentum.

    When short-term momentum is weaker than long-term (ratio < 1) the
    curve is effectively flattening or inverting → credit to risk-off.
    """
    short_mom = _momentum(close, 20)
    long_mom = _momentum(close, 126)
    if np.isnan(short_mom) or np.isnan(long_mom):
        return np.nan
    # Guard against division-by-zero
    denom = abs(long_mom) if abs(long_mom) > 1e-6 else long_mom
    return short_mom / denom


def _growth_proxy(close: pd.Series) -> float:
    """Return proxy for economic growth: 60-day momentum minus 126-day."""
    m60 = _momentum(close, 60)
    m126 = _momentum(close, 126)
    if np.isnan(m60) or np.isnan(m126):
        return np.nan
    return float(m60 - m126)


# ── Regime Classifier ─────────────────────────────────────────────────


def classify_regime(
    df: pd.DataFrame,
    config: MacroConfig | None = None,
    symbol: str | None = None,
) -> Regime:
    """Classify the current macro regime from price data.

    Parameters
    ----------
    df :
        DataFrame with at least ``Close`` column, indexed by datetime,
        sorted ascending.  Should contain ≥ ``config.min_obs`` rows.
    config :
        Tuning parameters.  Defaults to conservative settings.
    symbol :
        Optional label for logging.

    Returns
    -------
    Regime
        One of ``Regime.RISK_ON``, ``Regime.NEUTRAL``, ``Regime.RISK_OFF``.
    """
    cfg = config or MacroConfig()
    sym = symbol or "<unknown>"

    close = df["Close"] if "Close" in df else df.iloc[:, 4]

    if len(close) < cfg.min_obs:
        logger.debug(
            "%s: insufficient data (%d obs < %d); returning NEUTRAL",
            sym,
            len(close),
            cfg.min_obs,
        )
        return Regime.NEUTRAL

    returns = close.pct_change().dropna()

    # Component scores: -1 (bearish) / 0 (neutral) / +1 (bullish)
    vix_ann = _annualised_vol(returns)
    if np.isnan(vix_ann):
        vol_score = 0
    elif vix_ann < cfg.vol_rules.low_thresh:
        vol_score = 1
    elif vix_ann >= cfg.vol_rules.high_thresh:
        vol_score = -1
    else:
        vol_score = 0  # middle band → neutral vote

    ycs = _yield_curve_slope(close)
    if np.isnan(ycs):
        yc_score = 0
    elif ycs >= cfg.yield_curve_rules.bull_thresh:
        yc_score = 1
    elif ycs <= cfg.yield_curve_rules.bear_thresh:
        yc_score = -1
    else:
        yc_score = 0

    gp = _growth_proxy(close)
    if np.isnan(gp):
        growth_score = 0
    elif gp >= cfg.growth_rules.strong_thresh:
        growth_score = 1
    elif gp <= cfg.growth_rules.weak_thresh:
        growth_score = -1
    else:
        growth_score = 0

    total = vol_score + yc_score + growth_score

    if total >= 2:
        regime = Regime.RISK_ON
    elif total <= -2:
        regime = Regime.RISK_OFF
    else:
        regime = Regime.NEUTRAL

    logger.debug(
        "%s: regime=%s (vol=%.2f→%d, ycs=%.3f→%d, gpr=%.3f→%d)",
        sym,
        regime.value,
        vix_ann if not np.isnan(vix_ann) else 0,
        vol_score,
        ycs if not np.isnan(ycs) else 0,
        yc_score,
        gp if not np.isnan(gp) else 0,
        growth_score,
    )
    return regime


# ── Sector Group Mapping ───────────────────────────────────────────────


#: Default mapping of ticker-prefix groups to sector labels.
#: Extend or override at runtime to match your universe.
_SECTOR_GROUPS: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "NVDA": "tech", "AMD": "tech", "AVGO": "tech", "QCOM": "tech",
    "INTC": "tech", "CRM": "tech", "ORCL": "tech",
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "BLK": "financials", "AXP": "financials",
    "WFC": "financials",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "AMZN": "consumer", "HD": "consumer", "MCD": "consumer",
    "NKE": "consumer", "SBUX": "consumer", "TGT": "consumer",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
    "CAT": "industrials", "HON": "industrials", "UNP": "industrials",
    "RTX": "industrials",
    "VALE": "materials", "NEM": "materials",
}


def assign_sectors(symbols: list[str]) -> dict[str, str]:
    """Map tickers → sector group names.

    Unknown tickers land in ``"other"``.
    """
    result: dict[str, str] = {}
    for sym in symbols:
        upper = sym.upper()
        matched: str | None = None
        for prefix, sector in _SECTOR_GROUPS.items():
            if upper.startswith(prefix):
                matched = sector
                break
        result[sym] = matched or "other"
    return result


# ── Target Weight Maps ────────────────────────────────────────────────


#: Per-regime base allocation to equity vs cash-equivalent bucket.
_EQUIFY_BUCKETS: dict[Regime, tuple[float, float]] = {
    Regime.RISK_ON:   (0.85, 0.15),
    Regime.NEUTRAL:   (0.60, 0.40),
    Regime.RISK_OFF:  (0.30, 0.70),
}


def get_target_weights(
    regime: Regime,
    symbols: list[str],
    config: MacroConfig | None = None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Return target weights for each symbol under *regime*.

    Algorithm
    ---------
    1. Assign each symbol to a sector group.
    2. Allocate the equity bucket among sectors using regime-specific tilts.
    3. Within each sector, distribute equally among its members.
    4. Cash-equivalent bucket gets zero weight per individual equity symbol
       (it would be allocated to an ETF proxy outside this module).

    Parameters
    ----------
    regime :
        Classified macro regime.
    symbols :
        Equity symbols to allocate across.
    config :
        Unused here but accepted for API symmetry.
    sector_map :
        Pre-computed sector map from ``assign_sectors()``.

    Returns
    -------
    dict[str, float]
        Normalised weights summing to the equity fraction of the regime
        bucket (rest goes to cash proxy).
    """
    eq_weight, cash_weight = _EQUIFY_BUCKETS[regime]

    sectors = sector_map or assign_sectors(symbols)
    sector_groups: dict[str, list[str]] = {}
    for sym, sector in sectors.items():
        sector_groups.setdefault(sector, []).append(sym)

    # Regime-specific sector tilts (relative multipliers)
    _tilts = {
        Regime.RISK_ON:   {"tech": 1.6, "consumer": 1.3, "financials": 1.2,
                           "industrials": 1.1, "healthcare": 0.9, "energy": 1.0,
                           "materials": 0.8, "utilities": 0.5, "consumer_staples": 0.6,
                           "other": 1.0},
        Regime.NEUTRAL:   {"tech": 1.2, "consumer": 1.0, "financials": 1.0,
                           "industrials": 1.0, "healthcare": 1.2, "energy": 0.8,
                           "materials": 0.7, "utilities": 1.0, "consumer_staples": 1.1,
                           "other": 1.0},
        Regime.RISK_OFF:  {"tech": 0.7, "consumer": 0.6, "financials": 0.5,
                           "industrials": 0.5, "healthcare": 1.3, "energy": 0.7,
                           "materials": 0.6, "utilities": 1.2, "consumer_staples": 1.5,
                           "other": 0.8},
    }
    tilts = _tilts.get(regime, _tilts[Regime.NEUTRAL])

    # Compute weighted score per sector
    sector_scores: dict[str, float] = {}
    for sector, members in sector_groups.items():
        tilt = tilts.get(sector, 1.0)
        count_weight = len(members) ** 0.5  # sqrt(N) to penalise concentration
        sector_scores[sector] = tilt * count_weight

    total_score = sum(sector_scores.values()) or 1.0
    weights: dict[str, float] = {}

    for sector, members in sector_groups.items():
        sector_alloc = eq_weight * (sector_scores[sector] / total_score)
        per_member = sector_alloc / len(members) if members else 0.0
        for sym in members:
            weights[sym] = round(per_member, 8)

    return weights


# ── Rebalance Signal Generator ────────────────────────────────────────


def generate_signals(
    current_alloc: dict[str, float],
    target_alloc: dict[str, float],
    config: MacroConfig | None = None,
) -> dict[str, int]:
    """Generate rebalance trade signals based on weight drift.

    Compares each symbol's current portfolio weight against the target
    weight prescribed by the regime model.  Signals:

    * ``1``  – buy / increase position (current < target by more than tolerance)
    * ``-1`` – sell / decrease position (current > target by more than tolerance)
    * ``0``  – hold (drift within tolerance)

    Parameters
    ----------
    current_alloc :
        Current weights, ideally normalised to sum ≤ 1.0.
    target_alloc :
        Target weights from ``get_target_weights()``.
    config :
        Drift-tolerance parameter.

    Returns
    -------
    dict[str, int]
        Symbol → signal (-1, 0, 1).
    """
    cfg = config or MacroConfig()
    tolerance = cfg.drift_tolerance

    all_symbols = set(current_alloc) | set(target_alloc)
    signals: dict[str, int] = {}

    for sym in all_symbols:
        curr = current_alloc.get(sym, 0.0)
        tgt = target_alloc.get(sym, 0.0)
        drift = tgt - curr

        if drift > tolerance:
            signals[sym] = 1          # accumulate
        elif drift < -tolerance:
            signals[sym] = -1         # trim
        else:
            signals[sym] = 0          # hold

    return signals


# ── Convenience Factory ────────────────────────────────────────────────


class MacroAllocator:
    """Facade combining regime classification, weight targeting, and
    rebalance signal generation.

    Usage::

        alloc = MacroAllocator()
        regime = alloc.classify_regime(df)
        targets = alloc.get_target_weights(regime, symbols)
        signals = alloc.generate_signals(current, targets)
    """

    def __init__(self, config: MacroConfig | None = None) -> None:
        self.config = config or MacroConfig()

    def classify_regime(self, df: pd.DataFrame, symbol: str | None = None) -> Regime:
        """Alias for module-level :func:`classify_regime`."""
        return classify_regime(df, self.config, symbol)

    def get_target_weights(
        self,
        regime: Regime,
        symbols: list[str],
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Alias for module-level :func:`get_target_weights`."""
        return get_target_weights(regime, symbols, self.config, sector_map)

    def generate_signals(
        self,
        current_alloc: dict[str, float],
        target_alloc: dict[str, float],
    ) -> dict[str, int]:
        """Alias for module-level :func:`generate_signals`."""
        return generate_signals(current_alloc, target_alloc, self.config)

    def full_rebalance(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        current_alloc: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """End-to-end pipeline: classify → weight → signal.

        Returns a dict with keys ``regime``, ``target_alloc``, ``signals``.
        """
        regime = self.classify_regime(df, symbol=symbols[0] if symbols else None)
        targets = self.get_target_weights(regime, symbols)
        cur = current_alloc or {}
        signals = self.generate_signals(cur, targets)
        return {
            "regime": regime,
            "target_alloc": targets,
            "signals": signals,
        }
