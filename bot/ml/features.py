"""Feature engineering for ML signal model.

Produces feature frames from OHLCV data, triple-barrier labels (Jansen /
de Prado pattern), Kelly fraction sizing helpers, and label distribution
analytics.  Intended as a utility library consumed by the ML pipeline and
strategy plugins.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Canonical feature column order ────────────────────────────────────
# Keep this explicit so train/persist/predict always use the same columns
# in the same order.  Tests and callers may reference FEATURE_COLUMNS.
FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "rsi_14",
    "macd_hist",
    "bb_pct",
    "ema_slope",
    "atr_pct",
]

# Import indicator functions from bot.indicators; lazy-import avoided
# because they are small pure-Pandas helpers used throughout this module.
from bot.indicators import atr, bollinger, ema, rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line − signal line)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _normalize_col(df: pd.DataFrame, upper: str, lower: str, pos: int) -> pd.Series:
    """Resolve a column by preferred upper/lower name, else positional."""
    if upper in df.columns:
        return df[upper]
    if lower in df.columns:
        return df[lower]
    return df.iloc[:, pos]


def build_feature_frame(
    df: pd.DataFrame,
    *,
    ret_windows: list[int] | None = None,
    vol_windows: list[int] | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    ema_period: int = 50,
    atr_period: int = 14,
    clip_infinite: bool = True,
) -> pd.DataFrame:
    """Build a feature DataFrame from OHLCV bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and at least Open/High/Low/Close/Volume
        columns (case-insensitive matching).
    ret_windows : list[int] | None
        Lookback periods for percentage returns (e.g. ``[1, 3, 5]``).
    vol_windows : list[int] | None
        Lookback periods for rolling volatility.
    rsi_period : int
        Period for RSI computation.
    macd_* : int
        Fast/slow/signal periods for MACD histogram.
    bb_period, bb_std : int, float
        Bollinger Band period and number of standard deviations.
    ema_period : int
        Lookback for EMA slope derivative.
    atr_period : int
        ATR period for volatility proxy.
    clip_infinite : bool
        Replace ``inf/-inf`` values introduced by pct_change with NaN.

    Returns
    -------
    pd.DataFrame
        Feature frame with one row per input bar (first rows are NaN
        due to rolling windows).  Column order follows ``FEATURE_COLUMNS``.
    """
    close = _normalize_col(df, "Close", "close", -1)
    high = _normalize_col(df, "High", "high", 1)
    low = _normalize_col(df, "Low", "low", 2)
    volume = _normalize_col(df, "Volume", "volume", 3)

    # Returns
    rets = {f"ret_{w}": close.pct_change(w) for w in (ret_windows or [1, 3, 5, 10])}

    # Rolling volatility
    vols = {}
    for w in (vol_windows or [10, 20]):
        vols[f"vol_{w}"] = close.pct_change().rolling(w).std()

    # RSI
    feats = pd.DataFrame({"rsi_14": rsi(close, rsi_period)}, index=df.index)

    # MACD histogram
    feats["macd_hist"] = _macd(close, macd_fast, macd_slow, macd_signal)

    # Bollinger %B
    bb = bollinger(close, bb_period, bb_std)
    bb_width = bb["upper"] - bb["lower"]
    feats["bb_pct"] = (close - bb["lower"]) / bb_width.replace(0, np.nan)

    # EMA slope (difference between adjacent EMAs normalized by price)
    e = ema(close, ema_period)
    feats["ema_slope"] = (e - e.shift(1)) / e.replace(0, np.nan)

    # ATR-based volatility measure
    feats["atr_pct"] = atr(df)[atrlab(atr_period)] / close

    all_features = {**rets, **vols}
    feat_df = pd.DataFrame(all_features, index=df.index)
    feat_df = pd.concat([feats, feat_df], axis=1)

    # Clean infinities from pct_change on flat prices
    if clip_infinite:
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    return feat_df


def atrlab(period: int) -> str:
    """Return the column name that ATR writes."""
    return f"ATR{period}"


# ── Triple-Barrier Labeling (Jansen ML for Trading / de Prado pattern) ──


class TripleBarrierLabels:
    """Generate supervised ML labels via upper barrier (TP), lower barrier (SL), time limit.

    Follows Stefan Jansen's ML4T implementation: three barriers are placed
    relative to each bar — take-profit above, stop-loss below, and an
    optional maximum holding period in bars.  The first barrier hit
    determines the label (+1, -1, or +μ / -μ for partial wins/losses).
    """

    def __init__(
        self,
        pt_scale: float = 1.0,
        sl_scale: float = 1.0,
        time_limit: int = 20,
        minimum_distance: float = 1e-4,
    ) -> None:
        self.pt_scale = pt_scale
        self.sl_scale = sl_scale
        self.time_limit = time_limit
        self.minimum_distance = minimum_distance

    def generate(
        self,
        df: pd.DataFrame,
        *,
        atr_col: str = "ATR14",
        verbose: bool = False,
    ) -> pd.Series:
        """Label each bar using the next three-barrier event.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a datetime index.  Must contain *atr_col*.
        atr_col : str
            Name of the ATR column used to scale barriers.
        verbose : bool
            Emit per-label progress logging.

        Returns
        -------
        pd.Series
            Labels aligned to the original index (length ``n-1`` because
            the last bar has no future observation window).
        """
        close = df["Close"].values.astype(float)
        n = len(close) - 1  # last bar has no horizon
        labels = np.zeros(n, dtype=np.float64)

        for i in range(n):
            base_price = close[i]
            if base_price <= 0:
                continue

            try:
                atr_val = float(df[atr_col].iloc[i])
            except Exception:
                atr_val = base_price * 0.02  # fallback 2% estimate

            # Barriers
            tp_dist = self.pt_scale * atr_val
            sl_dist = self.sl_scale * atr_val

            # Search forward up to time_limit bars
            horizon = min(self.time_limit, n - i)
            future_high = close[i + 1 : i + 1 + horizon]
            future_low = close[i + 1 : i + 1 + horizon]

            # Check which barrier is hit first
            tp_hit = future_high >= base_price + tp_dist
            sl_hit = future_low <= base_price - sl_dist

            if not (tp_hit.any() or sl_hit.any()):
                # Reached time limit without hitting either barrier
                labels[i] = np.nan
                continue

            tp_idx = np.argmax(tp_hit) if tp_hit.any() else horizon
            sl_idx = np.argmax(sl_hit) if sl_hit.any() else horizon

            if tp_idx < sl_idx:
                # Take-profit hit first — calculate profit % scaled by μ
                exit_price = close[i + 1 + tp_idx]
                ret_pct = (exit_price - base_price) / base_price
                # Partial P&L scaling factor from de Prado
                mu = abs(ret_pct) / (self.pt_scale * atr_val / base_price) if atr_val > 0 else 1.0
                labels[i] = min(mu, 1.0)
            elif sl_idx < tp_idx:
                labels[i] = -1.0
            else:
                # Simultaneous — prefer SL
                labels[i] = -1.0

            if verbose and i % 50 == 0:
                logger.debug("Labelled bar %d/%d", i, n)

        return pd.Series(labels[:n], index=df.index[:n], name="label_3bar")


def build_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    pt_scale: float = 1.0,
    sl_scale: float = 1.0,
    time_limit: int = 20,
    atr_col: str = "ATR14",
) -> pd.Series:
    tvm = TripleBarrierLabels(pt_scale=pt_scale, sl_scale=sl_scale, time_limit=time_limit)
    return tvm.generate(df, atr_col=atr_col)


# ── Asymmetric Triple-Barrier Labels ─────────────────────────────────


class AsymmetricTripleBarrierLabels:
    """Triple-barrier labels where take-profit and stop-loss have independent scales.

    Like the standard variant but allows different multiplier factors for
    upside capture vs downside protection (e.g. ``pt_scale=2.0, sl_scale=1.0``
    seeks 2× the ATR upside before accepting 1× ATR loss).
    """

    def __init__(
        self,
        pt_scale: float = 2.0,
        sl_scale: float = 1.0,
        time_limit: int = 20,
    ) -> None:
        self.pt_scale = pt_scale
        self.sl_scale = sl_scale
        self.time_limit = time_limit

    def generate(
        self,
        df: pd.DataFrame,
        *,
        atr_col: str = "ATR14",
        verbose: bool = False,
    ) -> pd.Series:
        close = df["Close"].values.astype(float)
        n = len(close) - 1
        labels = np.zeros(n, dtype=np.float64)

        for i in range(n):
            base_price = close[i]
            if base_price <= 0:
                continue

            try:
                atr_val = float(df[atr_col].iloc[i])
            except Exception:
                atr_val = base_price * 0.02

            tp_dist = self.pt_scale * atr_val
            sl_dist = self.sl_scale * atr_val

            horizon = min(self.time_limit, n - i)
            future_high = close[i + 1 : i + 1 + horizon]
            future_low = close[i + 1 : i + 1 + horizon]

            tp_hit = future_high >= base_price + tp_dist
            sl_hit = future_low <= base_price - sl_dist

            if not (tp_hit.any() or sl_hit.any()):
                labels[i] = np.nan
                continue

            tp_idx = np.argmax(tp_hit) if tp_hit.any() else horizon
            sl_idx = np.argmax(sl_hit) if sl_hit.any() else horizon

            if tp_idx < sl_idx:
                exit_price = close[i + 1 + tp_idx]
                labels[i] = 1.0
            else:
                labels[i] = -1.0

            if verbose and i % 50 == 0:
                logger.debug("Asym labelled bar %d/%d", i, n)

        return pd.Series(labels[:n], index=df.index[:n], name="label_3bar_asym")


def build_asymmetric_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    pt_scale: float = 2.0,
    sl_scale: float = 1.0,
    time_limit: int = 20,
    atr_col: str = "ATR14",
) -> pd.Series:
    """Convenience wrapper around AsymmetricTripleBarrierLabels."""
    tvm = AsymmetricTripleBarrierLabels(pt_scale=pt_scale, sl_scale=sl_scale, time_limit=time_limit)
    return tvm.generate(df, atr_col=atr_col)


# ── Volatility-Adjusted Position Sizing ───────────────────────────────


def vol_adjusted_position_size(
    account_equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 0.10,
    min_position_value: float = 1.0,
) -> dict:
    """Compute position size capped by volatility (ATR-style distance to SL).

    Parameters
    ----------
    account_equity : float
        Current portfolio NAV.
    risk_per_trade : float
        Fraction of equity to risk on this trade (e.g. 0.01 = 1 %).
    entry_price : float
        Estimated fill price.
    stop_loss_price : float
        Hard stop-loss level — must differ from entry.
    max_position_pct : float, optional
        Upper bound on position as fraction of equity (default 10 %).
    min_position_value : float, optional
        Floor in currency units; round-trip below this is skipped.

    Returns
    -------
    dict
        ``{"shares": int, "value": float, "risk_currency": float,
         "risk_pct_of_equity": float}``

    Notes
    -----
    This follows the classic de Prado / Jansen approach: size the position so
    that a move to the stop-loss costs exactly *risk_per_trade* of equity.
    The result is then **capped** at *max_position_pct* to avoid concentration.
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        logger.warning("Invalid prices — entry=%.4f, SL=%.4f", entry_price, stop_loss_price)
        return {"shares": 0, "value": 0.0, "risk_currency": 0.0, "risk_pct_of_equity": 0.0}

    price_risk = abs(entry_price - stop_loss_price)
    if price_risk < 1e-6:
        logger.warning("Entry equals stop-loss — cannot size position")
        return {"shares": 0, "value": 0.0, "risk_currency": 0.0, "risk_pct_of_equity": 0.0}

    # Shares allowed by raw risk budget
    dollar_risk_target = account_equity * risk_per_trade
    shares_by_risk = dollar_risk_target / price_risk

    # Shares allowed by concentration cap
    max_shares = int(account_equity * max_position_pct / entry_price)

    # FIX: use min() to ENFORCE the cap (was incorrectly max())
    shares = min(int(shares_by_risk), max_shares)
    shares = max(shares, 0)

    position_value = shares * entry_price
    actual_risk = shares * price_risk

    # Minimum position floor check
    if position_value < min_position_value and shares > 0:
        shares = 0
        position_value = 0.0
        actual_risk = 0.0

    return {
        "shares": shares,
        "value": round(position_value, 2),
        "risk_currency": round(actual_risk, 2),
        "risk_pct_of_equity": round((actual_risk / account_equity) * 100, 2) if account_equity > 0 else 0.0,
    }


# ── Kelly Re-export (de-duped into bot.kelly) ───────────────────────────

# Backwards-compatible alias: consumers of ``bot.ml.features.kelly_fraction``
# continue working; the implementation is now centralized in :py:mod:`bot.kelly`.
from bot.kelly import compute_winrate_payoff_kelly as kelly_fraction  # noqa: E402, F401


# ── Feature Correlation Matrix Utility ────────────────────────────────


def feature_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    threshold: float = 0.85,
) -> dict:
    """Compute Pearson/Spearman/Kendall correlation matrix and flag collinear pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame (rows = bars, cols = features).
    columns : list[str] | None
        Subset of columns to include; defaults to all numeric columns.
    method : str
        One of ``pearson``, ``spearman``, ``kendall`` (pandas ``corr`` values).
    threshold : float
        Absolute correlation above which a pair is flagged as multicollinear.

    Returns
    -------
    dict
        ``{
            "correlation_matrix": DataFrame,
            "collinear_pairs": [(str, str, float), ...],
            "flags_removed": list[str],
        }``
    """
    cols = columns or [c for c in df.columns if df[c].dtype in ("float64", "int64")]
    subset = df[cols].dropna()

    if subset.empty:
        return {"correlation_matrix": pd.DataFrame(), "collinear_pairs": [], "flags_removed": []}

    corr_matrix = subset.corr(method=method)

    # Find pairs with absolute correlation > threshold
    collinear_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                collinear_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], round(val, 4)))

    # Auto-flag highest-correlated column within each cluster
    flags_removed = _flag_collinear(corr_matrix, threshold)

    return {
        "correlation_matrix": corr_matrix,
        "collinear_pairs": collinear_pairs,
        "flags_removed": flags_removed,
    }


def _flag_collinear(
    corr_matrix: pd.DataFrame, threshold: float
) -> list[str]:
    """Identify and suggest removal of redundant highly-correlated features.

    For each connected component in the high-correlation graph, keep the
    feature with the lowest average absolute correlation to others and flag
    the rest for removal.
    """
    removed: list[str] = []
    already_flagged: set[str] = set()

    indices = list(range(len(corr_matrix)))
    visited = set()

    while indices:
        start = indices.pop(0)
        if start in visited:
            continue
        # BFS to find connected component
        queue = [start]
        component: list[int] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for other in range(len(corr_matrix)):
                if other not in visited and other not in component:
                    val = abs(corr_matrix.iloc[node, other])
                    if val > threshold:
                        queue.append(other)

        if len(component) > 1:
            # Keep feature with lowest average |corr| with others in component
            scores = {}
            for idx in component:
                names = corr_matrix.index[idx]
                avg_corr = sum(
                    abs(corr_matrix.iloc[idx, oidx])
                    for oidx in component
                    if oidx != idx
                ) / (len(component) - 1)
                scores[names] = avg_corr
            keep_name = min(scores, key=scores.get)
            for idx in component:
                name = corr_matrix.index[idx]
                if name != keep_name:
                    removed.append(name)
                    already_flagged.add(name)

    return removed


# ── Label Distribution Analyzer ───────────────────────────────────────


class LabelDistributionAnalyzer:
    """Analyse class balance of triple-barrection labels across splits.

    Provides summary statistics per split (train/test/OOS) including
    positive/negative/neutral counts and ratios, as well as asymmetry
    diagnostics (whether the strategy skews toward losses).
    """

    def analyze(
        self,
        labels: pd.Series,
        *,
        name: str = "dataset",
    ) -> dict:
        """Compute distribution statistics for a single label series.

        Parameters
        ----------
        labels : pd.Series
            Triple-barrier labels (+1, -1, or NaN for time-limit exits).
        name : str
            Dataset identifier used in log messages.

        Returns
        -------
        dict
            ``{"total": int, "positive": int, "negative": int,
               "neutral": int, "pos_ratio": float, "neg_ratio": float,
               "asymmetry": float}``
        """
        valid = labels.dropna()
        total = len(valid)
        positive = (valid == 1).sum()
        negative = (valid == -1).sum()
        neutral = 0  # No explicit neutral label in 3-bar scheme

        pos_ratio = positive / total if total > 0 else 0.0
        neg_ratio = negative / total if total > 0 else 0.0
        # Asymmetry: how much more likely we are to win than lose
        asymmetry = pos_ratio - neg_ratio

        logger.info(
            "Label distribution %s: N=%d pos=%d (%.1f%%) neg=%d (%.1f%%)",
            name, total, positive, pos_ratio * 100, negative, neg_ratio * 100,
        )

        return {
            "total": int(total),
            "positive": int(positive),
            "negative": int(negative),
            "neutral": int(neutral),
            "pos_ratio": round(pos_ratio, 4),
            "neg_ratio": round(neg_ratio, 4),
            "asymmetry": round(asymmetry, 4),
        }


def analyze_label_distribution(
    labels: pd.Series,
    *,
    name: str = "dataset",
) -> dict:
    """Convenience wrapper around LabelDistributionAnalyzer."""
    return LabelDistributionAnalyzer().analyze(labels, name=name)


def compare_train_test_labels(
    train_labels: pd.Series,
    test_labels: pd.Series,
    train_name: str = "train",
    test_name: str = "test",
) -> dict:
    """Compare label distributions between two splits.

    Parameters
    ----------
    train_labels, test_labels : pd.Series
        Label series from each split.
    train_name, test_name : str
        Identifiers for logging.

    Returns
    -------
    dict
        Aggregate statistics from both splits plus a stability score.
    """
    train_stats = LabelDistributionAnalyzer().analyze(train_labels, name=train_name)
    test_stats = LabelDistributionAnalyzer().analyze(test_labels, name=test_name)

    # Stability: difference in positive ratios (smaller = better generalization)
    stability = 1.0 - abs(train_stats["pos_ratio"] - test_stats["pos_ratio"])

    return {
        "train": train_stats,
        "test": test_stats,
        "stability_score": round(stability, 4),
        "label_shift_detected": stability < 0.7,
    }
