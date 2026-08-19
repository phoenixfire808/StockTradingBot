"""Fractional Kelly Criterion position sizing + Hierarchical Risk Parity (HRP) allocation.

Implements:
  - KellySizer: fractional Kelly position sizing with dynamic confidence scaling
  - HRPPortfolioManager: hierarchical clustering-based portfolio allocation per Lopez de Prado (ML4T)
  - weighted_allocation: -style WeightedAllocation pattern adapter for equity portfolios

Reference: Jansen, ML for Trading, Ch. 3 (Kelly criterion) + Ch. 5 (HRP).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KellySizer — fractional Kelly with dynamic confidence scaling
# ---------------------------------------------------------------------------


class KellySizer:
    """Fractional Kelly Criterion position sizer.

    f* = (p*b - q) / b       -- classical Kelly fraction
        = p - q/b              -- equivalent form where b = avg_win/avg_loss

    Applies a configurable fraction of f* to dampen volatility drag,
    then scales by signal confidence (0..1) before clamping to [0, max_allocation].
    """

    def __init__(
        self,
        default_fraction: float = 0.5,
        max_allocation: float = 0.25,
    ) -> None:
        self.default_fraction = max(0.0, min(1.0, default_fraction))
        self.max_allocation = max(0.0, min(1.0, max_allocation))

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _edge(avg_win_pct: float, avg_loss_pct: float) -> float:
        """Win–loss ratio b.  Returns inf when avg_loss_pct == 0 (unbounded edge)."""
        if avg_loss_pct <= 0:
            return float("inf")
        return avg_win_pct / avg_loss_pct

    @staticmethod
    def _expected_return(win_prob: float, avg_win_pct: float, avg_loss_pct: float) -> float:
        """Expected return per unit risked: p*avg_win - q*avg_loss."""
        q = 1.0 - win_prob
        return win_prob * avg_win_pct - q * avg_loss_pct

    # -- public API -----------------------------------------------------------

    def calculate_position_size(
        self,
        symbol: str,
        win_prob: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        confidence: float = 1.0,
        fraction: Optional[float] = None,
        max_allocation: Optional[float] = None,
    ) -> float:
        """Return optimal fraction of portfolio to allocate to *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument identifier (logged only; not used in computation).
        win_prob : float
            Probability of a winning trade (0-1).
        avg_win_pct : float
            Average magnitude of a winning trade in percent.
        avg_loss_pct : float
            Average magnitude of a losing trade in percent (positive value).
        confidence : float
            Signal confidence scaled 0-1; position shrinks proportionally.
        fraction : float, optional
            Fraction of full-Kelly to use (default class-level).
        max_allocation : float, optional
            Absolute upper bound on allocation (default class-level).

        Returns
        -------
        float
            Portfolio fraction in ``[0, max_allocation]``.
        """
        frac = fraction if fraction is not None else self.default_fraction
        cap = max_allocation if max_allocation is not None else self.max_allocation

        win_prob = max(0.0, min(1.0, float(win_prob)))
        confidence = max(0.0, min(1.0, float(confidence)))

        q = 1.0 - win_prob
        b = self._edge(avg_win_pct, avg_loss_pct)

        # Classical Kelly: f* = p - q/b  (handles b=inf → f*=p cleanly)
        if np.isinf(b):
            full_kelly = win_prob          # unbounded edge → take whole win prob
        else:
            full_kelly = win_prob - q / b

        # Fractional Kelly + confidence scaling
        effective_fraction = frac * max(0.0, min(1.0, confidence))
        adjusted = full_kelly * effective_fraction

        # Clamp to [0, max_allocation]
        result = max(0.0, min(cap, adjusted))

        logger.debug(
            "Kelly[%s]: p=%.3f  b=%.3f  f*=%.4f  conf=%.2f  frac=%.2f  size=%.4f",
            symbol, win_prob, b, full_kelly, confidence, effective_fraction, result,
        )
        return result


# ---------------------------------------------------------------------------
# HRPPortfolioManager — Hierarchical Risk Parity
# ---------------------------------------------------------------------------


class HRPPortfolioManager:
    """Hierarchical Risk Parity portfolio allocator (Lopez de Prado, 2016).

    Allocates capital inversely proportional to cluster variance via:
      1. Correlation matrix → distance matrix
      2. Ward linkage clustering (tree/dendrogram)
      3. Recursive quaternary bisection with reverse-variance weighting within each node
    Falls back to inverse-variance when scipy/sklearn are unavailable.
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        self.risk_aversion = risk_aversion

    # -- core allocation entry-point ------------------------------------------

    def allocate(
        self,
        returns_df: pd.DataFrame,
        weights: str = "equal",
    ) -> Dict[str, float]:
        """Allocate portfolio weights using HRP.

        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame of asset returns (columns = symbols, rows = dates).
        weights : str
            Allocation style within clusters: ``"equal"`` or ``"risk_parity"``.
            Currently both delegate to reverse-variance (the canonical HRP rule).

        Returns
        -------
        dict[str, float]
            Symbol → normalized weight mapping (sum ≈ 1.0).
        """
        # Import scipy hierarchy for Ward linkage
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform

            _deps_available = True
        except ImportError:
            _deps_available = False
            logger.warning(
                "HRP: scipy not available; falling back to inverse-variance weighting.",
            )

        if not _deps_available:
            return self._inverse_variance_weights(returns_df)

        # 1. Compute sample covariance & diagonal (variances)
        cov = returns_df.cov()
        diag = np.diag(cov.values)

        # Guard against zero-variance assets
        zero_var_mask = diag == 0
        if zero_var_mask.any():
            nonzero_idx = np.where(~zero_var_mask)[0]
            if len(nonzero_idx) < 2:
                raise ValueError(
                    "HRP requires at least two assets with non-zero variance."
                )
            corr = returns_df.iloc[:, nonzero_idx].corr().values
            diag = diag[nonzero_idx]
            logger.warning(
                "HRP: dropped %d asset(s) with zero variance.", zero_var_mask.sum(),
            )
        else:
            corr = returns_df.corr().values

        n_assets = len(diag)
        sigmas = np.sqrt(diag)
        corr = np.clip(corr, -1.0, 1.0)

        # 2. Correlation -> distance matrix -> Ward linkage clustering -> tree order
        dist_matrix = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="ward")
        ordered_idx = list(dendrogram(Z, no_plot=True)["leaves"])

        # 3. Reverse variance recursive weighting over ordered leaves
        w_hrp = self._reverse_variance_recursive(corr, ordered_idx, sigmas)

        # Normalize so weights sum to 1
        total = w_hrp.sum()
        if total <= 0:
            n = len(w_hrp)
            w_hrp = np.ones(n) / n
            total = 1.0

        w_normalized = w_hrp / total

        # Reconstruct dict with original indices
        all_symbols = list(returns_df.columns)
        if zero_var_mask.any():
            active_symbols = [all_symbols[i] for i in range(len(all_symbols)) if not zero_var_mask[i]]
        else:
            active_symbols = all_symbols

        weights_map: Dict[str, float] = {}
        for sym, w in zip(active_symbols, w_normalized):
            weights_map[sym] = round(float(w), 10)

        logger.info(
            "HRP allocation (%d assets): sum=%.6f, min=%.4f, max=%.4f",
            len(weights_map), sum(weights_map.values()),
            min(weights_map.values()), max(weights_map.values()),
        )
        return weights_map

    @staticmethod
    def _reverse_variance_recursive(
        corr: np.ndarray,
        ordered_idx: List[int],
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """Recursive bisection of the clustering tree yielding raw pre-normalize weights.

        At each split, partitions {1..q} vs {q+1..n} and allocates weight
        proportional to inverse variance share between the two sides.
        Recurses within each side.
        """
        n = len(ordered_idx)
        if n <= 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        idx_list = list(ordered_idx)

        var_total = sum(sigmas[idx] ** 2 for idx in idx_list)

        cum_var_left = 0.0
        best_q = n // 2
        best_diff = 0.0

        for q_idx in range(1, n):
            cum_var_left += sigmas[idx_list[q_idx - 1]] ** 2
            cum_var_right = var_total - cum_var_left
            diff = abs(cum_var_left - cum_var_right)
            if diff > best_diff:
                best_diff = diff
                best_q = q_idx

        q = max(1, min(best_q, n - 1))

        left_idx = idx_list[:q]
        right_idx = idx_list[q:]

        w_left = np.array(HRPPortfolioManager._reverse_variance_recursive(
            corr, left_idx, sigmas
        ))
        w_right = np.array(HRPPortfolioManager._reverse_variance_recursive(
            corr, right_idx, sigmas
        ))

        inv_var_left = sum(sigmas[idx] ** (-2) for idx in left_idx) if left_idx else 1.0
        inv_var_right = sum(sigmas[idx] ** (-2) for idx in right_idx) if right_idx else 1.0

        total_inv = inv_var_left + inv_var_right
        alpha = inv_var_left / total_inv if total_inv > 0 else 0.5

        w_total = np.zeros(n)
        w_total[:len(w_left)] = alpha * w_left
        w_total[len(w_left):] = (1.0 - alpha) * w_right

        return w_total

    @staticmethod
    def _inverse_variance_weights(returns_df: pd.DataFrame) -> Dict[str, float]:
        """Fallback: inverse-variance weighting when scipy unavailable."""
        var = returns_df.var()
        nonzero = var[var > 0]
        if len(nonzero) == 0:
            n = len(returns_df.columns)
            return {sym: 1.0 / n for sym in returns_df.columns}
        w = 1.0 / nonzero
        total = w.sum()
        return {sym: round(float(val / total), 10) for sym, val in w.items()}


# ---------------------------------------------------------------------------
# Helper: weighted_allocation —  WeightedAllocation pattern
# ---------------------------------------------------------------------------


def weighted_allocation(
    symbols: Sequence[str],
    weights_dict: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> List[float]:
    """Return a weight list matching  sWeightedAllocation interface.

    If *weights_dict* is provided, look up each symbol; otherwise produce
    equal weights. When *normalize* is True the returned list sums to 1.0.

    Usage::

        pairlist_weights = weighted_allocation(my_symbols, my_weights_dict)
        # trade_amount = total_trade_amount * pairlist_weights[i] / sum(pairlist_weights)

    Parameters
    ----------
    symbols : sequence[str]
        Asset identifiers.
    weights_dict : dict[str, float], optional
        Mapping symbol → raw weight. Missing keys receive weight 1.0.
    normalize : bool
        If True, scale weights so they sum to 1.0.

    Returns
    -------
    list[float]
        Weights aligned to *symbols*.
    """
    symbols = list(symbols)
    wd = weights_dict if weights_dict is not None else {}

    weights = []
    for sym in symbols:
        w = wd.get(sym, 1.0)
        weights.append(max(0.0, w))

    total = sum(weights)
    if normalize and total > 0:
        weights = [w / total for w in weights]
    elif normalize:
        n = len(symbols)
        weights = [1.0 / n] * n if n else []

    return weights
