"""Portfolio allocation — Kelly criterion, risk parity, and equal-weight capital
allocation across multiple trading strategies.

Self-contained: depends only on pandas/numpy and the standard library. The
multi-strategy engine consumes the public functions and ``PortfolioState`` to
decide how to split capital across registered strategies.

Kelly fraction (univariate, per strategy):
    f* = mean(r) / var(r)
Derivation: maximize E[log(1 + f·r)] ≈ f·E[r] − f²·Var[r]/2  →  f* = E[r]/Var[r].

Fractional Kelly (default 0.25 = quarter-Kelly) scales f* down to reduce
volatility and drawdown at the cost of lower geometric growth.

Edge-case handling:
    - Negative expected edge (mean < 0)  → f* capped at 0 (no shorting)
    - All-negative returns               → falls back to equal weight
    - NaN / inf in returns               → dropped before computation
    - Zero variance / insufficient data  → f* = 0, excluded from normalization
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.kelly import (
    KellyConfig,
    compute_returns_kelly,
    scale_risk_with_kelly,
)

logger = logging.getLogger(__name__)

# Default persistence path (mirrors EngineState convention in bot/engine.py).
DEFAULT_STATE_FILE = Path("logs/portfolio_state.json")

# Minimum return observations needed for a trustworthy Kelly estimate.
# Below this → statistics unreliable → weight falls back to 0.
_MIN_SAMPLES = 10


# ── Helpers (backwards compat) ───────────────────────────────────────


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Coerce to float, drop NaN/inf, return a clean Series.

    Kept as local alias so that existing code importing this directly still works.
    Delegates to :py:func:`bot.kelly._clean_returns`.
    """
    from bot.kelly import _clean_returns as _kr
    return _kr(returns)


def _kelly_fraction(returns: pd.Series) -> float:
    """Full Kelly fraction for one strategy: f* = mean(r) / var(r).

    Backwards-compatible wrapper around :py:func:`bot.kelly.compute_returns_kelly`
    using a default config with `_MIN_SAMPLES` as the minimum observation count.

    .. deprecated::
        New code should call :py:func:`bot.kelly.compute_returns_kelly` directly.
    """
    cfg = KellyConfig(min_samples=_MIN_SAMPLES, fractional=1.0, enabled=True)
    return compute_returns_kelly(returns, cfg)


# ── Public allocation functions ─────────────────────────────────────


def allocate_kelly(
    returns_by_strategy: dict[str, pd.Series],
    fractional: float = 0.25,
) -> dict[str, float]:
    """Allocate capital across strategies using fractional Kelly.

    Each strategy's univariate Kelly fraction f* = mean/var is computed,
    scaled by ``fractional`` (default 0.25 = quarter-Kelly), floored at 0,
    and normalized so weights sum to 1.

    If every strategy has non-positive edge or insufficient data, falls back
    to equal weighting so capital is still deployed.

    Parameters
    ----------
    returns_by_strategy : dict[str, pd.Series]
        Strategy name → Series of per-period returns (e.g. daily pct change).
    fractional : float, default 0.25
        Fraction of full Kelly to use (0, 1]. Quarter-Kelly is a conservative
        default: sacrifices ~25 % of geometric growth for a large cut in
        volatility and drawdown.

    Returns
    -------
    dict[str, float]
        Strategy name → portfolio weight (sums to 1.0).
    """
    if not returns_by_strategy:
        logger.warning("allocate_kelly: empty input — returning empty dict")
        return {}

    if not 0 < fractional <= 1.0:
        logger.warning(
            "allocate_kelly: fractional=%.4f outside (0,1] — clamping to 0.25", fractional,
        )
        fractional = 0.25

    raw: dict[str, float] = {}
    for name, returns in returns_by_strategy.items():
        f = _kelly_fraction(returns)
        raw[name] = f * fractional

    total = sum(raw.values())
    logger.info(
        "allocate_kelly: raw Kelly weights (pre-normalize) = %s (sum=%.4f)", raw, total,
    )

    if total <= 0:
        names = list(returns_by_strategy.keys())
        logger.info(
            "allocate_kelly: all edges ≤ 0 — falling back to equal weight across %d strategies",
            len(names),
        )
        return allocate_equal_weight(names)

    weights = {name: f / total for name, f in raw.items()}
    logger.info("allocate_kelly: final normalized weights = %s (fractional=%.2f)", weights, fractional)
    return weights


def allocate_equal_weight(strategies: list[str]) -> dict[str, float]:
    """Equal-weight allocation: 1/N per strategy."""
    if not strategies:
        logger.warning("allocate_equal_weight: empty list — returning empty dict")
        return {}

    w = 1.0 / len(strategies)
    weights = {s: w for s in strategies}
    logger.info("allocate_equal_weight: %d strategies → %.4f each", len(strategies), w)
    return weights


def allocate_risk_parity(
    strategies: list[str],
    volatilities: dict[str, float] | pd.Series,
) -> dict[str, float]:
    """Risk-parity allocation: weight inversely proportional to volatility.

    Each strategy gets weight ∝ 1/vol, so every strategy contributes equal
    risk to the portfolio. Strategies with zero / NaN / ≤ 0 volatility are
    excluded; if all are invalid, falls back to equal weight.

    Parameters
    ----------
    strategies : list[str]
        Strategy names to allocate across.
    volatilities : dict[str, float] | pd.Series
        Volatility per strategy (any consistent period — annualized or daily).

    Returns
    -------
    dict[str, float]
        Strategy name → portfolio weight (sums to 1.0; excluded → 0.0).
    """
    if not strategies:
        logger.warning("allocate_risk_parity: empty strategies — returning empty dict")
        return {}

    inv_vols: dict[str, float] = {}
    excluded: list[str] = []

    for s in strategies:
        vol = volatilities.get(s, 0.0) if hasattr(volatilities, "get") else volatilities[s]
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            vol = 0.0

        if math.isnan(vol) or vol <= 0:
            excluded.append(s)
            logger.warning("allocate_risk_parity: %s has invalid vol (%r) — excluding", s, vol)
            continue

        inv_vols[s] = 1.0 / vol

    total = sum(inv_vols.values())
    if total <= 0:
        logger.info("allocate_risk_parity: all vols invalid — falling back to equal weight")
        return allocate_equal_weight(strategies)

    weights = {s: iv / total for s, iv in inv_vols.items()}
    # Explicit zero weight for excluded strategies
    for s in strategies:
        if s not in weights:
            weights[s] = 0.0

    logger.info("allocate_risk_parity: weights = %s (excluded=%s)", weights, excluded)
    return weights


# ── Persistence ────────────────────────────────────────────────────


class PortfolioState:
    """Persists target strategy allocations to ``logs/portfolio_state.json``.

    Mirrors the persistence pattern in ``EngineState`` (bot/engine.py):
    JSON dump with ``indent=2``, UTC ISO timestamps, parent-dir auto-create.

    Usage::

        state = PortfolioState()
        state.save(weights, method="kelly", fractional=0.25)
        loaded = state.read()
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STATE_FILE

    @property
    def path(self) -> Path:
        """Resolved on-disk path for the state file."""
        return self._path

    def read(self) -> dict[str, Any]:
        """Load persisted portfolio state. Returns ``{}`` if missing or corrupt."""
        if not self._path.exists():
            logger.debug("PortfolioState: %s not found — returning empty dict", self._path)
            return {}
        try:
            with open(self._path) as f:
                data = json.load(f)
            logger.info(
                "PortfolioState: loaded allocations=%s from %s",
                data.get("allocations"), self._path,
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("PortfolioState: failed to read %s — %s", self._path, exc)
            return {}

    def save(
        self,
        allocations: dict[str, float],
        method: str = "kelly",
        fractional: float | None = None,
    ) -> None:
        """Persist target allocations with metadata.

        Parameters
        ----------
        allocations : dict[str, float]
            Strategy name → portfolio weight (should sum to 1.0).
        method : str
            Allocation method used (``"kelly"``, ``"equal_weight"``, …).
        fractional : float | None
            Kelly fraction applied, if applicable.
        """
        data = {
            "allocations": allocations,
            "method": method,
            "fractional": fractional,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "PortfolioState: saved allocations=%s method=%s → %s",
            allocations, method, self._path,
        )



# ── Hierarchical Risk Parity (HRP) — Stefan Jansen, ML4T §6.2.3 ───────────────
#
# Algorithm:
#   1. Compute Pearson correlation r(X,X).
#   2. Convert to distances d(i,j) = sqrt((1-r)/2) and cluster (Ward linkage).
#   3. Recursively bisect the dendrogram; at each binary split allocate capital
#      between children inversely proportional to their within-cluster variance.
#
# Why HRP over mean-variance?  No matrix inversion required; handles singular cov,
# stable out-of-sample, no estimate error amplification.
#
# See: https://jaseg.dev – ML4T source, Ch 6.
# ────────────────────────────────────────────────────────────────────────────────


from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import squareform


def _reorder_cluster_weights(linkage_matrix: np.ndarray, assets: list[str]) -> list[str]:
    """Reorder ``assets`` according to the dendrogram leaf ordering.

    HRP allocates across contiguous blocks in clustered order; without this
    reordering the recursive bisection would not respect the hierarchy.

    Parameters
    ----------
    linkage_matrix : ndarray shape (n-1, 4)
        Output of ``scipy.cluster.hierarchy.linkage``.
    assets : list[str]
        Asset labels in the original (pre-cluster) order.

    Returns
    -------
    list[str]
        Assets reordered by ``leaves_list(linkage_matrix)``.
    """
    ordered_indices = leaves_list(linkage_matrix)
    return [assets[i] for i in ordered_indices]


class HRPAllocator:
    """Hierarchical Risk Parity allocator using scipy hierarchical clustering.

    Implements the non-parametric clustering-based allocation strategy from
    Jansen (ML4T, Ch. 6).  Instead of inverting the covariance matrix, HRP
    builds a tree from the correlation structure and distributes risk down the
    branches -- yielding robust, near-optimal out-of-sample weights with no
    tuning parameters beyond the choice of clustering method.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.HRPAllocator")

    def allocate(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """Compute HRP portfolio weights from period returns.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Columns = assets, rows = time periods, values = fractional returns
            (e.g. daily percentage changes expressed as decimals).

        Returns
        -------
        dict[str, float]
            ``{symbol: weight}`` where weights sum to ~1.0.

        Raises
        ------
        ValueError
            If the DataFrame is empty or contains fewer than 2 assets.

        Examples
        --------
        >>> returns = pd.DataFrame({  # doctest: +SKIP
        ...     'AAPL': [0.01, -0.02, 0.015, -0.005],
        ...     'GOOG': [-0.005, 0.01, 0.008, -0.012],
        ... })
        >>> weights = HRPAllocator().allocate(returns)
        >>> round(sum(weights.values()), 9)
        1.0
        """
        if returns_df.empty:
            raise ValueError("Returns DataFrame is empty")

        assets = list(returns_df.columns)
        if len(assets) < 2:
            raise ValueError(f"Need >= 2 assets for HRP, got {len(assets)}")

        # --- Step 1: correlation -> distance matrix ----------------------------------
        corr = returns_df.corr()
        dist_matrix = self._corr_to_distance(corr)

        # --- Step 2: Ward linkage hierarchical clustering ---------------------------
        condensed_dist = squareform(dist_matrix)
        link = linkage(condensed_dist, method="ward")
        self._logger.info("HRP: linkage matrix shape=%s", link.shape)

        # --- Step 3: reorder assets to match dendrogram leaves ----------------------
        ordered_assets = _reorder_cluster_weights(link, assets)

        # --- Step 4: recursive bisection for inverse-variance weights ---------------
        raw_weights = self._recursive_bisection(link, ordered_assets, corr)

        # Normalise to sum=1 (guard against floating-point drift)
        total = abs(sum(raw_weights))
        if total < 1e-15:
            # Degenerate case: fallback to equal weight
            self._logger.warning(
                "HRP: near-zero total weight (%.2e) -- falling back to equal weight", total
            )
            n = len(ordered_assets)
            return dict(zip(ordered_assets, [1.0 / n] * n))

        final_weights = [w / total for w in raw_weights]
        self._logger.debug(
            "HRP: weights=%s",
            dict(zip(ordered_assets, [round(w, 4) for w in final_weights])),
        )

        return dict(zip(ordered_assets, final_weights))

    @staticmethod
    def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Convert a correlation matrix to a metric distance matrix.

        Uses the trigonometric identity:  d(i,j) = sqrt((1 - r_ij) / 2),
        which satisfies the triangle inequality for positive-semidefinite
        correlation matrices.
        """
        d = np.sqrt(np.clip((1 - corr.values) / 2, 0, None))
        return pd.DataFrame(d, index=corr.index, columns=corr.columns)

    # -- recursive bisection core ------------------------------------------------

    def _recursive_bisection(
        self,
        link: np.ndarray,
        ordered: list[str],
        corr: pd.DataFrame,
    ) -> list[float]:
        """Assign weights by recursively splitting the dendrogram top-down.

        At each binary split (left vs. right subtree) we compute the portfolio
        variance of each side under an **equal-weight** assumption, then allocate
        capital inversely proportional to that variance (Eq. 6.9-6.10 in Jansen).

        Parameters
        ----------
        link : ndarray
            Linkage matrix from ``linkage()``.
        ordered : list[str]
            Assets in ``leaves_list`` order.
        corr : pd.DataFrame
            Full correlation matrix indexed by asset name.

        Returns
        -------
        list[float]
            Raw (un-normalised) weights matching *ordered*.
        """
        n = len(ordered)
        weights = np.ones(n)

        # The linkage matrix records merges bottom-up; we iterate **backwards**
        # so the first split processed is the root-level bifurcation.
        for merge_idx in range(len(link) - 1, -1, -1):
            node_i = int(link[merge_idx][0])
            node_j = int(link[merge_idx][1])

            # Identify which leaf indices live under each child.
            members_i = self._get_leaf_indices(node_i, len(link), n)
            members_j = self._get_leaf_indices(node_j, len(link), n)

            # Cluster variance under equal-weight assumption:
            # Var(cluster) = (1/K^2) * sum_{a,b in cluster} Corr(a,b)
            var_i = self._cluster_variance(corr, members_i)
            var_j = self._cluster_variance(corr, members_j)

            total_var = var_i + var_j
            if total_var > 0:
                alpha = 1.0 - var_i / total_var  # weight fraction for LEFT child
            else:
                alpha = 0.5

            # Apply multiplicative adjustment: lower-variance side gets more weight
            weights[members_i] *= alpha
            weights[members_j] *= (1.0 - alpha)

        return weights.tolist()

    @staticmethod
    def _get_leaf_indices(node_id: int, n_merges: int, n_leaves: int) -> np.ndarray:
        """Return the array of original-leaf indices beneath ``node_id``.

        Internal (merged) nodes have index ``>= n_leaves``; leaf nodes have
        index ``< n_leaves``.  This walks the linkage matrix bottom-up.
        """
        offset = n_leaves  # new clusters start at index n_leaves
        stack = [node_id]
        leaves = []
        while stack:
            cur = stack.pop()
            if cur < n_leaves:
                leaves.append(cur)
            else:
                row_idx = cur - offset
                if 0 <= row_idx < n_merges:
                    left = int(link[row_idx][0])
                    right = int(link[row_idx][1])
                    stack.append(left)
                    stack.append(right)
        return np.array(sorted(leaves), dtype=int)

    @staticmethod
    def _cluster_variance(corr: pd.DataFrame, member_indices: np.ndarray) -> float:
        """Compute the variance of an equal-weighted portfolio within a cluster.

        Given a subset of asset indices (into the original return series) this
        computes Var = (1/K^2) * sum of all pairwise correlations in the subset.
        """
        n_members = len(member_indices)
        if n_members == 0:
            return 1.0

        # Get the sub-correlation matrix for these assets
        col_names = list(corr.columns)
        assets_in_cluster = [col_names[i] for i in sorted(member_indices)]
        sub = corr.loc[assets_in_cluster, assets_in_cluster].values

        # Equal-weight portfolio variance
        K = float(n_members)
        var = float(np.sum(sub)) / (K * K)
        return max(var, 1e-12)  # avoid exact zero from identical assets



# ── Hierarchical Risk Parity (HRP) — Stefan Jansen, ML4T §6.2.3 ───────────────
#
# Algorithm:
#   1. Compute Pearson correlation r(X,X).
#   2. Convert to distances d(i,j) = sqrt((1-r)/2) and cluster (Ward linkage).
#   3. Recursively bisect the dendrogram; at each binary split allocate capital
#      between children inversely proportional to their within-cluster variance.
#
# Why HRP over mean-variance?  No matrix inversion required; handles singular cov,
# stable out-of-sample, no estimate error amplification.
#
# See: https://jaseg.dev – ML4T source, Ch 6.
# ────────────────────────────────────────────────────────────────────────────────


from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import squareform


def _reorder_cluster_weights(linkage_matrix: np.ndarray, assets: list[str]) -> list[str]:
    """Reorder ``assets`` according to the dendrogram leaf ordering.

    HRP allocates across contiguous blocks in clustered order; without this
    reordering the recursive bisection would not respect the hierarchy.

    Parameters
    ----------
    linkage_matrix : ndarray shape (n-1, 4)
        Output of ``scipy.cluster.hierarchy.linkage``.
    assets : list[str]
        Asset labels in the original (pre-cluster) order.

    Returns
    -------
    list[str]
        Assets reordered by ``leaves_list(linkage_matrix)``.
    """
    ordered_indices = leaves_list(linkage_matrix)
    return [assets[i] for i in ordered_indices]


class HRPAllocator:
    """Hierarchical Risk Parity allocator using scipy hierarchical clustering.

    Implements the non-parametric clustering-based allocation strategy from
    Jansen (ML4T, Ch. 6).  Instead of inverting the covariance matrix, HRP
    builds a tree from the correlation structure and distributes risk down the
    branches -- yielding robust, near-optimal out-of-sample weights with no
    tuning parameters beyond the choice of clustering method.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.HRPAllocator")

    def allocate(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """Compute HRP portfolio weights from period returns.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Columns = assets, rows = time periods, values = fractional returns
            (e.g. daily percentage changes expressed as decimals).

        Returns
        -------
        dict[str, float]
            ``{symbol: weight}`` where weights sum to ~1.0.

        Raises
        ------
        ValueError
            If the DataFrame is empty or contains fewer than 2 assets.

        Examples
        --------
        >>> returns = pd.DataFrame({  # doctest: +SKIP
        ...     'AAPL': [0.01, -0.02, 0.015, -0.005],
        ...     'GOOG': [-0.005, 0.01, 0.008, -0.012],
        ... })
        >>> weights = HRPAllocator().allocate(returns)
        >>> round(sum(weights.values()), 9)
        1.0
        """
        if returns_df.empty:
            raise ValueError("Returns DataFrame is empty")

        assets = list(returns_df.columns)
        if len(assets) < 2:
            raise ValueError(f"Need >= 2 assets for HRP, got {len(assets)}")

        # --- Step 1: correlation -> distance matrix ----------------------------------
        corr = returns_df.corr()
        dist_matrix = self._corr_to_distance(corr)

        # --- Step 2: Ward linkage hierarchical clustering ---------------------------
        condensed_dist = squareform(dist_matrix)
        link = linkage(condensed_dist, method="ward")
        self._logger.info("HRP: linkage matrix shape=%s", link.shape)

        # --- Step 3: reorder assets to match dendrogram leaves ----------------------
        ordered_assets = _reorder_cluster_weights(link, assets)

        # --- Step 4: recursive bisection for inverse-variance weights ---------------
        raw_weights = self._recursive_bisection(link, ordered_assets, corr)

        # Normalise to sum=1 (guard against floating-point drift)
        total = abs(sum(raw_weights))
        if total < 1e-15:
            # Degenerate case: fallback to equal weight
            self._logger.warning(
                "HRP: near-zero total weight (%.2e) -- falling back to equal weight", total
            )
            n = len(ordered_assets)
            return dict(zip(ordered_assets, [1.0 / n] * n))

        final_weights = [w / total for w in raw_weights]
        self._logger.debug(
            "HRP: weights=%s",
            dict(zip(ordered_assets, [round(w, 4) for w in final_weights])),
        )

        return dict(zip(ordered_assets, final_weights))

    @staticmethod
    def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Convert a correlation matrix to a metric distance matrix.

        Uses the trigonometric identity:  d(i,j) = sqrt((1 - r_ij) / 2),
        which satisfies the triangle inequality for positive-semidefinite
        correlation matrices.
        """
        d = np.sqrt(np.clip((1 - corr.values) / 2, 0, None))
        return pd.DataFrame(d, index=corr.index, columns=corr.columns)

    # -- recursive bisection core ------------------------------------------------

    def _recursive_bisection(
        self,
        link: np.ndarray,
        ordered: list[str],
        corr: pd.DataFrame,
    ) -> list[float]:
        """Assign weights by recursively splitting the dendrogram top-down.

        At each binary split (left vs. right subtree) we compute the portfolio
        variance of each side under an **equal-weight** assumption, then allocate
        capital inversely proportional to that variance (Eq. 6.9-6.10 in Jansen).

        Parameters
        ----------
        link : ndarray
            Linkage matrix from ``linkage()``.
        ordered : list[str]
            Assets in ``leaves_list`` order.
        corr : pd.DataFrame
            Full correlation matrix indexed by asset name.

        Returns
        -------
        list[float]
            Raw (un-normalised) weights matching *ordered*.
        """
        n = len(ordered)
        weights = np.ones(n)

        # The linkage matrix records merges bottom-up; we iterate **backwards**
        # so the first split processed is the root-level bifurcation.
        for merge_idx in range(len(link) - 1, -1, -1):
            node_i = int(link[merge_idx][0])
            node_j = int(link[merge_idx][1])

            # Identify which leaf indices live under each child.
            members_i = self._get_leaf_indices(node_i, link, n)
            members_j = self._get_leaf_indices(node_j, link, n)

            # Cluster variance under equal-weight assumption:
            # Var(cluster) = (1/K^2) * sum_{a,b in cluster} Corr(a,b)
            var_i = self._cluster_variance(corr, members_i)
            var_j = self._cluster_variance(corr, members_j)

            total_var = var_i + var_j
            if total_var > 0:
                alpha = 1.0 - var_i / total_var  # weight fraction for LEFT child
            else:
                alpha = 0.5

            # Apply multiplicative adjustment: lower-variance side gets more weight
            weights[members_i] *= alpha
            weights[members_j] *= (1.0 - alpha)

        return weights.tolist()

    def _get_leaf_indices(self, node_id: int, link: np.ndarray, n_leaves: int) -> np.ndarray:
        """Return the array of original-leaf indices beneath ``node_id``.

        Internal (merged) nodes have index ``>= n_leaves``; leaf nodes have
        index ``< n_leaves``.  This walks the linkage matrix bottom-up.
        """
        offset = n_leaves
        stack = [node_id]
        leaves = []
        while stack:
            cur = stack.pop()
            if cur < n_leaves:
                leaves.append(cur)
            else:
                row_idx = cur - offset
                if 0 <= row_idx < len(link):
                    left = int(link[row_idx][0])
                    right = int(link[row_idx][1])
                    stack.append(left)
                    stack.append(right)
        return np.array(sorted(leaves), dtype=int)

    @staticmethod
    def _cluster_variance(corr: pd.DataFrame, member_indices: np.ndarray) -> float:
        """Compute the variance of an equal-weighted portfolio within a cluster.

        Given a subset of asset indices (into the original return series) this
        computes Var = (1/K^2) * sum of all pairwise correlations in the subset.
        """
        n_members = len(member_indices)
        if n_members == 0:
            return 1.0

        # Get the sub-correlation matrix for these assets
        col_names = list(corr.columns)
        assets_in_cluster = [col_names[i] for i in sorted(member_indices)]
        sub = corr.loc[assets_in_cluster, assets_in_cluster].values

        # Equal-weight portfolio variance
        K = float(n_members)
        var = float(np.sum(sub)) / (K * K)
        return max(var, 1e-12)  # avoid exact zero from identical assets

