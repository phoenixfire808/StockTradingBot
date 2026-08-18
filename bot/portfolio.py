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

logger = logging.getLogger(__name__)

# Default persistence path (mirrors EngineState convention in bot/engine.py).
DEFAULT_STATE_FILE = Path("logs/portfolio_state.json")

# Minimum return observations needed for a trustworthy Kelly estimate.
# Below this → statistics unreliable → weight falls back to 0.
_MIN_SAMPLES = 10


# ── Helpers ─────────────────────────────────────────────────────────


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Coerce to float, drop NaN/inf, return a clean Series."""
    cleaned = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return cleaned


def _kelly_fraction(returns: pd.Series) -> float:
    """Full Kelly fraction for one strategy: f* = mean(r) / var(r).

    Returns 0.0 on negative edge, zero variance, or insufficient data.
    """
    r = _clean_returns(returns)
    n = len(r)

    if n < _MIN_SAMPLES:
        logger.warning("Kelly: only %d samples (need ≥%d) — returning 0.0", n, _MIN_SAMPLES)
        return 0.0

    mean = float(r.mean())
    var = float(r.var(ddof=1))  # sample variance

    if math.isclose(var, 0.0, abs_tol=1e-12):
        logger.warning("Kelly: zero variance over %d samples — returning 0.0", n)
        return 0.0

    f = mean / var

    if f <= 0:
        logger.info(
            "Kelly: non-positive edge (mean=%.6f, var=%.6f → f*=%.4f) — capping at 0",
            mean, var, f,
        )
        return 0.0

    logger.debug("Kelly: mean=%.6f var=%.6f → f*=%.4f (n=%d)", mean, var, f, n)
    return f


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
