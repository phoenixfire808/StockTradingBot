"""Kelly Criterion — core computations and configuration.

Two Kelly formulas supported:

1. Returns-based (portfolio allocation):  f* = mean(r) / var(r)
   Derivation: maximize E[log(1 + f*r)] ~ f*E[r] - f^2*Var[r]/2 => f* = E[r]/Var[r].
   Used by ``allocate_kelly`` in :py:mod:`bot.portfolio` to split capital across strategies.

2. Win-rate / payoff (trade-level sizing): f* = w - (1-w) / b
   where w = win-rate, b = avg_win / avg_loss (payoff ratio).
   Used to scale per-trade risk budgets upward when historical performance justifies it.

Both are scaled by a configurable fractional factor (default 0.25 = quarter-Kelly)
and clamped to [0, max_fraction] to prevent overbetting.

Usage::

    from bot.kelly import KellyConfig, compute_returns_kelly, compute_winrate_payoff_kelly

    cfg = KellyConfig(fractional=0.25)

    # Strategy allocation via rolling returns
    f_star = compute_returns_kelly(strategy_returns_series, cfg)

    # Per-trade sizing via empirical trade stats
    f_star = compute_winrate_payoff_kelly(win_rate, avg_win, avg_loss, cfg)

References
----------
* Kelly, J. L. (1956). "A New Interpretation of Information Rate."
* Thorp, E. O. (2008). "The King of Optimization."
* Zagmunk, F. (2011). "An Introduction to the Kelly Criterion."
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────

_DEFAULT_FRACTIONAL = 0.25        # quarter-Kelly (conservative industry standard)
_DEFAULT_MAX_FRACTION = 0.50      # half-Kelly hard cap (prevents overbetting)
_DEFAULT_MIN_SAMPLES = 30         # min observations before trusting estimate
_DEFAULT_METHOD = "returns"       # "returns" | "winrate_payoff" | "disabled"
_DEFAULT_ENABLED = False          # feature-flag: off-by-default for safety
_DEFAULT_TRACK_RETURNS_WINDOW = 90  # rolling-window length for returns-based Kelly


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class KellyConfig:
    """Configuration controlling Kelly criterion behavior.

    Parameters
    ----------
    method : str
        Which Kelly formula to use. Options:
        - ``"returns"``: f* = mean/var from rolling returns series (portfolio allocation)
        - ``"winrate_payoff"``: f* = w - (1-w)/b from empirical trade stats (per-trade sizing)
        - ``"disabled"``: skip Kelly entirely, fall back to base risk params
    fractional : float
        Fraction of full Kelly to apply (0, 1]. Quarter-Kelly (0.25) is the default;
        industry best practice. Full Kelly is rarely used in production due to
        estimation error and fat-tail risk.
    max_fraction : float
        Absolute ceiling on output f* (prevents overbetting even if raw computation
        yields large values). Default 0.5 (half-Kelly convention from Thorp & Zagmunk).
    min_samples : int
        Minimum number of observations/trades required before using Kelly estimate.
        Below this threshold, functions return 0.0 and callers fall back to base risk.
    track_returns_window : int
        Rolling window length (in bars/cycles) for returns-based Kelly. Only used
        when callers supply truncated Series; the function itself does not fetch data.
    enabled : bool
        Feature flag. When False, callers should check early and skip Kelly paths.

    Attributes
    ----------
    kelly_method : str
        Synonym for ``method`` kept for backwards compat with old naming.
    """
    method: str = _DEFAULT_METHOD
    kelly_method: str = field(default=_DEFAULT_METHOD, repr=False, init=False)
    fractional: float = _DEFAULT_FRACTIONAL
    max_fraction: float = _DEFAULT_MAX_FRACTION
    min_samples: int = _DEFAULT_MIN_SAMPLES
    track_returns_window: int = _DEFAULT_TRACK_RETURNS_WINDOW
    enabled: bool = _DEFAULT_ENABLED

    def __post_init__(self) -> None:
        # Sync old-style attr for backwards compat
        object.__setattr__(self, "kelly_method", self.method)

        # Clamp fractional to valid range; log and adjust if invalid
        if not 0 < self.fractional <= 1.0:
            logger.warning("KellyConfig: fractional=%.4f outside (0,1] — clamping to %s",
                           self.fractional, _DEFAULT_FRACTIONAL)
            object.__setattr__(self, "fractional", _DEFAULT_FRACTIONAL)

        # Validate max_fraction
        if not 0 < self.max_fraction <= 1.0:
            logger.warning("KellyConfig: max_fraction=%.4f outside (0,1] — clamping to %s",
                           self.max_fraction, _DEFAULT_MAX_FRACTION)
            object.__setattr__(self, "max_fraction", _DEFAULT_MAX_FRACTION)

        # Clamp method to known values
        if self.method not in ("returns", "winrate_payoff", "disabled"):
            logger.warning("KellyConfig: unknown method=%r — resetting to %s",
                           self.method, _DEFAULT_METHOD)
            object.__setattr__(self, "method", _DEFAULT_METHOD)
            object.__setattr__(self, "kelly_method", _DEFAULT_METHOD)

        # Min samples sanity check
        if self.min_samples < 1:
            object.__setattr__(self, "min_samples", _DEFAULT_MIN_SAMPLES)

    @property
    def disabled(self) -> bool:
        """Return True if Kelly is effectively disabled."""
        return not self.enabled or self.method == "disabled"

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        d = asdict(self)
        # Remove deprecated duplicate attr from output
        d.pop("kelly_method", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "KellyConfig":
        """Deserialize from dict / JSON object."""
        clean = {k: v for k, v in data.items() if k != "kelly_method"}
        return cls(**clean)

    @classmethod
    def from_env(cls) -> "KellyConfig":
        """Read Kelly config from environment variables with sane defaults.

        Environment variables recognized:
        - ``KELLY_ENABLED``: "true"/"false"
        - ``KELLY_METHOD``: "returns"|"winrate_payoff"|"disabled"
        - ``KELLY_FRACTIONAL``: float in (0, 1]
        - ``KELLY_MAX_FRACTION``: float in (0, 1]
        - ``KELLY_MIN_SAMPLES``: int >= 1
        - ``KELLY_TRACK_RETURNS_WINDOW``: int >= 1
        """
        import os

        enabled = os.getenv("KELLY_ENABLED", "false").lower() == "true"
        method = os.getenv("KELLY_METHOD", _DEFAULT_METHOD)
        try:
            fractional = float(os.getenv("KELLY_FRACTIONAL", str(_DEFAULT_FRACTIONAL)))
        except ValueError:
            fractional = _DEFAULT_FRACTIONAL
        try:
            max_frac = float(os.getenv("KELLY_MAX_FRACTION", str(_DEFAULT_MAX_FRACTION)))
        except ValueError:
            max_frac = _DEFAULT_MAX_FRACTION
        try:
            min_samp = int(os.getenv("KELLY_MIN_SAMPLES", str(_DEFAULT_MIN_SAMPLES)))
        except ValueError:
            min_samp = _DEFAULT_MIN_SAMPLES
        try:
            ret_win = int(os.getenv("KELLY_TRACK_RETURNS_WINDOW",
                                    str(_DEFAULT_TRACK_RETURNS_WINDOW)))
        except ValueError:
            ret_win = _DEFAULT_TRACK_RETURNS_WINDOW

        cfg = cls(
            enabled=enabled,
            method=method,
            fractional=fractional,
            max_fraction=max_frac,
            min_samples=min_samp,
            track_returns_window=ret_win,
        )
        logger.info("KellyConfig loaded from env: enabled=%s method=%s fractional=%.3f max=%.3f min_samples=%d",
                     cfg.enabled, cfg.method, cfg.fractional, cfg.max_fraction, cfg.min_samples)
        return cfg


# ── Returns-Based Kelly ───────────────────────────────────────────────


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Coerce to float, drop NaN/inf, return a clean Series.

    Identical logic to ``bot.portfolio._clean_returns`` — callers who only need
    the returns path can import this directly without depending on portfolio.
    """
    cleaned = (
        pd.to_numeric(returns, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
    )
    return cleaned


def compute_returns_kelly(returns: pd.Series, cfg: Optional[KellyConfig] = None) -> float:
    """Full Kelly fraction from a Series of periodic returns: f* = mean(r) / var(r).

    Derivation: maximizing E[log(1 + f*r)] under a second-order Taylor approximation
    gives f* = E[r] / Var[r].  See Kelly (1956).

    Edge-case handling:
        - Negative expected edge (mean < 0)           → 0.0 (no shorting)
        - Zero variance                                → 0.0
        - Insufficient samples (< cfg.min_samples)     → 0.0
        - NaN / inf in input                           → dropped silently
        - Computed f* > cfg.max_fraction               → capped

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (e.g. daily pct change) for a single strategy or asset.
    cfg : KellyConfig | None
        Configuration; defaults to conservative quarter-Kelly when None.

    Returns
    -------
    float
        Optimal fraction in [0, cfg.max_fraction].  0.0 means no bet advised.
    """
    if cfg is None:
        cfg = KellyConfig()

    r = _clean_returns(returns)
    n = len(r)

    if n < cfg.min_samples:
        logger.debug(
            "Returns-Kelly: only %d samples (need >=%d) — returning 0.0", n, cfg.min_samples
        )
        return 0.0

    mean_val = float(r.mean())
    var_val = float(r.var(ddof=1))  # sample variance (Bessel-corrected)

    if math.isclose(var_val, 0.0, abs_tol=1e-12):
        logger.warning("Returns-Kelly: zero variance over %d samples — returning 0.0", n)
        return 0.0

    f_full = mean_val / var_val

    if f_full <= 0:
        logger.info(
            "Returns-Kelly: non-positive edge (mean=%.6f var=%.6f f*=%.4f) — capping at 0",
            mean_val, var_val, f_full,
        )
        return 0.0

    # Apply fractional scaling
    f_scaled = f_full * cfg.fractional

    # Clamp to absolute maximum
    if f_scaled > cfg.max_fraction:
        logger.info(
            "Returns-Kelly: f*=%.4f exceeds max (%.2f after fractional=%.2f) — capping at %.2f",
            f_scaled, cfg.max_fraction, cfg.fractional, cfg.max_fraction,
        )
        f_scaled = cfg.max_fraction

    logger.debug(
        "Returns-Kelly: mean=%.6f var=%.6f n=%d f_full=%.4f f_scaled=%.4f",
        mean_val, var_val, n, f_full, f_scaled,
    )
    return f_scaled


# ── Win-Rate / Payoff Kelly ──────────────────────────────────────────


def compute_winrate_payoff_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cfg: Optional[KellyConfig] = None,
) -> float:
    """Full Kelly fraction from empirical win rate and payoff ratio: f* = w - (1-w)/b.

    Derivation: given probability w of winning payoff b and probability (1-w)
    of losing 1 unit, the optimal fraction maximizing log wealth is
    f* = w - (1-w)/b.  See Kelly (1956), Thorp (2008).

    This version is suited for **per-trade** sizing: each signal triggers a
    recalculation based on how that specific strategy+symbol combo historically performs.

    Edge-case handling:
        - avg_loss <= epsilon   → 0.0 (undefined payoff ratio)
        - win_rate <= 0 or >= 1 → 0.0 (degenerate cases)
        - Infinite payoff (all wins) → f approaches w, still clamped to max_fraction
        - Computed f* > cfg.max_fraction → capped

    Parameters
    ----------
    win_rate : float
        Fraction of trades that were winners (0, 1).
    avg_win : float
        Mean profit magnitude of winning trades (absolute, positive).
    avg_loss : float
        Mean loss magnitude of losing trades (absolute, positive).
    cfg : KellyConfig | None
        Configuration; defaults to conservative quarter-Kelly when None.

    Returns
    -------
    float
        Optimal fraction in [0, cfg.max_fraction].  0.0 means no bet advised.

    Examples
    --------
    >>> compute_winrate_payoff_kelly(0.58, 200.0, 100.0)
    0.359999...
    # 58% win rate, 2:1 payoff → ~36% Kelly (clamped to max_fraction=0.5)
    """
    if cfg is None:
        cfg = KellyConfig()

    # Guard degenerate inputs
    if avg_loss <= 1e-6:
        logger.warning(
            "WinRate-Kelly: avg_loss=%.4f (near-zero) — returning 0.0", avg_loss
        )
        return 0.0
    if win_rate <= 0 or win_rate >= 1:
        logger.warning(
            "WinRate-Kelly: win_rate=%.4f at boundary — returning 0.0", win_rate
        )
        return 0.0

    # Payoff ratio: average win divided by average loss
    b = avg_win / avg_loss

    # Kelly formula: f* = w - (1 - w) / b
    f_full = win_rate - (1.0 - win_rate) / b

    if f_full <= 0:
        logger.info(
            "WinRate-Kelly: non-positive edge (w=%.4f b=%.2f f*=%.4f) — capping at 0",
            win_rate, b, f_full,
        )
        return 0.0

    # Apply fractional scaling
    f_scaled = f_full * cfg.fractional

    # Clamp to absolute maximum
    if f_scaled > cfg.max_fraction:
        logger.info(
            "WinRate-Kelly: f*=%.4f exceeds max (%.2f after fractional=%.2f) — capping at %.2f",
            f_scaled, cfg.max_fraction, cfg.fractional, cfg.max_fraction,
        )
        f_scaled = cfg.max_fraction

    logger.debug(
        "WinRate-Kelly: w=%.4f avg_win=%.2f avg_loss=%.2f b=%.2f f_full=%.4f f_scaled=%.4f",
        win_rate, avg_win, avg_loss, b, f_full, f_scaled,
    )
    return f_scaled


# ── Risk Scaling ─────────────────────────────────────────────────────


def scale_risk_with_kelly(
    base_risk_pct: float,
    kelly_fraction: float,
    minimum_risk_pct: float = 1e-3,
) -> float:
    """Blend Kelly fraction with base risk budget to produce effective risk.

    Effective risk = max(base_risk_pct * (1 + kelly_fraction), minimum_risk_pct).

    When kelly_fraction <= 0, falls back cleanly to base_risk_pct.
    When kelly_fraction > 0, scales risk proportionally — up to a practical cap.

    Parameters
    ----------
    base_risk_pct : float
        Base risk-per-trade percentage (e.g. 0.01 = 1%). Acts as floor.
    kelly_fraction : float
        Kelly-derived fraction from either returns or winrate/payoff calculation.
        Should be in [0, max_fraction] (already clamped by caller).
    minimum_risk_pct : float
        Hard minimum fraction below which risk never falls. Prevents micro-positions.

    Returns
    -------
    float
        Effective risk-per-trade percentage.

    Examples
    --------
    >>> scale_risk_with_kelly(0.01, 0.0)   # no Kelly signal
    0.01
    >>> round(scale_risk_with_kelly(0.01, 0.25), 4)  # quarter-Kelly boost
    0.0125
    >>> scale_risk_with_kelly(0.01, -0.1)  # negative Kelly (edge eroded)
    0.01
    """
    if kelly_fraction <= 0:
        return max(base_risk_pct, minimum_risk_pct)

    effective = base_risk_pct * (1.0 + kelly_fraction)
    effective = max(effective, minimum_risk_pct)

    logger.debug(
        "Risk scale: base=%.4f kelly=%.4f effective=%.4f",
        base_risk_pct, kelly_fraction, effective,
    )
    return effective


# ── Empirical Trade Statistics ────────────────────────────────────────


@dataclass
class TradeOutcome:
    """Cumulative trade outcome statistics for a (strategy, symbol) pair."""
    wins: int = 0
    losses: int = 0
    total_win_amount: float = 0.0
    total_loss_amount: float = 0.0  # negative sum of losing trades
    last_updated: str = ""

    @property
    def trade_count(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.wins / self.trade_count

    @property
    def avg_win(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.total_win_amount / self.wins

    @property
    def avg_loss(self) -> float:
        if self.losses == 0:
            return 0.0
        return abs(self.total_loss_amount) / self.losses

    @property
    def payoff_ratio(self) -> float:
        if self.avg_loss <= 1e-6:
            return float("inf")
        return self.avg_win / self.avg_loss

    def record_trade(self, pnl: float) -> None:
        """Record a realized trade P&L (positive = win, negative = loss)."""
        now_str = datetime.now(timezone.utc).isoformat()
        if pnl > 0:
            self.wins += 1
            self.total_win_amount += pnl
        elif pnl < 0:
            self.losses += 1
            self.total_loss_amount += pnl
        self.last_updated = now_str

    def to_dict(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "total_win_amount": round(self.total_win_amount, 2),
            "total_loss_amount": round(self.total_loss_amount, 2),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "payoff_ratio": round(self.payoff_ratio, 4) if self.payoff_ratio != float("inf") else "inf",
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeOutcome":
        return cls(
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            total_win_amount=data.get("total_win_amount", 0.0),
            total_loss_amount=data.get("total_loss_amount", 0.0),
            last_updated=data.get("last_updated", ""),
        )


class TradeOutcomesStore:
    """Lightweight JSON-backed store for trade-outcome statistics per (strategy, symbol).

    Stores data in ``logs/trade_outcomes.json``, keyed by strategy name then symbol.
    Uses JSON with indent=2, auto-creates parent directories, mirrors the persistence
    pattern from EngineState and PortfolioState.

    Usage::

        store = TradeOutcomesStore()
        store.record("ema_cross_rsi", "AAPL", 150.0)  # $150 win
        store.record("ema_cross_rsi", "AAPL", -80.0)   # $80 loss
        outcomes = store.read("ema_cross_rsi", "AAPL")
        f = compute_winrate_payoff_kelly(outcomes.win_rate, outcomes.avg_win, outcomes.avg_loss)
    """

    DEFAULT_PATH = Path("logs/trade_outcomes.json")

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else self.DEFAULT_PATH

    def read_all(self) -> dict[str, dict[str, dict]]:
        """Load all stored outcomes. Returns nested dict[str,str,TradeOutcomeDict]."""
        if not self._path.exists():
            logger.debug("TradeOutcomesStore: %s not found — returning empty dict", self._path)
            return {}
        try:
            with open(self._path) as f:
                data = json.load(f)
            logger.info("TradeOutcomesStore: loaded from %s (%d strategies)", self._path, len(data))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("TradeOutcomesStore: failed to read %s — %s", self._path, exc)
            return {}

    def write_all(self, data: dict[str, dict[str, dict]]) -> None:
        """Atomically overwrite the outcomes file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._path)  # atomic on POSIX; works on Windows too
        logger.info("TradeOutcomesStore: wrote %d strategies to %s", len(data), self._path)

    def read(self, strategy: str, symbol: str) -> Optional[TradeOutcome]:
        """Load outcomes for one (strategy, symbol) pair."""
        data = self.read_all()
        strat_data = data.get(strategy, {})
        sym_data = strat_data.get(symbol)
        if sym_data is None:
            return None
        return TradeOutcome.from_dict(sym_data)

    def record(self, strategy: str, symbol: str, pnl: float) -> None:
        """Record a realized trade P&L for (strategy, symbol)."""
        data = self.read_all()
        strat_data = data.setdefault(strategy, {})
        sym_data = strat_data.get(symbol)
        if sym_data is None:
            strat_data[symbol] = {}
            sym_data = {}
        outcome = TradeOutcome.from_dict(sym_data)
        outcome.record_trade(pnl)
        strat_data[symbol] = outcome.to_dict()
        self.write_all(data)

    def get_kelly_params(
        self,
        strategy: str,
        symbol: str,
        cfg: Optional[KellyConfig] = None,
    ) -> Optional[float]:
        """Convenience: load trade stats and return Kelly fraction directly.

        Returns None if insufficient data.
        """
        if cfg is None:
            cfg = KellyConfig()
        if cfg.disabled:
            logger.debug("TradeOutcomesStore: Kelly disabled — skipping")
            return None

        outcomes = self.read(strategy, symbol)
        if outcomes is None or outcomes.trade_count < cfg.min_samples:
            logger.debug(
                "TradeOutcomesStore: %s/%s has %d trades (need >=%d) — skipping Kelly",
                strategy, symbol,
                outcomes.trade_count if outcomes else 0,
                cfg.min_samples,
            )
            return None

        f = compute_winrate_payoff_kelly(
            outcomes.win_rate, outcomes.avg_win, outcomes.avg_loss, cfg,
        )
        return f
