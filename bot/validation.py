"""Out-of-sample (OOS) validation for StockTradingBot strategies.

Provides date-based train/test splitting, signal-driven OOS backtesting,
performance metric utilities (Sharpe, max drawdown), and a multi-trial
:class:`OOSValidator` that runs several OOS windows, tracks metrics across
trials, and reports pass/fail against configurable criteria.

Design notes
------------
* Pure-signal backtest path (``backtest_oos``) so we never *require*
  ``backtesting.py`` to be installed for the validator to function — useful
  for unit tests, smoke tests, and lightweight CI.
* Graceful integration with ``bot.ml.model`` if/when that module lands —
  see :func:`score_predictions_oos` and ``OOSValidator.add_ml_score``.
* Detailed logging via the ``bot.validation`` logger; one log line per
  trial metric so post-mortem analysis is trivial.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _to_timestamp(value: Any) -> pd.Timestamp:
    """Coerce ``value`` to :class:`pandas.Timestamp`. Logs the conversion."""
    ts = pd.Timestamp(value)
    logger.debug("Coerced cutoff %r → %s", value, ts.isoformat())
    return ts


def _coerce_equity(equity: pd.Series | pd.DataFrame | Iterable[float]) -> pd.Series:
    """Normalize an equity curve input to a numeric :class:`pandas.Series`.

    Accepts a Series, a DataFrame with an ``Equity`` column (the convention
    used by ``bot.backtest``), or any iterable of floats.
    """
    if isinstance(equity, pd.DataFrame):
        if "Equity" in equity.columns:
            return pd.Series(equity["Equity"]).astype(float)
        if "equity" in equity.columns:
            return pd.Series(equity["equity"]).astype(float)
        # Fall back to the first numeric column
        for col in equity.columns:
            if pd.api.types.is_numeric_dtype(equity[col]):
                return pd.Series(equity[col]).astype(float)
        raise ValueError("Equity DataFrame has no numeric column")
    if isinstance(equity, pd.Series):
        return equity.astype(float)
    return pd.Series(list(equity), dtype=float)


def _coerce_returns(returns: pd.Series | Iterable[float]) -> pd.Series:
    """Normalize a returns iterable to a numeric Series with NaNs dropped."""
    if isinstance(returns, pd.Series):
        out = returns.astype(float)
    else:
        out = pd.Series(list(returns), dtype=float)
    return out.dropna()


# ──────────────────────────────────────────────────────────────────────
# Train/test split
# ──────────────────────────────────────────────────────────────────────


def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff: str | datetime | pd.Timestamp,
    embargo_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/test slices using ``cutoff`` as the boundary.

    Parameters
    ----------
    df:
        Input data. Must have a :class:`~pandas.DatetimeIndex`.
    cutoff:
        Anything :class:`pandas.Timestamp` can parse: ISO string,
        ``datetime``, ``pd.Timestamp``.
    embargo_bars:
        Number of bars to *exclude* on each side of the cutoff to prevent
        look-ahead bias. Defaults to 0 (split is exact).

    Returns
    -------
    (df_train, df_test) : tuple of DataFrame

    Notes
    -----
    The train slice contains bars **strictly before** the cutoff
    (or before ``cutoff - embargo`` when ``embargo_bars > 0``).
    The test slice contains bars **on or after** ``cutoff + embargo``.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame must have a DatetimeIndex for date-based OOS split; "
            f"got {type(df.index).__name__}"
        )

    cutoff_ts = _to_timestamp(cutoff)
    if cutoff_ts.tzinfo is None and df.index.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(df.index.tz)

    if embargo_bars > 0:
        # Walk forward ``embargo_bars`` rows past the cutoff on each side.
        df_sorted = df.sort_index()
        idx = df_sorted.index.get_indexer([cutoff_ts], method="bfill")[0]
        lower = df_sorted.index[max(idx - embargo_bars, 0)] if idx >= 0 else cutoff_ts
        upper = df_sorted.index[min(idx + embargo_bars - 1, len(df_sorted) - 1)] if idx >= 0 else cutoff_ts
        df_train = df_sorted.loc[df_sorted.index < lower].copy()
        df_test = df_sorted.loc[df_sorted.index > upper].copy()
        logger.info(
            "Split %d rows by cutoff=%s with embargo=%d → train=%d, test=%d "
            "(gap=[%s, %s])",
            len(df),
            cutoff_ts.isoformat(),
            embargo_bars,
            len(df_train),
            len(df_test),
            lower.isoformat() if hasattr(lower, "isoformat") else lower,
            upper.isoformat() if hasattr(upper, "isoformat") else upper,
        )
        return df_train, df_test

    df_train = df.loc[df.index < cutoff_ts].copy()
    df_test = df.loc[df.index >= cutoff_ts].copy()
    logger.info(
        "Split %d rows by cutoff=%s → train=%d, test=%d",
        len(df),
        cutoff_ts.isoformat(),
        len(df_train),
        len(df_test),
    )
    if len(df_train) == 0:
        logger.warning("Train slice is empty; check cutoff vs df range")
    if len(df_test) == 0:
        logger.warning("Test slice is empty; check cutoff vs df range")
    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────────────────


def sharpe_ratio(
    returns: pd.Series | Iterable[float],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio of a per-bar return stream.

    Parameters
    ----------
    returns:
        Per-bar simple returns. NaNs are dropped.
    periods_per_year:
        Number of return bars per year (252 for daily, 252*5 for hourly, etc.).
    risk_free_rate:
        Annualized risk-free rate. Defaults to 0.

    Returns
    -------
    float
        Annualized Sharpe. Returns 0.0 for empty / zero-variance inputs.
    """
    r = _coerce_returns(returns)
    if r.empty:
        logger.debug("sharpe_ratio called on empty series → 0.0")
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year if periods_per_year else 0.0
    excess = r - rf_per_period
    std = excess.std(ddof=1)
    if std is None or std == 0 or math.isnan(std):
        logger.debug("sharpe_ratio: zero/NaN std → 0.0")
        return 0.0
    sr = float(excess.mean() / std * math.sqrt(periods_per_year))
    logger.debug(
        "sharpe_ratio: n=%d mean=%.6f std=%.6f periods=%d → %.4f",
        len(r),
        excess.mean(),
        std,
        periods_per_year,
        sr,
    )
    return sr


def max_drawdown(
    equity_curve: pd.Series | pd.DataFrame | Iterable[float],
) -> float:
    """Maximum drawdown of an equity curve (returned as a negative fraction).

    Examples
    --------
    >>> max_drawdown(pd.Series([100, 110, 105, 95, 100]))
    -0.13636363636363635
    """
    equity = _coerce_equity(equity_curve)
    if equity.empty:
        logger.debug("max_drawdown called on empty series → 0.0")
        return 0.0

    running_max = equity.cummax()
    # Avoid /0 when running_max == 0
    safe_max = running_max.replace(0, np.nan)
    drawdown = (equity - running_max) / safe_max
    drawdown = drawdown.fillna(0.0)
    mdd = float(drawdown.min())
    logger.debug("max_drawdown: min=%.4f from peak=%.4f", mdd, equity.cummax().max())
    return mdd


def cagr(equity_curve: pd.Series | pd.DataFrame | Iterable[float], periods_per_year: int = 252) -> float:
    """Compound annual growth rate (CAGR) from an equity curve."""
    equity = _coerce_equity(equity_curve)
    if equity.empty or len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        logger.warning("cagr: non-positive starting equity %.2f → 0.0", start)
        return 0.0
    n_periods = len(equity)
    if n_periods <= 1:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


# ──────────────────────────────────────────────────────────────────────
# Signal-driven OOS backtest
# ──────────────────────────────────────────────────────────────────────


@dataclass
class OOSBacktestResult:
    """Container for a single train/test backtest result."""

    train: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    train_equity: list[float] = field(default_factory=list)
    test_equity: list[float] = field(default_factory=list)
    train_signals: list[int] = field(default_factory=list)
    test_signals: list[int] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""

    def to_dict(self, include_equity: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "train": self.train,
            "test": self.test,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_window": [self.train_start, self.train_end],
            "test_window": [self.test_start, self.test_end],
        }
        if include_equity:
            d["train_equity"] = self.train_equity
            d["test_equity"] = self.test_equity
        return d


def _signal_returns(
    close: pd.Series,
    signals: pd.Series,
    commission: float = 0.0005,
) -> tuple[pd.Series, pd.Series]:
    """Compute per-bar returns and the equity curve from a signal series.

    Position is assumed to be held from signal bar *t* through bar *t+1*.
    A flat (0) signal closes the position on the same bar.
    """
    sig = signals.fillna(0).astype(int).clip(-1, 1)
    # Position held from bar t closes its return on bar t+1.
    position = sig.shift(1).fillna(0)
    bar_ret = close.pct_change().fillna(0)
    strat_ret = position * bar_ret
    # Apply commission on bar transitions into a long.
    enters = (position == 1) & (position.shift(1).fillna(0) != 1)
    strat_ret = strat_ret.copy()
    strat_ret[enters] = strat_ret[enters] - commission
    equity = (1 + strat_ret).cumprod()
    return strat_ret, equity


def backtest_oos(
    strategy,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    commission: float = 0.0005,
) -> OOSBacktestResult:
    """Run a signal-driven OOS backtest for ``strategy`` on train/test data.

    The strategy is expected to expose ``generate_signals(df) -> pd.Series``
    matching the :class:`~bot.strategy.Strategy` interface. Both ``df_train``
    and ``df_test`` must have a ``Close`` column.

    Returns an :class:`OOSBacktestResult` with both train and test metrics.
    """
    result = OOSBacktestResult()

    for label, df in (("train", df_train), ("test", df_test)):
        if df is None or df.empty:
            logger.warning("backtest_oos: %s slice is empty", label)
            if label == "train":
                result.train = _empty_metrics()
            else:
                result.test = _empty_metrics()
            continue
        if "Close" not in df.columns:
            raise ValueError(
                f"{label} DataFrame is missing required 'Close' column "
                f"(columns={list(df.columns)})"
            )
        sigs = strategy.generate_signals(df)
        strat_ret, equity = _signal_returns(df["Close"], sigs, commission=commission)
        metrics = _compute_metrics(strat_ret, equity)
        if label == "train":
            result.train = metrics
            result.train_equity = [round(float(v), 6) for v in equity.tolist()]
            result.train_signals = [int(v) for v in sigs.fillna(0).astype(int).tolist()]
            result.n_train = len(df)
            if len(df):
                result.train_start = str(df.index[0])
                result.train_end = str(df.index[-1])
        else:
            result.test = metrics
            result.test_equity = [round(float(v), 6) for v in equity.tolist()]
            result.test_signals = [int(v) for v in sigs.fillna(0).astype(int).tolist()]
            result.n_test = len(df)
            if len(df):
                result.test_start = str(df.index[0])
                result.test_end = str(df.index[-1])
        logger.info(
            "backtest_oos[%s] rows=%d return_pct=%.2f sharpe=%.3f max_dd=%.2f trades=%d",
            label,
            len(df),
            metrics.get("total_return_pct", 0.0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown_pct", 0.0),
            metrics.get("trades", 0),
        )

    return result


def _empty_metrics() -> dict[str, float]:
    return {
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "trades": 0,
        "avg_bar_return_pct": 0.0,
        "cagr": 0.0,
    }


def _compute_metrics(strat_ret: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Build a metrics dict from a strategy return / equity series."""
    if strat_ret.empty:
        return _empty_metrics()

    # Trade counting: a trade is a contiguous run of non-zero positions.
    position = (strat_ret != 0).astype(int)
    position_change = position.diff().fillna(position.iloc[0] if len(position) else 0)
    entries = int(((position == 1) & (position_change == 1)).sum())
    exits = int(((position == 0) & (position_change == -1)).sum())
    n_trades = max(entries, exits)

    # Per-trade returns: aggregate consecutive non-zero bars into one PnL.
    grouped = (1 + strat_ret).cumprod()
    trade_pnls: list[float] = []
    cur_start_val: float | None = None
    prev_pos = 0
    for pos, val in zip(position.tolist(), grouped.tolist()):
        if pos == 1 and prev_pos == 0:
            cur_start_val = val / (1 + strat_ret.iloc[0]) if cur_start_val is None else val
            cur_start_val = val / (1 + strat_ret.iloc[0]) if len(trade_pnls) == 0 else cur_start_val
        if pos == 1 and cur_start_val is None:
            cur_start_val = val
        if pos == 0 and prev_pos == 1 and cur_start_val is not None:
            trade_pnls.append(val / cur_start_val - 1)
            cur_start_val = None
        prev_pos = pos

    wins = sum(1 for x in trade_pnls if x > 0)
    win_rate = (wins / len(trade_pnls) * 100) if trade_pnls else 0.0

    # Sortino: downside deviation only.
    downside = strat_ret[strat_ret < 0]
    if len(downside) >= 2 and downside.std(ddof=1) and downside.std(ddof=1) > 0:
        sortino = float(strat_ret.mean() / downside.std(ddof=1) * math.sqrt(252))
    else:
        sortino = sharpe_ratio(strat_ret) * 1.25 if strat_ret.std(ddof=1) else 0.0

    total_return_pct = (float(equity.iloc[-1]) - 1.0) * 100 if len(equity) else 0.0
    return {
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe_ratio(strat_ret), 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "win_rate_pct": round(float(win_rate), 2),
        "trades": int(n_trades),
        "avg_bar_return_pct": round(float(strat_ret.mean() * 100), 6),
        "cagr": round(cagr(equity), 4),
    }


# ──────────────────────────────────────────────────────────────────────
# ML integration (graceful: works without bot.ml.model)
# ──────────────────────────────────────────────────────────────────────


def score_predictions_oos(
    y_true: pd.Series | Iterable[float],
    y_pred: pd.Series | Iterable[float],
) -> dict[str, float]:
    """Score ML model predictions against ground-truth labels on OOS data.

    Returns a dict of regression / classification-friendly metrics. Always
    includes ``mse``, ``mae``, ``directional_accuracy``; ``r2`` is included
    when sklearn is importable (otherwise logged and set to ``NaN``).
    """
    yt = pd.Series(list(y_true) if not isinstance(y_true, pd.Series) else y_true).astype(float).reset_index(drop=True)
    yp = pd.Series(list(y_pred) if not isinstance(y_pred, pd.Series) else y_pred).astype(float).reset_index(drop=True)
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred length mismatch ({len(yt)} vs {len(yp)})")
    if yt.empty:
        logger.warning("score_predictions_oos: empty inputs")
        return {"mse": 0.0, "mae": 0.0, "directional_accuracy": 0.0, "r2": float("nan")}

    err = yp - yt
    mse = float((err ** 2).mean())
    mae = float(err.abs().mean())

    # Directional accuracy: did the sign match the prior bar?
    yt_dir = yt.diff().fillna(0)
    yp_dir = yp.diff().fillna(0)
    valid = yt_dir != 0
    if valid.sum() > 0:
        dir_acc = float(((yt_dir[valid].sign() == yp_dir[valid].sign()).sum()) / valid.sum() * 100)
    else:
        dir_acc = 0.0

    metrics: dict[str, float] = {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "directional_accuracy": round(dir_acc, 2),
    }
    try:
        from sklearn.metrics import r2_score  # type: ignore
        metrics["r2"] = round(float(r2_score(yt, yp)), 6)
    except Exception as exc:  # pragma: no cover - sklearn optional
        logger.debug("sklearn not available for r2_score (%s)", exc)
        metrics["r2"] = float("nan")

    logger.info(
        "score_predictions_oos: n=%d mse=%.6f mae=%.6f dir_acc=%.2f%% r2=%s",
        len(yt),
        metrics["mse"],
        metrics["mae"],
        metrics["directional_accuracy"],
        metrics["r2"],
    )
    return metrics


def try_load_ml_model(path: str | Path | None = None) -> Any:
    """Attempt to load an ML model from ``bot.ml.model`` if it exists.

    Returns the model object on success, ``None`` otherwise. This is purely
    a graceful convenience — validation must not require an ML model.
    """
    try:
        from bot.ml import model as _model_mod  # type: ignore
    except Exception as exc:
        logger.info("bot.ml.model not available (%s); ML integration skipped", exc)
        return None
    loader = getattr(_model_mod, "load_model", None)
    if loader is None:
        logger.info("bot.ml.model has no load_model(); ML integration skipped")
        return None
    try:
        mdl = loader(path) if path is not None else loader()
    except Exception as exc:
        logger.warning("Failed to load ML model via bot.ml.model.load_model: %s", exc)
        return None
    logger.info("Loaded ML model: %s", type(mdl).__name__)
    return mdl


# ──────────────────────────────────────────────────────────────────────
# OOSValidator
# ──────────────────────────────────────────────────────────────────────


DEFAULT_CRITERIA: dict[str, float] = {
    "min_oos_sharpe": 0.5,
    "min_oos_return_pct": 0.0,
    "max_oos_drawdown_pct": -25.0,
    "min_sharpe_retention": 0.5,   # OOS sharpe / train sharpe >= 0.5
    "max_train_test_spread_pct": 100.0,  # (train - test) <= 100% absolute
    "min_oos_win_rate_pct": 35.0,
}


@dataclass
class TrialResult:
    """Result of a single OOS trial."""

    trial_index: int
    train_window: list[str]
    test_window: list[str]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    passed: bool
    failure_reasons: list[str]


class OOSValidator:
    """Run multiple OOS trials and report pass/fail against criteria.

    Parameters
    ----------
    strategy:
        Object exposing ``generate_signals(df) -> pd.Series`` matching
        :class:`~bot.strategy.Strategy`.
    df:
        Full OHLCV DataFrame with DatetimeIndex.
    n_trials:
        How many rolling/walk-forward trials to run.
    train_frac:
        Fraction of bars (or rows) used for the train slice per trial.
    rolling:
        If True, use a rolling-window split that advances the start of each
        trial. If False (default), trials share the same start but use
        progressively later cutoffs.
    criteria:
        Dict of pass/fail thresholds. See :data:`DEFAULT_CRITERIA`.
    embargo_bars:
        Number of bars to skip at each cutoff (anti-leakage).
    commission:
        Per-trade commission used by :func:`backtest_oos`.
    periods_per_year:
        Annualization factor for Sharpe.
    """

    def __init__(
        self,
        strategy,
        df: pd.DataFrame,
        n_trials: int = 5,
        train_frac: float = 0.7,
        rolling: bool = False,
        criteria: dict[str, float] | None = None,
        embargo_bars: int = 0,
        commission: float = 0.0005,
        periods_per_year: int = 252,
    ) -> None:
        if df is None or len(df) == 0:
            raise ValueError("OOSValidator: df is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OOSValidator: df must have a DatetimeIndex")
        if not (0 < train_frac < 1):
            raise ValueError("train_frac must be in (0, 1)")
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")

        self.strategy = strategy
        self.df = df.sort_index()
        self.n_trials = int(n_trials)
        self.train_frac = float(train_frac)
        self.rolling = bool(rolling)
        self.criteria = dict(DEFAULT_CRITERIA)
        if criteria:
            self.criteria.update(criteria)
        self.embargo_bars = int(embargo_bars)
        self.commission = float(commission)
        self.periods_per_year = int(periods_per_year)

        self.trials: list[TrialResult] = []
        self._ml_metrics: dict[str, float] | None = None
        self._report: dict[str, Any] | None = None

        logger.info(
            "OOSValidator init: n_trials=%d train_frac=%.2f rolling=%s criteria=%s",
            self.n_trials,
            self.train_frac,
            self.rolling,
            self.criteria,
        )

    # ── Trial execution ────────────────────────────────────────────────

    def _trial_cutoffs(self) -> list[pd.Timestamp]:
        """Compute the cutoffs for each trial."""
        n = len(self.df)
        if self.rolling:
            # Window length = train + test = (n // (n_trials+1)) * n_trials.
            window = max(int(n / (self.n_trials + 1)), 2)
            cutoffs = []
            for i in range(self.n_trials):
                start_idx = i * window
                cutoff_idx = start_idx + int(window * self.train_frac)
                if cutoff_idx >= n:
                    break
                cutoffs.append(self.df.index[cutoff_idx])
            return cutoffs[: self.n_trials]
        # Expanding-window style: keep the same start, push cutoff forward.
        train_size = max(int(n * self.train_frac), 1)
        # Leave room for n_trials test slices.
        remaining = n - train_size
        step = max(int(remaining / self.n_trials), 1)
        cutoffs = []
        for i in range(self.n_trials):
            idx = train_size + i * step
            if idx >= n:
                break
            cutoffs.append(self.df.index[idx])
        return cutoffs[: self.n_trials]

    def run(self) -> "OOSValidator":
        """Execute all trials. Returns ``self`` for chaining."""
        cutoffs = self._trial_cutoffs()
        if not cutoffs:
            logger.error("OOSValidator: no valid cutoffs could be derived from data")
            self._report = self._empty_report()
            return self

        logger.info("OOSValidator.run: %d trials, cutoffs=%s", len(cutoffs), [str(c) for c in cutoffs])
        for i, cutoff in enumerate(cutoffs):
            try:
                df_train, df_test = train_test_split_by_date(
                    self.df, cutoff, embargo_bars=self.embargo_bars
                )
            except Exception as exc:
                logger.exception("Trial %d: split failed at cutoff=%s (%s)", i, cutoff, exc)
                continue
            if df_train.empty or df_test.empty:
                logger.warning("Trial %d: empty train/test slice, skipping", i)
                continue
            try:
                res = backtest_oos(
                    self.strategy, df_train, df_test, commission=self.commission
                )
            except Exception as exc:
                logger.exception("Trial %d: backtest_oos failed (%s)", i, exc)
                continue
            passed, reasons = self._evaluate_trial(res)
            trial = TrialResult(
                trial_index=i,
                train_window=[res.train_start, res.train_end],
                test_window=[res.test_start, res.test_end],
                train_metrics=res.train,
                test_metrics=res.test,
                passed=passed,
                failure_reasons=reasons,
            )
            self.trials.append(trial)
            logger.info(
                "Trial %d [%s → %s] PASS=%s train_sharpe=%.3f oos_sharpe=%.3f "
                "oos_dd=%.2f%% reasons=%s",
                i,
                res.train_start,
                res.test_end,
                passed,
                res.train.get("sharpe_ratio", 0.0),
                res.test.get("sharpe_ratio", 0.0),
                res.test.get("max_drawdown_pct", 0.0),
                reasons or "[]",
            )
        self._report = self._build_report()
        return self

    # ── Pass/fail evaluation ───────────────────────────────────────────

    def _evaluate_trial(self, res: OOSBacktestResult) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        c = self.criteria
        t = res.test
        train_sharpe = res.train.get("sharpe_ratio", 0.0)

        if t.get("sharpe_ratio", 0.0) < c["min_oos_sharpe"]:
            reasons.append(
                f"oos_sharpe={t.get('sharpe_ratio', 0.0):.3f} < {c['min_oos_sharpe']}"
            )
        if t.get("total_return_pct", 0.0) < c["min_oos_return_pct"]:
            reasons.append(
                f"oos_return={t.get('total_return_pct', 0.0):.2f}% < {c['min_oos_return_pct']}%"
            )
        if t.get("max_drawdown_pct", 0.0) < c["max_oos_drawdown_pct"]:
            reasons.append(
                f"oos_maxdd={t.get('max_drawdown_pct', 0.0):.2f}% > {c['max_oos_drawdown_pct']}%"
            )
        if t.get("win_rate_pct", 0.0) < c["min_oos_win_rate_pct"]:
            reasons.append(
                f"oos_win_rate={t.get('win_rate_pct', 0.0):.2f}% < {c['min_oos_win_rate_pct']}%"
            )
        if train_sharpe > 0:
            retention = t.get("sharpe_ratio", 0.0) / train_sharpe if train_sharpe else 0.0
            if retention < c["min_sharpe_retention"]:
                reasons.append(
                    f"sharpe_retention={retention:.2f} < {c['min_sharpe_retention']}"
                )
        spread = res.train.get("total_return_pct", 0.0) - t.get("total_return_pct", 0.0)
        if abs(spread) > c["max_train_test_spread_pct"]:
            reasons.append(
                f"train_test_spread={spread:.2f}% > ±{c['max_train_test_spread_pct']}%"
            )
        return (len(reasons) == 0), reasons

    # ── ML integration (optional) ──────────────────────────────────────

    def add_ml_score(
        self,
        model: Any | None,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Optionally fit/score an ML model on the train+test windows.

        ``model`` may be a fresh estimator (we'll fit it), or a pre-fit
        model with a ``predict`` method. If ``model`` is ``None``, we try
        ``bot.ml.model.load_model``. Returns a metrics dict; the last
        call's result is stored on the validator.
        """
        if model is None:
            model = try_load_ml_model()
        if model is None:
            logger.info("add_ml_score: no model available; skipping ML scoring")
            self._ml_metrics = {}
            return {}

        try:
            if hasattr(model, "fit") and not getattr(model, "_is_fitted", False):
                model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning("add_ml_score: model.fit failed (%s)", exc)

        if not hasattr(model, "predict"):
            logger.warning("add_ml_score: model has no predict(); cannot score")
            self._ml_metrics = {}
            return {}

        try:
            y_pred_train = pd.Series(model.predict(X_train))
            y_pred_test = pd.Series(model.predict(X_test))
        except Exception as exc:
            logger.exception("add_ml_score: predict failed (%s)", exc)
            self._ml_metrics = {}
            return {}

        train_metrics = score_predictions_oos(y_train, y_pred_train)
        test_metrics = score_predictions_oos(y_test, y_pred_test)
        combined = {f"train_{k}": v for k, v in train_metrics.items()}
        combined.update({f"test_{k}": v for k, v in test_metrics.items()})
        self._ml_metrics = combined
        logger.info("add_ml_score: %s", combined)
        return combined

    # ── Reporting ──────────────────────────────────────────────────────

    def _empty_report(self) -> dict[str, Any]:
        return {
            "n_trials": 0,
            "trials": [],
            "aggregate": {},
            "passed": False,
            "criteria": self.criteria,
            "ml_metrics": self._ml_metrics,
        }

    def _build_report(self) -> dict[str, Any]:
        if not self.trials:
            return self._empty_report()

        agg = self._aggregate()
        passed = all(t.passed for t in self.trials)
        report = {
            "n_trials": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
            "aggregate": agg,
            "passed": passed,
            "criteria": self.criteria,
            "ml_metrics": self._ml_metrics,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        logger.info(
            "OOSValidator report: n_trials=%d passed=%s aggregate=%s",
            len(self.trials),
            passed,
            {k: round(v, 3) if isinstance(v, float) else v for k, v in agg.items()},
        )
        return report

    def _aggregate(self) -> dict[str, float]:
        keys = ["sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                "win_rate_pct", "total_return_pct", "trades"]
        agg: dict[str, float] = {}
        for k in keys:
            train_vals = [t.train_metrics.get(k, 0.0) for t in self.trials]
            test_vals = [t.test_metrics.get(k, 0.0) for t in self.trials]
            agg[f"train_{k}_mean"] = round(float(np.mean(train_vals)), 4)
            agg[f"train_{k}_std"] = round(float(np.std(train_vals, ddof=0)), 4)
            agg[f"oos_{k}_mean"] = round(float(np.mean(test_vals)), 4)
            agg[f"oos_{k}_std"] = round(float(np.std(test_vals, ddof=0)), 4)

        # Stability: ratio of mean OOS to mean train (only meaningful for sharpe)
        sharpe_train = agg.get("train_sharpe_ratio_mean", 0.0)
        sharpe_oos = agg.get("oos_sharpe_ratio_mean", 0.0)
        if sharpe_train > 0:
            agg["sharpe_retention"] = round(sharpe_oos / sharpe_train, 4)
        else:
            agg["sharpe_retention"] = 0.0
        agg["n_passed_trials"] = int(sum(1 for t in self.trials if t.passed))
        agg["n_failed_trials"] = int(sum(1 for t in self.trials if not t.passed))
        return agg

    def report(self) -> dict[str, Any]:
        """Return the full report dict (lazy-builds if needed)."""
        if self._report is None:
            self._report = self._build_report()
        return self._report

    def passed(self) -> bool:
        """``True`` iff every trial passed its criteria."""
        return bool(self.report().get("passed", False))

    def failure_summary(self) -> list[str]:
        """Flattened list of unique failure reasons across all trials."""
        seen: set[str] = set()
        out: list[str] = []
        for t in self.trials:
            for r in t.failure_reasons:
                if r not in seen:
                    seen.add(r)
                    out.append(r)
        return out

    def to_json(self, path: str | Path, include_equity: bool = False) -> None:
        """Persist the report as JSON. Optionally inline equity curves."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # We don't keep raw equity per-trial in the dataclass; include_equity
        # is provided for API symmetry but is a no-op here unless enriched.
        path.write_text(json.dumps(self.report(), indent=2, default=_json_default))
        logger.info("Saved OOSValidator report to %s (%d bytes)", path, path.stat().st_size)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if not math.isnan(f) else None
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


__all__ = [
    "DEFAULT_CRITERIA",
    "OOSBacktestResult",
    "OOSValidator",
    "TrialResult",
    "backtest_oos",
    "cagr",
    "max_drawdown",
    "score_predictions_oos",
    "sharpe_ratio",
    "train_test_split_by_date",
    "try_load_ml_model",
]