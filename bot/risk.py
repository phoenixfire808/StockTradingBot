"""Risk management: position sizing, stop/target computation, daily loss guard."""

import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


def position_size(
    equity: float,
    price: float,
    stop_distance: float,
    risk_per_trade: float,
) -> int:
    """Compute share count from equity and risk parameters.

    Shares = floor((equity * risk_per_trade) / stop_distance), capped at 25 % of equity.
    Returns 0 if stop_distance <= 0 (invalid).
    """
    if stop_distance <= 0 or price <= 0:
        logger.warning("Invalid stop_distance=%.4f or price=%.4f - returning 0", stop_distance, price)
        return 0

    raw_shares = (equity * risk_per_trade) / stop_distance
    equity_cap = int(equity * 0.25 / price)
    shares = min(int(raw_shares), equity_cap)
    return max(shares, 0)


def stop_loss(entry_price: float, atr: float) -> float:
    """Stop-loss: entry minus 2xATR."""
    return entry_price - 2 * atr


def take_profit(entry_price: float, atr: float) -> float:
    """Take-profit: entry plus 3xATR (1.5:1 R:R)."""
    return entry_price + 3 * atr


@dataclass
class PositionState:
    """Internal tracking for a single position."""
    symbol: str
    qty: int
    entry_price: float
    entry_ts: str
    stop: float
    target: float


class KillSwitch:
    """Halt trading when daily drawdown exceeds threshold.

    Usage: call ``reset_day(starting_equity)`` once per trading day,
    then call ``check(current_equity)`` each cycle.
    Also respects the flag file ``logs/kill_switch.flag``.
    """

    def __init__(self, max_daily_loss_pct: float = 3.0) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.day_start_equity: float | None = None
        self.tripped: bool = False

    def reset_day(self, equity: float) -> None:
        self.day_start_equity = equity
        self.tripped = False
        # Clear the UI-writable flag on day reset
        flag = Path("logs/kill_switch.flag")
        if flag.exists():
            flag.unlink()
            logger.info("Kill-switch re-armed: cleared logs/kill_switch.flag")

    def check(self, current_equity: float) -> bool:
        """Return True if trading should be halted this cycle."""
        if self.day_start_equity is None or self.day_start_equity == 0:
            return False

        drawdown_pct = ((self.day_start_equity - current_equity) / self.day_start_equity) * 100
        if drawdown_pct >= self.max_daily_loss_pct:
            self.tripped = True
            logger.critical(
                "KILLSWITCH TRIPPED - daily loss %.2f%% exceeds %.1f%% cap",
                drawdown_pct, self.max_daily_loss_pct,
            )
            Path("logs/kill_switch.flag").touch()

        # Check UI manual emergency stop
        if Path("logs/kill_switch.flag").exists():
            self.tripped = True

        return self.tripped
# ---------------------------------------------------------------------------
# Trailing stop-loss with multi-tier thresholds
# ---------------------------------------------------------------------------


class TrailingStopLoss:
    """Manages trailing stops with configurable tier behavior.

    Inspired by 's ``trailing_stop`` pattern: the stop price
    trails behind the *highest* observed price (the draw high), but
    uses **different** trail distances depending on how far into profit
    the position currently is.

    Tier example::

        tiers = {
            0.05: 0.02,   # at +5 % profit trail by 2 %
            0.10: 0.03,   # at +10 % profit trail by 3 %
            0.20: 0.05,   # at +20 % profit trail by 5 %
        }

    The *lowest* applicable tier (based on current profit) governs the
    active trail distance -- so a +7 % position would use the 5 % tier's
    2 % distance.

    Parameters
    ----------
    trail_pct : float
        Default fixed trailing distance (used when no tiers are set).
    tiers : dict[float, float] | None
        Mapping of ``{profit_threshold_pct : trail_pct}``.
    activate_at_profit : float
        Minimum profit required before trailing activates (default 0).
    """

    def __init__(
        self,
        *,
        trail_pct: float = 0.05,
        tiers: dict[float, float] | None = None,
        activate_at_profit: float = 0.0,
    ) -> None:
        self.trail_pct = trail_pct
        self.tiers: dict[float, float] = tiers or {}
        self.activate_at_profit = activate_at_profit
        self.highest_price: float = 0.0
        self.stop_price: float = 0.0
        self._active_trail: float = trail_pct

    # -- public API -------------------------------------------------------

    def register_position(self, entry_price: float) -> None:
        """Call once per open position."""
        self.highest_price = entry_price
        self.stop_price = entry_price
        self._recalculate(entry_price, entry_price)

    def update(
        self,
        current_price: float,
        entry_price: float,
    ) -> None:
        """Update state for each bar/tick; mutates ``stop_price`` in place."""
        if current_price > self.highest_price:
            self.highest_price = current_price
        self._recalculate(current_price, entry_price)

    @property
    def current_profit_pct(self) -> float:
        """Return (current - entry) / entry based on highest price."""
        if self.stop_price == 0:
            return 0.0
        return (self.highest_price - self.stop_price) / self.highest_price

    @property
    def trail_distance(self) -> float:
        """Active trailing distance considering tier configuration."""
        return self._active_trail

    # -- internal helpers --------------------------------------------------

    def _get_active_tier(self, profit_pct: float) -> float:
        """Return the trail pct for the given profit percentage.

        Per  trailing-stop convention, the tier whose profit
        threshold is the highest value still less-than-or-equal to the
        current profit governs -- i.e. the tightest bracket the position
        has climbed into.
        """
        if not self.tiers:
            return self.trail_pct

        # Find all tiers met by current profit, pick highest threshold
        best_thresh = None
        for thresh, tp in sorted(self.tiers.items()):
            if profit_pct >= thresh:
                best_thresh = thresh
            else:
                break

        tiers_dict = dict(sorted(self.tiers.items()))
        return tiers_dict[best_thresh] if best_thresh is not None else self.trail_pct

    def _recalculate(self, current_price: float, entry_price: float) -> None:
        """Compute new stop from highest-watermark and active trail."""
        current_profit_pct = (current_price - entry_price) / entry_price

        if current_profit_pct < self.activate_at_profit:
            # Not profitable enough -- keep stop at entry (never trails below)
            if self.stop_price == 0 or current_profit_pct >= self.activate_at_profit:
                self.stop_price = max(self.stop_price, entry_price)
            else:
                self.stop_price = entry_price
            return

        self._active_trail = self._get_active_tier(current_profit_pct)
        candidate = self.highest_price * (1.0 - self._active_trail)
        # Never move stop down; floor at least at entry price
        stop_floor = max(entry_price, self.stop_price)
        effective_candidate = max(candidate, stop_floor)
        if effective_candidate > self.stop_price:
            self.stop_price = effective_candidate


# ---------------------------------------------------------------------------
# ROI-based exit ( roi_thresholds pattern)
# ---------------------------------------------------------------------------


def roi_based_exit(
    current_price: float,
    entry_price: float,
    roi_table: dict[float, int],
    hold_minutes: int = 0,
) -> bool:
    """Determine whether an ROI-based exit signal should fire.

    Parameters
    ----------
    current_price : float
        Latest market price.
    entry_price : float
        Position fill price.
    roi_table : dict[float, int]
        ``{profit_pct : max_hold_minutes}`` mapping per 
        ``roi_thresholds`` convention.  A value of ``0`` for minutes means
        *exit immediately* once that profit threshold is crossed.
    hold_minutes : int
        Minutes the position has been held since entry.

    Returns
    -------
    bool
        ``True`` when the exit condition is satisfied; ``False`` otherwise.

    Examples
    --------
    >>> roi_table = {0.05: 60, 0.10: 0}  # hold up to 60 min at +5 %
    ... roi_based_exit(108, 100, roi_table, hold_minutes=70)
    True
    ... roi_based_exit(108, 100, roi_table, hold_minutes=30)
    False
    """
    if entry_price <= 0 or current_price <= 0:
        return False

    profit_pct = (current_price - entry_price) / entry_price

    # Check ALL tiers whose threshold is met by current profit.
    # Any single tier firing returns True immediately.
    for threshold, max_hold in sorted(roi_table.items()):
        if profit_pct >= threshold:
            if max_hold == 0:
                return True
            if hold_minutes >= max_hold:
                return True

    return False
