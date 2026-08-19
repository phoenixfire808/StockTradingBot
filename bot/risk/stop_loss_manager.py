"""Stop-loss and take-profit management for active positions.
Implements three exit modes per  patterns:
  - Fixed percentage stop-loss
  - Trailing dollar stop-loss
  - Trailing percent stop-loss
Plus time-based stops and multi-tier take-profit targets.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class StopMode(str, Enum):
    FIXED_PERCENT = "fixed_percent"
    TRAILING_DOLLAR = "trailing_dollar"
    TRAILING_PERCENT = "trailing_percent"
class ExitReason(str, Enum):
    FIXED_STOP_HIT = "fixed_stop_hit"
    TRAILING_STOP_HIT = "trailing_stop_hit"
    TIME_LIMIT_HIT = "time_limit_hit"
    ROI_TIER_EXIT = "roi_tier_exit"
    MULTI_TIER_TP = "multi_tier_take_profit"
# ---------------------------------------------------------------------------
# Internal state tracker
# ---------------------------------------------------------------------------
@dataclass
class _TradeState:
    """Mutable per-position state kept by StopLossManager."""
    symbol: str
    entry_price: float
    mode: StopMode
    # Fixed stop loss parameters
    fixed_stop_pct: float = 0.05
    # Trailing stop parameters
    trail_pct: float = 0.05
    trail_dollar: float = 1.0
    # Time-based parameters
    max_hold_minutes: float = 60.0
    # Take-profit parameters
    roi_tiers: dict[float, int] | None = None
    multi_tier_peak_drawdown: float = 0.03
    # Runtime state (updated each tick)
    highest_price: float = 0.0
    entry_time: float = 0.0
    current_stop: float = 0.0
    last_roi_check_minutes: float = 0.0
    peak_profit: float = 0.0
    def update_current_price(self, price: float) -> None:
        """Update highest_price watermark on every bar/tick call."""
        if price > self.highest_price:
            self.highest_price = price
            self.peak_profit = (price - self.entry_price) / self.entry_price
    def fixed_stop_price(self) -> float:
        return self.entry_price * (1.0 - self.fixed_stop_pct)
    def trailing_stop_price(self) -> float:
        if self.mode == StopMode.TRAILING_DOLLAR:
            return self.highest_price - self.trail_dollar
        else:
            return self.highest_price * (1.0 - self.trail_pct)
# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------
class StopLossManager:
    """Manages stop-loss, trailing-stop, and time-based exits.
    Parameters
    ----------
    stop_mode : StopMode
        Which stop-loss strategy to use for new positions.
    default_fixed_pct : float
        Default percentage below entry for fixed stops.
    default_trail_pct : float
        Default percentage below highest price for trailing percent stops.
    default_trail_dollar : float
        Default dollar distance for trailing dollar stops.
    default_max_hold_minutes : float
        Max minutes to hold before forced exit.
    roi_tiers : dict[float, int] | None
        {profit_pct: max_hold_minutes} tier table per  roi_thresholds pattern.
    multi_tier_peak_drawdown : float
        Exit if profit falls this far below peak (e.g. 0.03 = 3%).
    """
    def __init__(
        self,
        *,
        stop_mode: StopMode = StopMode.FIXED_PERCENT,
        default_fixed_pct: float = 0.05,
        default_trail_pct: float = 0.05,
        default_trail_dollar: float = 1.0,
        default_max_hold_minutes: float = 60.0,
        roi_tiers: Optional[dict[float, int]] = None,
        multi_tier_peak_drawdown: float = 0.03,
    ) -> None:
        self.stop_mode = stop_mode
        self.default_fixed_pct = default_fixed_pct
        self.default_trail_pct = default_trail_pct
        self.default_trail_dollar = default_trail_dollar
        self.default_max_hold_minutes = default_max_hold_minutes
        self.roi_tiers = roi_tiers or {}
        self.multi_tier_peak_drawdown = multi_tier_peak_drawdown
        self._states: dict[str, _TradeState] = {}
    def new_position(
        self,
        trade_key: str,
        entry_price: float,
        mode: Optional[StopMode] = None,
        stop_pct: Optional[float] = None,
        trail_pct: Optional[float] = None,
        trail_dollar: Optional[float] = None,
        max_hold_minutes: Optional[float] = None,
        roi_tiers: Optional[dict[float, int]] = None,
        multi_tier_peak_drawdown: Optional[float] = None,
    ) -> str:
        """Register a new position. Returns trade_key."""
        m = mode or self.stop_mode
        sp = stop_pct if stop_pct is not None else self.default_fixed_pct
        tp = trail_pct if trail_pct is not None else self.default_trail_pct
        td = trail_dollar if trail_dollar is not None else self.default_trail_dollar
        mh = max_hold_minutes if max_hold_minutes is not None else self.default_max_hold_minutes
        rt = roi_tiers or self.roi_tiers
        mtdd = multi_tier_peak_drawdown if multi_tier_peak_drawdown is not None else self.multi_tier_peak_drawdown
        state = _TradeState(
            symbol=trade_key,
            entry_price=entry_price,
            mode=m,
            fixed_stop_pct=sp,
            trail_pct=tp,
            trail_dollar=td,
            max_hold_minutes=mh,
            roi_tiers=rt,
            multi_tier_peak_drawdown=mtdd,
            highest_price=entry_price,
            entry_time=time.time(),
        )
        self._states[trade_key] = state
        logger.info(
            "New position tracked: %s mode=%s entry=%.2f stop=%.2f",
            trade_key, m.value, entry_price, state.fixed_stop_price(),
        )
        return trade_key
    def update_for_position(
        self,
        trade_key: str,
        current_price: float,
    ) -> dict[str, Any]:
        """Evaluate exit conditions for an open position.
        Updates internal state and returns diagnostics dict with keys:
          - ``highest_price``: updated high watermark
          - ``current_stop``: latest computed stop-loss level
          - ``peak_profit``: best profit % since entry
          - ``exit_signal``: None or ExitReason if an exit condition fired
          - ``minutes_held``: approximate holding time in minutes
        """
        state = self._states.get(trade_key)
        if state is None:
            logger.warning("update_for_position called for unknown trade: %s", trade_key)
            return {}
        # Update watermarks
        state.update_current_price(current_price)
        # Compute stop based on mode
        if state.mode == StopMode.FIXED_PERCENT:
            new_stop = state.fixed_stop_price()
        else:
            new_stop = state.trailing_stop_price()
        state.current_stop = new_stop
        elapsed_min = (time.time() - state.entry_time) / 60.0
        state.last_roi_check_minutes = elapsed_min
        exit_signal: Optional[ExitReason] = None
        # Check hard stop hit
        if current_price <= new_stop:
            exit_signal = ExitReason.TRAILING_STOP_HIT if state.mode != StopMode.FIXED_PERCENT else ExitReason.FIXED_STOP_HIT
        # Check time limit
        if state.max_hold_minutes > 0 and elapsed_min >= state.max_hold_minutes and exit_signal is None:
            exit_signal = ExitReason.TIME_LIMIT_HIT
        # Check ROI tiers
        if exit_signal is None and state.roi_tiers:
            for profit_thresh, max_hold in sorted(state.roi_tiers.items()):
                if state.peak_profit >= profit_thresh:
                    if elapsed_min >= max_hold:
                        exit_signal = ExitReason.ROI_TIER_EXIT
                    break
        # Check multi-tier peak drawdown
        if exit_signal is None and state.roi_tiers is None:
            if state.peak_profit > 0.01:  # Only care if we have real gains
                current_drawdown = state.peak_profit - ((current_price - state.entry_price) / state.entry_price)
                if current_drawdown >= state.multi_tier_peak_drawdown:
                    exit_signal = ExitReason.MULTI_TIER_TP
        result: dict[str, Any] = {
            "highest_price": state.highest_price,
            "current_stop": new_stop,
            "peak_profit": round(state.peak_profit, 4),
            "exit_signal": exit_signal,
            "minutes_held": round(elapsed_min, 1),
        }
        return result
    def remove_position(self, trade_key: str) -> bool:
        """Remove tracking for a closed position."""
        was_present = trade_key in self._states
        self._states.pop(trade_key, None)
        if was_present:
            logger.debug("Removed tracked position: %s", trade_key)
        return was_present
    @property
    def active_positions(self) -> list[str]:
        """Return list of currently tracked trade keys."""
        return list(self._states.keys())
    @property
    def count(self) -> int:
        """Number of positions currently being tracked."""
        return len(self._states)
    def summary(self) -> list[dict]:
        """Return diagnostic info for all active positions."""
        items: list[dict] = []
        for key, state in self._states.items():
            items.append({
                "key": key,
                "mode": state.mode.value,
                "entry_price": state.entry_price,
                "current_stop": state.current_stop,
                "highest_price": state.highest_price,
                "peak_profit": round(state.peak_profit, 4),
                "elapsed_minutes": round((time.time() - state.entry_time) / 60, 1),
            })
        return items
def _parse_roi_string(value: str) -> dict[float, int]:
    """Parse a -style roi_thresholds string into dict[float, int].
    Format: ``{"100": 0, "30": 60, "0.1": 120}`` → `{100.0: 0, 30.0: 60, 0.1: 120}`
    """
    try:
        raw = json.loads(value) if isinstance(value, str) else value
        return {float(k): int(v) for k, v in raw.items()}
    except Exception:
        logger.warning("Failed to parse ROI string: %s", value)
        return {}
# ---------------------------------------------------------------------------
# Convenience helpers matching legacy bot.risk API
# ---------------------------------------------------------------------------
def stop_loss(entry_price: float, atr: float) -> float:
    """Legacy stop-loss: entry minus 2x ATR."""
    return entry_price - 2 * atr
def take_profit(entry_price: float, atr: float) -> float:
    """Legacy take-profit: entry plus 3x ATR (1.5:1 R:R)."""
    return entry_price + 3 * atr



# ---------------------------------------------------------------------------
# TrailingStopLoss & ROIBasedExit - -style dynamic trailing + ROI exits
# ---------------------------------------------------------------------------


class TrailingStopLoss:
    """Dynamic trailing stop-loss with profit-tier logic.

    Implements the same pattern that  calls trailing_stop /
    trailing_stop: as a position moves into profit the trail tightens
    in discrete tiers.  The stop price only tightens (moves up); it never
    relaxes back down.

    Default tier behaviour::

        Below +5 % profit   -> no change from initial stop
        +5 %-+10 % profit   -> trail 2 % below current high
        Above +10 % profit  -> trail 5 % below current high

    Parameters
    ----------
    initial_stop_pct : float
        Percentage below entry for the *initial* absolute stop.
    tiers : dict[float, float] | None
        Mapping ``{profit_threshold : trail_distance}``.
    activate_at_profit : float
        Minimum profit fraction before trailing kicks in (default 0.05).
    """

    DEFAULT_TIERS = [(0.05, 0.02), (0.10, 0.05)]

    def __init__(
        self,
        initial_stop_pct: float = 0.10,
        tiers: dict | None = None,
        activate_at_profit: float = 0.05,
    ) -> None:
        self.initial_stop_pct = initial_stop_pct
        self.tiers: dict[float, float] = dict(tiers or self.DEFAULT_TIERS)
        self.activate_at_profit = activate_at_profit
        self._entry_price: float = 0.0
        self._highest_price: float = 0.0
        self._stop_price: float = 0.0

    def register_position(self, entry_price: float) -> None:
        """Call once when a new position opens."""
        self._entry_price = entry_price
        self._highest_price = entry_price
        self._stop_price = entry_price * (1.0 - self.initial_stop_pct)

    def dynamic_stop(self, trade_info: dict) -> tuple:
        """Returns ``(should_update, new_stop_price)``.

        trade_info must contain: ``current_price``, ``entry_price``.
        Implements dynamic trailing based on current profit tiers:

          - Below 5 % profit:      no change (returns static initial stop)
          - 5 %-10 % profit:       trail at 2 % below current price
          - Above 10 % profit:     trail at 5 % below current price
        """
        current_price = trade_info["current_price"]
        entry_price = trade_info["entry_price"]

        # Update highest watermark
        if current_price > self._highest_price:
            self._highest_price = current_price

        current_profit = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        if current_profit < self.activate_at_profit:
            # Not profitable enough -- return initial stop unchanged
            static_stop = entry_price * (1.0 - self.initial_stop_pct)
            return (False, round(static_stop, 4))

        # Pick the tightest trail among applicable tiers
        applicable_tails = [
            tail for threshold, tail in sorted(self.tiers.items())
            if current_profit >= threshold
        ]
        active_trail = min(applicable_tails) if applicable_tails else self._highest_price * 0.05
        candidate = self._highest_price * (1.0 - active_trail)

        # Only tighten upward
        if candidate > self._stop_price:
            self._stop_price = candidate

        should_update = candidate > (entry_price * (1.0 - self.initial_stop_pct))
        return (should_update, round(candidate, 4))

    @property
    def stop_price(self) -> float:
        """Current trailing stop level."""
        return self._stop_price

    @property
    def current_profit_pct(self) -> float:
        """Profit percentage based on highest price."""
        if self._entry_price <= 0:
            return 0.0
        return (self._highest_price - self._entry_price) / self._entry_price


class ROIBasedExit:
    """ROI-based exit conditions per  interface.

    Accepts a table like ``{100: 0, 30: 60, 10: 120}`` meaning:

      - At 100 % profit: exit immediately
      - At 30 % profit:   exit after 60 minutes held
      - At 10 % profit:   exit after 120 minutes held
    """

    def __init__(self, roi_table: dict[float, int]) -> None:
        """
        Parameters
        ----------
        roi_table : dict[float, int]
            Mapping ``{profit_pct : max_hold_minutes}`` per  roi_thresholds.
            A minute value of 0 means exit immediately upon reaching that tier.
        """
        self.roi_table = dict(sorted(roi_table.items()))

    def should_exit(self, current_price: float, entry_price: float, hold_minutes: int = 0) -> bool:
        """Return True if the ROI exit condition fires right now.

        Parameters
        ----------
        current_price : float
            Current market price of the position.
        entry_price : float
            Price at which the position was opened.
        hold_minutes : int
            Minutes already held since entry.

        Returns
        -------
        bool
            True => exit signal; False => keep holding.
        """
        if entry_price <= 0 or current_price <= 0:
            return False

        profit_pct = (current_price - entry_price) / entry_price

        # Iterate from highest threshold down so the tightest constraint wins.
        # E.g. +100% profit with {100:0, 30:60, 10:120} checks 100% first -> immediate exit.
        for threshold, max_hold in reversed(list(self.roi_table.items())):
            threshold_frac = threshold / 100.0  # table uses percent, profit_pct is fraction
            if profit_pct >= threshold_frac:
                if max_hold == 0:
                    return True
                if hold_minutes >= max_hold:
                    return True
                continue  # keep checking lower tiers for additional constraints

        return False

    def next_action_hint(self, current_price: float, entry_price: float, hold_minutes: int = 0) -> str:
        """Human-readable hint about what will happen next."""
        if entry_price <= 0 or current_price <= 0:
            return "invalid prices"

        profit_pct = (current_price - entry_price) / entry_price

        # Check from highest tier down for most urgent hint
        for threshold, max_hold in reversed(list(self.roi_table.items())):
            threshold_frac = threshold / 100.0
            if profit_pct >= threshold_frac:
                if max_hold == 0:
                    return f"EXIT NOW ({threshold:.1f}% threshold reached)"
                remaining = max_hold - hold_minutes
                return f"hold until {threshold:.1f}% (+{remaining} more min)"
            remaining_gain = (threshold - profit_pct * 100)
            return f"hold until {threshold:.1f}% ({remaining_gain:.1f}% more gain needed)"

        return "hold -- no ROI constraints"
