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
        logger.warning("Invalid stop_distance=%.4f or price=%.4f — returning 0", stop_distance, price)
        return 0

    raw_shares = (equity * risk_per_trade) / stop_distance
    equity_cap = int(equity * 0.25 / price)
    shares = min(int(raw_shares), equity_cap)
    return max(shares, 0)


def stop_loss(entry_price: float, atr: float) -> float:
    """Stop-loss: entry minus 2×ATR."""
    return entry_price - 2 * atr


def take_profit(entry_price: float, atr: float) -> float:
    """Take-profit: entry plus 3×ATR (1.5:1 R:R)."""
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
                "KILLSWITCH TRIPPED — daily loss %.2f%% exceeds %.1f%% cap",
                drawdown_pct, self.max_daily_loss_pct,
            )
            Path("logs/kill_switch.flag").touch()

        # Check UI manual emergency stop
        if Path("logs/kill_switch.flag").exists():
            self.tripped = True

        return self.tripped
