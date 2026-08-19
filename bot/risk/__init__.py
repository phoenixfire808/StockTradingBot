"""Risk management sub-package."""

from __future__ import annotations

import importlib.util
import os

# stop_loss_manager module exports
from bot.risk.stop_loss_manager import (
    ExitReason,
    ROIBasedExit,
    StopLossManager,
    StopMode,
    TrailingStopLoss,
)

# Re-export legacy symbols from bot/risk.py (the standalone file, not the package)
_risk_file_path = os.path.join(os.path.dirname(__file__), "..", "risk.py")
_spec = importlib.util.spec_from_file_location("_bot_risk_file", _risk_file_path)
_risk_file = importlib.util.module_from_spec(_spec)  # type: ignore[assignment]
_spec.loader.exec_module(_risk_file)  # type: ignore[union-attr]

position_size = _risk_file.position_size
stop_loss = _risk_file.stop_loss
take_profit = _risk_file.take_profit
PositionState = _risk_file.PositionState
KillSwitch = _risk_file.KillSwitch

# Clean up namespace
del _risk_file, _spec, _risk_file_path

__all__ = [
    # From stop_loss_manager
    "ExitReason",
    "ROIBasedExit",
    "StopLossManager",
    "StopMode",
    "TrailingStopLoss",
    # From risk.py
    "KillSwitch",
    "PositionState",
    "position_size",
    "stop_loss",
    "take_profit",
]
