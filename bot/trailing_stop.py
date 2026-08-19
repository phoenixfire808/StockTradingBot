"""Trailing stop-loss management with profit-based tier progression.

Derived from ::strategy::interface trailing_stop() pattern.

Implements progressive stop-loss tightening as positions move in your favor:
  - Entry: static ATR-based stop (already set via bot.risk.stop_loss)
  - Tier 1: When profit >= threshold_pct, lock stop at breakeven + buffer
  - Tier 2+: Progressive trailing tighter as price rises further
  - Always tracks highest price reached; trail level = highest * (1 - trail_pct)

Position dict keys added on BUY:
  - trail_pct: float — base trailing percentage from high-water mark
  - highest_price: float — running high-water mark (updated each cycle)
  - tier_level: int — how many profit tiers have been unlocked

Integration: check_trailing_stop() called in engine exit loop BEFORE
the existing static stop-check. If trailing stop is hit, returns True
with reason 'trailing_stop_tier_N'.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Default profit-tier configuration ─────────────────────────────

_DEFAULT_TIER_CONFIG = [
    {"threshold_pct": 0.02, "trail_pct": 0.015},   # Tier 1: 2% profit → 1.5% trail
    {"threshold_pct": 0.05, "trail_pct": 0.025},   # Tier 2: 5% profit → 2.5% trail
    {"threshold_pct": 0.10, "trail_pct": 0.035},   # Tier 3: 10% profit → 3.5% trail
    {"threshold_pct": 0.20, "trail_pct": 0.04},    # Tier 4: 20% profit → 4.0% trail
]


def _current_profit(entry_price: float, current_price: float) -> float:
    """Compute fractional profit for a position."""
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    return (current_price - entry_price) / entry_price


def update_high_water(position: dict, current_price: float) -> float:
    """Update the running high-water mark for a position.

    Args:
        position: The internal_pos entry for the symbol.
        current_price: Current market price of the position.

    Returns:
        Updated highest_price value.
    """
    old_highest = position.get("highest_price", 0.0)
    new_highest = max(old_highest, current_price)
    if new_highest > old_highest:
        logger.debug(
            "HIGHEST PRICE updated for %s: %.2f → %.2f",
            position.get("symbol", "?"),
            old_highest,
            new_highest,
        )
    position["highest_price"] = new_highest
    return new_highest


def compute_trail_level(highest_price: float, trail_pct: float) -> float:
    """Compute the trailing stop level from a high-water mark.

    trail_level = highest_price × (1 - trail_pct)
    Never drops below the original static stop loss.
    """
    base_stop = highest_price * (1.0 - trail_pct)
    return round(base_stop, 6)


def check_trailing_stop(
    position: dict,
    current_price: float,
    tier_config: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check whether the trailing stop has been hit.

    Works in addition to (before) the static stop-loss check.
    Updates the high-water mark and raises the stop progressively
    through profit tiers.

    Args:
        position: Internal position dict containing entry_price, highest_price,
                  stop (original static stop), and optionally tier_level.
        current_price: Latest market price.
        tier_config: Profit tier thresholds; uses defaults if None.

    Returns:
        (is_trailing_stop_hit, reason_string)
        reason_string is e.g. "trailing_stop_tier_0" (0-indexed).
    """
    if tier_config is None:
        tier_config = _DEFAULT_TIER_CONFIG

    entry_price = position.get("entry_price", 0.0)
    if entry_price <= 0 or current_price <= 0:
        return False, ""

    # Update high-water mark
    update_high_water(position, current_price)
    highest = position["highest_price"]

    # Original static stop (entry - 2×ATR)
    orig_stop = position.get("stop", 0.0)
    if orig_stop <= 0:
        return False, ""

    # Determine current profit fraction
    profit_pct = _current_profit(entry_price, highest)

    # Find the applicable tier based on highest profit reached
    applied_tier = -1
    for i, tier in enumerate(tier_config):
        threshold = tier["threshold_pct"]
        if profit_pct >= threshold:
            applied_tier = i

    # Compute the trail level using the tightest applicable tier
    if applied_tier >= 0:
        trail_pct = tier_config[applied_tier]["trail_pct"]
    else:
        # No tier reached yet — no trailing, just use static stop
        return False, ""

    trail_level = compute_trail_level(highest, trail_pct)

    # Never go below the original stop — trail can only tighten, never loosen
    trail_level = max(trail_level, orig_stop)

    # Update the position stop if trailing is above static
    if trail_level > position.get("stop", 0.0):
        position["stop"] = trail_level
        position["tier_level"] = applied_tier
        position["last_trail_reason"] = f"trailing_stop_tier_{applied_tier}"

    # Check if current price has broken through the trailing stop
    if current_price <= trail_level:
        reason = f"trailing_stop_tier_{applied_tier}"
        return True, reason

    return False, ""


def get_current_trailing_info(position: dict) -> dict:
    """Return human-readable trailing-stop diagnostics for dashboard/logging.

    Returns dict with:
      current_profit_pct, highest_price, current_stop, tier_reached, trail_pct_applied
    """
    entry = position.get("entry_price", 0)
    highest = position.get("highest_price", entry)
    current_stop = position.get("stop", 0)
    tier_level = position.get("tier_level", -1)

    profit_pct = _current_profit(entry, highest) if entry > 0 else 0.0

    trail_pct_applied = 0.0
    if tier_level >= 0 and tier_level < len(_DEFAULT_TIER_CONFIG):
        trail_pct_applied = _DEFAULT_TIER_CONFIG[tier_level]["trail_pct"]

    return {
        "current_profit_pct": round(profit_pct, 4),
        "highest_price": round(highest, 4),
        "current_stop": round(current_stop, 4),
        "tier_reached": tier_level,
        "trail_pct_applied": trail_pct_applied,
    }
