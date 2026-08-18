"""Shared theme utilities for Streamlit dashboard.

Provides CSS injection for dark terminal look, styled metric cards,
P&L colorization, broker badges, and data sourcing helpers.
"""

from typing import Any

import streamlit as st


# ── CSS Injection ──────────────────────────────────────────────────

_DARK_CSS = """
<style>
/* Dark terminal backgrounds */
[data-testid="stSidebar"] { background: #121520; }
.stMetric { background: #1A1F2E; border-radius: 8px; padding: 12px; }
/* Number coloring */
.pnl-pos { color: #00C896; font-weight: bold; }
.pnl-neg { color: #FF5252; font-weight: bold; }
.pnl-zero { color: #888; }
/* Status indicator dots */
.dot-green { display: inline-block; width: 10px; height: 10px; background: #00C896; border-radius: 50%; margin-right: 6px; }
.dot-red { display: inline-block; width: 10px; height: 10px; background: #FF5252; border-radius: 50%; margin-right: 6px; }
.dot-amber { display: inline-block; width: 10px; height: 10px; background: #FFB74D; border-radius: 50%; margin-right: 6px; }
/* Card borders */
.card-bordered { border: 1px solid #2a3040; border-radius: 8px; padding: 10px; background: #1A1F2E; }
/* Table styling */
.stDataFrame { background: #0E1117; color: #E6E9EF; }
.killswitch-banner { background: #4A0000; border: 2px solid #FF5252; border-radius: 8px; padding: 10px; text-align: center; font-size: 1.1em; color: #FF5252; font-weight: bold; }
.armed-banner { background: #003322; border: 2px solid #00C896; border-radius: 8px; padding: 10px; text-align: center; font-size: 1.1em; color: #00C896; font-weight: bold; }
</style>
"""


@st.cache_resource
def inject_css() -> None:
    """Inject dark terminal CSS once per session."""
    st.markdown(_DARK_CSS, unsafe_allow_html=True)


# ── Styling Helpers ────────────────────────────────────────────────

def pnl_color(value: float | int) -> str:
    """Return CSS class for P&L value: green / red / neutral."""
    if value > 0:
        return "pnl-pos"
    elif value < 0:
        return "pnl-neg"
    return "pnl-zero"


def pnl_badge(value: float | int) -> str:
    """Render colored P&L badge."""
    cls = pnl_color(value)
    sign = "+" if value >= 0 else ""
    return f'<span class="{cls}">{sign}{value}</span>'


def status_dot(color: str) -> str:
    """Render status indicator dot."""
    return f'<span class="dot-{color}"></span>'


def broker_badge(mode: str = "mock") -> str:
    """Render broker mode badge."""
    if mode == "live":
        return status_dot("red") + '<b>LIVE — AGENTIC ACCOUNT</b>'
    elif mode == "stdio":
        return status_dot("green") + '<b>STDIO PROXY — AGENTIC ACCOUNT</b>'
    return status_dot("amber") + '<b>MOCK DATA (NO BROKER)</b>'


def risk_metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Styled risk metric card."""
    st.metric(label=label, value=value, delta=delta)


def kill_switch_banner(is_tripped: bool, reason: str = "") -> None:
    """Display kill-switch status banner."""
    if is_tripped:
        msg = f"TRIPPED — trading halted"
        if reason:
            msg += f" ({reason})"
        st.markdown(f'<div class="killswitch-banner">🛑 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="armed-banner">✅ ARMED — trading active</div>', unsafe_allow_html=True)


def auto_refresh(seconds: int = 5, key: str = "autorefresh") -> None:
    """Auto-refresh toggle using session state."""
    enabled = st.toggle("Auto-refresh", value=True, key=key)
    if enabled:
        import time
        time.sleep(seconds)
        st.rerun()


def get_engine_status(path: str = "logs/engine_state.json") -> dict[str, Any] | None:
    """Read current engine state from disk."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        import json
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def get_positions_file(path: str = "logs/positions_state.json") -> dict[str, Any] | None:
    """Read positions state from disk."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        import json
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None
