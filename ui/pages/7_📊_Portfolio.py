"""Portfolio allocation dashboard — pie chart of weights, per-strategy equity
curves, Kelly fractions, and risk-parity breakdown.

Reads from:
  * ``bot.portfolio.PortfolioState`` → ``logs/portfolio_state.json``
  * ``logs/equity_by_strategy.json`` → per-strategy equity curves
    (written by the engine; format: {"strategies": {name: [{"ts": iso, "equity": float}, ...]}})
  * ``logs/equity_history.csv`` → aggregate equity for context
  * ``logs/engine_state.json`` → current mode / strategy

All data sources are optional. The page degrades gracefully when files are
missing (e.g., before the engine has ever run) and shows an info banner instead
of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.theme import auto_refresh, inject_css, pnl_color

inject_css()

st.set_page_config(page_title="Portfolio", layout="wide", page_icon="🥧")
st.title("🥧 Portfolio Allocation")

# ── Auto-refresh ────────────────────────────────────────────────────────
auto_refresh(seconds=10, key="portfolio_autorefresh")

# ── Paths ────────────────────────────────────────────────────────────────
PORTFOLIO_STATE = Path("logs/portfolio_state.json")
EQUITY_BY_STRATEGY = Path("logs/equity_by_strategy.json")
EQUITY_HISTORY_CSV = Path("logs/equity_history.csv")
ENGINE_STATE = Path("logs/engine_state.json")


# ── Loaders (cached; tolerate missing/corrupt files) ────────────────────
@st.cache_data(ttl=2, show_spinner=False)
def load_portfolio_state(path: str = str(PORTFOLIO_STATE)) -> dict[str, Any]:
    """Read ``PortfolioState`` payload — empty dict if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        st.warning(f"Could not read {p.name}: {exc}")
        return {}


@st.cache_data(ttl=2, show_spinner=False)
def load_equity_by_strategy(path: str = str(EQUITY_BY_STRATEGY)) -> dict[str, list[dict[str, Any]]]:
    """Read per-strategy equity curves: ``{"strategies": {name: [{ts, equity}, ...]}}``."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and "strategies" in raw:
            return raw["strategies"]
        if isinstance(raw, dict):
            return raw
    except (json.JSONDecodeError, OSError) as exc:
        st.warning(f"Could not read {p.name}: {exc}")
    return {}


@st.cache_data(ttl=2, show_spinner=False)
def load_equity_history(path: str = str(EQUITY_HISTORY_CSV)) -> pd.DataFrame | None:
    """Read aggregate equity curve as a DataFrame."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as exc:
        st.warning(f"Could not read {p.name}: {exc}")
        return None


def load_engine_state(path: str = str(ENGINE_STATE)) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ── Data ────────────────────────────────────────────────────────────────
state = load_portfolio_state()
allocations: dict[str, float] = dict(state.get("allocations") or {})
method = state.get("method", "—")
fractional = state.get("fractional")
updated_at = state.get("updated_at")

equity_curves = load_equity_by_strategy()
equity_history_df = load_equity_history()
engine = load_engine_state()

# ── Header summary ─────────────────────────────────────────────────────
mode = (engine.get("mode") or "—").upper()
strategy_now = engine.get("strategy") or "—"

header_cols = st.columns(4)
header_cols[0].metric("Mode", mode)
header_cols[1].metric("Active Strategy", strategy_now)
header_cols[2].metric("Allocation Method", str(method).upper())
header_cols[3].metric(
    "Strategies",
    f"{len(allocations)}" if allocations else "0",
)

# Normalize MagicMock-like values from streamlit stub (test environments)
# to safe primitives; real Streamlit returns None for missing dict keys.
if not isinstance(allocations, dict):
    allocations = {}
if not isinstance(method, str):
    method = "—"
if not isinstance(fractional, (int, float)):
    fractional = None
if not isinstance(updated_at, str):
    updated_at = None

if updated_at:
    st.caption(f"Last portfolio state update: `{updated_at}`")
    if fractional is not None:
        st.caption(f"Kelly fraction: **{fractional:.2f}** (quarter-Kelly = 0.25)")

st.divider()
# ── Empty-state guard ──────────────────────────────────────────────────
if not allocations and not equity_curves and equity_history_df is None:
    st.info(
        "No portfolio data yet. The portfolio page populates once the bot:\n"
        "  • persists `logs/portfolio_state.json` (Kelly / risk-parity / equal-weight)\n"
        "  • writes `logs/equity_by_strategy.json` (per-strategy equity curves)\n\n"
        "Run `python main.py dry-run` or `python main.py live` to begin."
    )
    st.stop()

# ── Allocation pie chart ──────────────────────────────────────────────
st.subheader("📊 Capital Allocation")

if allocations:
    # Drop zero-weight entries for the pie; show them in a table below.
    nonzero = {k: float(v) for k, v in allocations.items() if v and float(v) > 0}

    pie_df = (
        pd.DataFrame(
            [{"strategy": k, "weight_pct": v * 100.0} for k, v in nonzero.items()]
        )
        if nonzero
        else None
    )

    if pie_df is not None and not pie_df.empty:
        fig = px.pie(
            pie_df,
            names="strategy",
            values="weight_pct",
            hole=0.45,
            title="Target Strategy Weights",
            color_discrete_sequence=px.colors.sequential.Mint,
        )
        fig.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent:.1%}",
            hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Capital: %{value:.2f}%<extra></extra>",
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#E6E9EF",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "All allocations are 0 — every strategy is excluded from the pie. "
            "This usually means Kelly fractions were non-positive. "
            "Increase observation history or switch to equal-weight."
        )

    # Allocation table — always show full list including zeros.
    st.markdown("##### Allocation Details")
    alloc_rows = [
        {
            "Strategy": k,
            "Weight": f"{v * 100:.2f}%",
            "Raw Fraction": f"{v:.4f}",
            "Status": "✅ Active" if v > 0 else "⚪ Zeroed",
        }
        for k, v in sorted(allocations.items(), key=lambda kv: -kv[1])
    ]
    st.dataframe(
        pd.DataFrame(alloc_rows),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No portfolio state file found at `logs/portfolio_state.json` — the engine "
        "hasn't computed an allocation yet. The Kelly / risk-parity helpers live in "
        "`bot.portfolio` and can be invoked manually if needed."
    )

st.divider()

# ── Per-strategy equity curves ────────────────────────────────────────
st.subheader("📈 Per-Strategy Equity Curves")

if equity_curves:
    curve_frames: dict[str, pd.DataFrame] = {}
    for name, points in equity_curves.items():
        if not points:
            continue
        df = pd.DataFrame(points)
        if "ts" not in df.columns or "equity" not in df.columns:
            continue
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
        if not df.empty:
            curve_frames[name] = df

    if curve_frames:
        # Combined overlay chart.
        combined = pd.concat(
            [df.assign(strategy=name) for name, df in curve_frames.items()],
            ignore_index=True,
        )
        fig = px.line(
            combined,
            x="ts",
            y="equity",
            color="strategy",
            title="Equity Over Time — by Strategy",
            labels={"ts": "Timestamp", "equity": "Equity ($)", "strategy": "Strategy"},
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#E6E9EF",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

        # Per-strategy normalized returns — useful for comparing strategies
        # at different starting equities.
        st.markdown("##### Normalized (Start = 100)")
        norm_fig = go.Figure()
        for name, df in curve_frames.items():
            if df.empty or df["equity"].iloc[0] == 0:
                continue
            base = float(df["equity"].iloc[0])
            normalized = (df["equity"] / base) * 100.0
            norm_fig.add_trace(
                go.Scatter(
                    x=df["ts"],
                    y=normalized,
                    mode="lines",
                    name=name,
                    line=dict(width=2),
                    hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Index: %{y:.2f}<extra></extra>",
                )
            )
        norm_fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#E6E9EF",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
            xaxis_title="Timestamp",
            yaxis_title="Equity Index (start = 100)",
        )
        st.plotly_chart(norm_fig, use_container_width=True)

        # Per-strategy summary table.
        st.markdown("##### Per-Strategy Summary")
        summary_rows = []
        for name, df in curve_frames.items():
            if df.empty:
                continue
            start_eq = float(df["equity"].iloc[0])
            end_eq = float(df["equity"].iloc[-1])
            pnl = end_eq - start_eq
            pct = (pnl / start_eq * 100.0) if start_eq else 0.0
            n_points = len(df)
            summary_rows.append(
                {
                    "Strategy": name,
                    "Start Equity": f"${start_eq:,.2f}",
                    "End Equity": f"${end_eq:,.2f}",
                    "P&L": f"{pnl:+,.2f}",
                    "Return %": f"{pct:+.2f}%",
                    "Observations": n_points,
                }
            )
        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "P&L": st.column_config.TextColumn(help="Absolute P&L"),
                    "Return %": st.column_config.TextColumn(help="Total return %"),
                },
            )
    else:
        st.info("Per-strategy equity curves are empty — no observations yet.")
else:
    st.info(
        "`logs/equity_by_strategy.json` not found. The engine writes this file as "
        "strategies accrue equity history. Run a backtest or dry-run to populate it."
    )

st.divider()

# ── Aggregate equity curve (for context) ───────────────────────────────
if equity_history_df is not None and not equity_history_df.empty:
    st.subheader("📉 Aggregate Equity (Bot-Level)")
    if "equity" in equity_history_df.columns:
        agg_fig = px.line(
            equity_history_df,
            x="timestamp" if "timestamp" in equity_history_df.columns else None,
            y="equity",
            title="Combined Account Equity",
            labels={"timestamp": "Timestamp", "equity": "Equity ($)"},
        )
        agg_fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#E6E9EF",
        )
        agg_fig.update_traces(line=dict(color="#00C896", width=2))
        st.plotly_chart(agg_fig, use_container_width=True)

        # Final P&L badge.
        if "equity" in equity_history_df.columns and len(equity_history_df) >= 2:
            start_eq = float(equity_history_df["equity"].iloc[0])
            end_eq = float(equity_history_df["equity"].iloc[-1])
            pnl = end_eq - start_eq
            cls = pnl_color(pnl)
            sign = "+" if pnl >= 0 else ""
            st.markdown(
                f'<div style="text-align:right; font-size:1.2em">'
                f'Total P&L: <span class="{cls}">{sign}${pnl:,.2f}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("`equity_history.csv` is missing the 'equity' column.")
else:
    st.caption("No aggregate equity history available.")

st.divider()

# ── Methodology footer ─────────────────────────────────────────────────
with st.expander("ℹ️ How are weights calculated?"):
    st.markdown(
        """
        **Allocation methods** (see `bot/portfolio.py`):

        - **Kelly criterion** (`allocate_kelly`): each strategy gets weight
          proportional to its full-Kelly fraction `f* = mean(r) / var(r)`,
          scaled by a configurable fraction (default **0.25** = quarter-Kelly).
          If every strategy has non-positive edge or fewer than 10 observations,
          falls back to **equal-weight** so capital is still deployed.
        - **Risk parity** (`allocate_risk_parity`): weight inversely proportional
          to each strategy's volatility (`w_i ∝ 1/σ_i`) so every strategy
          contributes equal risk. Strategies with zero/NaN/≤ 0 volatility are
          excluded; full fallback to equal-weight if all are invalid.
        - **Equal weight** (`allocate_equal_weight`): `1/N` per strategy.

        Persisted to `logs/portfolio_state.json` by `PortfolioState.save()`
        with `method`, `fractional`, and `updated_at` for auditability.
        """
    )