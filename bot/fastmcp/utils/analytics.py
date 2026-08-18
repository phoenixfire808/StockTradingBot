"""Analytics helpers for backtest results, trade journal, portfolio, and risk."""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _safe_json(path: Path, default: dict = None) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return default or {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default or {}


def query_trades(symbol: str | None = None, date_from: str | None = None,
                 date_to: str | None = None, side: str | None = None,
                 reason: str | None = None, limit: int = 100) -> dict:
    """Query trades CSV with filters."""
    trades_path = PROJECT_ROOT / "logs" / "trades.csv"
    if not trades_path.exists():
        return {"trades": [], "count": 0, "message": "No trade log found"}
    
    try:
        import pandas as pd
        df = pd.read_csv(trades_path)
        
        if symbol:
            df = df[df["symbol"].str.upper() == symbol.upper()]
        if side:
            df = df[df["side"].str.upper() == side.upper()]
        if reason:
            df = df[df["reason"].str.contains(reason, case=False, na=False)]
        
        if date_from or date_to:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if date_from:
                df = df[df["timestamp"] >= pd.to_datetime(date_from)]
            if date_to:
                df = df[df["timestamp"] <= pd.to_datetime(date_to)]
        
        total = len(df)
        buys = len(df[df["side"].str.upper() == "BUY"])
        sells = len(df[df["side"].str.upper() == "SELL"])
        
        top_reasons = df["reason"].value_counts().head(5).to_dict() if "reason" in df.columns else {}
        
        filtered = df.head(limit)
        records = filtered.to_dict(orient="records")
        
        return {
            "total_matches": total,
            "showing": len(records),
            "buy_count": buys,
            "sell_count": sells,
            "top_reasons": top_reasons,
            "trades": records,
        }
    except Exception as e:
        return {"error": str(e), "trades": []}


def portfolio_summary() -> dict:
    """Current positions + equity snapshot."""
    positions_data = _safe_json(PROJECT_ROOT / "logs" / "positions_state.json")
    state = _safe_json(PROJECT_ROOT / "logs" / "engine_state.json")
    
    equity = state.get("day_start_equity", 0) or 0
    mode = state.get("mode", "unknown")
    strategy = state.get("strategy", "none")
    kill_switch = state.get("kill_switch", False)
    
    pos_list = []
    for sym, pos in positions_data.items():
        pos_list.append({
            "symbol": sym,
            "qty": pos.get("qty", 0),
            "entry_price": pos.get("entry_price", 0),
            "stop": pos.get("stop", 0),
            "target": pos.get("target", 0),
        })
    
    return {
        "equity": equity,
        "mode": mode,
        "active_strategy": strategy,
        "kill_switch_active": kill_switch,
        "open_positions": pos_list,
        "position_count": len(pos_list),
    }


def equity_curve_stats() -> dict:
    """Equity history summary statistics."""
    eq_path = PROJECT_ROOT / "logs" / "equity_history.csv"
    if not eq_path.exists() or eq_path.stat().st_size == 0:
        return {"error": "No equity history data"}
    
    try:
        import pandas as pd
        df = pd.read_csv(eq_path, names=["timestamp", "equity"], header=0)
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
        
        if df.empty or df["equity"].isna().all():
            return {"error": "Invalid equity data"}
        
        values = df["equity"].dropna()
        returns = values.pct_change().dropna()
        
        daily_returns = values.diff().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * math.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Running max drawdown
        cumulative_max = values.cummax()
        drawdowns = ((values - cumulative_max) / cumulative_max * 100)
        max_dd = drawdowns.min()
        
        return {
            "data_points": len(values),
            "start_equity": float(values.iloc[0]),
            "end_equity": float(values.iloc[-1]),
            "current_equity": float(values.iloc[-1]),
            "total_gain": float(values.iloc[-1] - values.iloc[0]),
            "total_gain_pct": round(float((values.iloc[-1] / values.iloc[0] - 1) * 100), 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_daily_pnl": round(float(daily_returns.mean()), 2),
            "std_daily_pnl": round(float(daily_returns.std()), 2),
            "days_active": len(values),
        }
    except Exception as e:
        return {"error": str(e)}


def daily_pnl_breakdown(window_days: int = 14) -> dict:
    """Daily P&L breakdown grouped by day."""
    trades_path = PROJECT_ROOT / "logs" / "trades.csv"
    if not trades_path.exists() or trades_path.stat().st_size == 0:
        return {"message": "No trades recorded"}
    
    try:
        import pandas as pd
        df = pd.read_csv(trades_path)
        
        if "timestamp" not in df.columns or "price" not in df.columns or "side" not in df.columns:
            return {"error": "Missing required columns"}
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["amount"] = df.apply(lambda r: -float(r["price"]) * float(r["qty"]) if r["side"].upper() == "BUY" else float(r["price"]) * float(r["qty"]), axis=1)
        
        daily = df.groupby("date").agg(
            total_pnl=("amount", "sum"),
            buy_count=("side", lambda x: (x.str.upper() == "BUY").sum()),
            sell_count=("side", lambda x: (x.str.upper() == "SELL").sum()),
        ).tail(window_days)
        
        best_day = daily["total_pnl"].idxmax() if len(daily) > 0 else None
        worst_day = daily["total_pnl"].idxmin() if len(daily) > 0 else None
        
        return {
            "daily_records": [{
                "date": str(day),
                "total_pnl": round(float(row["total_pnl"]), 2),
                "buys": int(row["buy_count"]),
                "sells": int(row["sell_count"]),
            } for day, row in daily.iterrows()],
            "best_day": str(best_day) if best_day else None,
            "worst_day": str(worst_day) if worst_day else None,
            "period_days": len(daily),
        }
    except Exception as e:
        return {"error": str(e)}


def risk_metrics() -> dict:
    """Portfolio risk metrics: Sharpe, Sortino, Calmar, beta proxy."""
    eq_stats = equity_curve_stats()
    if "error" in eq_stats:
        return eq_stats
    
    try:
        import pandas as pd
        eq_path = PROJECT_ROOT / "logs" / "equity_history.csv"
        df = pd.read_csv(eq_path, names=["timestamp", "equity"], header=0)
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce").dropna()
        if df.empty:
            return {"error": "Insufficient equity data"}
        
        values = df["equity"]
        returns = values.pct_change().dropna()
        
        # Annualized Sharpe (assuming 252 trading days)
        annualized_vol = returns.std() * math.sqrt(252)
        annualized_return = returns.mean() * 252
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        downside_std = downside.std() * math.sqrt(252) if len(downside) > 0 else 0
        sortino = annualized_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        cumulative_max = values.cummax()
        drawdowns = (values - cumulative_max) / cumulative_max * 100
        max_dd = drawdowns.min()
        calmar = (annualized_return / abs(max_dd)) if max_dd != 0 else 0
        
        # Max consecutive losing days
        losses = (returns < 0).astype(int)
        max_consec_loss = 0
        current_streak = 0
        for loss in losses:
            if loss:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0
        
        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "annualized_return_pct": round(annualized_return, 2),
            "annualized_volatility_pct": round(annualized_vol * 100, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "max_consecutive_losses": max_consec_loss,
            "data_points": len(values),
        }
    except Exception as e:
        return {"error": str(e)}


def kill_switch_stats() -> dict:
    """Kill switch tripping frequency and conditions."""
    flag_file = PROJECT_ROOT / "logs" / "kill_switch.flag"
    state = _safe_json(PROJECT_ROOT / "logs" / "engine_state.json")
    
    is_active = flag_file.exists()
    
    return {
        "currently_active": is_active,
        "total_trips": state.get("kill_switch_trips", 0),
        "last_trip_timestamp": state.get("last_kill_switch_time"),
        "mean_equity_at_trip": state.get("avg_equity_at_trip"),
        "common_symbols": state.get("most_common_symbol_at_trip", []),
        "recommendation": "Re-arm the kill switch and review recent trade performance." if is_active else "Kill switch is armed.",
    }
