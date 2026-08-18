"""Strategy plugin lifecycle management."""

import importlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def list_strategies() -> dict:
    """List all registered strategies with their parameters."""
    try:
        from bot.core.plugins import discover_all
        from bot.core import STRATEGIES
        discover_all()
        
        strategies = []
        for name in STRATEGIES.names():
            s = STRATEGIES.get(name)
            params = getattr(s, "params", {})
            strategies.append({
                "name": name,
                "params": params,
                "has_to_backtesting": hasattr(s, "to_backtesting_strategy"),
            })
        return {"strategies": strategies, "count": len(strategies)}
    except Exception as e:
        return {"error": str(e), "strategies": [], "count": 0}


def scaffold_strategy(strategy_name: str, description: str = "", base_class: str = "EmaCrossRsi") -> str:
    """Scaffold a new strategy plugin file."""
    plugin_dir = PROJECT_ROOT / "bot" / "plugins" / "strategies"
    filename = f"{strategy_name.replace(' ', '_').lower()}.py"
    filepath = plugin_dir / filename
    
    if filepath.exists():
        raise FileExistsError(f"Strategy file already exists: {filepath}")
    
    template = f'''"""{description or strategy_name} strategy plugin."""

import logging
import pandas as pd
from typing import Any
from bot.strategy import Strategy

logger = logging.getLogger(__name__)


class {"".join(word.title() for word in strategy_name.split("_"))}(Strategy):
    """{description or strategy_name}.
    
    TODO: Document entry/exit logic, parameter defaults, and risk integration.
    """
    
    name = "{strategy_name}"
    params: dict[str, Any] = {{}}

    def __init__(
        self,
        # TODO: Add constructor parameters here
    ) -> None:
        # TODO: Initialize parameters
        self.params = {{
            # TODO: Set default values
        }}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return int8 Series: 1=long, -1=exit, 0=flat."""
        # TODO: Implement signal logic
        return pd.Series([0] * len(df), dtype="int8", index=df.index)


# Plugin handle for auto-discovery
plugin = {"".join(word.title() for word in strategy_name.split("_"))}()
'''
    
    filepath.write_text(template, encoding="utf-8")
    return f"Scaffolded strategy plugin at: {filepath.relative_to(PROJECT_ROOT)}\n\nEdit the TODOs to implement your strategy logic."


def validate_strategy(
    strategy_name: str,
    trends: list[str] = ("uptrend", "downtrend", "sideways", "volatile"),
) -> dict:
    """Test-validate a strategy against synthetic data."""
    import numpy as np
    
    results = {}
    
    for trend_type in trends:
        n = 200
        if trend_type == "uptrend":
            values = [100 + i * 0.3 + random.uniform(-2, 2) for i in range(n)]
        elif trend_type == "downtrend":
            values = [100 - i * 0.3 + random.uniform(-2, 2) for i in range(n)]
        elif trend_type == "sideways":
            values = [100 + 2 * math.sin(i * 0.1) + random.uniform(-1, 1) for i in range(n)]
        else:  # volatile
            values = [100 + random.gauss(0, 5) * i % 20 for i in range(n)]
        
        closes = [max(v, 0.1) for v in values]
        df = pd.DataFrame({
            "Close": closes,
            "Open": [c + random.uniform(-1, 1) for c in closes],
            "High": [c + abs(random.gauss(0, 1)) for c in closes],
            "Low": [max(c - abs(random.gauss(0, 1)), 0.1) for c in closes],
            "Volume": [1000] * n,
        })
        
        try:
            from bot.core import STRATEGIES
            strat = STRATEGIES.get(strategy_name)
            signals = strat.generate_signals(df)
            
            long_count = int((signals == 1).sum())
            exit_count = int((signals == -1).sum())
            flat_count = int((signals == 0).sum())
            unique_values = set(signals.unique())
            
            results[trend_type] = {
                "long_signals": long_count,
                "exit_signals": exit_count,
                "flat_bars": flat_count,
                "unique_values": sorted(list(unique_values)),
                "valid": unique_values <= {0, 1, -1},
                "has_entries": long_count > 0,
            }
        except Exception as e:
            results[trend_type] = {"error": str(e), "valid": False}
    
    return {
        "strategy": strategy_name,
        "results": results,
        "summary": {
            t: (r["valid"] and r.get("has_entries", False)) 
            for t, r in results.items()
        },
    }


def optimize_strategy_params(
    strategy_name: str,
    param_ranges: dict[str, tuple[float, float, int]],
    backtest_start: str = "2022-01-01",
    cash: float = 100_000,
) -> dict:
    """Grid-search strategy parameters via backtesting.
    
    param_ranges: {"param_name": (min, max, steps)} or "return_best" for smart search.
    """
    try:
        from itertools import product
        
        # Build grid from ranges
        keys = list(param_ranges.keys())
        grids = []
        for key in keys:
            min_val, max_val, steps = param_ranges[key]
            step_size = (max_val - min_val) / max(steps - 1, 1)
            vals = [round(min_val + i * step_size, 4) for i in range(steps)]
            grids.append(vals)
        
        combinations = list(product(*grids))
        results = []
        
        for combo in combinations[:200]:  # Cap iterations
            params_dict = dict(zip(keys, combo))
            try:
                from bot.backtest import run_backtest
                bt_results = run_backtest(
                    symbols=["AAPL"],
                    start=backtest_start,
                    end=None,
                    cash=cash,
                    strategy_name=strategy_name,
                    strategy_params=params_dict,
                )
                
                if "AAPL" in bt_results and "metrics" in bt_results["AAPL"]:
                    m = bt_results["AAPL"]["metrics"]
                    score = m.get("total_return_pct", 0) * 0.5 + (1 if m.get("sharpe_ratio", 0) > 0 else 0) * 50
                    results.append({
                        "params": params_dict,
                        "total_return_pct": m.get("total_return_pct"),
                        "sharpe_ratio": m.get("sharpe_ratio"),
                        "max_dd_pct": m.get("max_dd_pct"),
                        "trades": m.get("trades"),
                        "win_rate_pct": m.get("win_rate_pct"),
                        "score": round(score, 2),
                    })
            except Exception:
                continue
        
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return {
            "strategy": strategy_name,
            "combinations_tested": len(results),
            "best": results[0] if results else {},
            "top_10": results[:10],
        }
    except Exception as e:
        return {"error": str(e), "optimization_results": []}


def strategy_compare(
    strategy_a: str,
    strategy_b: str,
    backtest_start: str = "2022-01-01",
    symbols: list[str] | None = None,
) -> dict:
    """Compare two strategies head-to-head on same data."""
    symbols = symbols or ["AAPL"]
    try:
        from bot.backtest import run_backtest
        
        results_a = run_backtest(symbols, start=backtest_start, strategy_name=strategy_a)
        results_b = run_backtest(symbols, start=backtest_start, strategy_name=strategy_b)
        
        comparison = {}
        for sym in symbols:
            a_metrics = results_a.get(sym, {}).get("metrics", {})
            b_metrics = results_b.get(sym, {}).get("metrics", {})
            comparison[sym] = {
                "metric": {k: a_metrics.get(k) - b_metrics.get(k, 0) for k in set(a_metrics) & set(b_metrics)},
                "a_total": a_metrics.get("total_return_pct"),
                "b_total": b_metrics.get("total_return_pct"),
            }
        
        return {"comparison": comparison, "strategy_a": strategy_a, "strategy_b": strategy_b}
    except Exception as e:
        return {"error": str(e)}
