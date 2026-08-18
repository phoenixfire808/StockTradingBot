"""Backtesting runner using the backtesting.py library.

Returns metrics dict per symbol for both CLI output and UI consumption.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _compute_calmar(stats: pd.Series, max_dd_pct: float) -> float:
    """Compute Calmar ratio: annualized return / max drawdown."""
    ann_return = stats.get("Return (Ann.) [%]", 0)
    if abs(max_dd_pct) < 0.01:
        return 0.0
    return round(float(ann_return / abs(max_dd_pct)), 4)


def _compute_sortino(stats: pd.Series) -> float:
    """Get Sortino ratio from backtesting stats."""
    if "Sortino Ratio" in stats.index:
        val = stats["Sortino Ratio"]
        return round(float(val), 4) if not pd.isna(val) else 0.0
    sharpe = stats.get("Sharpe Ratio", 0)
    if pd.isna(sharpe):
        return 0.0
    return round(float(sharpe * 1.25), 4)


def run_backtest(
    symbols: list[str],
    start: str = "2022-01-01",
    end: str | None = None,
    cash: float = 100_000,
    strategy_name: str = "ema_cross_rsi",
    strategy_params: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run backtest for each symbol. Returns {symbol: {metrics, trades_df, equity_curve}}."""
    results: dict[str, dict[str, Any]] = {}

    try:
        from backtesting import Backtest, Strategy
    except ImportError:
        logger.error("backtesting.py not installed. Run: pip install backtesting")
        raise RuntimeError("backtesting.py required")

    # Import strategy plugin from registry
    from bot.core import DATASOURCES, STRATEGIES
    from bot.core.plugins import discover_all
    discover_all()

    # Get strategy instance from plugin registry
    try:
        strategy_obj = STRATEGIES.get(strategy_name)
    except KeyError:
        available = STRATEGIES.names()
        logger.error(f"Unknown strategy '{strategy_name}'. Available: {available}")
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {', '.join(available)}")

    # Merge default params with any overrides, then create fresh instance
    if strategy_params is None:
        strategy_params = {}
    defaults = getattr(strategy_obj, "params", {})
    merged_params = {**defaults, **strategy_params}
    strategy_instance = type(strategy_obj)(**merged_params)
    bt_class = strategy_instance.to_backtesting_strategy()

    for sym in symbols:
        logger.info(f"Backtesting {sym} [{start} → {end or 'latest'}]")
        try:
            df = _fetch_for_backtest(sym, start, end)
            if df is None or df.empty:
                logger.warning(f"No data for {sym}, skipping")
                continue
        except Exception as e:
            logger.warning(f"Data fetch failed for {sym}: {e}")
            continue

        try:
            bt = Backtest(df, bt_class, cash=cash, commission=0.0005, exclusive_orders=True)
            stats = bt.run()

            # Extract core metrics
            max_dd = float(stats["Max. Drawdown [%]"])
            metrics = {
                "total_return_pct": round(float(stats["Return [%]"]), 2),
                "buy_hold_pct": round(float(stats["Buy & Hold Return [%]"]), 2),
                "sharpe_ratio": round(float(stats.get("Sharpe Ratio", 0)), 4),
                "sortino_ratio": _compute_sortino(stats),
                "calmar_ratio": _compute_calmar(stats, max_dd),
                "max_dd_pct": round(max_dd, 2),
                "trades": int(stats["# Trades"]),
                "win_rate_pct": round(float(stats.get("Win Rate [%]", 0)), 1),
                "profit_factor": round(float(stats.get("Profit Factor", 0)), 2),
                "avg_trade_dur": round(float(stats.get("Avg. Trade Duration", 0)), 1),
                "best_trade_pct": round(float(stats.get("Best Trade [%]", 0)), 2),
                "worst_trade_pct": round(float(stats.get("Worst Trade [%]", 0)), 2),
            }

            # Get trades dataframe
            trades_df = stats.get("_trades", pd.DataFrame())
            if not trades_df.empty:
                trades_df["PnL"] = round(trades_df["PnL"], 2)
                if "ReturnPct" in trades_df.columns:
                    trades_df["return_pct"] = round(trades_df["ReturnPct"] * 100, 2)

            # Save HTML report
            report_path = f"reports/{sym}_backtest.html"
            bt.plot(filename=report_path, open_browser=False)

            results[sym] = {
                "metrics": metrics,
                "trades_df": trades_df if not trades_df.empty else pd.DataFrame(),
                "equity_curve": stats.get("_equity_curve", pd.DataFrame()).reset_index(drop=True),
            }
            logger.info(f"Backtest {sym}: {metrics['total_return_pct']}% return, {metrics['trades']} trades, Sharpe={metrics['sharpe_ratio']}, Sortino={metrics['sortino_ratio']}, Calmar={metrics['calmar_ratio']}")

        except Exception as e:
            logger.exception(f"Backtest failed for {sym}: {e}")
            results[sym] = {"metrics": {}, "trades_df": pd.DataFrame(), "error": str(e)}

    return results


def compare_strategies(symbols: list[str], strategies: dict[str, dict], start: str, end: str | None, cash: float = 100_000) -> dict[str, dict]:
    """Compare multiple strategies head-to-head on same symbols/date range."""
    comparison = {}

    for strat_name, strat_params in strategies.items():
        logger.info(f"Comparing strategy: {strat_name}")
        strat_results = run_backtest(
            symbols=symbols,
            start=start,
            end=end,
            cash=cash,
            strategy_name=strat_name,
            strategy_params=strat_params,
        )

        for sym in symbols:
            if sym not in strat_results:
                continue
            m = strat_results[sym].get("metrics", {})
            key = f"{strat_name}::{sym}"
            comparison[key] = {
                **m,
                "strategy": strat_name,
                "symbol": sym,
            }

    return comparison


def print_backtest_table(results: dict[str, dict[str, Any]]) -> None:
    """Print formatted metrics table to stdout."""
    if not results:
        logger.info("No backtest results to display.")
        return

    headers = ["Symbol", "Total %", "BH %", "Sharpe", "Sortino", "Calmar", "MaxDD %", "Trades", "WinRate %"]
    print(f"\n{''.join(h.rjust(12) for h in headers)}")
    print("=" * (len(headers) * 12))

    for sym, res in sorted(results.items()):
        m = res.get("metrics", {})
        row = [sym,
               f'{m.get("total_return_pct", 0):8.2f}',
               f'{m.get("buy_hold_pct", 0):6.2f}',
               f'{m.get("sharpe_ratio", 0):6.3f}',
               f'{m.get("sortino_ratio", 0):6.3f}',
               f'{m.get("calmar_ratio", 0):6.3f}',
               f'{m.get("max_dd_pct", 0):8.2f}',
               f'{m.get("trades", 0):5d}',
               f'{m.get("win_rate_pct", 0):8.1f}']
        print("".join(h.rjust(12) for h in row))
    print()


def _fetch_for_backtest(symbol: str, start: str, end: str | None = None) -> pd.DataFrame | None:
    """Fetch data for backtest using DataHub."""
    from bot.data import fetch_history
    try:
        return fetch_history(symbol, start, end, interval="1d")
    except Exception:
        return None
