"""Backtesting runner using the backtesting.py library.

Returns metrics dict per symbol for both CLI output and UI consumption.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


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
    try:
        strategy_cls = STRATEGIES.get(strategy_name)
    except KeyError:
        available = STRATEGIES.names()
        logger.error(f"Unknown strategy '{strategy_name}'. Available: {available}")
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {', '.join(available)}")

    # Create strategy instance with params
    if strategy_params is None:
        strategy_params = {}
    params = {**strategy_cls.params, **strategy_params}
    strategy_instance = strategy_cls(**params)

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

            # Extract metrics
            metrics = {
                "total_return_pct": round(float(stats["Return [%]"]), 2),
                "buy_hold_pct": round(float(stats["Buy & Hold Return [%]"]), 2),
                "sharpe_ratio": round(float(stats.get("Sharpe Ratio", 0)), 4),
                "max_dd_pct": round(float(stats["Max Drawdown [%]"]), 2),
                "trades": int(stats["# Trades"]),
                "win_rate_pct": round(float(stats.get("Win Rate [%]", 0)), 1),
            }

            # Get trades dataframe
            trades = bt.trades()
            if not trades.empty:
                trades["pnl"] = round(trades["pnl"], 2)
                trades["return_pct"] = round(trades["return_pct"] * 100, 2)

            # Save HTML report
            report_path = f"reports/{sym}_backtest.html"
            bt.plot(filename=report_path, open_browser=False)

            results[sym] = {
                "metrics": metrics,
                "trades_df": trades if not trades.empty else pd.DataFrame(),
                "equity_curve": bt.equity_curve,
            }
            logger.info(f"Backtest {sym}: {metrics['total_return_pct']}% return, {metrics['trades']} trades")

        except Exception as e:
            logger.exception(f"Backtest failed for {sym}: {e}")
            results[sym] = {"metrics": {}, "trades_df": pd.DataFrame(), "error": str(e)}

    return results


def print_backtest_table(results: dict[str, dict[str, Any]]) -> None:
    """Print formatted metrics table to stdout."""
    if not results:
        logger.info("No backtest results to display.")
        return

    print("\n" + "=" * 80)
    print(f"{'Symbol':<10} {'Total %':<12} {'BH %':<10} {'Sharpe':<10} {'MaxDD %':<10} {'Trades':<8} {'WinRate %':<10}")
    print("=" * 80)
    for sym, res in results.items():
        m = res.get("metrics", {})
        print(f"{sym:<10} {m.get('total_return_pct', 0):<12} {m.get('buy_hold_pct', 0):<10} {m.get('sharpe_ratio', 0):<10} {m.get('max_dd_pct', 0):<10} {m.get('trades', 0):<8} {m.get('win_rate_pct', 0):<10}")
    print("=" * 80 + "\n")


def _fetch_for_backtest(symbol: str, start: str, end: str | None = None) -> pd.DataFrame | None:
    """Fetch data for backtest using DataHub."""
    from bot.data import fetch_history
    try:
        return fetch_history(symbol, start, end, interval="1d")
    except Exception:
        return None
