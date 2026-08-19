"""Backtesting runner using the backtesting.py library.

Returns metrics dict per symbol for both CLI output and UI consumption.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transaction cost helpers
# ---------------------------------------------------------------------------

STD_EQTY_COMMISSION_RATE = 0.005  # $ per share
STD_MINIMUM_COMMISSION = 1.00      # minimum $1 per order

BASE_SLIPPAGE = 0.001             # base proportional slippage
VOLUME_PENALTY = 0.0005           # extra slippage per volume-ratio unit
DAILY_RANGE_FRACTION = 0.002      # ~0.2% typical daily range as fraction of price


def commission_cost(ticker: str, qty: int, price: float, side: str) -> float:
    """Standard equity commission: $0.005/share, min $1.00 per order."""
    calc = STD_EQTY_COMMISSION_RATE * qty * price
    return max(calc, STD_MINIMUM_COMMISSION)


def slippage_model(volume: float, avg_volume: float) -> float:
    """Proportional slippage based on volume ratio.

    Returns the expected fractional slippage applied to a fill.
    """
    vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    return BASE_SLIPPAGE + VOLUME_PENALTY * vol_ratio


def bid_ask_spread(ticker: str, close_price: float) -> float:
    """~0.5 x average daily range as spread estimate."""
    return close_price * DAILY_RANGE_FRACTION * 0.5


class TransactionCostModel:
    """Realistic transaction cost estimator.

    Models three cost components on every trade:
      1. Commission: fixed % per side of order value.
      2. Slippage: proportional to daily volatility (ATR-based).
      3. Bid-ask spread: estimated from average daily range.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,       # 10 bps per side (20 bps round-trip)
        slippage_atr_mult: float = 0.25,       # fraction of ATR lost to slippage per fill
        spread_avg_daily_range_frac: float = 0.5,  # half-spread approx 0.5 x ADR / 2
    ) -> None:
        self.commission_rate = commission_rate
        self.slippage_atr_mult = slippage_atr_mult
        self.spread_adr_frac = spread_avg_daily_range_frac

    def estimate_entry_price(self, bar: pd.Series) -> float:
        """Return the effective entry price including slippage + spread overhead."""
        atr = self._atr_from_bar(bar)
        if atr <= 0:
            return float(bar["Close"])
        premium = atr * (self.slippage_atr_mult + self.spread_adr_frac)
        return float(bar["Close"]) + premium

    def estimate_exit_price(self, bar: pd.Series) -> float:
        """Return the effective exit price including slippage + spread drag."""
        atr = self._atr_from_bar(bar)
        if atr <= 0:
            return float(bar["Close"])
        drag = atr * (self.slippage_atr_mult + self.spread_adr_frac)
        return float(bar["Close"]) - drag

    def compute_trade_costs(
        self,
        entry_bar: pd.Series,
        exit_bar: pd.Series,
        n_shares: int,
        side: str = "long",
    ) -> dict[str, float]:
        """Compute per-trade cost breakdown given entry/exit bars and shares."""
        entry_price = self.estimate_entry_price(entry_bar)
        exit_price = self.estimate_exit_price(exit_bar)

        gross_pnl = n_shares * (exit_price - entry_price) if side == "long" else n_shares * (entry_price - exit_price)

        comm_entry = self.commission_rate * entry_price * n_shares
        comm_exit = self.commission_rate * exit_price * n_shares
        total_commission = comm_entry + comm_exit

        slippage_cost = abs(float(entry_bar["High"] - entry_price) + float(exit_price - exit_bar["Low"])) * n_shares

        avg_dr = float(entry_bar["High"] - entry_bar["Low"]) + float(exit_bar["High"] - exit_bar["Low"])
        half_spread_each_side = avg_dr * self.spread_adr_frac * 0.5
        spread_cost = half_spread_each_side * n_shares * 2

        total_cost = total_commission + slippage_cost + spread_cost
        net_pnl = gross_pnl - total_cost

        return {
            "entry_price_raw": round(float(entry_bar["Close"]), 4),
            "entry_price_effective": round(entry_price, 4),
            "exit_price_raw": round(float(exit_bar["Close"]), 4),
            "exit_price_effective": round(exit_price, 4),
            "n_shares": n_shares,
            "side": side,
            "gross_pnl": round(gross_pnl, 4),
            "commission": round(total_commission, 4),
            "slippage": round(slippage_cost, 4),
            "spread_cost": round(spread_cost, 4),
            "total_transaction_cost": round(total_cost, 4),
            "net_pnl": round(net_pnl, 4),
        }

    @staticmethod
    def _atr_from_bar(bar: pd.Series) -> float:
        """Approximate ATR from a single OHLCV bar (use High-Low as proxy)."""
        hl_range = float(bar.get("High", 0) - bar.get("Low", 0))
        if hl_range > 0:
            return hl_range
        close = float(bar.get("Close", 0))
        prev_close = float(bar.get("_prev_close", 0))
        return abs(close - prev_close) if prev_close > 0 else hl_range


# ---------------------------------------------------------------------------
# Internal metric helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


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
        logger.info(f"Backtesting {sym} [{start} to {end or 'latest'}]")
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

            # --- Per-trade cost analysis -----------------------------------------
            total_commission = 0.0
            total_slippage = 0.0
            gross_pnl_sum = 0.0
            net_pnl_sum = 0.0
            trades_data = stats.get("_trades", pd.DataFrame())

            if not trades_data.empty and len(trades_data) > 0:
                # Collect avg volume across whole dataset for slippage calc
                avg_vol = float(df["Volume"].mean()) if "Volume" in df.columns else 1_000_000

                for _, t in trades_data.iterrows():
                    try:
                        n_shares = t["Size"] if "Size" in t else t.get("size", 0)
                        entry_price = t["EntryPrice"] if "EntryPrice" in t else t.get("entry_price", 0)
                        exit_price = t["ExitPrice"] if "ExitPrice" in t else t.get("exit_price", 0)

                        # Commission on both sides
                        comm_entry = commission_cost(sym, int(n_shares), float(entry_price), "long")
                        comm_exit = commission_cost(sym, int(n_shares), float(exit_price), "long")
                        trade_commission = comm_entry + comm_exit

                        # Slippage proportional to volume ratio
                        vol_in = float(t.get("volume", 0)) or 0
                        slip = slippage_model(vol_in, avg_vol)
                        mid_price = (float(entry_price) + float(exit_price)) / 2
                        trade_slippage = slip * n_shares * mid_price

                        # Bid-ask spread impact
                        spread_amt = bid_ask_spread(sym, float(entry_price)) + bid_ask_spread(sym, float(exit_price))
                        trade_spread = spread_amt * n_shares

                        gross_trade_pnl = (float(exit_price) - float(entry_price)) * n_shares
                        gross_pnl_sum += gross_trade_pnl
                        net_trade_pnl = gross_trade_pnl - trade_commission - trade_slippage - trade_spread
                        net_pnl_sum += net_trade_pnl
                        total_commission += trade_commission
                        total_slippage += trade_slippage
                    except Exception:
                        # Skip individual trade on parse failure
                        continue

            gross_return = gross_pnl_sum / cash if cash > 0 else 0.0
            net_return = net_pnl_sum / cash if cash > 0 else 0.0
            cost_ratio = total_commission / gross_pnl_sum if gross_pnl_sum != 0 else 0.0

            # --- Extract core metrics --------------------------------------------
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
                "gross_return": round(float(gross_return), 4),
                "net_return": round(float(net_return), 4),
                "total_commission": round(float(total_commission), 4),
                "total_slippage": round(float(total_slippage), 4),
                "cost_ratio": round(float(cost_ratio), 6),
            }

            # Get trades dataframe
            trades_df = stats.get("_trades", pd.DataFrame())
            if not trades_df.empty:
                trades_df["PnL"] = round(trades_df["PnL"], 2)
                if "ReturnPct" in trades_df.columns:
                    trades_df["return_pct"] = round(trades_df["ReturnPct"] * 100, 2)

            # Save HTML report
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_path = reports_dir / f"{sym}_backtest.html"
            bt.plot(filename=str(report_path), open_browser=False)

            results[sym] = {
                "metrics": metrics,
                "trades_df": trades_df if not trades_df.empty else pd.DataFrame(),
                "equity_curve": stats.get("_equity_curve", pd.DataFrame()).reset_index(drop=True),
            }
            logger.info(f"Backtest {sym}: {metrics['total_return_pct']}% return, {metrics['trades']} trades, Sharpe={metrics['sharpe_ratio']}, Sortino={metrics['sortino_ratio']}, Calmar={metrics['calmar_ratio']}")

        except Exception as e:
            logger.exception(f"Backtest failed for {sym}: {e}")
            results[sym] = {"metrics": {"gross_return": 0.0, "net_return": 0.0, "total_commission": 0.0, "total_slippage": 0.0, "cost_ratio": 0.0}, "trades_df": pd.DataFrame(), "error": str(e)}

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
