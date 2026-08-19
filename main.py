#!/usr/bin/env python3
"""Stock Trading Bot — CLI entry point.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bot.config import load_settings, setup_logging
from bot.core import discover_all


def main():
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── backtest ─────────────────────────────────────────────────────
    p_bt = sub.add_parser("backtest", help="Run backtests")
    p_bt.add_argument("--symbols", nargs="+", default=None, help="Symbols to test")
    p_bt.add_argument("--start", default="2022-01-01", help="Start date")
    p_bt.add_argument("--end", default=None, help="End date")
    p_bt.add_argument("--cash", type=float, default=100_000, help="Starting cash")
    p_bt.add_argument("--strategy", default="ema_cross_rsi", help="Strategy name")

    # ── dry-run ──────────────────────────────────────────────────────
    p_dry = sub.add_parser("dry-run", help="Dry-run engine (MockBroker)")
    p_dry.add_argument("--strategy", default="ema_cross_rsi", help="Strategy to run")
    p_dry.add_argument("--symbols", nargs="+", default=None, help="Override symbols")
    p_dry.add_argument(
        "--duration", type=int, default=30,
        help="Auto-shutdown the engine after N seconds (default: 30). Set to 0 to run forever.",
    )

    # ── live ─────────────────────────────────────────────────────────
    p_live = sub.add_parser("live", help="Live trading via Robinhood MCP")
    p_live.add_argument("--strategy", default="ema_cross_rsi", help="Strategy to confirm and run")
    p_live.add_argument("--symbols", nargs="+", default=None, help="Override symbols")
    p_live.add_argument("--confirm-only", action="store_true", help="Only save confirmation, don't start engine")

    # ── sentiment ────────────────────────────────────────────────────
    p_sent = sub.add_parser("sentiment", help="Quick sentiment check")
    p_sent.add_argument("symbols", nargs="+", help="Symbols to check")
    p_sent.add_argument("--hours", type=int, default=24, help="Lookback window in hours")

    # ── ui ───────────────────────────────────────────────────────────
    p_ui = sub.add_parser("ui", help="Launch Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501, help="Port number")

    # ── optimize ────────────────────────────────────────────────────
    p_opt = sub.add_parser("optimize", help="Walk-forward strategy optimization")
    p_opt.add_argument("--strategy", default="ema_cross_rsi", help="Strategy to optimize")
    p_opt.add_argument("--symbols", nargs="+", default=["AAPL"], help="Symbols to optimize on")
    p_opt.add_argument("--start", default="2023-01-01", help="Start date")
    p_opt.add_argument("--end", default=None, help="End date")

    # ── multi (multi-strategy) ─────────────────────────────────────
    p_multi = sub.add_parser("multi", help="Run multiple strategies with capital allocation")
    p_multi.add_argument("--strategy", required=True,
                         help='JSON string or file path: {"ema_cross_rsi": {"symbols": ["AAPL"], "weight": 0.5}}')
    p_multi.add_argument("--cash", type=float, default=100_000, help="Starting cash (default from env)")
    p_multi.add_argument("--dry-run", action="store_true", help="Use MockBroker instead of live broker")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Bootstrap logging
    settings = load_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    # Discover all plugins
    discovered = discover_all()
    logger.info("Plugin discovery: %s", discovered)

    if args.command == "backtest":
        _cmd_backtest(args, settings, logger)
    elif args.command == "dry-run":
        _cmd_dry_run(args, settings, logger)
    elif args.command == "live":
        _cmd_live(args, settings, logger)
    elif args.command == "sentiment":
        _cmd_sentiment(args, settings, logger)
    elif args.command == "optimize":
        _cmd_optimize(args, settings, logger)
    elif args.command == "ui":
        _cmd_ui(args, settings, logger)
    elif args.command == "multi":
        _cmd_multi(args, settings, logger)


def _cmd_backtest(args, settings, logger):
    """Run backtests."""
    from bot.backtest import run_backtest, print_backtest_table

    symbols = args.symbols or settings.symbols
    results = run_backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        cash=args.cash,
        strategy_name=args.strategy,
    )
    print_backtest_table(results)

    # Save summary to logs
    summary_path = Path("logs/backtest_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for sym, res in results.items():
        m = res.get("metrics", {})
        trades_df = res.get("trades_df", None)
        trade_list = []
        if isinstance(trades_df, object) and hasattr(trades_df, 'to_dict'):
            try:
                trade_list = trades_df.to_dict(orient="records")
            except Exception:
                pass
        serializable[sym] = {"metrics": m, "num_trades": len(trade_list)}
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Backtest summary saved to %s", summary_path)


def _cmd_dry_run(args, settings, logger):
    """Start engine with MockBroker.

    ``args.duration`` defaults to 30 seconds. Passing 0 disables
    auto-shutdown and restores the legacy "run forever until Ctrl+C" behavior.
    """
    from bot.broker import MockBroker
    from bot.engine import run_engine, EngineState

    strategy_name = args.strategy
    symbols = args.symbols or settings.symbols
    effective_params = {
        "fast": 9,
        "slow": 21,
        "rsi_period": 14,
        "rsi_entry_max": 70.0,
        "rsi_exit": 75.0,
    }
    engine_state = EngineState()
    engine_state.save_strategy_confirmation(strategy_name, effective_params, symbols)

    broker = MockBroker(starting_equity=settings.cash)
    duration_seconds = args.duration if args.duration and args.duration > 0 else None

    async def _start():
        await broker.test_connection()
        if duration_seconds is None:
            print("[DRY-RUN] Broker connected (mock mode). Press Ctrl+C to stop.")
        else:
            print(
                f"[DRY-RUN] Broker connected (mock mode). "
                f"Engine will auto-stop after {duration_seconds}s."
            )
        run_engine(
            broker,
            settings,
            strategy_name=strategy_name,
            strategy_params=effective_params,
            duration_seconds=duration_seconds,
            symbols=symbols,
        )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_start())
    except KeyboardInterrupt:
        print("\n[DRY-RUN] Engine stopped.")


def _cmd_live(args, settings, logger):
    """Start live trading via Robinhood MCP."""
    from bot.broker import RobinhoodMcpBroker, MockBroker
    from bot.engine import EngineState

    strategy_name = args.strategy
    symbols = args.symbols or settings.symbols

    effective_params = {
        "fast": 9,
        "slow": 21,
        "rsi_period": 14,
        "rsi_entry_max": 70.0,
        "rsi_exit": 75.0,
    }

    # Build broker from settings
    broker = RobinhoodMcpBroker(settings)

    # Check connection first
    print(f"\nTesting connection to Robinhood Agentic Trading MCP...")
    is_connected = asyncio.get_event_loop().run_until_complete(broker.test_connection())

    if not is_connected:
        logger.error("Cannot connect to Robinhood MCP. Check your credentials and network.")
        logger.error("See references/robinhood/mcp-setup-guide.md for configuration steps.")
        sys.exit(2)

    print("✓ Connected to Robinhood Agentic Trading MCP")

    # Strategy confirmation
    engine_state = EngineState()
    confirmed = engine_state.read_strategy_confirmation()
    needs_confirmation = True

    if confirmed and confirmed.get("strategy") == strategy_name:
        # Check if strategy params match
        existing_params = confirmed.get("params", {})
        params_match = all(existing_params.get(k) == v for k, v in effective_params.items())
        if params_match:
            needs_confirmation = False

    if needs_confirmation:
        print("\n" + "=" * 60)
        print("LIVE TRADING CONFIRMATION")
        print("=" * 60)
        print(f"Strategy : {strategy_name}")
        print(f"Params   : {effective_params}")
        print(f"Symbols  : {', '.join(symbols)}")
        print(f"Risk/trade: {settings.risk_per_trade*100}%")
        print(f"Max daily loss: {settings.max_daily_loss_pct}%")
        print("-" * 60)
        confirm = input("\nTrade live on the Robinhood Agentic account with this strategy? [y/N] ").strip().lower()

        if confirm != "y":
            logger.info("User declined. No orders placed. Exiting.")
            return

        engine_state.save_strategy_confirmation(strategy_name, effective_params, symbols)
        print("✓ Strategy confirmed and saved.")

    print("\nStarting live engine...\n")

    run_engine(broker, settings, strategy_name=strategy_name, strategy_params=effective_params)


def _cmd_ui(args, settings, logger):
    """Launch Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard on port %d...", args.port)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/app.py",
        f"--server.port={args.port}",
        "--server.headless=true",
    ], check=False)


def _cmd_optimize(args, settings, logger):
    """Walk-forward strategy parameter optimization."""
    from bot.optimization import walk_forward_optimize

    # Parse optional param grid JSON
    param_grid = None
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
            logger.info("Using custom param grid: %s", param_grid)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --param-grid JSON: %s", exc)
            sys.exit(1)

    logger.info(
        "Starting walk-forward optimization: strategy=%s symbols=%s "
        "train=%dd test=%dd range=%s→%s",
        args.strategy, args.symbols, args.train_window, args.test_window,
        args.start, args.end or "now",
    )

    result = walk_forward_optimize(
        strategy_name=args.strategy,
        param_grid=param_grid,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        train_window=args.train_window,
        test_window=args.test_window,
        cash=args.cash,
    )
    print("=" * 70)
    print(f"  Best params:        {result.get('best_params', {})}")
    print(f"  Best OOS score:     {result.get('best_score', 'N/A')}")
    print(f"  Best train score:   {result.get('best_train_score', 'N/A')}")
    print(f"  Total combinations: {result.get('total_combinations', 0)}")
    print(f"  Folds:              {len(result.get('folds', []))}")
    print()

    for fold in result.get("folds", []):
        print(
            f"  Fold {fold['fold']}: "
            f"train={fold['train_start']}→{fold['train_end']} "
            f"test={fold['test_start']}→{fold['test_end']} "
            f"score={fold['test_score']:.2f}"
        )
        m = fold.get("test_metrics", {})
        if m:
            print(
                f"    → ret={m.get('total_return_pct', 0):.1f}% "
                f"sharpe={m.get('sharpe_ratio', 0):.2f} "
                f"maxdd={m.get('max_dd_pct', 0):.1f}%"
            )
    print()

    # Save full results JSON
    results_path = Path("reports") / f"optimize_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Optimization results saved to %s", results_path)


def _cmd_multi(args, settings, logger):
    """Start multi-strategy engine — splits capital across strategies."""
    from bot.broker import MockBroker
    from bot.engine import run_multi_strategy
    from bot.equity_tracker import EquityTracker

    # Parse strategy allocations from JSON string or file
    alloc_str = args.strategy
    if Path(alloc_str).is_file():
        with open(alloc_str) as f:
            alloc_str = f.read()

    try:
        strategy_allocations = json.loads(alloc_str)
    except json.JSONDecodeError as exc:
        logger.error("Invalid --strategy JSON: %s", exc)
        sys.exit(1)

    if not strategy_allocations:
        logger.error("Empty strategy allocations — nothing to run")
        sys.exit(1)

    effective_cash = args.cash or settings.cash
    logger.info("Multi-strategy mode: %d strategies, cash=$%.2f", len(strategy_allocations), effective_cash)

    # Build broker
    broker = MockBroker(starting_equity=effective_cash) if args.dry_run else None

    async def _start():
        await broker.test_connection()
        print("[MULTI-STRATEGY] Broker connected (mock mode). Press Ctrl+C to stop.")

        tracker = EquityTracker()
        run_multi_strategy(
            broker=broker,
            settings=settings,
            strategy_allocations=strategy_allocations,
            equity_tracker=tracker,
        )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_start())
    except KeyboardInterrupt:
        print("\n[MULTI-STRATEGY] Engine stopped.")


if __name__ == "__main__":
    main()
