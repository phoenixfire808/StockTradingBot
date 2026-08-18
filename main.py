#!/usr/bin/env python3
"""Stock Trading Bot — CLI entry point.

Subcommands:
  backtest  — Run historical backtests on configured symbols
  dry-run   — Start engine with MockBroker (no auth needed)
  live      — Start live trading via Robinhood MCP broker
  sentiment — Quick sentiment score lookup for a ticker
  ui        — Launch Streamlit dashboard at localhost:8501
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
    sub.add_parser("dry-run", help="Dry-run engine (MockBroker)")

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
    elif args.command == "ui":
        _cmd_ui(args, settings, logger)


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
    """Start engine with MockBroker."""
    from bot.broker import MockBroker
    from bot.engine import run_engine

    broker = MockBroker(starting_equity=settings.cash)

    async def _start():
        await broker.test_connection()
        print("[DRY-RUN] Broker connected (mock mode). Press Ctrl+C to stop.")
        run_engine(broker, settings, strategy_name="ema_cross_rsi")

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
    effective_params = {
        "fast": 9,
        "slow": 21,
        "rsi_period": 14,
        "rsi_entry_max": 70.0,
        "rsi_exit": 75.0,
    }

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


def _cmd_sentiment(args, settings, logger):
    """Quick sentiment score check."""
    from bot.sentiment import SentimentEngine

    engine = SentimentEngine()
    for sym in args.symbols:
        score = engine.score(sym, hours=args.hours)
        print(f"\n{sym} [{args.hours}h]: mentions={score.mentions} bullish={score.bullish} bearish={score.bearish} net_score={score.net_score:.3f}")
        for post in score.top_posts[:3]:
            emoji = "🟢" if post.score > 0 else "🔴" if post.score < 0 else "⚪"
            print(f"  {emoji} [{post.source}] {post.text[:80]}... ({post.score:+.3f})")


def _cmd_ui(args, settings, logger):
    """Launch Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard on port %d...", args.port)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/app.py",
        f"--server.port={args.port}",
        "--server.headless=true",
    ], check=False)


if __name__ == "__main__":
    main()
