"""Live trading engine — APScheduler loop with full account management.

Cycle order each iteration:
  1. Kill-switch check (daily-loss guard + emergency flag file)
  2. Position reconciliation (adopt external positions, drop stale ones)
  3. Exit management (stop/target/signal exits for open positions)
  4. Stale-order cleanup (cancel unmatched orders)
  5. Entry signals (new buys on positive signals)
  6. Persist state to logs
"""

import json
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EngineState:
    """Tracks engine lifecycle state and persists to disk."""

    def __init__(self) -> None:
        self._state_file = Path("logs/engine_state.json")
        self._positions_file = Path("logs/positions_state.json")
        self._equity_file = Path("logs/equity_history.csv")
        self._trades_file = Path("logs/trades.csv")
        self._confirm_file = Path("logs/strategy_confirmed.json")

    def read_strategy_confirmation(self) -> dict[str, Any] | None:
        if not self._confirm_file.exists():
            return None
        try:
            with open(self._confirm_file) as f:
                return json.load(f)
        except Exception:
            return None

    def save_strategy_confirmation(self, strategy_name: str, params: dict, symbols: list[str]) -> None:
        data = {
            "strategy": strategy_name,
            "params": params,
            "symbols": symbols,
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._confirm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._confirm_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Strategy confirmed: %s params=%s", strategy_name, params)

    def write_state(self, mode: str, strategy: str, params: dict, kill_switch: bool, equity: float, ts: str) -> None:
        state = {
            "mode": mode,
            "strategy": strategy,
            "params": params,
            "kill_switch": kill_switch,
            "day_start_equity": equity,
            "last_cycle_ts": ts,
        }
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def append_equity(self, equity: float, ts: str) -> None:
        self._equity_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._equity_file, "a") as f:
            f.write(f"{ts},{equity}\n")

    def append_trade(self, timestamp: str, symbol: str, side: str, qty: int, price: float, reason: str) -> None:
        self._trades_file.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._trades_file.exists() or self._trades_file.stat().st_size == 0
        with open(self._trades_file, "a") as f:
            if needs_header:
                f.write("timestamp,symbol,side,qty,price,reason\n")
            f.write(f"{timestamp},{symbol},{side},{qty},{price:.4f},{reason}\n")

    def read_positions(self) -> dict[str, dict]:
        if not self._positions_file.exists():
            return {}
        try:
            with open(self._positions_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def save_positions(self, positions: dict[str, dict]) -> None:
        self._positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._positions_file, "w") as f:
            json.dump(positions, f, indent=2)


def run_engine(broker, settings, strategy_name="ema_cross_rsi", strategy_params=None) -> None:
    """Main live trading loop via APScheduler. Manages entire Agentic account."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from bot.broker import MockBroker

    scheduler = BlockingScheduler()
    engine_state = EngineState()

    # Confirm strategy before starting
    confirmed = engine_state.read_strategy_confirmation()
    if confirmed is None:
        logger.error("No strategy confirmed. Run: python main.py live --strategy ema_cross_rsi")
        return

    effective_params = {**confirmed.get("params", {}), **(strategy_params or {})}

    # Import strategy
    from bot.core import STRATEGIES
    try:
        strategy_cls = STRATEGIES.get(strategy_name)
    except KeyError:
        logger.error(f"Strategy '{strategy_name}' not registered. Available: {STRATEGIES.names()}")
        return

    strategy_instance = strategy_cls(**effective_params)
    logger.info("Engine started: strategy=%s params=%s symbols=%s", strategy_name, effective_params, settings.symbols)

    # Risk management
    from bot.risk import KillSwitch
    kill_switch_mgr = KillSwitch(settings.max_daily_loss_pct)

    # Signal the running state
    last_equity = 0.0
    cycle_counter = [0]

    def on_sigint(signum, frame):
        logger.info("SIGINT received — shutting down...")
        import asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        try:
            async def _cancel():
                await broker.cancel_all()
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_cancel())
            finally:
                loop.close()
        except Exception:
            pass
        logger.info("Engine stopped. Orders cancelled.")
        exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    async def _cycle():
        nonlocal last_equity, cycle_counter

        now_str = datetime.utcnow().isoformat()
        cycle_counter[0] += 1

        # Step 1: Check if market is open (unless mock)
        if not isinstance(broker, MockBroker):
            if not broker.is_market_open():
                logger.debug("Market closed, skipping cycle %d", cycle_counter[0])
                return

        # Read current equity
        try:
            equity = await broker.get_equity()
            last_equity = equity
        except Exception:
            logger.warning("Could not fetch equity, continuing with previous value")
            equity = last_equity

        # Reset kill switch for new day
        day_start = datetime.utcnow().strftime("%Y-%m-%d")
        day_file = Path("logs/.current_day")
        try:
            saved_day = day_file.read_text().strip()
        except FileNotFoundError:
            saved_day = ""

        if saved_day != day_start:
            kill_switch_mgr.reset_day(equity)
            day_file.write_text(day_start)

        # Check kill switch
        is_killed = kill_switch_mgr.check(equity)
        if is_killed:
            logger.warning("Kill switch active — skipping trading this cycle")
            engine_state.write_state("live", strategy_name, effective_params, True, equity, now_str)
            return

        # Write engine state
        engine_state.write_state("live", strategy_name, effective_params, False, equity, now_str)

        # Step 2: Reconcile positions with broker
        internal_pos = engine_state.read_positions()
        try:
            broker_pos = await broker.get_positions()
        except Exception as exc:
            logger.warning("Position fetch failed: %s", exc)
            broker_pos = {}

        # Adopt external positions
        for sym, qty in broker_pos.items():
            if qty > 0 and sym not in internal_pos:
                logger.info("Adopted external position: %s qty=%d", sym, qty)
                internal_pos[sym] = {
                    "qty": qty,
                    "entry_price": 0,  # will be updated when we get quotes
                    "entry_ts": now_str,
                    "stop": 0,
                    "target": 0,
                }

        # Drop positions no longer on broker
        for sym in list(internal_pos.keys()):
            if sym not in broker_pos or broker_pos.get(sym, 0) <= 0:
                internal_pos.pop(sym, None)

        # Step 3: Manage exits for each position
        for sym, pos in list(internal_pos.items()):
            # Skip symbols we're not tracking
            if sym not in settings.symbols:
                continue

            try:
                quotes_result = await broker.get_quotes([sym])
                quote_info = quotes_result.get(sym, {})
                last_price = quote_info.get("last", quote_info.get("price", 0))

                if last_price <= 0:
                    continue

                # Check stop loss
                if pos.get("stop", 0) > 0 and last_price <= pos["stop"]:
                    logger.info("STOP hit: selling %s @ %.2f (stop=%.2f)", sym, last_price, pos["stop"])
                    await broker.submit_order(sym, pos["qty"], "SELL")
                    engine_state.append_trade(now_str, sym, "SELL", pos["qty"], last_price, "stop_loss")
                    internal_pos.pop(sym)
                    continue

                # Check take profit
                if pos.get("target", 0) > 0 and last_price >= pos["target"]:
                    logger.info("TAKE PROFIT: selling %s @ %.2f (target=%.2f)", sym, last_price, pos["target"])
                    await broker.submit_order(sym, pos["qty"], "SELL")
                    engine_state.append_trade(now_str, sym, "SELL", pos["qty"], last_price, "take_profit")
                    internal_pos.pop(sym)
                    continue

            except Exception as exc:
                logger.warning("Error checking exit for %s: %s", sym, exc)

        # Update entry prices from quotes for newly adopted positions
        if any(p.get("entry_price", 0) == 0 for p in internal_pos.values()):
            tracked = [s for s in settings.symbols if s in internal_pos]
            if tracked:
                try:
                    quotes_result = await broker.get_quotes(tracked)
                    for sym in tracked:
                        if internal_pos[sym].get("entry_price", 0) == 0:
                            q = quotes_result.get(sym, {})
                            internal_pos[sym]["entry_price"] = q.get("last", q.get("price", 0))
                except Exception:
                    pass

        # Step 4: Entry signals for tracked symbols
        from bot.data import fetch_latest_bars
        from bot.risk import position_size, stop_loss as calc_stop, take_profit as calc_tp

        for sym in settings.symbols:
            try:
                bars = fetch_latest_bars(sym, lookback=100)
                if bars is None or bars.empty or len(bars) < 21:
                    continue

                signals = strategy_instance.generate_signals(bars)
                last_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

                has_position = sym in internal_pos

                if last_signal == 1 and not has_position:
                    # Buy signal
                    last_close = float(bars["Close"].iloc[-1])
                    atr_val = 0
                    if len(bars) > 14:
                        from bot.indicators import atr as _atr
                        atr_val = _atr(bars)["ATRr14_14"].iloc[-1] if "ATRr14_14" in _atr(bars) else float(bars["High"].diff().rolling(14).max().iloc[-1]) if "High" in bars else 1.0

                    if atr_val <= 0:
                        atr_val = last_close * 0.02  # fallback 2% estimate

                    stop_dist = atr_val * 2
                    qty = position_size(equity, last_close, stop_dist, settings.risk_per_trade)
                    if qty <= 0:
                        continue

                    sl = calc_stop(last_close, atr_val)
                    tp = calc_tp(last_close, atr_val)

                    order_id = await broker.submit_order(sym, qty, "BUY", stop=sl, target=tp)
                    logger.info("BUY %s qty=%d @%.2f stop=%.2f target=%.2f (order=%s)", sym, qty, last_close, sl, tp, order_id)
                    engine_state.append_trade(now_str, sym, "BUY", qty, last_close, "signal")

                    internal_pos[sym] = {
                        "qty": qty,
                        "entry_price": last_close,
                        "entry_ts": now_str,
                        "stop": sl,
                        "target": tp,
                    }

                elif last_signal == -1 and has_position:
                    # Sell signal
                    qty = internal_pos[sym]["qty"]
                    try:
                        quotes_result = await broker.get_quotes([sym])
                        last_price = quotes_result.get(sym, {}).get("last", last_close)
                    except Exception:
                        last_price = last_close

                    await broker.submit_order(sym, qty, "SELL")
                    logger.info("SELL %s qty=%d @%.2f (signal exit)", sym, qty, last_price)
                    engine_state.append_trade(now_str, sym, "SELL", qty, last_price, "signal_exit")
                    internal_pos.pop(sym)

            except Exception as exc:
                logger.warning("Error processing %s: %s", sym, exc)
                continue

        # Step 5: Persist positions state
        engine_state.save_positions(internal_pos)

        # Log equity
        engine_state.append_equity(equity, now_str)

        logger.info(
            "Cycle %d complete: equity=%.2f positions=%d",
            cycle_counter[0], equity, len(internal_pos),
        )

    scheduler.add_job(_cycle, "interval", minutes=settings.engine_interval_minutes, id="engine_cycle")

    logger.info("Engine starting — monitoring every %d min for symbols: %s", settings.engine_interval_minutes, settings.symbols)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down engine")
    finally:
        logger.info("Engine shutdown complete")
