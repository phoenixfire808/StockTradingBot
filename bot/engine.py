"""Live trading engine — APScheduler loop with full account management.

Cycle order each iteration:
  1. Kill-switch check (daily-loss guard + emergency flag file)
  2. Position reconciliation (adopt external positions, drop stale ones)
  3. Exit management (stop/target/signal exits for open positions)
  4. Stale-order cleanup (cancel unmatched orders)
  5. Entry signals (new buys on positive signals)
  6. Daily rebalance at market open
  7. Persist state to logs
"""

import json
import logging
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.trade_store import TradeStore, sqlite_enabled
logger = logging.getLogger(__name__)


class EngineState:
    """Tracks engine lifecycle state and persists to disk.

    Backward-compatible facade: when SQLite is enabled (env TRADE_STORE=sqlite,
    or auto-detected when logs/trade_store.db exists) every append/save delegates
    to bot.trade_store.TradeStore. Otherwise it falls back to the original CSV/JSON
    files so existing deployments keep working unchanged. The CSV-writing code is
    retained as the fallback path.
    """

    def __init__(self) -> None:
        self._state_file = Path("logs/engine_state.json")
        self._positions_file = Path("logs/positions_state.json")
        self._equity_file = Path("logs/equity_history.csv")
        self._trades_file = Path("logs/trades.csv")
        self._signal_file = Path("logs/signals.csv")
        self._confirm_file = Path("logs/strategy_confirmed.json")
        # SQLite store — active when feature flag is on (see sqlite_enabled()).
        self._use_sqlite = sqlite_enabled()
        self._store: TradeStore | None = TradeStore() if self._use_sqlite else None
        logger.info("EngineState init: use_sqlite=%s", self._use_sqlite)

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
        if self._use_sqlite and self._store is not None:
            self._store.upsert_state(mode, strategy, params, kill_switch, equity, ts)
            return
        # ── CSV/JSON fallback ──
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
        if self._use_sqlite and self._store is not None:
            self._store.insert_equity(ts, equity)
            return
        # ── CSV fallback ──
        self._equity_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._equity_file, "a") as f:
            f.write(f"{ts},{equity}\n")

    def append_trade(self, timestamp: str, symbol: str, side: str, qty: int, price: float, reason: str) -> None:
        if self._use_sqlite and self._store is not None:
            self._store.insert_trade(timestamp, symbol, side, qty, price, reason)
            return
        # ── CSV fallback ──
        self._trades_file.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._trades_file.exists() or self._trades_file.stat().st_size == 0
        with open(self._trades_file, "a") as f:
            if needs_header:
                f.write("timestamp,symbol,side,qty,price,reason\n")
            f.write(f"{timestamp},{symbol},{side},{qty},{price:.4f},{reason}\n")

    def append_signal(self, timestamp: str, symbol: str, signal_val: int, reason: str) -> None:
        if self._use_sqlite and self._store is not None:
            self._store.insert_signal(timestamp, symbol, signal_val, reason)
            return
        # ── CSV fallback ──
        self._signal_file.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._signal_file.exists() or self._signal_file.stat().st_size == 0
        with open(self._signal_file, "a") as f:
            if needs_header:
                f.write("timestamp,symbol,signal,reason\n")
            f.write(f"{timestamp},{symbol},{signal_val},{reason}\n")

    def read_positions(self) -> dict[str, dict]:
        if self._use_sqlite and self._store is not None:
            return self._store.get_positions()
        # ── JSON fallback ──
        if not self._positions_file.exists():
            return {}
        try:
            with open(self._positions_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def save_positions(self, positions: dict[str, dict]) -> None:
        if self._use_sqlite and self._store is not None:
            self._store.upsert_positions(positions)
            return
        # ── JSON fallback ──
        self._positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._positions_file, "w") as f:
            json.dump(positions, f, indent=2)


def run_engine(broker, settings, strategy_name="ema_cross_rsi", strategy_params=None, duration_seconds: int | None = None, symbols: list[str] | None = None) -> None:
    """Main live trading loop via APScheduler. Manages entire Agentic account.

    If ``duration_seconds`` is provided, the engine schedules a daemon
    ``threading.Timer`` that calls ``scheduler.shutdown(wait=False)`` after
    the given number of seconds. ``None`` (the default) preserves the
    legacy "run forever" behavior used by the ``live`` command.

    The ``symbols`` parameter (when provided) overrides the symbol set used by
    the trading loop. When omitted, the engine falls back to the symbol list
    stored in the confirmation file (``logs/strategy_confirmed.json``), and
    finally to ``settings.symbols``. This guarantees that
    ``python main.py dry-run --symbols AAPL`` actually monitors only AAPL
    instead of the configured default.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from bot.broker import MockBroker

    scheduler = BlockingScheduler()
    engine_state = EngineState()

    # Optional duration shutdown — fired from a separate thread so the
    # main thread's ``scheduler.start()`` returns cleanly when the timer pops.
    duration_timer: threading.Timer | None = None
    if duration_seconds is not None:
        def _duration_shutdown():
            logger.info("Duration limit (%ss) reached — shutting down engine", duration_seconds)
            try:
                scheduler.shutdown(wait=False)
            except Exception as exc:
                logger.warning("Scheduler shutdown during duration limit failed: %s", exc)

        duration_timer = threading.Timer(duration_seconds, _duration_shutdown)
        duration_timer.daemon = True
        duration_timer.start()
        logger.info("Engine will auto-stop after %s seconds", duration_seconds)

    # Confirm strategy before starting
    confirmed = engine_state.read_strategy_confirmation()
    if confirmed is None:
        logger.error("No strategy confirmed. Run: python main.py live --strategy ema_cross_rsi")
        return

    effective_params = {**confirmed.get("params", {}), **(strategy_params or {})}

    # Resolve the active symbol set with explicit precedence:
    #   1. caller-supplied ``symbols`` argument
    #   2. symbols stored in the confirmation file
    #   3. configured ``settings.symbols`` fallback
    confirmed_symbols = confirmed.get("symbols")
    if symbols is not None:
        active_symbols = list(symbols)
    elif confirmed_symbols:
        active_symbols = list(confirmed_symbols)
    else:
        active_symbols = list(settings.symbols)

    # Import strategy
    from bot.core import STRATEGIES
    try:
        strategy_cls = STRATEGIES.get(strategy_name)
    except KeyError:
        logger.error(f"Strategy '{strategy_name}' not registered. Available: {STRATEGIES.names()}")
        return

    strategy_instance = type(strategy_cls)(**effective_params)
    logger.info("Engine started: strategy=%s params=%s symbols=%s", strategy_name, effective_params, active_symbols)

    # Risk management
    from bot.risk import KillSwitch
    kill_switch_mgr = KillSwitch(settings.max_daily_loss_pct)

    # Alert notifications (Discord webhook + SMTP email)
    from bot.alerts import AlertManager
    alerts = AlertManager.from_env()
    logger.info(
        "Alerts configured: discord=%s smtp=%s",
        alerts.config.discord_enabled, alerts.config.smtp_enabled,
    )

    # Track kill-switch trip so we only alert once per trip (re-armed on day reset)
    kill_switch_alerted = [False]
    # Track drawdown threshold crossings so each threshold fires once per day
    drawdown_alerted_pct = [0.0]

    # Signal the running state
    last_equity = 0.0
    cycle_counter = [0]
    prev_day = ""
    has_rebalanced_today = False

    def on_sigint(signum, frame):
        logger.info("SIGINT received — shutting down...")
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
        nonlocal last_equity, cycle_counter, prev_day, has_rebalanced_today
        now_str = datetime.now(timezone.utc).isoformat()
        now_dt = datetime.now(timezone.utc)
        current_day = now_dt.strftime("%Y-%m-%d")
        cycle_counter[0] += 1

        try:
            equity = await broker.get_equity()
            last_equity = equity
        except Exception:
            logger.warning("Could not fetch equity, continuing with previous value")
            equity = last_equity

        # Reset kill switch for new day
        if current_day != prev_day:
            # Send daily summary for the prior day before resetting
            if prev_day and kill_switch_mgr.day_start_equity is not None:
                _prev_positions = engine_state.read_positions()
                alerts.send_daily_summary(
                    equity=equity,
                    day_start_equity=kill_switch_mgr.day_start_equity,
                    positions=_prev_positions,
                    cycle_count=cycle_counter[0],
                )
            kill_switch_mgr.reset_day(equity)
            prev_day = current_day
            has_rebalanced_today = False
            kill_switch_alerted[0] = False
            drawdown_alerted_pct[0] = 0.0
            logger.info("New trading day %s — alerts re-armed", current_day)
        is_killed = kill_switch_mgr.check(equity)
        if is_killed:
            logger.warning("Kill switch active — skipping trading this cycle")
            if not kill_switch_alerted[0]:
                _dd_for_alert = 0.0
                if kill_switch_mgr.day_start_equity and kill_switch_mgr.day_start_equity != 0:
                    _dd_for_alert = ((kill_switch_mgr.day_start_equity - equity)
                                      / kill_switch_mgr.day_start_equity) * 100
                alerts.send_kill_switch_alert(
                    "daily_loss_exceeded" if _dd_for_alert >= settings.max_daily_loss_pct
                    else "manual_stop",
                    drawdown_pct=_dd_for_alert,
                    equity=equity,
                )
                kill_switch_alerted[0] = True
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
                    "entry_price": 0,
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
            if sym not in active_symbols:
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
                    engine_state.append_signal(now_str, sym, -1, "stop_loss_hit")
                    alerts.send_fill_alert(sym, "SELL", pos["qty"], last_price, "stop_loss", equity=equity)
                    internal_pos.pop(sym)
                    continue
                # Check take profit
                if pos.get("target", 0) > 0 and last_price >= pos["target"]:
                    logger.info("TAKE PROFIT: selling %s @ %.2f (target=%.2f)", sym, last_price, pos["target"])
                    await broker.submit_order(sym, pos["qty"], "SELL")
                    engine_state.append_trade(now_str, sym, "SELL", pos["qty"], last_price, "take_profit")
                    engine_state.append_signal(now_str, sym, -1, "take_profit_hit")
                    alerts.send_fill_alert(sym, "SELL", pos["qty"], last_price, "take_profit", equity=equity)
                    internal_pos.pop(sym)
                    continue

            except Exception as exc:
                logger.warning("Error checking exit for %s: %s", sym, exc)

        # Update entry prices for newly adopted positions
        if any(p.get("entry_price", 0) == 0 for p in internal_pos.values()):
            tracked = [s for s in active_symbols if s in internal_pos]
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

        for sym in active_symbols:
            try:
                bars = fetch_latest_bars(sym, lookback=100)
                if bars is None or bars.empty or len(bars) < 21:
                    continue

                signals = strategy_instance.generate_signals(bars)
                last_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

                # Log signal
                if last_signal != 0:
                    reason = f"strategy_signal_{strategy_name}"
                    engine_state.append_signal(now_str, sym, last_signal, reason)

                has_position = sym in internal_pos

                if last_signal == 1 and not has_position:
                    # Buy signal
                    last_close = float(bars["Close"].iloc[-1])
                    atr_val = 0
                    if len(bars) > 14:
                        from bot.indicators import atr as _atr
                        atr_result = _atr(bars)
                        atr_col = "ATRr14_14" if "ATRr14_14" in atr_result else "ATRr14_14"
                        if atr_col in atr_result:
                            atr_val = float(atr_result[atr_col].iloc[-1])
                        elif "High" in bars and "Low" in bars:
                            atr_val = float(bars["High"].diff().rolling(14).max().iloc[-1])
                        else:
                            atr_val = 1.0

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
                    engine_state.append_signal(now_str, sym, 1, "signal_entry")
                    alerts.send_fill_alert(sym, "BUY", qty, last_close, "signal_entry", equity=equity)

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
                    engine_state.append_signal(now_str, sym, -1, "signal_exit")
                    alerts.send_fill_alert(sym, "SELL", qty, last_price, "signal_exit", equity=equity)
                    internal_pos.pop(sym)

            except Exception as exc:
                logger.warning("Error processing %s: %s", sym, exc)
                continue

        # Step 5: Daily rebalance at market open (once per day)
        if current_day != prev_day and not has_rebalanced_today:
            has_rebalanced_today = True
            await _daily_rebalance(broker, internal_pos, active_symbols, equity, now_str)

        # Step 6: Persist positions state
        engine_state.save_positions(internal_pos)

        # Log equity
        engine_state.append_equity(equity, now_str)

        logger.info(
            "Cycle %d complete: equity=%.2f positions=%d rebalanced=%s",
            cycle_counter[0], equity, len(internal_pos), has_rebalanced_today,
        )

    async def _daily_rebalance(broker_obj, positions_dict, symbols_list, eq, ts_now):
        """Rebalance portfolio weights at market open. Ensures all tracked symbols
        have proportional exposure based on target percentages."""
        target_pct = 1.0 / max(len(symbols_list), 1)
        total_equity = eq
        target_value = total_equity * target_pct
        logger.info("Daily rebalance triggered: target_weight=%.1f%% per symbol", target_pct * 100)

        for sym in symbols_list:
            if sym not in positions_dict:
                continue
            pos = positions_dict[sym]
            qty = pos.get("qty", 0)
            if qty <= 0:
                continue
            try:
                quotes_result = await broker_obj.get_quotes([sym])
                price = quotes_result.get(sym, {}).get("last", 0)
                if price <= 0:
                    continue
                current_value = qty * price
                diff = target_value - current_value
                if abs(diff) / target_value > 0.15:  # 15% tolerance band
                    trade_qty = int(abs(diff) / price)
                    direction = "BUY" if diff > 0 else "SELL"
                    await broker_obj.submit_order(sym, trade_qty, direction)
                    logger.info("REBALANCE: %s %s %d @%.2f (curr_val=%.0f target=%.0f diff=%.0f)",
                                sym, direction, trade_qty, price, current_value, target_value, diff)
            except Exception as e:
                logger.warning("Rebalance error for %s: %s", sym, e)

    scheduler.add_job(_cycle, "interval", minutes=settings.engine_interval_minutes, id="engine_cycle")

    logger.info("Engine starting — monitoring every %d min for symbols: %s", settings.engine_interval_minutes, active_symbols)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down engine")
    finally:
        logger.info("Engine shutdown complete")
def run_multi_strategy(
    broker,
    settings,
    strategy_allocations: dict[str, dict],
    equity_tracker=None,
) -> None:
    """Multi-strategy engine — splits capital across strategies via portfolio allocator.

    Each strategy receives its share of ``settings.cash`` based on *strategy_allocations*,
    runs over its own symbol set, and tracks per-strategy equity independently.

    Parameters
    ----------
    broker : BotBroker
        Broker instance (MockBroker for dry-run).
    settings : Settings
        Application settings (cash, risk params, interval, alerts config).
    strategy_allocations : dict[str, dict]
        Parsed allocation map from ``Settings.parse_strategy_allocations()`` or CLI.
        Each entry: ``{name: {"symbols": [...], "weight": float}, ...}``.
        Weights are normalised so they sum to 1.0 if not already.
    equity_tracker : EquityTracker | None
        Pre-created tracker; auto-created when None.

    Backward-compatible with single-strategy flow: uses existing KillSwitch, AlertManager,
    position reconciliation, and APScheduler loop — just parameterised per strategy.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from bot.broker import MockBroker
    from bot.core import STRATEGIES
    from bot.risk import KillSwitch
    from bot.alerts import AlertManager
    from bot.equity_tracker import EquityTracker

    # ── Validate allocations ────────────────────────────────────────
    if not strategy_allocations:
        logger.error("No strategy allocations provided — cannot start multi-strategy engine")
        return

    # Normalise weights so they sum to 1.0
    total_weight = sum(v.get("weight", 0) for v in strategy_allocations.values())
    if total_weight <= 0:
        logger.error("Sum of strategy weights is zero — cannot split capital")
        return

    allocs = {}
    for name, spec in strategy_allocations.items():
        weight = spec.get("weight", 0) / total_weight
        symbols = spec.get("symbols", [])
        params = spec.get("params", {})
        if not symbols:
            logger.warning("Strategy '%s' has no symbols — skipping", name)
            continue
        try:
            strategy_cls = STRATEGIES.get(name)
        except KeyError as exc:
            logger.error("Strategy '%s' not found in registry. Available: %s", name, STRATEGIES.names())
            continue
        if strategy_cls is None:
            logger.error("Strategy '%s' not found in registry. Available: %s", name, STRATEGIES.names())
            continue
        allocs[name] = {"class": strategy_cls, "symbols": symbols, "weight": weight, "params": params}


    if not allocs:
        logger.error("No valid strategy allocations after filtering — aborting")
        return

    scheduler = BlockingScheduler()
    engine_state = EngineState()
    tracker = equity_tracker or EquityTracker()
    kill_switch_mgr = KillSwitch(settings.max_daily_loss_pct)
    alerts = AlertManager.from_env()

    # ── Per-strategy tracking state ─────────────────────────────────
    # Positions keyed by (strategy_name, symbol) → pos_dict
    internal_pos: dict[tuple[str, str], dict] = {}
    last_equity = [0.0]
    cycle_counter = [0]
    prev_day = [""]
    has_rebalanced_today = [False]
    kill_switch_alerted = [False]
    drawdown_alerted_pct = [0.0]
    trade_counts_per_day: dict[str, int] = {}  # day_key → count

    # Split cash per strategy based on weight
    strategy_cash: dict[str, float] = {
        name: settings.cash * data["weight"] for name, data in allocs.items()
    }
    logger.info(
        "Multi-strategy engine started: %d strategies, total_weight=%.2f, cash=%s",
        len(allocs), total_weight, strategy_cash,
    )
    for sname, scash in strategy_cash.items():
        sym_str = ", ".join(allocs[sname]["symbols"])
        w = allocs[sname]["weight"] * 100
        logger.info("  Strategy '%s': weight=%.1f%% cash=$%.2f symbols=[%s]",
                    sname, w, scash, sym_str)

    def on_sigint(signum, frame):
        logger.info("SIGINT received — shutting down multi-strategy engine...")
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
        nonlocal last_equity, trade_counts_per_day

        now_str = datetime.now(timezone.utc).isoformat()
        now_dt = datetime.now(timezone.utc)
        current_day = now_dt.strftime("%Y-%m-%d")
        cycle_counter[0] += 1

        # ── New-day reset ─────────────────────────────────────────────
        if current_day != prev_day[0]:
            if prev_day[0] and kill_switch_mgr.day_start_equity is not None:
                _all_positions = {}
                for (sn, ss), sp in internal_pos.items():
                    _all_positions[ss] = sp
                alerts.send_daily_summary(
                    equity=last_equity[0],
                    day_start_equity=kill_switch_mgr.day_start_equity,
                    positions=_all_positions,
                    cycle_count=cycle_counter[0],
                    trade_count=trade_counts_per_day.get(prev_day[0], 0),
                )
            trade_counts_per_day.clear()
            kill_switch_mgr.reset_day(last_equity[0])
            prev_day[0] = current_day
            has_rebalanced_today[0] = False
            kill_switch_alerted[0] = False
            drawdown_alerted_pct[0] = 0.0
            logger.info("New trading day %s — alerts re-armed", current_day)

        # Read total account equity
        try:
            equity = await broker.get_equity()
            last_equity[0] = equity
        except Exception:
            logger.warning("Could not fetch equity, continuing with previous value")
            equity = last_equity[0]

        # Drawdown advisory alert
        if kill_switch_mgr.day_start_equity and kill_switch_mgr.day_start_equity != 0:
            _dd_pct = ((kill_switch_mgr.day_start_equity - equity)
                       / kill_switch_mgr.day_start_equity) * 100
            _alert_thresh = alerts.config.drawdown_alert_pct
            if _dd_pct >= _alert_thresh and _dd_pct > drawdown_alerted_pct[0]:
                alerts.send_drawdown_alert(_dd_pct, equity=equity)
                drawdown_alerted_pct[0] = _dd_pct

        # Kill-switch check
        is_killed = kill_switch_mgr.check(equity)
        if is_killed:
            logger.warning("Kill switch active — skipping all strategy cycles")
            if not kill_switch_alerted[0]:
                _dd_for_alert = 0.0
                if kill_switch_mgr.day_start_equity and kill_switch_mgr.day_start_equity != 0:
                    _dd_for_alert = ((kill_switch_mgr.day_start_equity - equity)
                                      / kill_switch_mgr.day_start_equity) * 100
                alerts.send_kill_switch_alert(
                    "daily_loss_exceeded" if _dd_for_alert >= settings.max_daily_loss_pct
                    else "manual_stop",
                    drawdown_pct=_dd_for_alert,
                    equity=equity,
                )
                kill_switch_alerted[0] = True
            engine_state.write_state("multi_strategy", list(allocs.keys()), {}, True, equity, now_str)
            return

        # Track trades for daily summary
        trade_counts_per_day[current_day] = trade_counts_per_day.get(current_day, 0) + sum(
            1 for (_, _sym), _pos in internal_pos.items() if _pos.get("qty", 0) > 0
        )

        # ── Per-strategy cycle ────────────────────────────────────────
        for strategy_name, sdata in allocs.items():
            await _strategy_cycle(
                strategy_name=strategy_name,
                broker=broker,
                alloc=sdata,
                strategy_cash=strategy_cash,
                internal_pos=internal_pos,
                engine_state=engine_state,
                alerts=alerts,
                tracker=tracker,
                now_str=now_str,
                equity=equity,
                settings=settings,
            )

        # Save consolidated positions
        _consolidated = {}
        for (sn, ss), sp in internal_pos.items():
            _consolidated[ss] = sp
        engine_state.save_positions(_consolidated)

        # Persist total equity
        engine_state.append_equity(equity, now_str)

        # Record per-strategy equity snapshots
        for sn, scash in strategy_cash.items():
            tracker.record(sn, scash, now_str)

        logger.info(
            "Multi-strategy cycle %d complete: equity=%.2f positions=%d strategies=%d rebalanced=%s",
            cycle_counter[0], equity, len(internal_pos), len(allocs), has_rebalanced_today[0],
        )

    async def _strategy_cycle(
        strategy_name, broker, alloc, strategy_cash, internal_pos,
        engine_state, alerts, tracker, now_str, equity, settings,
    ):
        """Run one strategy's full cycle: reconcile exits, entries."""
        sym_list = alloc["symbols"]
        weight = alloc["weight"]
        params = alloc["params"]
        strat_cash = strategy_cash.get(strategy_name, settings.cash * weight)

        strategy_cls = alloc["class"]
        strategy_instance = type(strategy_cls)(**params)
        logger.debug("[%s] Starting cycle (cash=$%.2f, symbols=%s)",
                     strategy_name, strat_cash, sym_list)

        # Position key prefix for this strategy
        def _k(sym):
            return (strategy_name, sym)

        # Reconcile positions for this strategy's symbols
        try:
            broker_pos = await broker.get_positions()
        except Exception as exc:
            logger.warning("[%s] Position fetch failed: %s", strategy_name, exc)
            broker_pos = {}

        # Adopt external positions
        for sym in sym_list:
            qty = broker_pos.get(sym, 0)
            if qty > 0 and _k(sym) not in internal_pos:
                logger.info("[%s] Adopted external position: %s qty=%d", strategy_name, sym, qty)
                internal_pos[_k(sym)] = {
                    "qty": qty, "entry_price": 0, "entry_ts": now_str,
                    "stop": 0, "target": 0, "strategy": strategy_name,
                }

        # Drop positions no longer on broker
        for sym in sym_list:
            k = _k(sym)
            if k in internal_pos and (sym not in broker_pos or broker_pos.get(sym, 0) <= 0):
                removed = internal_pos.pop(k)
                logger.info("[%s] Dropped stale position: %s qty=%d", strategy_name, sym, removed["qty"])

        # Step 3: Manage exits (stops/targets/signals)
        for sym in list(sym_list):
            k = _k(sym)
            pos = internal_pos.get(k)
            if not pos:
                continue
            try:
                quotes_result = await broker.get_quotes([sym])
                quote_info = quotes_result.get(sym, {})
                last_price = quote_info.get("last", quote_info.get("price", 0))
                if last_price <= 0:
                    continue

                if pos.get("stop", 0) > 0 and last_price <= pos["stop"]:
                    logger.info("[%s] STOP hit: selling %s @ %.2f (stop=%.2f)",
                                strategy_name, sym, last_price, pos["stop"])
                    await broker.submit_order(sym, pos["qty"], "SELL")
                    engine_state.append_trade(now_str, sym, "SELL", pos["qty"], last_price, f"{strategy_name}_stop_loss")
                    engine_state.append_signal(now_str, sym, -1, f"{strategy_name}_stop_loss_hit")
                    alerts.send_fill_alert(sym, "SELL", pos["qty"], last_price,
                                           f"{strategy_name}_stop_loss", equity=equity)
                    tracker.record(strategy_name, strat_cash, now_str)
                    del internal_pos[k]
                    continue

                if pos.get("target", 0) > 0 and last_price >= pos["target"]:
                    logger.info("[%s] TAKE PROFIT: selling %s @ %.2f (target=%.2f)",
                                strategy_name, sym, last_price, pos["target"])
                    await broker.submit_order(sym, pos["qty"], "SELL")
                    engine_state.append_trade(now_str, sym, "SELL", pos["qty"], last_price, f"{strategy_name}_take_profit")
                    engine_state.append_signal(now_str, sym, -1, f"{strategy_name}_take_profit_hit")
                    alerts.send_fill_alert(sym, "SELL", pos["qty"], last_price,
                                           f"{strategy_name}_take_profit", equity=equity)
                    tracker.record(strategy_name, strat_cash, now_str)
                    del internal_pos[k]
                    continue
            except Exception as exc:
                logger.warning("[%s] Error checking exit for %s: %s", strategy_name, sym, exc)

        # Update entry prices
        tracked_syms = [s for s in sym_list if _k(s) in internal_pos and internal_pos[_k(s)].get("entry_price", 0) == 0]
        if tracked_syms:
            try:
                quotes_result = await broker.get_quotes(tracked_syms)
                for sym in tracked_syms:
                    q = quotes_result.get(sym, {})
                    internal_pos[_k(s)]["entry_price"] = q.get("last", q.get("price", 0))
            except Exception:
                pass

        # Step 4: Generate signals and execute entries/exits
        from bot.data import fetch_latest_bars
        from bot.risk import position_size, stop_loss as calc_stop, take_profit as calc_tp

        for sym in sym_list:
            try:
                bars = fetch_latest_bars(sym, lookback=100)
                if bars is None or bars.empty or len(bars) < 21:
                    continue

                signals = strategy_instance.generate_signals(bars)
                last_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

                if last_signal != 0:
                    reason = f"strategy_signal_{strategy_name}"
                    engine_state.append_signal(now_str, sym, last_signal, reason)

                has_position = _k(sym) in internal_pos

                if last_signal == 1 and not has_position:
                    last_close = float(bars["Close"].iloc[-1])
                    atr_val = 0
                    if len(bars) > 14:
                        from bot.indicators import atr as _atr_func
                        atr_result = _atr_func(bars)
                        atr_col = "ATRr14_14"
                        if atr_col in atr_result:
                            atr_val = float(atr_result[atr_col].iloc[-1])
                        elif "High" in bars and "Low" in bars:
                            atr_val = float(bars["High"].diff().rolling(14).max().iloc[-1])
                        else:
                            atr_val = 1.0

                    if atr_val <= 0:
                        atr_val = last_close * 0.02

                    stop_dist = atr_val * 2
                    qty = position_size(strat_cash, last_close, stop_dist, settings.risk_per_trade)
                    if qty <= 0:
                        continue

                    sl = calc_stop(last_close, atr_val)
                    tp = calc_tp(last_close, atr_val)

                    order_id = await broker.submit_order(sym, qty, "BUY", stop=sl, target=tp)
                    logger.info("[%s] BUY %s qty=%d @%.2f stop=%.2f target=%.2f (order=%s)",
                                strategy_name, sym, qty, last_close, sl, tp, order_id)
                    engine_state.append_trade(now_str, sym, "BUY", qty, last_close, f"{strategy_name}_signal_entry")
                    engine_state.append_signal(now_str, sym, 1, f"{strategy_name}_signal_entry")
                    alerts.send_fill_alert(sym, "BUY", qty, last_close,
                                           f"{strategy_name}_signal_entry", equity=equity)
                    internal_pos[_k(sym)] = {
                        "qty": qty, "entry_price": last_close, "entry_ts": now_str,
                        "stop": sl, "target": tp, "strategy": strategy_name,
                    }

                elif last_signal == -1 and has_position:
                    qty = internal_pos[_k(sym)]["qty"]
                    try:
                        quotes_result = await broker.get_quotes([sym])
                        last_price = quotes_result.get(sym, {}).get("last", last_close)
                    except Exception:
                        last_price = last_close
                    await broker.submit_order(sym, qty, "SELL")
                    logger.info("[%s] SELL %s qty=%d @%.2f (signal exit)", strategy_name, sym, qty, last_price)
                    engine_state.append_trade(now_str, sym, "SELL", qty, last_price, f"{strategy_name}_signal_exit")
                    engine_state.append_signal(now_str, sym, -1, f"{strategy_name}_signal_exit")
                    alerts.send_fill_alert(sym, "SELL", qty, last_price, f"{strategy_name}_signal_exit", equity=equity)
                    del internal_pos[_k(sym)]

            except Exception as exc:
                logger.warning("[%s] Error processing %s: %s", strategy_name, sym, exc)
                continue

    scheduler.add_job(_cycle, "interval", minutes=settings.engine_interval_minutes, id="multi_engine_cycle")

    logger.info("Multi-strategy engine starting — monitoring every %d min across %d strategies",
                settings.engine_interval_minutes, len(allocs))

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down multi-strategy engine")
    finally:
        logger.info("Multi-strategy engine shutdown complete")
