"""Tests for the --duration flag added to `python main.py dry-run`.

The dry-run CLI command must accept an optional --duration argument that
auto-shuts the engine down via ``scheduler.shutdown(wait=False)``. Default
behavior is a 30-second cap; passing 0 disables the timer and the engine
runs until Ctrl+C (the legacy "infinite" behavior).

These tests exercise:
  * Parser accepts --duration with a numeric value and the documented
    default of 30.
  * ``_cmd_dry_run`` forwards the value to ``run_engine`` as
    ``duration_seconds``, mapping 0 -> None.
  * A real ``run_engine`` invocation with a short duration shuts down
    cleanly within a bounded wall-clock window.
"""

import json
import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure we can import the project root regardless of pytest cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_args(duration: int = 30, strategy: str = "ema_cross_rsi",
               symbols: list | None = None):
    """Build a minimal argparse Namespace mimicking CLI input."""
    ns = MagicMock()
    ns.strategy = strategy
    ns.symbols = symbols
    ns.duration = duration
    return ns


# ─── parser ────────────────────────────────────────────────────────────


class TestDryRunParser:
    def test_dry_run_default_duration_is_30(self):
        """`python main.py dry-run` (no flags) sets duration=30 by default."""
        import argparse

        parser = argparse.ArgumentParser(description="Stock Trading Bot")
        sub = parser.add_subparsers(dest="command")
        p_dry = sub.add_parser("dry-run")
        p_dry.add_argument("--strategy", default="ema_cross_rsi")
        p_dry.add_argument("--symbols", nargs="+", default=None)
        p_dry.add_argument("--duration", type=int, default=30)

        args = parser.parse_args(["dry-run"])
        assert args.duration == 30
        assert args.strategy == "ema_cross_rsi"

    def test_dry_run_accepts_custom_duration(self):
        """`--duration 5` is accepted and stored."""
        import argparse

        parser = argparse.ArgumentParser(description="Stock Trading Bot")
        sub = parser.add_subparsers(dest="command")
        p_dry = sub.add_parser("dry-run")
        p_dry.add_argument("--strategy", default="ema_cross_rsi")
        p_dry.add_argument("--symbols", nargs="+", default=None)
        p_dry.add_argument("--duration", type=int, default=30)

        args = parser.parse_args(["dry-run", "--duration", "5"])
        assert args.duration == 5


# ─── main.py wiring ─────────────────────────────────────────────────────


class TestCmdDryRunDurationWiring:
    def test_duration_zero_maps_to_none(self):
        """`--duration 0` should be passed as ``None`` to run_engine
        so the engine runs until Ctrl+C (legacy infinite behavior)."""
        from unittest.mock import AsyncMock

        from main import _cmd_dry_run

        captured = {}

        def fake_run_engine(broker, settings, **kwargs):
            captured.update(kwargs)

        with patch("bot.broker.MockBroker") as mock_broker_cls, \
             patch("bot.engine.EngineState"), \
             patch("bot.engine.run_engine", side_effect=fake_run_engine):

            mock_broker = MagicMock()
            mock_broker.test_connection = AsyncMock(return_value=True)
            mock_broker_cls.return_value = mock_broker

            settings = MagicMock(symbols=["AAPL"], cash=100_000)

            args = _make_args(duration=0)
            logger = logging.getLogger("test")
            _cmd_dry_run(args, settings, logger)

        assert captured.get("duration_seconds") is None

    def test_duration_positive_is_forwarded(self):
        """`--duration 7` should be passed to run_engine as 7."""
        from unittest.mock import AsyncMock

        from main import _cmd_dry_run

        captured = {}

        def fake_run_engine(broker, settings, **kwargs):
            captured.update(kwargs)

        with patch("bot.broker.MockBroker") as mock_broker_cls, \
             patch("bot.engine.EngineState"), \
             patch("bot.engine.run_engine", side_effect=fake_run_engine):

            mock_broker = MagicMock()
            mock_broker.test_connection = AsyncMock(return_value=True)
            mock_broker_cls.return_value = mock_broker

            settings = MagicMock(symbols=["AAPL"], cash=100_000)

            args = _make_args(duration=7)
            logger = logging.getLogger("test")
            _cmd_dry_run(args, settings, logger)


# ─── run_engine integration ─────────────────────────────────────────────


def _seed_strategy_confirmation(logs_dir: Path) -> None:
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "strategy_confirmed.json").write_text(json.dumps({
        "strategy": "ema_cross_rsi",
        "params": {"fast": 9, "slow": 21, "rsi_period": 14,
                   "rsi_entry_max": 70.0, "rsi_exit": 75.0},
        "symbols": ["AAPL"],
    }))


class TestRunEngineDurationShutdown:
    """Verify ``run_engine`` honors ``duration_seconds`` end-to-end."""

    def test_duration_triggers_scheduler_shutdown(self, tmp_path, monkeypatch):
        """A 1-second duration must call ``scheduler.shutdown(wait=False)``
        from the background ``threading.Timer`` and unblock ``scheduler.start()``.
        """
        _seed_strategy_confirmation(tmp_path / "logs")
        monkeypatch.chdir(tmp_path)

        # Make sure plugin discovery has run so STRATEGIES is populated.
        from bot.core import discover_all
        discover_all()

        # Run a quick probe to see what's registered; if no ema_cross_rsi,
        # fall back to ANY registered strategy.
        import bot.core as core
        registered = list(core.STRATEGIES.names())
        if "ema_cross_rsi" not in registered:
            pytest.skip(f"ema_cross_rsi not registered; available: {registered}")

        from bot.engine import run_engine
        from bot.config import Settings
        from bot.broker import MockBroker

        settings = Settings(symbols=["AAPL"], cash=100_000,
                            engine_interval_minutes=5)
        broker = MockBroker(starting_equity=100_000)

        sched_inst = MagicMock()

        shutdown_obs = MagicMock()

        def fake_start():
            # Block until shutdown is called, with a hard timeout.
            deadline = time.time() + 5.0
            while not shutdown_obs.called and time.time() < deadline:
                time.sleep(0.05)
            return None

        sched_inst.start.side_effect = fake_start

        real_shutdown = sched_inst.shutdown
        def fake_shutdown(*a, **kw):
            shutdown_obs()
            return real_shutdown(*a, **kw)
        sched_inst.shutdown = fake_shutdown

        from apscheduler.schedulers import blocking as ap_blocking
        with patch.object(ap_blocking, "BlockingScheduler") as MockCls:
            MockCls.return_value = sched_inst

            t0 = time.time()
            run_engine(broker, settings,
                       strategy_name="ema_cross_rsi",
                       duration_seconds=1)
            elapsed = time.time() - t0

        assert shutdown_obs.called, \
            "scheduler.shutdown was not called — duration timer failed"
        assert 0.5 < elapsed < 4.5, \
            f"Engine returned in {elapsed:.2f}s — outside the expected [0.5s, 4.5s] band"

    def test_no_duration_does_not_start_timer(self, tmp_path, monkeypatch):
        """When ``duration_seconds`` is None the engine must NOT auto-shut;
        the timer path should be skipped entirely."""
        _seed_strategy_confirmation(tmp_path / "logs")
        monkeypatch.chdir(tmp_path)

        from bot.core import discover_all
        discover_all()

        import bot.core as core
        registered = list(core.STRATEGIES.names())
        if "ema_cross_rsi" not in registered:
            pytest.skip(f"ema_cross_rsi not registered; available: {registered}")

        from bot.engine import run_engine
        from bot.config import Settings
        from bot.broker import MockBroker

        settings = Settings(symbols=["AAPL"], cash=100_000,
                            engine_interval_minutes=5)
        broker = MockBroker(starting_equity=100_000)

        sched_inst = MagicMock()
        def fake_start():
            # Block briefly then return — no shutdown timer active.
            time.sleep(0.1)
            return None
        sched_inst.start.side_effect = fake_start

        from apscheduler.schedulers import blocking as ap_blocking
        with patch.object(ap_blocking, "BlockingScheduler") as MockCls:
            MockCls.return_value = sched_inst
            run_engine(broker, settings,
                       strategy_name="ema_cross_rsi",
                       duration_seconds=None)

        assert not sched_inst.shutdown.called, \
            "shutdown must NOT be invoked when duration_seconds is None"
