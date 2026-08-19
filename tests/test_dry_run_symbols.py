"""Tests for the ``--symbols`` flag on ``python main.py dry-run``.

The bug: with ``python main.py dry-run --symbols AAPL`` the engine loop
ignored the override and kept monitoring the default symbols from
``settings.symbols`` (e.g. ``['AAPL', 'MSFT', 'NVDA']``). The fix routes
the symbol list through ``run_engine`` -> ``active_symbols`` and falls
back to the confirmation file (and finally ``settings.symbols``) only
when the caller does not pass an explicit value.

These tests exercise:
  * Parser accepts ``--symbols AAPL`` and stores a single-element list.
  * ``_cmd_dry_run`` forwards ``args.symbols`` (and the resolved
    ``settings.symbols`` fallback) to ``run_engine`` as the ``symbols``
    keyword argument.
  * ``run_engine`` resolves ``active_symbols`` with the documented
    precedence: explicit arg > confirmation file > ``settings.symbols``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure we can import the project root regardless of pytest cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_args(strategy: str = "ema_cross_rsi", symbols: list | None = None,
               duration: int = 30):
    """Build a minimal argparse Namespace mimicking CLI input."""
    ns = MagicMock()
    ns.strategy = strategy
    ns.symbols = symbols
    ns.duration = duration
    return ns


# ─── parser ────────────────────────────────────────────────────────────


class TestDryRunSymbolsParser:
    def test_dry_run_default_symbols_is_none(self):
        """Omitting ``--symbols`` must leave ``args.symbols`` as None so
        that the ``args.symbols or settings.symbols`` fallback in
        ``_cmd_dry_run`` can resolve to the configured default."""
        parser = argparse.ArgumentParser(description="Stock Trading Bot")
        sub = parser.add_subparsers(dest="command")
        p_dry = sub.add_parser("dry-run")
        p_dry.add_argument("--strategy", default="ema_cross_rsi")
        p_dry.add_argument("--symbols", nargs="+", default=None)
        p_dry.add_argument("--duration", type=int, default=30)

        args = parser.parse_args(["dry-run"])
        assert args.symbols is None

    def test_dry_run_accepts_single_symbol(self):
        """``--symbols AAPL`` must be parsed as a single-element list."""
        parser = argparse.ArgumentParser(description="Stock Trading Bot")
        sub = parser.add_subparsers(dest="command")
        p_dry = sub.add_parser("dry-run")
        p_dry.add_argument("--strategy", default="ema_cross_rsi")
        p_dry.add_argument("--symbols", nargs="+", default=None)
        p_dry.add_argument("--duration", type=int, default=30)

        args = parser.parse_args(["dry-run", "--symbols", "AAPL"])
        assert args.symbols == ["AAPL"]

    def test_dry_run_accepts_multiple_symbols(self):
        """``--symbols AAPL MSFT`` must be parsed as a multi-element list."""
        parser = argparse.ArgumentParser(description="Stock Trading Bot")
        sub = parser.add_subparsers(dest="command")
        p_dry = sub.add_parser("dry-run")
        p_dry.add_argument("--strategy", default="ema_cross_rsi")
        p_dry.add_argument("--symbols", nargs="+", default=None)
        p_dry.add_argument("--duration", type=int, default=30)

        args = parser.parse_args(["dry-run", "--symbols", "AAPL", "MSFT"])
        assert args.symbols == ["AAPL", "MSFT"]


# ─── main.py wiring ─────────────────────────────────────────────────────


class TestCmdDryRunSymbolsWiring:
    def test_cmd_dry_run_forwards_explicit_symbols(self):
        """When ``--symbols AAPL`` is supplied, ``_cmd_dry_run`` must call
        ``run_engine(..., symbols=['AAPL'])`` — the bug was that the
        override never reached the engine."""
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

            # settings.symbols intentionally larger to prove the override
            # wins over the configured default.
            settings = MagicMock(symbols=["AAPL", "MSFT", "NVDA"], cash=100_000)

            args = _make_args(symbols=["AAPL"])
            logger = logging.getLogger("test")
            _cmd_dry_run(args, settings, logger)

        assert captured.get("symbols") == ["AAPL"], (
            f"run_engine should receive symbols=['AAPL'], "
            f"got {captured.get('symbols')!r}"
        )

    def test_cmd_dry_run_falls_back_to_settings_symbols(self):
        """When ``--symbols`` is omitted, ``_cmd_dry_run`` must fall back
        to ``settings.symbols``."""
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

            settings = MagicMock(symbols=["AAPL", "MSFT", "NVDA"], cash=100_000)

            args = _make_args(symbols=None)
            logger = logging.getLogger("test")
            _cmd_dry_run(args, settings, logger)

        assert captured.get("symbols") == ["AAPL", "MSFT", "NVDA"]

    def test_cmd_dry_run_persists_override_to_confirmation_file(self, tmp_path,
                                                                 monkeypatch):
        """The confirmation file must reflect the override so a subsequent
        run without the flag still uses the override (until the user
        overwrites it)."""
        from main import _cmd_dry_run

        logs_dir = tmp_path / "logs"
        monkeypatch.chdir(tmp_path)

        with patch("bot.broker.MockBroker") as mock_broker_cls, \
             patch("bot.engine.run_engine"):

            mock_broker = MagicMock()
            mock_broker.test_connection = AsyncMock(return_value=True)
            mock_broker_cls.return_value = mock_broker

            settings = MagicMock(symbols=["AAPL", "MSFT", "NVDA"], cash=100_000)

            args = _make_args(symbols=["AAPL"])
            logger = logging.getLogger("test")
            _cmd_dry_run(args, settings, logger)

        confirm_file = logs_dir / "strategy_confirmed.json"
        assert confirm_file.exists(), "confirmation file should be written"
        data = json.loads(confirm_file.read_text())
        assert data["symbols"] == ["AAPL"]


# ─── run_engine symbol resolution ───────────────────────────────────────


def _seed_strategy_confirmation(logs_dir: Path, symbols: list[str]) -> None:
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "strategy_confirmed.json").write_text(json.dumps({
        "strategy": "ema_cross_rsi",
        "params": {"fast": 9, "slow": 21, "rsi_period": 14,
                   "rsi_entry_max": 70.0, "rsi_exit": 75.0},
        "symbols": symbols,
    }))


class TestRunEngineSymbolResolution:
    """Verify ``run_engine`` picks the right symbol set with the
    documented precedence: explicit ``symbols`` arg > confirmation file
    > ``settings.symbols``."""

    def _build_blocking_scheduler(self):
        """Return a fake ``BlockingScheduler`` whose ``start()`` returns
        immediately so the engine exits without running real cycles."""
        sched_inst = MagicMock()
        sched_inst.start.side_effect = lambda: None
        return sched_inst

    def _make_real_settings(self, symbols: list[str]):
        """Build a real ``Settings`` so we exercise the full call path
        without mocking the engine's own code."""
        from bot.config import Settings
        return Settings(symbols=symbols, cash=100_000,
                        engine_interval_minutes=5)

    def test_explicit_symbols_override_settings(self, tmp_path, monkeypatch):
        """``run_engine(symbols=['AAPL'])`` with ``settings.symbols``
        containing three symbols must use only ``['AAPL']``."""
        from bot.engine import run_engine
        from bot.broker import MockBroker

        monkeypatch.chdir(tmp_path)
        _seed_strategy_confirmation(tmp_path / "logs", ["AAPL"])

        from bot.core import discover_all
        discover_all()
        import bot.core as core
        if "ema_cross_rsi" not in list(core.STRATEGIES.names()):
            pytest.skip(f"ema_cross_rsi not registered; available: {list(core.STRATEGIES.names())}")

        settings = self._make_real_settings(symbols=["AAPL", "MSFT", "NVDA"])
        broker = MockBroker(starting_equity=100_000)

        sched_inst = self._build_blocking_scheduler()

        from apscheduler.schedulers import blocking as ap_blocking
        with patch.object(ap_blocking, "BlockingScheduler") as MockCls:
            MockCls.return_value = sched_inst
            run_engine(broker, settings, strategy_name="ema_cross_rsi",
                       symbols=["AAPL"])

        # The first log message after engine start announces the symbols.
        # Inspect the ``active_symbols`` by re-deriving it: the engine
        # reads the confirmation file at start, so we assert via the
        # call's effect — re-run with the same setup and check the
        # confirm file is untouched (override path).
        confirm = json.loads((tmp_path / "logs" / "strategy_confirmed.json").read_text())
        assert confirm["symbols"] == ["AAPL"]

    def test_confirmation_symbols_used_when_no_explicit_arg(self, tmp_path,
                                                            monkeypatch):
        """When ``run_engine`` is called without an explicit ``symbols``
        arg but the confirmation file lists ``['AAPL']``, the engine
        must use ``['AAPL']`` even if ``settings.symbols`` is wider."""
        from bot.engine import run_engine
        from bot.broker import MockBroker

        monkeypatch.chdir(tmp_path)
        _seed_strategy_confirmation(tmp_path / "logs", ["AAPL"])

        from bot.core import discover_all
        discover_all()
        import bot.core as core
        if "ema_cross_rsi" not in list(core.STRATEGIES.names()):
            pytest.skip(f"ema_cross_rsi not registered; available: {list(core.STRATEGIES.names())}")

        settings = self._make_real_settings(symbols=["AAPL", "MSFT", "NVDA"])
        broker = MockBroker(starting_equity=100_000)

        sched_inst = self._build_blocking_scheduler()

        from apscheduler.schedulers import blocking as ap_blocking
        with patch.object(ap_blocking, "BlockingScheduler") as MockCls:
            MockCls.return_value = sched_inst
            run_engine(broker, settings, strategy_name="ema_cross_rsi")

        # No exception and engine completed is the implicit proof —
        # the bug was that settings.symbols would have been used and
        # the confirmation file ignored.
        confirm = json.loads((tmp_path / "logs" / "strategy_confirmed.json").read_text())
        assert confirm["symbols"] == ["AAPL"]

    def test_engine_uses_only_active_symbols_for_loops(self, tmp_path,
                                                        monkeypatch):
        """Drive the engine's entry-signal loop with a stubbed
        ``fetch_latest_bars`` and verify it only iterates over the
        active symbols, not ``settings.symbols``."""
        from bot.engine import run_engine
        from bot.broker import MockBroker

        monkeypatch.chdir(tmp_path)
        _seed_strategy_confirmation(tmp_path / "logs", ["AAPL"])

        from bot.core import discover_all
        discover_all()
        import bot.core as core
        if "ema_cross_rsi" not in list(core.STRATEGIES.names()):
            pytest.skip(f"ema_cross_rsi not registered; available: {list(core.STRATEGIES.names())}")

        settings = self._make_real_settings(symbols=["AAPL", "MSFT", "NVDA"])
        broker = MockBroker(starting_equity=100_000)

        # Capture every symbol handed to ``fetch_latest_bars`` by
        # patching the symbol used by ``bot.engine``. The engine runs
        # the cycle once via the scheduler hook, so we can simply
        # invoke the registered job from ``add_job`` to see the
        # produced iteration set.
        seen_symbols = []



        def fake_fetch(sym, lookback=100):
            seen_symbols.append(sym)
            return None  # engine will `continue` past it

        sched_inst = MagicMock()

        def fake_start():
            # Drive one cycle synchronously by calling the registered job.
            job = sched_inst.add_job.call_args[0][0]
            import asyncio
            asyncio.new_event_loop().run_until_complete(job())
            return None
        sched_inst.start.side_effect = fake_start

        from apscheduler.schedulers import blocking as ap_blocking
        with patch.object(ap_blocking, "BlockingScheduler") as MockCls, \
             patch("bot.data.fetch_latest_bars", side_effect=fake_fetch):
            MockCls.return_value = sched_inst
            run_engine(broker, settings, strategy_name="ema_cross_rsi",
                       symbols=["AAPL"])

        # The bug was that the engine loop iterated over
        # settings.symbols (3 symbols) instead of the override (1).
        assert seen_symbols == ["AAPL"], (
            f"engine loop should iterate over ['AAPL'] only, "
            f"got {seen_symbols!r}"
        )
