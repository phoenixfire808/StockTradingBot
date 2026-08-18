"""Tests for bot.alerts — Discord/SMTP alert manager.

Covers:
  * AlertConfig.from_env — env var parsing, defaults, type coercion.
  * AlertManager — all four send_* methods dispatch via injected notifier.
  * Dry-run / disabled-transport paths never raise.
  * Custom drawdown threshold is honored.
  * Integration with the bot.engine import path (smoke import).
"""

from __future__ import annotations

import pytest

from bot.alerts import AlertConfig, AlertManager


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def captured_alerts() -> list[tuple[str, str, str]]:
    """List of (title, message, level) tuples captured by the test notifier."""
    return []


@pytest.fixture
def manager(captured_alerts) -> AlertManager:
    """An AlertManager wired to a capturing notifier — no real Discord/SMTP."""

    def _notifier(title: str, message: str, level: str) -> None:
        captured_alerts.append((title, message, level))

    return AlertManager(notifier=_notifier)


# ── AlertConfig.from_env ────────────────────────────────────────────────


class TestAlertConfigFromEnv:
    def test_defaults_when_no_env(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_TO_ADDRS", raising=False)

        cfg = AlertConfig.from_env()
        assert cfg.discord_enabled is False
        assert cfg.smtp_enabled is False
        assert cfg.any_enabled is False
        assert cfg.smtp_port == 587
        assert cfg.smtp_use_tls is True
        assert cfg.drawdown_alert_pct == pytest.approx(2.0)
        assert cfg.discord_username == "StockTradingBot"

    def test_discord_enabled_with_webhook(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook/abc")
        monkeypatch.delenv("SMTP_HOST", raising=False)

        cfg = AlertConfig.from_env()
        assert cfg.discord_enabled is True
        assert cfg.any_enabled is True

    def test_smtp_to_addrs_parsed_csv(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_TO_ADDRS", "a@x.com, b@x.com ,c@x.com")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        cfg = AlertConfig.from_env()
        assert cfg.smtp_enabled is True
        assert cfg.smtp_to_addrs == ["a@x.com", "b@x.com", "c@x.com"]

    def test_drawdown_threshold_overridden(self, monkeypatch):
        monkeypatch.setenv("ALERT_DRAWDOWN_PCT", "5.5")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        cfg = AlertConfig.from_env()
        assert cfg.drawdown_alert_pct == pytest.approx(5.5)

    def test_dry_run_flag(self, monkeypatch):
        monkeypatch.setenv("ALERT_DRY_RUN", "1")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        cfg = AlertConfig.from_env()
        assert cfg.dry_run is True

    def test_invalid_port_raises_value_error(self, monkeypatch):
        """Documented behaviour: a non-numeric SMTP_PORT raises ValueError
        (alerts.from_env does not silently swallow type errors)."""
        monkeypatch.setenv("SMTP_PORT", "not-a-number")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        with pytest.raises(ValueError):
            AlertConfig.from_env()


# ── AlertManager (no transports — notifier-captured) ───────────────────


class TestAlertManagerSend:
    def test_send_fill_alert(self, manager, captured_alerts):
        manager.send_fill_alert(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.25,
            reason="signal",
            equity=101502.50,
        )
        assert len(captured_alerts) == 1
        title, message, level = captured_alerts[0]
        assert title == "Fill: BUY AAPL"
        assert level == "info"
        assert "BUY" in message
        assert "AAPL" in message
        assert "10" in message
        assert "150.25" in message
        assert "Equity" in message
        assert "BUY" in title  # side coerced to uppercase

    def test_send_kill_switch_alert(self, manager, captured_alerts):
        manager.send_kill_switch_alert(
            reason="daily_loss_exceeded",
            drawdown_pct=3.2,
            equity=97500.0,
        )
        assert len(captured_alerts) == 1
        title, message, level = captured_alerts[0]
        assert title == "Kill Switch Tripped"
        assert level == "critical"
        assert "KILL SWITCH" in message
        assert "daily_loss_exceeded" in message
        assert "3.20%" in message

    def test_send_drawdown_alert_uses_config_threshold_by_default(
        self, manager, captured_alerts
    ):
        manager.send_drawdown_alert(drawdown_pct=2.5, equity=99000.0)
        title, _, level = captured_alerts[0]
        assert title == "Drawdown Alert"
        assert level == "warning"

    def test_send_drawdown_alert_honors_explicit_threshold(
        self, manager, captured_alerts
    ):
        manager.send_drawdown_alert(
            drawdown_pct=5.0, equity=99000.0, threshold=4.5
        )
        _, message, _ = captured_alerts[0]
        assert "4.50%" in message

    def test_send_daily_summary(self, manager, captured_alerts):
        positions = {
            "AAPL": {"qty": 10, "entry_price": 150.0},
            "MSFT": {"qty": 5, "entry_price": 300.0},
        }
        manager.send_daily_summary(
            equity=102500.0,
            day_start_equity=100000.0,
            positions=positions,
            trade_count=7,
            cycle_count=42,
        )
        assert len(captured_alerts) == 1
        title, message, level = captured_alerts[0]
        assert title == "Daily Summary"
        assert level == "info"
        assert "$102,500.00" in message
        assert "+$2,500.00" in message  # P&L positive
        assert "+2.50%" in message  # pct positive
        assert "7" in message  # trade count
        assert "42" in message  # cycle count
        assert "AAPL" in message
        assert "MSFT" in message

    def test_send_daily_summary_no_day_start(self, manager, captured_alerts):
        manager.send_daily_summary(equity=100000.0, day_start_equity=None)
        _, message, _ = captured_alerts[0]
        assert "N/A" in message  # P&L marked N/A without day start


# ── AlertManager (no transports, no notifier — should NOT raise) ────────


class TestAlertManagerSafeWhenUnconfigured:
    def test_no_transport_no_notifier_is_silent(self):
        """With no notifier and no transports, dispatch is a no-op — must
        not raise and must not make any network calls."""
        mgr = AlertManager(config=AlertConfig(), notifier=None)
        # None of these should raise.
        mgr.send_fill_alert("AAPL", "buy", 10, 150.0)
        mgr.send_kill_switch_alert("manual_stop")
        mgr.send_drawdown_alert(2.0)
        mgr.send_daily_summary(equity=100000.0)

    def test_dry_run_does_not_call_transports(self):
        from unittest.mock import MagicMock

        cfg = AlertConfig(
            discord_webhook_url="https://discord.example/webhook/x",
            smtp_host="smtp.example.com",
            smtp_to_addrs=["a@x.com"],
            dry_run=True,
        )
        mgr = AlertManager(config=cfg, notifier=None)
        # Patch the transport methods to ensure they're never called.
        mgr._send_discord = MagicMock()
        mgr._send_smtp = MagicMock()
        mgr.send_fill_alert("AAPL", "buy", 1, 100.0)
        mgr._send_discord.assert_not_called()
        mgr._send_smtp.assert_not_called()

    def test_transport_failure_is_swallowed(self):
        """A Discord webhook 500 should be logged + swallowed — engine
        must never crash because of an alert."""
        from unittest.mock import MagicMock

        cfg = AlertConfig(
            discord_webhook_url="https://discord.example/webhook/x",
            dry_run=False,
        )
        mgr = AlertManager(config=cfg, notifier=None)
        mgr._send_discord = MagicMock(side_effect=RuntimeError("network down"))
        # Must not raise.
        mgr.send_fill_alert("AAPL", "buy", 1, 100.0)


# ── Engine integration smoke ────────────────────────────────────────────


class TestEngineImport:
    def test_alert_manager_constructible_via_from_env(self):
        """Mirror the call site in bot/engine.py to make sure the factory
        path used by the engine still works after refactors."""
        mgr = AlertManager.from_env()
        assert mgr is not None
        assert hasattr(mgr, "send_fill_alert")
        assert hasattr(mgr, "send_kill_switch_alert")
        assert hasattr(mgr, "send_drawdown_alert")
        assert hasattr(mgr, "send_daily_summary")