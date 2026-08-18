"""Alert notifications — Discord webhook and SMTP email.

Sends push notifications for key trading events:
  - Order fills (BUY/SELL)
  - Kill-switch trip (daily-loss guard or manual emergency stop)
  - Large drawdown (configurable threshold)
  - Daily summary (P&L, positions, cycle count)

Transport:
  - Discord: webhook URL POST with JSON payload (requests).
  - Email:   SMTP with optional TLS + auth (smtplib).

Configuration is read from environment variables via AlertConfig, with
sensible disabled-by-default behaviour so the bot runs fine without any
alerts configured.  AlertManager is safe to call even when unconfigured:
every method logs at INFO and returns silently when no transport is enabled.

Design goals:
  - Never block the engine: all sends wrapped in try/except, logged on failure.
  - Idempotent: calling the same alert twice sends twice (no dedup); the engine
    is responsible for choosing when to call.
  - Testable: transports are pluggable via the ``_send`` hook; tests inject
    a fake.

Integration points (bot/engine.py):
  - send_fill_alert      → after every BUY / SELL / stop / target fill
  - send_kill_switch_alert → when KillSwitch.check() first trips
  - send_drawdown_alert    → when intra-day drawdown crosses threshold
  - send_daily_summary     → scheduled once per UTC day
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

# Default drawdown % that triggers an alert when not otherwise configured.
_DEFAULT_DRAWDOWN_ALERT_PCT = 2.0


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class AlertConfig:
    """Alert transport configuration from environment.

    All fields default to "disabled" so the bot runs without any alert
    configuration.  Setting ``discord_webhook_url`` or ``smtp_host`` enables
    the respective transport.
    """

    discord_webhook_url: str = ""
    discord_username: str = "StockTradingBot"
    discord_avatar_url: str = ""

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_from_addr: str = ""
    smtp_to_addrs: list[str] = field(default_factory=list)

    # Drawdown threshold (% of day-start equity) that triggers an alert.
    drawdown_alert_pct: float = _DEFAULT_DRAWDOWN_ALERT_PCT

    # When True, no transport is actually called — used by tests.
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """Load alert config from environment variables.

        Recognised env vars:
          DISCORD_WEBHOOK_URL, DISCORD_USERNAME, DISCORD_AVATAR_URL
          SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
          SMTP_USE_TLS, SMTP_FROM_ADDR, SMTP_TO_ADDRS (comma-separated)
          ALERT_DRAWDOWN_PCT
          ALERT_DRY_RUN (1/true → dry-run mode)
        """
        def _env(key: str, default: str = "") -> str:
            return os.getenv(key, default)

        def _bool(key: str, default: bool = True) -> bool:
            val = os.getenv(key)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        to_addrs_raw = _env("SMTP_TO_ADDRS", "")
        to_addrs = [a.strip() for a in to_addrs_raw.split(",") if a.strip()]

        return cls(
            discord_webhook_url=_env("DISCORD_WEBHOOK_URL"),
            discord_username=_env("DISCORD_USERNAME", "StockTradingBot"),
            discord_avatar_url=_env("DISCORD_AVATAR_URL"),
            smtp_host=_env("SMTP_HOST"),
            smtp_port=int(_env("SMTP_PORT", "587") or 587),
            smtp_username=_env("SMTP_USERNAME"),
            smtp_password=_env("SMTP_PASSWORD"),
            smtp_use_tls=_bool("SMTP_USE_TLS", True),
            smtp_from_addr=_env("SMTP_FROM_ADDR"),
            smtp_to_addrs=to_addrs,
            drawdown_alert_pct=float(
                _env("ALERT_DRAWDOWN_PCT", str(_DEFAULT_DRAWDOWN_ALERT_PCT))
                or _DEFAULT_DRAWDOWN_ALERT_PCT
            ),
            dry_run=_bool("ALERT_DRY_RUN", False),
        )

    @property
    def discord_enabled(self) -> bool:
        return bool(self.discord_webhook_url)

    @property
    def smtp_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_to_addrs)

    @property
    def any_enabled(self) -> bool:
        return self.discord_enabled or self.smtp_enabled


# ── Alert Manager ────────────────────────────────────────────────────


class AlertManager:
    """Sends trading alerts via Discord webhook and/or SMTP email.

    Construct with an AlertConfig (or call ``AlertManager.from_env()``).
    Each ``send_*_alert`` method builds a title + message, routes it through
    the enabled transports, and logs the outcome.  Failures are logged at
    WARNING and swallowed — alerts must never break the trading engine.

    Parameters
    ----------
    config : AlertConfig
        Transport settings.
    notifier : Callable[[str, str, str], None] | None
        Test hook: if provided, called as ``notifier(title, message, level)``
        instead of the real transports.  Level is "info" | "warning" | "critical".
    """

    def __init__(
        self,
        config: AlertConfig | None = None,
        notifier: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.config = config or AlertConfig.from_env()
        self._notifier = notifier
        logger.info(
            "AlertManager init: discord=%s smtp=%s dry_run=%s drawdown_alert_pct=%.2f",
            self.config.discord_enabled,
            self.config.smtp_enabled,
            self.config.dry_run,
            self.config.drawdown_alert_pct,
        )

    @classmethod
    def from_env(cls) -> "AlertManager":
        """Build an AlertManager from environment variables."""
        return cls(config=AlertConfig.from_env())

    # ── Public alert methods ──────────────────────────────────────────

    def send_fill_alert(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        reason: str = "",
        equity: float | None = None,
    ) -> None:
        """Notify that an order was filled / executed.

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. ``"AAPL"``.
        side : str
            ``"BUY"`` or ``"SELL"``.
        qty : int
            Number of shares.
        price : float
            Execution price.
        reason : str
            Why the fill happened (``"signal"``, ``"stop_loss"``, …).
        equity : float | None
            Account equity after the fill, if known.
        """
        side_upper = (side or "").upper()
        emoji = "[BUY]" if side_upper == "BUY" else "[SELL]"
        eq_str = f"\n**Equity:** ${equity:,.2f}" if equity is not None else ""
        message = (
            f"{emoji} **{side_upper} {qty} {symbol} @ ${price:.2f}**\n"
            f"**Reason:** {reason or 'signal'}{eq_str}\n"
            f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`"
        )
        title = f"Fill: {side_upper} {symbol}"
        self._dispatch(title, message, level="info")
        logger.info(
            "Fill alert sent: %s %s qty=%d price=%.2f reason=%s",
            side_upper, symbol, qty, price, reason,
        )

    def send_kill_switch_alert(
        self,
        reason: str,
        drawdown_pct: float | None = None,
        equity: float | None = None,
    ) -> None:
        """Notify that the kill switch has tripped.

        Parameters
        ----------
        reason : str
            What triggered the halt (``"daily_loss_exceeded"``, ``"manual_stop"``).
        drawdown_pct : float | None
            Current daily drawdown percentage, if known.
        equity : float | None
            Current account equity, if known.
        """
        dd_str = f"\n**Drawdown:** {drawdown_pct:.2f}%" if drawdown_pct is not None else ""
        eq_str = f"\n**Equity:** ${equity:,.2f}" if equity is not None else ""
        message = (
            f"[STOP] **KILL SWITCH TRIPPED**\n"
            f"**Reason:** {reason}{dd_str}{eq_str}\n"
            f"Trading is **halted** until the next day reset or manual re-arm.\n"
            f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`"
        )
        title = "Kill Switch Tripped"
        self._dispatch(title, message, level="critical")
        logger.warning("Kill-switch alert sent: reason=%s dd=%.2f", reason, drawdown_pct or 0)

    def send_drawdown_alert(
        self,
        drawdown_pct: float,
        equity: float | None = None,
        threshold: float | None = None,
    ) -> None:
        """Notify that intra-day drawdown crossed the alert threshold.

        This is *advisory* — the kill switch itself halts trading; this alert
        fires earlier so the operator is aware of worsening losses.

        Parameters
        ----------
        drawdown_pct : float
            Current daily drawdown percentage.
        equity : float | None
            Current account equity, if known.
        threshold : float | None
            Threshold that was crossed (defaults to config value).
        """
        thresh = threshold if threshold is not None else self.config.drawdown_alert_pct
        eq_str = f"\n**Equity:** ${equity:,.2f}" if equity is not None else ""
        message = (
            f"[WARN] **DRAWDOWN ALERT**\n"
            f"**Drawdown:** {drawdown_pct:.2f}% (threshold: {thresh:.2f}%)\n"
            f"Trading continues — monitoring closely.{eq_str}\n"
            f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`"
        )
        title = "Drawdown Alert"
        self._dispatch(title, message, level="warning")
        logger.warning(
            "Drawdown alert sent: dd=%.2f threshold=%.2f", drawdown_pct, thresh,
        )

    def send_daily_summary(
        self,
        equity: float,
        day_start_equity: float | None = None,
        positions: dict[str, dict[str, Any]] | None = None,
        trade_count: int = 0,
        cycle_count: int = 0,
    ) -> None:
        """Send the end-of-day summary.

        Parameters
        ----------
        equity : float
            Final equity for the day.
        day_start_equity : float | None
            Equity at day start (for P&L calc).
        positions : dict | None
            Open positions keyed by symbol.
        trade_count : int
            Number of trades executed today.
        cycle_count : int
            Number of engine cycles run today.
        """
        pnl = None
        pnl_str = "N/A"
        if day_start_equity is not None and day_start_equity != 0:
            pnl = equity - day_start_equity
            pnl_pct = (pnl / day_start_equity) * 100
            sign = "+" if pnl >= 0 else ""
            pnl_str = f"{sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)"

        pos_count = len(positions) if positions else 0
        pos_lines = ""
        if positions:
            lines = []
            for sym, p in positions.items():
                qty = p.get("qty", 0)
                ep = p.get("entry_price", 0)
                lines.append(f"  • {sym}: {qty} @ ${ep:.2f}")
            pos_lines = "\n".join(lines)

        message = (
            f"[SUMMARY] **DAILY SUMMARY**\n"
            f"**Equity:** ${equity:,.2f}\n"
            f"**Day P&L:** {pnl_str}\n"
            f"**Trades today:** {trade_count}\n"
            f"**Cycles:** {cycle_count}\n"
            f"**Open positions:** {pos_count}"
        )
        if pos_lines:
            message += f"\n**Positions:**\n{pos_lines}"
        message += f"\n`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`"

        title = "Daily Summary"
        self._dispatch(title, message, level="info")
        logger.info("Daily summary sent: equity=%.2f pnl=%s trades=%d", equity, pnl_str, trade_count)

    # ── Internal dispatch ─────────────────────────────────────────────

    def _dispatch(self, title: str, message: str, level: str = "info") -> None:
        """Route an alert to all enabled transports.

        Honours ``dry_run`` and the ``notifier`` test hook.
        """
        if self._notifier is not None:
            # Test mode: delegate to the injected notifier.
            self._notifier(title, message, level)
            return

        if not self.config.any_enabled:
            logger.debug("Alert dispatch (no transports): %s", title)
            return

        if self.config.dry_run:
            logger.info("[DRY-RUN] Alert: %s — %s", title, message)
            return

        # Discord
        if self.config.discord_enabled:
            try:
                self._send_discord(title, message)
            except Exception as exc:
                logger.warning("Discord alert failed for %r: %s", title, exc)

        # SMTP
        if self.config.smtp_enabled:
            try:
                self._send_smtp(title, message)
            except Exception as exc:
                logger.warning("SMTP alert failed for %r: %s", title, exc)

    # ── Transports ────────────────────────────────────────────────────

    def _send_discord(self, title: str, message: str) -> None:
        """POST a webhook payload to Discord."""
        payload: dict[str, Any] = {
            "username": self.config.discord_username,
            "content": f"**{title}**\n{message}",
        }
        if self.config.discord_avatar_url:
            payload["avatar_url"] = self.config.discord_avatar_url

        resp = requests.post(
            self.config.discord_webhook_url,
            json=payload,
            timeout=10,
        )
        logger.debug("Discord webhook %s → HTTP %d", title, resp.status_code)
        if not resp.ok:
            logger.warning(
                "Discord webhook non-2xx: %d %s", resp.status_code, resp.text[:200],
            )

    def _send_smtp(self, title: str, message: str) -> None:
        """Send an email via SMTP with optional TLS + auth."""
        cfg = self.config
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[StockTradingBot] {title}"
        msg["From"] = cfg.smtp_from_addr or cfg.smtp_username
        msg["To"] = ", ".join(cfg.smtp_to_addrs)
        msg["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

        # Plain-text body — Discord markdown stripped for email readability.
        plain = message.replace("**", "").replace("`", "")
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(message, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
            if cfg.smtp_use_tls:
                server.starttls(context=context)
            if cfg.smtp_username and cfg.smtp_password:
                server.login(cfg.smtp_username, cfg.smtp_password)
            server.sendmail(
                cfg.smtp_from_addr or cfg.smtp_username,
                cfg.smtp_to_addrs,
                msg.as_string(),
            )
        logger.debug("SMTP alert sent: %s → %s", title, cfg.smtp_to_addrs)
