"""Service-mode helpers for installing/running the bot as a Windows scheduled task.

This module is intentionally framework-light: it only inspects state on disk
and exposes a tiny surface for callers (install scripts, health probes, tests)
to ask "is the service installed?", "is it running?", "what's the last
result?" without having to shell out to ``schtasks.exe``.

All functions are pure helpers — they do NOT register or modify the scheduled
task. Use ``scripts/install_service.ps1`` / ``scripts/uninstall_service.ps1``
to manage the task itself; this module just queries.

Windows-only ``schtasks.exe`` parsing is gated behind ``IS_WINDOWS``; on other
platforms the helpers return ``None`` / ``False`` so they can be imported
safely from cross-platform code (CI, Docker, Linux dev boxes).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_TASK_NAME = "StockTradingBot-Live"

# Default log files written by install_service.ps1.
DEFAULT_STDOUT_LOG = Path("logs/service_stdout.log")
DEFAULT_STDERR_LOG = Path("logs/service_stderr.log")

IS_WINDOWS = platform.system().lower() == "windows"


# ── Service status dataclass ──────────────────────────────────────────


@dataclass
class ServiceStatus:
    """Snapshot of the Windows scheduled-task state for the bot."""

    installed: bool
    running: bool
    task_name: str
    last_run_time: datetime | None
    last_result_code: int | None
    next_run_time: datetime | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (datetimes → ISO strings)."""
        return {
            "installed": self.installed,
            "running": self.running,
            "task_name": self.task_name,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_result_code": self.last_result_code,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
        }


# ── Helpers ────────────────────────────────────────────────────────────


def _parse_schtasks_query(text: str) -> tuple[bool, dict[str, str]]:
    """Parse the relevant fields out of a ``schtasks /Query`` output block.

    The output for a single task looks roughly like::

        TaskName:        \\StockTradingBot-Live
        Status:          Running
        Last Run Time:   8/18/2025 9:30:00 AM
        Last Result:     0
        Next Run Time:   8/19/2025 9:30:00 AM

    We accept either "FOO: value" or "FOO    : value" (PowerShell widens
    the spacing) and pull a small fixed schema.
    """
    fields: dict[str, str] = {}
    rx = re.compile(r"^(?P<k>[A-Za-z ]+?)\s*:\s*(?P<v>.*?)\s*$")
    for line in text.splitlines():
        m = rx.match(line)
        if not m:
            continue
        key = m.group("k").strip()
        val = m.group("v").strip()
        # First wins; schtasks prints some keys twice (verbose + summary).
        fields.setdefault(key, val)
    installed = "TaskName" in fields and bool(fields.get("TaskName"))
    return installed, fields


def _coerce_datetime(s: str | None) -> datetime | None:
    """Best-effort parse of a ``schtasks``-formatted date/time string."""
    if not s or s.lower().startswith("n/a"):
        return None
    # Common schtasks formats: "8/18/2025 9:30:00 AM", "8/18/2025 9:30:00"
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    logger.debug("Could not parse datetime %r", s)
    return None


# ── Public API ────────────────────────────────────────────────────────


def get_service_status(task_name: str = DEFAULT_TASK_NAME) -> ServiceStatus:
    """Return a :class:`ServiceStatus` snapshot.

    On non-Windows or if ``schtasks`` is unavailable, returns a status with
    ``installed=False`` and ``running=False`` rather than raising.
    """
    if not IS_WINDOWS:
        logger.debug("get_service_status: non-Windows host — returning empty status")
        return ServiceStatus(
            installed=False,
            running=False,
            task_name=task_name,
            last_run_time=None,
            last_result_code=None,
            next_run_time=None,
        )

    try:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name, "/V", "/FO", "LIST"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("schtasks not available: %s", exc)
        return ServiceStatus(False, False, task_name, None, None, None)

    if proc.returncode != 0:
        # 1 = task not found. Anything else: log + return uninstalled.
        msg = proc.stderr.strip() or proc.stdout.strip()
        logger.info("schtasks query for %r returned %d: %s", task_name, proc.returncode, msg)
        return ServiceStatus(False, False, task_name, None, None, None)

    installed, fields = _parse_schtasks_query(proc.stdout)
    if not installed:
        return ServiceStatus(False, False, task_name, None, None, None)

    status = fields.get("Status", "").lower()
    last_result_raw = fields.get("Last Result", "").strip()
    try:
        last_result = int(last_result_raw) if last_result_raw else None
    except ValueError:
        last_result = None

    return ServiceStatus(
        installed=True,
        running=(status == "running"),
        task_name=task_name,
        last_run_time=_coerce_datetime(fields.get("Last Run Time")),
        last_result_code=last_result,
        next_run_time=_coerce_datetime(fields.get("Next Run Time")),
    )


def tail_service_log(path: Path | str, n_bytes: int = 4096) -> str:
    """Return the last ``n_bytes`` of a service log file (best-effort).

    Returns an empty string if the file is missing or unreadable.
    """
    p = Path(path)
    try:
        if not p.exists():
            return ""
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError as exc:
        logger.debug("tail_service_log(%s): %s", p, exc)
        return ""


def write_health_record(path: Path | str, status: ServiceStatus) -> None:
    """Persist the latest service status JSON for the dashboard to consume."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(status.as_dict(), indent=2))
    except OSError as exc:
        logger.warning("write_health_record(%s): %s", p, exc)


def load_health_record(path: Path | str = "logs/service_status.json") -> dict[str, Any] | None:
    """Read the latest :func:`write_health_record` payload, or ``None``."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("load_health_record(%s): %s", p, exc)
        return None


# Allow `python -m bot.service` smoke test.
if __name__ == "__main__":  # pragma: no cover
    s = get_service_status()
    print(json.dumps(s.as_dict(), indent=2))