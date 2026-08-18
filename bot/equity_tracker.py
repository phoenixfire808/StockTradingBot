"""Per-strategy equity tracking with CSV + JSON persistence.

Records an equity snapshot per strategy per cycle so the multi-strategy
engine can track how each strategy contributes to total account equity.

Storage layout (mirrors EngineState convention — everything under logs/):
  - CSV  : logs/equity_{strategy}.csv   (timestamp,equity per line)
  - JSON : logs/equity_curves.json       ({strategy: [{ts, equity}, ...]})

The CSV is the append-friendly primary store (one file per strategy).
The JSON is a query-friendly snapshot rebuilt from CSVs on read or
flushed incrementally on record(). Both are kept in sync.

Self-contained: stdlib only (csv, json, logging, pathlib).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default directory for equity files — mirrors EngineState / TradeStore.
DEFAULT_LOGS_DIR = Path("logs")

# Sanitise strategy names for safe filenames: keep alnum + underscore + dash.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_filename(strategy: str) -> str:
    """Return a filesystem-safe stem for a strategy name."""
    cleaned = _SAFE_NAME.sub("_", strategy.strip())
    return cleaned or "unknown"


class EquityTracker:
    """Per-strategy equity curve tracker with CSV + JSON persistence.

    Usage::

        tracker = EquityTracker()
        tracker.record("ema_cross", 100_000.0, "2024-01-01T10:00:00Z")
        curve = tracker.get_curve("ema_cross")   # [{timestamp, equity}, ...]
        all_curves = tracker.get_all_curves()     # {strategy: [...]}
        tracker.reset()
    """

    def __init__(self, logs_dir: Path | str | None = None) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir else DEFAULT_LOGS_DIR
        self._json_path = self._logs_dir / "equity_curves.json"
        logger.debug("EquityTracker init: logs_dir=%s", self._logs_dir)

    # ── paths ────────────────────────────────────────────────────────

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir

    def _csv_path(self, strategy: str) -> Path:
        return self._logs_dir / f"equity_{_safe_filename(strategy)}.csv"

    # ── write ───────────────────────────────────────────────────────

    def record(self, strategy: str, equity: float, ts: str) -> None:
        """Append an equity snapshot for *strategy* at timestamp *ts*.

        Writes to the per-strategy CSV and updates the JSON snapshot.
        """
        if not strategy:
            logger.warning("EquityTracker.record: empty strategy name — skipping")
            return
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._csv_path(strategy)

        # ── CSV append ──
        needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(["timestamp", "equity"])
            writer.writerow([ts, f"{float(equity):.4f}"])
        logger.info(
            "equity recorded strategy=%s equity=%.2f ts=%s → %s",
            strategy, equity, ts, csv_path.name,
        )

        # ── JSON incremental update ──
        self._append_json(strategy, ts, float(equity))

    def _append_json(self, strategy: str, ts: str, equity: float) -> None:
        """Update the JSON snapshot by appending one point."""
        curves = self._read_json()
        points = curves.setdefault(strategy, [])
        points.append({"timestamp": ts, "equity": equity})
        self._write_json(curves)

    def _read_json(self) -> dict[str, list[dict]]:
        if not self._json_path.exists():
            return {}
        try:
            with open(self._json_path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("EquityTracker: corrupt JSON %s — %s", self._json_path, exc)
            return {}

    def _write_json(self, curves: dict[str, list[dict]]) -> None:
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._json_path, "w") as f:
            json.dump(curves, f, indent=2)

    # ── read ─────────────────────────────────────────────────────────

    def get_curve(self, strategy: str) -> list[dict]:
        """Return [{timestamp, equity}, ...] for *strategy*, ordered by write."""
        # Prefer CSV (source of truth) when it exists; fall back to JSON.
        csv_path = self._csv_path(strategy)
        if csv_path.exists():
            return self._read_csv(csv_path)
        return self._read_json().get(strategy, [])

    def get_all_curves(self) -> dict[str, list[dict]]:
        """Return {strategy: [{timestamp, equity}, ...]} for every tracked strategy."""
        curves: dict[str, list[dict]] = {}

        # Scan CSV files to discover strategies
        if self._logs_dir.is_dir():
            for p in self._logs_dir.glob("equity_*.csv"):
                # Recover strategy name from filename
                stem = p.stem  # "equity_{strategy}"
                if not stem.startswith("equity_"):
                    continue
                strategy = stem[len("equity_"):]
                curves[strategy] = self._read_csv(p)

        # Merge any JSON-only strategies (shouldn't normally exist, but safe)
        json_curves = self._read_json()
        for strat, points in json_curves.items():
            if strat not in curves:
                curves[strat] = points

        logger.debug("EquityTracker.get_all_curves: %d strategies", len(curves))
        return curves

    @staticmethod
    def _read_csv(csv_path: Path) -> list[dict]:
        """Parse a per-strategy equity CSV into [{timestamp, equity}, ...]."""
        points: list[dict] = []
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        eq = float(row.get("equity", 0))
                    except (ValueError, TypeError):
                        logger.warning("EquityTracker: bad equity in %s row %r", csv_path, row)
                        continue
                    points.append({"timestamp": row.get("timestamp", ""), "equity": eq})
        except OSError as exc:
            logger.error("EquityTracker: failed reading %s — %s", csv_path, exc)
        return points

    # ── maintenance ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Delete all per-strategy equity CSVs and the JSON snapshot."""
        removed = 0
        if self._logs_dir.is_dir():
            for p in self._logs_dir.glob("equity_*.csv"):
                try:
                    p.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("EquityTracker: could not delete %s — %s", p, exc)
        if self._json_path.exists():
            try:
                self._json_path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("EquityTracker: could not delete %s — %s", self._json_path, exc)
        logger.info("EquityTracker reset: removed %d files from %s", removed, self._logs_dir)
