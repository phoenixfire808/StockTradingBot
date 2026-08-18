"""SQLite-backed trade store — replaces CSV logging with a queryable DB.

Schema tables:
  - trades         : one row per executed order (per-strategy)
  - positions       : current open positions (upserted by symbol, per-strategy)
  - equity_history : equity snapshot per cycle (per-strategy)
  - signals        : strategy signal log (per-strategy)
  - engine_state   : latest engine lifecycle state (single-row upsert)

Design:
  - sqlite3 stdlib only (no extra deps).
  - WAL journal mode for concurrent read while engine writes.
  - Context managers (with sqlite3.connect(...)) for every transaction.
  - Thread-safe: connections are opened per-call (sqlite3 default check_same_thread=True).
  - EngineState in bot/engine.py delegates here when SQLite is enabled (auto-detected
    via env TRADE_STORE=sqlite, or an explicit flag), falling back to CSV otherwise.

Per-strategy support (backward compatible):
  - trades / equity_history / signals / positions carry a ``strategy`` column.
  - All insert/get methods accept an optional ``strategy`` arg defaulting to ``""``
    so legacy single-strategy callers keep working unchanged.
  - A lightweight migration (ALTER TABLE ADD COLUMN) runs on schema init for DBs
    created before the per-strategy era.

Path: logs/trade_store.db (relative to project root, like the existing CSV files).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Default DB path mirrors the existing CSV location (logs/).
DEFAULT_DB_PATH = Path("logs/trade_store.db")

# Feature flag: "sqlite"  -> force SQLite store
#               "csv"     -> force CSV fallback (legacy)
#               unset    -> auto-detect (enabled if DB file exists or env says sqlite)
_TRADE_STORE_ENV = "TRADE_STORE"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,        -- BUY | SELL
    qty             INTEGER NOT NULL,
    price           REAL    NOT NULL,
    reason          TEXT    NOT NULL,
    strategy        TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts        ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT    PRIMARY KEY,
    qty             INTEGER NOT NULL,
    entry_price     REAL    NOT NULL DEFAULT 0,
    entry_ts        TEXT,
    stop            REAL    NOT NULL DEFAULT 0,
    target          REAL    NOT NULL DEFAULT 0,
    strategy        TEXT    NOT NULL DEFAULT '',
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);

CREATE TABLE IF NOT EXISTS equity_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    equity          REAL    NOT NULL,
    strategy        TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_equity_ts       ON equity_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_strategy ON equity_history(strategy);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    signal          INTEGER NOT NULL,        -- 1 buy, -1 sell, 0 neutral
    reason          TEXT    NOT NULL,
    strategy        TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol   ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_ts        ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_strategy  ON signals(strategy);

CREATE TABLE IF NOT EXISTS engine_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    mode                TEXT,
    strategy            TEXT,
    params_json         TEXT,
    kill_switch         INTEGER NOT NULL DEFAULT 0,
    day_start_equity    REAL,
    last_cycle_ts       TEXT,
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# ── Migration: add `strategy` column to legacy DBs ──────────────────
# Runs after _SCHEMA_SQL. Each ALTER is guarded by a column-exists check
# so it's a no-op on fresh DBs (which already have the column from CREATE).
_MIGRATION_STRATEGY_COLUMN = [
    ("trades", "strategy"),
    ("positions", "strategy"),
    ("equity_history", "strategy"),
    ("signals", "strategy"),
]


class TradeStore:
    """SQLite store for trades, positions, equity, signals, and engine state.

    Thread-safe via per-call connections; a module-level lock serialises
    schema init so two threads racing on first-use both see a ready DB.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._init_lock = threading.Lock()
        self._initialised = False
        logger.debug("TradeStore configured: db_path=%s", self._db_path)

    # ── connection / schema ──────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection with sensible pragmas."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,            # wait on locks
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, rollback on error.

        Uses the default (deferred) isolation level so Python auto-begins a
        transaction before DML. We commit/rollback explicitly. This is the
        standard sqlite3 pattern and handles executescript (which issues its
        own implicit commit) gracefully — a no-op commit with no active txn.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Transaction rolled back: db=%s", self._db_path)
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:  # double-check after acquiring lock
                return
            with self._txn() as conn:
                conn.executescript(_SCHEMA_SQL)
                # ── migration: add strategy column to legacy tables ──
                self._migrate_add_strategy_column(conn)
            self._initialised = True
            logger.info("TradeStore schema ready: %s", self._db_path)

    @staticmethod
    def _migrate_add_strategy_column(conn: sqlite3.Connection) -> None:
        """Add `strategy TEXT NOT NULL DEFAULT ''` to legacy tables lacking it.

        Idempotent: checks pragma table_info before each ALTER. Fresh DBs
        created by _SCHEMA_SQL already have the column → no-op.
        """
        for table, col in _MIGRATION_STRATEGY_COLUMN:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if col not in cols:
                logger.info("Migration: adding column %s.%s", table, col)
                conn.execute(f'ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ""')

    # ── writes ────────────────────────────────────────────────────────

    def insert_trade(
        self, timestamp: str, symbol: str, side: str, qty: int, price: float, reason: str,
        strategy: str = "",
    ) -> int:
        """Insert a trade row. Returns the new row id.

        *strategy* tags the trade with its originating strategy (default ``""``
        for backward-compatible single-strategy callers).
        """
        self._ensure_schema()
        with self._txn() as conn:
            cur = conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, qty, price, reason, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, symbol, side.upper(), int(qty), float(price), reason, strategy),
            )
            row_id = int(cur.lastrowid)
        logger.info(
            "trade inserted id=%d ts=%s sym=%s side=%s qty=%d price=%.4f reason=%s strategy=%s",
            row_id, timestamp, symbol, side, qty, price, reason, strategy or "(default)",
        )
        return row_id

    def insert_signal(self, timestamp: str, symbol: str, signal_val: int, reason: str, strategy: str = "") -> int:
        """Insert a signal row. Returns the new row id.

        *strategy* tags the signal with its originating strategy.
        """
        self._ensure_schema()
        with self._txn() as conn:
            cur = conn.execute(
                "INSERT INTO signals (timestamp, symbol, signal, reason, strategy) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, symbol, int(signal_val), reason, strategy),
            )
            row_id = int(cur.lastrowid)
        logger.info(
            "signal inserted id=%d ts=%s sym=%s signal=%d reason=%s strategy=%s",
            row_id, timestamp, symbol, signal_val, reason, strategy or "(default)",
        )
        return row_id

    def insert_equity(self, timestamp: str, equity: float, strategy: str = "") -> int:
        """Insert an equity-history row. Returns the new row id.

        *strategy* tags the equity point with its originating strategy. An
        empty string (default) represents total/account-level equity for
        backward compatibility with single-strategy callers.
        """
        self._ensure_schema()
        with self._txn() as conn:
            cur = conn.execute(
                "INSERT INTO equity_history (timestamp, equity, strategy) VALUES (?, ?, ?)",
                (timestamp, float(equity), strategy),
            )
            row_id = int(cur.lastrowid)
        logger.info("equity inserted id=%d ts=%s equity=%.2f strategy=%s", row_id, timestamp, equity, strategy or "(default)")
        return row_id

    def upsert_position(self, symbol: str, qty: int, entry_price: float, entry_ts: str, stop: float, target: float) -> None:
        """Insert or update a position row by symbol (UPSERT)."""
        self._ensure_schema()
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, qty, entry_price, entry_ts, stop, target, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "  qty=excluded.qty, entry_price=excluded.entry_price, "
                "  entry_ts=excluded.entry_ts, stop=excluded.stop, "
                "  target=excluded.target, updated_at=datetime('now')",
                (symbol, int(qty), float(entry_price), entry_ts, float(stop), float(target)),
            )
        logger.info(
            "position upserted sym=%s qty=%d entry=%.4f stop=%.2f target=%.2f",
            symbol, qty, entry_price, stop, target,
        )

    def upsert_positions(self, positions: dict[str, dict]) -> None:
        """Bulk upsert the full position set (used by EngineState.save_positions).

        Symbols absent from `positions` are removed so the DB matches the
        in-memory view — mirrors the old "overwrite the JSON file" semantics.
        """
        self._ensure_schema()
        with self._txn() as conn:
            if positions:
                rows = [
                    (
                        sym,
                        int(p.get("qty", 0)),
                        float(p.get("entry_price", 0)),
                        p.get("entry_ts", ""),
                        float(p.get("stop", 0)),
                        float(p.get("target", 0)),
                    )
                    for sym, p in positions.items()
                ]
                conn.executemany(
                    "INSERT INTO positions (symbol, qty, entry_price, entry_ts, stop, target, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
                    "ON CONFLICT(symbol) DO UPDATE SET "
                    "  qty=excluded.qty, entry_price=excluded.entry_price, "
                    "  entry_ts=excluded.entry_ts, stop=excluded.stop, "
                    "  target=excluded.target, updated_at=datetime('now')",
                    rows,
                )
            kept = set(positions.keys())
            if kept:
                placeholders = ",".join("?" for _ in kept)
                conn.execute(f"DELETE FROM positions WHERE symbol NOT IN ({placeholders})", list(kept))
            else:
                conn.execute("DELETE FROM positions;")
        logger.info("positions upserted count=%d", len(positions))

    def upsert_state(
        self, mode: str, strategy: str, params: dict, kill_switch: bool, equity: float, ts: str
    ) -> None:
        """Upsert the singleton engine_state row (id=1)."""
        self._ensure_schema()
        params_json = json.dumps(params) if params else "{}"
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO engine_state (id, mode, strategy, params_json, kill_switch, "
                "  day_start_equity, last_cycle_ts, updated_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  mode=excluded.mode, strategy=excluded.strategy, "
                "  params_json=excluded.params_json, kill_switch=excluded.kill_switch, "
                "  day_start_equity=excluded.day_start_equity, "
                "  last_cycle_ts=excluded.last_cycle_ts, updated_at=datetime('now')",
                (mode, strategy, params_json, int(bool(kill_switch)), float(equity), ts),
            )
        logger.info(
            "engine_state upserted mode=%s strategy=%s kill=%s equity=%.2f ts=%s",
            mode, strategy, kill_switch, equity, ts,
        )

    # ── reads ──────────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, dict]:
        """Return {symbol: {qty, entry_price, entry_ts, stop, target}}."""
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, qty, entry_price, entry_ts, stop, target FROM positions"
            ).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            out[r["symbol"]] = {
                "qty": int(r["qty"]),
                "entry_price": float(r["entry_price"]),
                "entry_ts": r["entry_ts"] or "",
                "stop": float(r["stop"]),
                "target": float(r["target"]),
            }
        return out

    def get_equity_curve(self) -> list[dict]:
        """Return list of {timestamp, equity} ordered ascending."""
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, equity FROM equity_history ORDER BY id ASC"
            ).fetchall()
        return [{"timestamp": r["timestamp"], "equity": float(r["equity"])} for r in rows]

    def get_trades(self, symbol: str | None = None, limit: int = 0) -> list[dict]:
        """Return trades, optionally filtered by symbol. limit<=0 = all."""
        self._ensure_schema()
        with self._connect() as conn:
            if symbol:
                sql = "SELECT id, timestamp, symbol, side, qty, price, reason FROM trades WHERE symbol=? ORDER BY id ASC"
                params: list[Any] = [symbol]
            else:
                sql = "SELECT id, timestamp, symbol, side, qty, price, reason FROM trades ORDER BY id ASC"
                params = []
            if limit and limit > 0:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "side": r["side"],
                "qty": int(r["qty"]),
                "price": float(r["price"]),
                "reason": r["reason"],
            }
            for r in rows
        ]

    def get_signals(self, symbol: str | None = None, limit: int = 0) -> list[dict]:
        """Return signals, optionally filtered by symbol. limit<=0 = all."""
        self._ensure_schema()
        with self._connect() as conn:
            if symbol:
                sql = "SELECT id, timestamp, symbol, signal, reason FROM signals WHERE symbol=? ORDER BY id ASC"
                params: list[Any] = [symbol]
            else:
                sql = "SELECT id, timestamp, symbol, signal, reason FROM signals ORDER BY id ASC"
                params = []
            if limit and limit > 0:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "signal": int(r["signal"]),
                "reason": r["reason"],
            }
            for r in rows
        ]

    def get_state(self) -> dict[str, Any] | None:
        """Return the singleton engine_state row as a dict, or None if absent."""
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT mode, strategy, params_json, kill_switch, day_start_equity, last_cycle_ts "
                "FROM engine_state WHERE id=1"
            ).fetchone()
        if row is None:
            return None
        try:
            params = json.loads(row["params_json"]) if row["params_json"] else {}
        except Exception:
            params = {}
        return {
            "mode": row["mode"],
            "strategy": row["strategy"],
            "params": params,
            "kill_switch": bool(row["kill_switch"]),
            "day_start_equity": float(row["day_start_equity"]) if row["day_start_equity"] is not None else None,
            "last_cycle_ts": row["last_cycle_ts"],
        }

    # ── maintenance ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe all data tables (keeps schema). Used for tests / fresh start."""
        with self._txn() as conn:
            conn.execute("DELETE FROM trades;")
            conn.execute("DELETE FROM positions;")
            conn.execute("DELETE FROM equity_history;")
            conn.execute("DELETE FROM signals;")
            conn.execute("DELETE FROM engine_state;")
            conn.execute("DELETE FROM sqlite_sequence;")  # reset AUTOINCREMENT
        self._initialised = True  # schema still good
        logger.warning("TradeStore reset: all tables cleared db=%s", self._db_path)

    def close(self) -> None:
        """No persistent connection to close; kept for API symmetry."""
        pass


# ── feature-flag helpers ──────────────────────────────────────────────


def sqlite_enabled() -> bool:
    """Decide whether the SQLite store is active.

    Precedence:
      1. env TRADE_STORE=sqlite -> True
      2. env TRADE_STORE=csv    -> False (legacy fallback)
      3. env TRADE_STORE unset  -> auto-detect: True if logs/trade_store.db exists,
         else fall back to CSV (so existing deployments keep working until DB is seeded).
    """
    mode = os.environ.get(_TRADE_STORE_ENV, "").strip().lower()
    if mode == "sqlite":
        return True
    if mode == "csv":
        return False
    # auto-detect
    return DEFAULT_DB_PATH.exists()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
