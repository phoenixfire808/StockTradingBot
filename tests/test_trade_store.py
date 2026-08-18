"""Tests for bot.trade_store — SQLite-backed trade/positions/equity/signals/state store."""

import json
import os
import threading
from pathlib import Path

import pytest

from bot.trade_store import DEFAULT_DB_PATH, TradeStore, sqlite_enabled


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Fresh TradeStore pointing at an isolated temp DB."""
    db = tmp_path / "trade_store.db"
    monkeypatch.setenv("TRADE_STORE", "sqlite")
    return TradeStore(db_path=db)


# ── schema / init ─────────────────────────────────────────────────────


class TestSchema:
    def test_ensure_schema_creates_all_tables(self, store):
        store._ensure_schema()
        with store._connect() as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        for t in ("trades", "positions", "equity_history", "signals", "engine_state"):
            assert t in tables, f"missing table: {t}"

    def test_ensure_schema_idempotent(self, store):
        store._ensure_schema()
        store._ensure_schema()  # second call must not raise
        with store._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert n == 0

    def test_db_file_created(self, store, tmp_path):
        store.insert_trade("2024-01-01T00:00:00", "AAPL", "BUY", 10, 150.0, "test")
        assert (tmp_path / "trade_store.db").exists()


# ── trades ────────────────────────────────────────────────────────────


class TestTrades:
    def test_insert_trade_returns_row_id(self, store):
        rid = store.insert_trade("2024-01-01", "AAPL", "BUY", 10, 150.0, "signal")
        assert isinstance(rid, int) and rid > 0

    def test_get_trades_roundtrip(self, store):
        store.insert_trade("2024-01-01T00", "AAPL", "BUY", 10, 150.0, "signal")
        store.insert_trade("2024-01-02T00", "MSFT", "SELL", 5, 400.0, "exit")
        trades = store.get_trades()
        assert len(trades) == 2
        assert trades[0]["symbol"] == "AAPL"
        assert trades[1]["symbol"] == "MSFT"
        assert trades[0]["side"] == "BUY"
        assert trades[0]["qty"] == 10
        assert abs(trades[0]["price"] - 150.0) < 1e-9

    def test_get_trades_filter_by_symbol(self, store):
        store.insert_trade("2024-01-01", "AAPL", "BUY", 10, 150.0, "s")
        store.insert_trade("2024-01-02", "MSFT", "BUY", 5, 400.0, "s")
        aapl = store.get_trades(symbol="AAPL")
        assert len(aapl) == 1
        assert aapl[0]["symbol"] == "AAPL"

    def test_get_trades_limit(self, store):
        for i in range(5):
            store.insert_trade(f"2024-01-0{i+1}", "AAPL", "BUY", 1, 100.0 + i, "s")
        assert len(store.get_trades(limit=2)) == 2

    def test_side_normalised_uppercase(self, store):
        store.insert_trade("2024-01-01", "AAPL", "buy", 1, 100.0, "s")
        assert store.get_trades()[0]["side"] == "BUY"


# ── signals ────────────────────────────────────────────────────────────


class TestSignals:
    def test_insert_and_get_signals(self, store):
        store.insert_signal("2024-01-01", "AAPL", 1, "entry")
        store.insert_signal("2024-01-02", "AAPL", -1, "exit")
        sigs = store.get_signals()
        assert len(sigs) == 2
        assert sigs[0]["signal"] == 1
        assert sigs[1]["signal"] == -1

    def test_get_signals_filter_symbol(self, store):
        store.insert_signal("2024-01-01", "AAPL", 1, "x")
        store.insert_signal("2024-01-01", "MSFT", 1, "x")
        assert len(store.get_signals(symbol="AAPL")) == 1

    def test_get_signals_limit(self, store):
        for i in range(4):
            store.insert_signal("2024-01-01", "AAPL", 1, "x")
        assert len(store.get_signals(limit=2)) == 2


# ── equity ────────────────────────────────────────────────────────────


class TestEquity:
    def test_insert_and_curve(self, store):
        store.insert_equity("2024-01-01T00:00:00", 100_000)
        store.insert_equity("2024-01-02T00:00:00", 101_500)
        curve = store.get_equity_curve()
        assert len(curve) == 2
        assert abs(curve[0]["equity"] - 100_000) < 1e-9
        assert abs(curve[1]["equity"] - 101_500) < 1e-9
        assert curve[0]["timestamp"] == "2024-01-01T00:00:00"

    def test_curve_order_is_insertion_order(self, store):
        # insert out of chronological order; must come back in id order
        store.insert_equity("2024-01-03", 3)
        store.insert_equity("2024-01-01", 1)
        store.insert_equity("2024-01-02", 2)
        curve = store.get_equity_curve()
        assert [c["equity"] for c in curve] == [3, 1, 2]


# ── positions ─────────────────────────────────────────────────────────


class TestPositions:
    def _pos(self, qty=10, entry=150.0):
        return {
            "qty": qty,
            "entry_price": entry,
            "entry_ts": "2024-01-01",
            "stop": 140.0,
            "target": 165.0,
        }

    def test_upsert_single_position(self, store):
        store.upsert_position("AAPL", 10, 150.0, "2024-01-01", 140.0, 165.0)
        pos = store.get_positions()
        assert "AAPL" in pos
        assert pos["AAPL"]["qty"] == 10
        assert abs(pos["AAPL"]["entry_price"] - 150.0) < 1e-9
        assert abs(pos["AAPL"]["stop"] - 140.0) < 1e-9

    def test_upsert_updates_existing(self, store):
        store.upsert_position("AAPL", 10, 150.0, "2024-01-01", 140.0, 165.0)
        store.upsert_position("AAPL", 20, 155.0, "2024-01-02", 142.0, 170.0)
        pos = store.get_positions()["AAPL"]
        assert pos["qty"] == 20
        assert abs(pos["entry_price"] - 155.0) < 1e-9

    def test_upsert_positions_bulk(self, store):
        positions = {
            "AAPL": self._pos(10, 150.0),
            "MSFT": self._pos(5, 400.0),
        }
        store.upsert_positions(positions)
        pos = store.get_positions()
        assert len(pos) == 2
        assert pos["AAPL"]["qty"] == 10
        assert pos["MSFT"]["qty"] == 5

    def test_upsert_positions_removes_absent(self, store):
        store.upsert_positions({"AAPL": self._pos(10, 150), "MSFT": self._pos(5, 400)})
        # second call drops MSFT
        store.upsert_positions({"AAPL": self._pos(12, 152)})
        pos = store.get_positions()
        assert len(pos) == 1
        assert "MSFT" not in pos
        assert pos["AAPL"]["qty"] == 12

    def test_upsert_positions_empty_clears_all(self, store):
        store.upsert_positions({"AAPL": self._pos(10, 150)})
        store.upsert_positions({})
        assert store.get_positions() == {}


# ── engine state ──────────────────────────────────────────────────────


class TestEngineState:
    def test_upsert_and_get_state(self, store):
        store.upsert_state("live", "ema_cross_rsi", {"fast": 9}, False, 100_000, "2024-01-01")
        st = store.get_state()
        assert st is not None
        assert st["mode"] == "live"
        assert st["strategy"] == "ema_cross_rsi"
        assert st["params"] == {"fast": 9}
        assert st["kill_switch"] is False
        assert abs(st["day_start_equity"] - 100_000) < 1e-9

    def test_upsert_state_overwrites_singleton(self, store):
        store.upsert_state("live", "s1", {}, False, 100_000, "t1")
        store.upsert_state("paper", "s2", {"x": 1}, True, 99_000, "t2")
        st = store.get_state()
        assert st["mode"] == "paper"
        assert st["strategy"] == "s2"
        assert st["kill_switch"] is True
        assert st["params"] == {"x": 1}

    def test_get_state_none_when_empty(self, store):
        assert store.get_state() is None


# ── reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_tables(self, store):
        store.insert_trade("2024-01-01", "AAPL", "BUY", 1, 100.0, "s")
        store.insert_signal("2024-01-01", "AAPL", 1, "s")
        store.insert_equity("2024-01-01", 100_000)
        store.upsert_position("AAPL", 1, 100.0, "2024-01-01", 90.0, 110.0)
        store.upsert_state("live", "s", {}, False, 100_000, "t")

        store.reset()

        assert store.get_trades() == []
        assert store.get_signals() == []
        assert store.get_equity_curve() == []
        assert store.get_positions() == {}
        assert store.get_state() is None

    def test_reset_restarts_autoincrement(self, store):
        store.insert_trade("2024-01-01", "AAPL", "BUY", 1, 100.0, "s")
        first_id = store.get_trades()[0]["id"]
        store.reset()
        store.insert_trade("2024-01-02", "AAPL", "BUY", 1, 100.0, "s")
        new_id = store.get_trades()[0]["id"]
        assert new_id == 1, f"autoincrement should reset, got {new_id} (first was {first_id})"


# ── feature flag / sqlite_enabled ─────────────────────────────────────


class TestFeatureFlag:
    def test_env_sqlite_forces_on(self, monkeypatch):
        monkeypatch.setenv("TRADE_STORE", "sqlite")
        assert sqlite_enabled() is True

    def test_env_csv_forces_off(self, monkeypatch):
        monkeypatch.setenv("TRADE_STORE", "csv")
        assert sqlite_enabled() is False

    def test_autodetect_when_db_exists(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TRADE_STORE", raising=False)
        # DEFAULT_DB_PATH points at logs/trade_store.db relative to cwd; we can't
        # easily redirect it, so test the logic by creating the file then cleaning.
        original = DEFAULT_DB_PATH
        db = Path("logs/trade_store.db")
        existed = db.exists()
        try:
            db.parent.mkdir(parents=True, exist_ok=True)
            db.touch()
            assert sqlite_enabled() is True
        finally:
            if not existed and db.exists():
                db.unlink()

    def test_autodetect_off_when_no_db(self, monkeypatch):
        monkeypatch.delenv("TRADE_STORE", raising=False)
        db = Path("logs/trade_store.db")
        existed = db.exists()
        if existed:
            db.unlink()
        try:
            assert sqlite_enabled() is False
        finally:
            if existed:
                db.parent.mkdir(parents=True, exist_ok=True)
                db.touch()


# ── context-manager / transaction safety ─────────────────────────────


class TestTransactions:
    def test_failed_insert_rolls_back(self, store):
        # insert a good row, then trigger a constraint violation on the next insert
        store.insert_trade("2024-01-01", "AAPL", "BUY", 1, 100.0, "s")
        with pytest.raises(Exception):
            # NOT NULL violation on side
            store.insert_trade("2024-01-02", "AAPL", None, 1, 100.0, "s")  # type: ignore[arg-type]
        trades = store.get_trades()
        assert len(trades) == 1, "failed insert must not leave a partial row"


# ── concurrency: WAL + per-call connections ────────────────────────────


class TestConcurrency:
    def test_parallel_inserts_all_persist(self, store):
        n_threads = 8
        per_thread = 20
        errors: list[Exception] = []

        def writer(tid: int) -> None:
            try:
                for i in range(per_thread):
                    store.insert_trade(
                        f"2024-01-{(tid * per_thread + i) % 28 + 1:02d}",
                        f"S{tid}",
                        "BUY",
                        1,
                        100.0 + i,
                        "threaded",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"errors during concurrent inserts: {errors}"
        trades = store.get_trades()
        assert len(trades) == n_threads * per_thread


# ── EngineState facade integration (SQLite path) ─────────────────────


class TestEngineStateFacade:
    def test_facade_delegates_to_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADE_STORE", "sqlite")
        # isolate DB so we don't touch the real logs/trade_store.db
        from bot.engine import EngineState

        es = EngineState()
        # swap the store's DB path to a temp file
        assert es._use_sqlite is True
        assert es._store is not None
        es._store._db_path = tmp_path / "facade.db"
        es._store._initialised = False

        es.append_trade("2024-01-01", "AAPL", "BUY", 10, 150.0, "signal")
        es.append_signal("2024-01-01", "AAPL", 1, "entry")
        es.append_equity(100_000, "2024-01-01")
        es.write_state("live", "ema", {"fast": 9}, False, 100_000, "2024-01-01")
        es.save_positions({"AAPL": {"qty": 10, "entry_price": 150.0, "entry_ts": "t", "stop": 140, "target": 165}})

        assert len(es._store.get_trades()) == 1
        assert len(es._store.get_signals()) == 1
        assert len(es._store.get_equity_curve()) == 1
        assert es._store.get_state()["mode"] == "live"
        assert es.read_positions()["AAPL"]["qty"] == 10

    def test_facade_falls_back_to_csv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADE_STORE", "csv")
        monkeypatch.chdir(tmp_path)
        from bot.engine import EngineState

        es = EngineState()
        assert es._use_sqlite is False
        assert es._store is None

        es.append_trade("2024-01-01", "AAPL", "BUY", 10, 150.0, "signal")
        es.append_signal("2024-01-01", "AAPL", 1, "entry")
        es.append_equity(100_000, "2024-01-01")
        es.write_state("live", "ema", {}, False, 100_000, "2024-01-01")
        es.save_positions({"AAPL": {"qty": 10, "entry_price": 150.0, "entry_ts": "t", "stop": 140, "target": 165}})

        # CSV/JSON files written
        assert (tmp_path / "logs" / "trades.csv").exists()
        assert (tmp_path / "logs" / "signals.csv").exists()
        assert (tmp_path / "logs" / "equity_history.csv").exists()
        assert (tmp_path / "logs" / "engine_state.json").exists()
        assert (tmp_path / "logs" / "positions_state.json").exists()
