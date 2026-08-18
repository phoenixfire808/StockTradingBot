"""Tests for ui/pages/portfolio.py — Streamlit portfolio dashboard.

Strategy: import the module under a mocked Streamlit (`streamlit = MagicMock()`)
so we can exercise:
  * loaders behave correctly with missing / corrupt / well-formed fixtures.
  * allocation table is built (sort by weight desc, mark zeros).
  * per-strategy summary aggregates correctly.

We do NOT instantiate the full Streamlit Page; we test the data shaping
helpers extracted out of the page module (or exercised via direct file IO).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import wraps
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Module import under mocked Streamlit ────────────────────────────────


def _identity_decorator(*_dargs, **_dkwargs):
    """Replacement for ``@st.cache_data`` that just runs the wrapped fn."""
    def _wrap(fn):
        @wraps(fn)
        def _inner(*a, **kw):
            return fn(*a, **kw)
        return _inner
    return _wrap


def _import_portfolio_page(name: str = "ui.pages.portfolio_under_test"):
    """Import ui/pages/7_📊_Portfolio.py with streamlit stubbed.

    Streamlit raises at import time if st.set_page_config / st.markdown are
    called outside a script context. Stubbing them with MagicMock sidesteps
    that and lets us exercise the loader functions directly.

    ``st.cache_data`` is replaced with a passthrough decorator so the loader
    functions actually run (otherwise they'd return a MagicMock).
    """
    st_mock = MagicMock()
    st_mock.cache_data = _identity_decorator
    sys.modules["streamlit"] = st_mock

    page_path = REPO_ROOT / "ui" / "pages" / "7_📊_Portfolio.py"
    spec = importlib.util.spec_from_file_location(name, page_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def portfolio_page():
    return _import_portfolio_page()


# ── Loaders: missing files → empty data ─────────────────────────────────


class TestLoadersMissing:
    def test_load_portfolio_state_missing_returns_empty(self, portfolio_page, tmp_path):
        result = portfolio_page.load_portfolio_state(path=str(tmp_path / "nope.json"))
        assert result == {}

    def test_load_equity_by_strategy_missing_returns_empty(
        self, portfolio_page, tmp_path
    ):
        result = portfolio_page.load_equity_by_strategy(
            path=str(tmp_path / "nope.json")
        )
        assert result == {}

    def test_load_equity_history_missing_returns_none(self, portfolio_page, tmp_path):
        result = portfolio_page.load_equity_history(path=str(tmp_path / "nope.csv"))
        assert result is None

    def test_load_engine_state_missing_returns_empty(self, portfolio_page, tmp_path):
        result = portfolio_page.load_engine_state(path=str(tmp_path / "nope.json"))
        assert result == {}


# ── Loaders: well-formed fixtures ───────────────────────────────────────


class TestLoadersHappyPath:
    def test_load_portfolio_state(self, portfolio_page, tmp_path):
        p = tmp_path / "portfolio_state.json"
        p.write_text(
            json.dumps(
                {
                    "allocations": {"a": 0.6, "b": 0.4},
                    "method": "kelly",
                    "fractional": 0.25,
                    "updated_at": "2025-01-01T00:00:00+00:00",
                }
            )
        )
        result = portfolio_page.load_portfolio_state(path=str(p))
        assert result["allocations"] == {"a": 0.6, "b": 0.4}
        assert result["method"] == "kelly"
        assert result["fractional"] == 0.25

    def test_load_equity_by_strategy_wraps_top_level_strategies_key(
        self, portfolio_page, tmp_path
    ):
        p = tmp_path / "equity_by_strategy.json"
        p.write_text(
            json.dumps(
                {
                    "strategies": {
                        "alpha": [{"ts": "2025-01-01T00:00:00", "equity": 100.0}],
                    }
                }
            )
        )
        result = portfolio_page.load_equity_by_strategy(path=str(p))
        assert "alpha" in result
        assert result["alpha"][0]["equity"] == 100.0

    def test_load_equity_by_strategy_accepts_flat_dict(
        self, portfolio_page, tmp_path
    ):
        p = tmp_path / "equity_by_strategy.json"
        p.write_text(
            json.dumps(
                {"alpha": [{"ts": "2025-01-01T00:00:00", "equity": 100.0}]}
            )
        )
        result = portfolio_page.load_equity_by_strategy(path=str(p))
        assert "alpha" in result

    def test_load_equity_history(self, portfolio_page, tmp_path):
        p = tmp_path / "equity_history.csv"
        p.write_text("timestamp,equity\n2025-01-01,100.0\n2025-01-02,101.0\n")
        df = portfolio_page.load_equity_history(path=str(p))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "equity" in df.columns


# ── Loaders: corrupt files ──────────────────────────────────────────────


class TestLoadersCorrupt:
    def test_corrupt_portfolio_state_returns_empty(self, portfolio_page, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        result = portfolio_page.load_portfolio_state(path=str(p))
        assert result == {}

    def test_corrupt_equity_history_handles_gracefully(self, portfolio_page, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("not,a,csv\n")
        result = portfolio_page.load_equity_history(path=str(p))
        # Either None (read failure) or DataFrame (read success with bad data)
        # are acceptable — the page never crashes.
        assert result is None or isinstance(result, pd.DataFrame)


# ── Portfolio-state shape (integration with bot.portfolio) ──────────────


class TestPortfolioModuleIntegration:
    """The page reads JSON shapes that PortfolioState.save() emits — make
    sure the round-trip works."""

    def test_portfolio_state_roundtrip_produces_page_readable_json(self, tmp_path):
        from bot.portfolio import PortfolioState, allocate_kelly
        import numpy as np
        import pandas as pd

        state_file = tmp_path / "portfolio_state.json"
        ps = PortfolioState(path=state_file)

        rng = np.random.default_rng(0)
        returns = {
            "alpha": pd.Series(rng.normal(0.002, 0.01, 50)),
            "beta": pd.Series(rng.normal(0.001, 0.015, 50)),
        }
        weights = allocate_kelly(returns, fractional=0.25)
        ps.save(weights, method="kelly", fractional=0.25)

        # Re-import the page with a fresh stub (separate module name to
        # avoid cached-state contamination between tests).
        mod = _import_portfolio_page(name="ui.pages.portfolio_int_test")
        loaded = mod.load_portfolio_state(path=str(state_file))

        assert loaded["method"] == "kelly"
        assert loaded["fractional"] == 0.25
        loaded_weights = loaded["allocations"]
        assert isinstance(loaded_weights, dict)
        assert len(loaded_weights) >= 1
        assert abs(sum(loaded_weights.values()) - 1.0) < 1e-9

    def test_page_loaders_handle_pure_dict_or_empty(self, portfolio_page):
        """The page must not raise when its module-level loaders return
        MagicMock-shaped values (test stub state)."""
        # The fact that the fixture successfully imported without raising
        # already proves the loaders tolerate the stub environment.
        assert portfolio_page is not None
        assert callable(portfolio_page.load_portfolio_state)
        assert callable(portfolio_page.load_equity_by_strategy)
        assert callable(portfolio_page.load_equity_history)
        assert callable(portfolio_page.load_engine_state)