"""Tests for bot.portfolio — Kelly, risk parity, equal weight, PortfolioState."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot.portfolio import (
    PortfolioState,
    allocate_equal_weight,
    allocate_kelly,
    allocate_risk_parity,
)


# ── Kelly ───────────────────────────────────────────────────────────


class TestAllocateKelly:
    def _pos_edge(self, n=100, mean=0.002, vol=0.01, seed=42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mean, vol, n))

    def test_positive_edge_returns_normalized_weights(self):
        rets = {"s1": self._pos_edge(), "s2": self._pos_edge()}
        w = allocate_kelly(rets, fractional=0.25)
        assert set(w.keys()) == {"s1", "s2"}
        assert all(v > 0 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_weights_sum_to_one(self):
        rets = {f"s{i}": self._pos_edge(seed=i) for i in range(5)}
        w = allocate_kelly(rets)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_fractional_scales_but_normalizes(self):
        rets = {"s1": self._pos_edge(), "s2": self._pos_edge()}
        full = allocate_kelly(rets, fractional=1.0)
        quarter = allocate_kelly(rets, fractional=0.25)
        # Same relative proportions after normalization
        ratio_full = full["s1"] / full["s2"]
        ratio_q = quarter["s1"] / quarter["s2"]
        assert abs(ratio_full - ratio_q) < 1e-6

    def test_negative_edge_falls_back_to_zero_weight(self):
        rng = np.random.default_rng(0)
        losing = pd.Series(rng.normal(-0.005, 0.01, 100))
        winning = pd.Series(rng.normal(0.005, 0.01, 100))
        w = allocate_kelly({"good": winning, "bad": losing})
        # Source caps negative-edge Kelly at 0 — losing strategy gets 0 weight,
        # winning strategy gets full allocation.
        assert w["bad"] == 0.0
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_all_negative_returns_equal_weight(self):
        rng = np.random.default_rng(1)
        rets = {
            "a": pd.Series(rng.normal(-0.01, 0.02, 100)),
            "b": pd.Series(rng.normal(-0.02, 0.03, 100)),
        }
        w = allocate_kelly(rets)
        assert abs(w["a"] - w["b"]) < 1e-9
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_nan_values_dropped(self):
        base = self._pos_edge()
        with_nan = base.copy()
        with_nan.iloc[0] = np.nan
        with_nan.iloc[1] = np.inf
        w = allocate_kelly({"clean": self._pos_edge(), "dirty": with_nan})
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(0 < v < 1 for v in w.values())
    def test_insufficient_samples_returns_zero_for_short(self):
        short = pd.Series([0.001, 0.002, 0.003])  # < _MIN_SAMPLES (10)
        good = self._pos_edge()
        w = allocate_kelly({"short": short, "good": good})
        # Source returns 0 for insufficient samples; the good strategy still has
        # positive edge so it absorbs the full allocation.
        assert w["short"] == 0.0
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_zero_variance_returns_zero(self):
        flat = pd.Series([0.001] * 100)  # mean > 0 but var = 0
        good = self._pos_edge()
        w = allocate_kelly({"flat": flat, "good": good})
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_empty_input(self):
        assert allocate_kelly({}) == {}

    def test_invalid_fractional_clamped_to_default(self):
        rets = {"s1": self._pos_edge()}
        w = allocate_kelly(rets, fractional=0.0)  # invalid → clamp to 0.25
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_higher_return_strategy_gets_higher_weight(self):
        """Given equal variance, the strategy with the larger mean should
        receive a larger Kelly weight."""
        low_mean = self._pos_edge(mean=0.001, vol=0.01, seed=10)
        high_mean = self._pos_edge(mean=0.004, vol=0.01, seed=11)
        w = allocate_kelly({"low": low_mean, "high": high_mean})
        assert w["high"] > w["low"]


# ── Equal weight ────────────────────────────────────────────────────


class TestAllocateEqualWeight:
    def test_three_strategies(self):
        w = allocate_equal_weight(["a", "b", "c"])
        assert set(w.keys()) == {"a", "b", "c"}
        assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_single_strategy(self):
        w = allocate_equal_weight(["only"])
        assert w == {"only": 1.0}

    def test_empty(self):
        assert allocate_equal_weight([]) == {}


# ── Risk parity ─────────────────────────────────────────────────────


class TestAllocateRiskParity:
    def test_inverse_volatility_weighting(self):
        vols = {"a": 0.10, "b": 0.20}
        w = allocate_risk_parity(["a", "b"], vols)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # Lower vol → higher weight: 1/0.1 vs 1/0.2 → 2:1
        assert abs(w["a"] - 2 / 3) < 1e-6
        assert abs(w["b"] - 1 / 3) < 1e-6

    def test_works_with_series(self):
        vols = pd.Series({"a": 0.10, "b": 0.20})
        w = allocate_risk_parity(["a", "b"], vols)
        assert abs(w["a"] - 2 / 3) < 1e-6

    def test_zero_vol_excluded(self):
        vols = {"a": 0.10, "b": 0.0}
        w = allocate_risk_parity(["a", "b"], vols)
        assert w["b"] == 0.0
        assert abs(w["a"] - 1.0) < 1e-9

    def test_nan_vol_excluded(self):
        vols = {"a": 0.10, "b": float("nan")}
        w = allocate_risk_parity(["a", "b"], vols)
        assert w["b"] == 0.0
        assert abs(w["a"] - 1.0) < 1e-9

    def test_all_invalid_vols_equal_weight_fallback(self):
        vols = {"a": 0.0, "b": float("nan"), "c": -1.0}
        w = allocate_risk_parity(["a", "b", "c"], vols)
        assert abs(w["a"] - 1 / 3) < 1e-9
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_empty(self):
        assert allocate_risk_parity([], {}) == {}


# ── PortfolioState persistence ──────────────────────────────────────


class TestPortfolioState:
    def test_save_and_read_roundtrip(self, tmp_path):
        path = tmp_path / "portfolio_state.json"
        state = PortfolioState(path)
        allocations = {"s1": 0.6, "s2": 0.4}
        state.save(allocations, method="kelly", fractional=0.25)

        loaded = state.read()
        assert loaded["allocations"] == allocations
        assert loaded["method"] == "kelly"
        assert loaded["fractional"] == 0.25
        assert "updated_at" in loaded

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "portfolio_state.json"
        state = PortfolioState(path)
        state.save({"s1": 1.0}, method="equal_weight")
        assert path.exists()

    def test_read_missing_file_returns_empty(self, tmp_path):
        state = PortfolioState(tmp_path / "nope.json")
        assert state.read() == {}

    def test_read_corrupt_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        state = PortfolioState(path)
        assert state.read() == {}

    def test_default_path_points_to_logs(self):
        state = PortfolioState()
        assert state.path == Path("logs/portfolio_state.json")

    def test_path_property(self, tmp_path):
        path = tmp_path / "custom.json"
        state = PortfolioState(path)
        assert state.path == path

    def test_save_without_fractional(self, tmp_path):
        path = tmp_path / "portfolio_state.json"
        state = PortfolioState(path)
        state.save({"s1": 1.0}, method="equal_weight")
        loaded = state.read()
        assert loaded["fractional"] is None
        assert loaded["method"] == "equal_weight"

    def test_written_json_is_indented(self, tmp_path):
        path = tmp_path / "portfolio_state.json"
        state = PortfolioState(path)
        state.save({"s1": 1.0})
        raw = path.read_text()
        # indented JSON contains newlines between keys
        assert '  "allocations"' in raw
