"""
tests/test_position_manager.py — PositionManager risk controls tests
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch

import pytest

from position_manager import (
    PositionManager,
    Position,
    TrailingStop,
    simulate_atr_stoploss,
    _state_path,
    _load_risk_state,
    _save_risk_state,
    _STATE_DIR,
)


# ── Kelly fraction ────────────────────────────────────────────────────────

class TestKellyWinRate:
    def test_win_rate_zero(self):
        pm = PositionManager(portfolio_value=100_000)
        size = pm.calculate_kelly_size(win_rate=0.0, profit_loss_ratio=1.5, capital=100_000, current_price=100.0)
        assert size == 0

    def test_win_rate_one(self):
        # Kelly with 100% win rate should return max position
        pm = PositionManager(portfolio_value=100_000)
        size = pm.calculate_kelly_size(win_rate=1.0, profit_loss_ratio=2.0, capital=100_000, current_price=100.0)
        assert size > 0

    def test_profit_loss_ratio_zero(self):
        pm = PositionManager(portfolio_value=100_000)
        size = pm.calculate_kelly_size(win_rate=0.5, profit_loss_ratio=0.0, capital=100_000, current_price=100.0)
        assert size == 0

    def test_negative_fraction_clamps_to_zero(self):
        # win_rate too low → Kelly formula gives negative → clamped to 0
        pm = PositionManager(portfolio_value=100_000)
        size = pm.calculate_kelly_size(win_rate=0.2, profit_loss_ratio=1.0, capital=100_000, current_price=100.0)
        assert size == 0


# ── Circuit breaker ─────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_single_day_loss_triggers(self):
        pm = PositionManager(
            portfolio_value=100_000,
            daily_loss_limit=0.05,
            max_consecutive_loss_days=3,
        )
        result = pm.check_circuit_breaker(-0.06, trade_date="2099-01-01")
        assert result["tripped"] is True
        assert "单日亏损" in result["reason"]

    def test_consecutive_losses_triggers(self):
        pm = PositionManager(
            portfolio_value=100_000,
            daily_loss_limit=0.05,
            max_consecutive_loss_days=3,
        )
        # Three consecutive loss days should trip
        pm.check_circuit_breaker(-0.03, trade_date="2099-01-01")
        pm.check_circuit_breaker(-0.03, trade_date="2099-01-02")
        result = pm.check_circuit_breaker(-0.03, trade_date="2099-01-03")
        assert result["tripped"] is True
        assert "连续亏损" in result["reason"]

    def test_same_day_debounced(self):
        pm = PositionManager(
            portfolio_value=100_000,
            daily_loss_limit=0.05,
            max_consecutive_loss_days=3,
        )
        r1 = pm.check_circuit_breaker(-0.06, trade_date="2099-01-01")
        r2 = pm.check_circuit_breaker(-0.01, trade_date="2099-01-01")
        # Same day — should not re-trigger, but return debounced result
        assert r2["consecutive_loss_days"] == r1["consecutive_loss_days"]


# ── ATR stop-loss ────────────────────────────────────────────────────────

class TestAtrStopLoss:
    def test_atr_stop_loss_trigger(self, atr_plunge_ohlcv):
        sig = pd.Series(1, index=atr_plunge_ohlcv.index)
        modified = simulate_atr_stoploss(
            atr_plunge_ohlcv, sig,
            atr_period=5,
            atr_multiplier=2.0,
        )
        n_stops = (modified == 0).sum()
        assert n_stops > 0, "ATR stop-loss should trigger on plunging data"

    def test_simulate_atr_stoploss_cooldown_bars(self, atr_plunge_ohlcv):
        sig = pd.Series(1, index=atr_plunge_ohlcv.index)
        modified = simulate_atr_stoploss(
            atr_plunge_ohlcv, sig,
            atr_period=5,
            atr_multiplier=2.0,
            cooldown_bars=5,
        )
        # After a stop, 5 cooldown bars should be 0
        first_stop_idx = atr_plunge_ohlcv.index.get_loc(modified[modified == 0].index[0])
        for i in range(first_stop_idx, min(first_stop_idx + 5, len(modified))):
            assert modified.iloc[i] == 0


# ── apply_risk_controls ──────────────────────────────────────────────────

class TestApplyRiskControls:
    def test_stop_loss_action(self):
        pm = PositionManager(
            portfolio_value=100_000,
            risk_config={
                "use_atr_stop": True,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "trailing_stop": True,
            },
        )
        pm.set_position(shares=200, avg_cost=60.0, current_price=65.0)
        # Price 64.5 with peak 68 and atr 1.5 → stop triggered
        result = pm.apply_risk_controls(
            signal=1,
            price=64.5,
            atr=1.5,
            entry_price=60.0,
            peak_price=68.0,
            today_pnl_pct=-0.01,
        )
        assert result["action"] == "止损卖出"

    def test_no_position_bull_signal(self):
        pm = PositionManager(portfolio_value=100_000)
        pm.set_position(shares=0, avg_cost=0, current_price=65.0)
        result = pm.apply_risk_controls(
            signal=1,
            price=65.0,
            atr=1.0,
            entry_price=65.0,
            peak_price=65.0,
            today_pnl_pct=0.0,
        )
        assert result["action"] == "买入"


# ── validate_position_size ───────────────────────────────────────────────

class TestValidatePositionSize:
    def test_caps_at_max_pct(self):
        pm = PositionManager(portfolio_value=100_000, risk_config={"max_position_pct": 0.25})
        # 100_000 * 0.25 / 50 = 500 shares max
        result = pm.validate_position_size(1000, 50.0, 100_000)
        assert result == 500


# ── BUG-6: per-ticker state isolation ────────────────────────────────────

class TestPerTickerStateIsolation:
    """Risk state files are isolated per-ticker; concurrent writes don't cross."""

    def test_different_tickers_use_different_files(self):
        path_a = _state_path("0700.HK")
        path_b = _state_path("0005.HK")
        assert path_a != path_b
        assert "0700_HK" in path_a
        assert "0005_HK" in path_b

    def test_none_ticker_uses_global_file(self):
        path = _state_path(None)
        assert "risk_state.json" in path
        assert "0700" not in path

    def test_save_and_load_roundtrip_per_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr("position_manager._STATE_DIR", str(tmp_path))
        # Patch _state_path to use tmp_path
        import position_manager as pm_mod
        orig_dir = pm_mod._STATE_DIR
        pm_mod._STATE_DIR = str(tmp_path)
        try:
            state_a = {"consecutive_loss_days": 2, "last_trade_date": "2026-01-01", "trailing_peak": 100.0}
            state_b = {"consecutive_loss_days": 5, "last_trade_date": "2026-01-02", "trailing_peak": 200.0}
            _save_risk_state(state_a, ticker="0700.HK")
            _save_risk_state(state_b, ticker="0005.HK")
            loaded_a = _load_risk_state(ticker="0700.HK")
            loaded_b = _load_risk_state(ticker="0005.HK")
            assert loaded_a["consecutive_loss_days"] == 2
            assert loaded_b["consecutive_loss_days"] == 5
            assert loaded_a["trailing_peak"] == 100.0
            assert loaded_b["trailing_peak"] == 200.0
        finally:
            pm_mod._STATE_DIR = orig_dir

    def test_circuit_breaker_state_isolated_between_tickers(self, tmp_path, monkeypatch):
        import position_manager as pm_mod
        orig_dir = pm_mod._STATE_DIR
        pm_mod._STATE_DIR = str(tmp_path)
        try:
            pm_a = PositionManager(portfolio_value=100_000, max_consecutive_loss_days=3, ticker="0700.HK")
            pm_b = PositionManager(portfolio_value=100_000, max_consecutive_loss_days=3, ticker="0005.HK")
            # Drive 0700 to 2 loss days
            pm_a.check_circuit_breaker(-0.02, trade_date="2026-01-01")
            pm_a.check_circuit_breaker(-0.02, trade_date="2026-01-02")
            # 0005 still at 0 loss days
            result_b = pm_b.check_circuit_breaker(-0.01, trade_date="2026-01-01")
            assert result_b["consecutive_loss_days"] == 1
            # 0700 should be at 2
            result_a = pm_a.check_circuit_breaker(-0.04, trade_date="2026-01-03")
            assert result_a["consecutive_loss_days"] == 3
        finally:
            pm_mod._STATE_DIR = orig_dir

    def test_atomic_write_leaves_no_tmp_files_on_success(self, tmp_path, monkeypatch):
        import position_manager as pm_mod
        pm_mod._STATE_DIR = str(tmp_path)
        try:
            _save_risk_state({"consecutive_loss_days": 1, "last_trade_date": "", "trailing_peak": None},
                             ticker="0700.HK")
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert tmp_files == [], f"Unexpected tmp files: {tmp_files}"
            json_files = list(tmp_path.glob("risk_state_*.json"))
            assert len(json_files) == 1
        finally:
            pm_mod._STATE_DIR = os.path.join(os.path.dirname(pm_mod.__file__), "data", "logs")

    def test_load_returns_default_on_corrupt_json(self, tmp_path, monkeypatch):
        import position_manager as pm_mod
        orig_dir = pm_mod._STATE_DIR
        pm_mod._STATE_DIR = str(tmp_path)
        try:
            bad_path = tmp_path / "risk_state_0700_HK.json"
            bad_path.write_text("{corrupt json!!!")
            state = _load_risk_state(ticker="0700.HK")
            assert state["consecutive_loss_days"] == 0
        finally:
            pm_mod._STATE_DIR = orig_dir
