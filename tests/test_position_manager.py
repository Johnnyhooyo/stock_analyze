"""
tests/test_position_manager.py — PositionManager risk controls tests
"""
import numpy as np
import pandas as pd
from datetime import datetime

import pytest

from position_manager import (
    PositionManager,
    Position,
    TrailingStop,
    simulate_atr_stoploss,
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
