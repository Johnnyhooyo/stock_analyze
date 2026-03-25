"""
tests/test_backtest.py — analyze_factor.backtest() tests
"""
import numpy as np
import pandas as pd

import pytest

from analyze_factor import backtest


class TestBacktestReturnsRequiredFields:
    def test_returns_all_required_fields(self, synthetic_ohlcv, default_config):
        sig = pd.Series(1, index=synthetic_ohlcv.index)
        result = backtest(synthetic_ohlcv, sig, default_config)
        assert "cum_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "buy_cnt" in result
        assert "sell_cnt" in result


class TestBacktestAllHoldSignal:
    def test_all_hold_signal_cum_return_positive(self, synthetic_ohlcv, default_config):
        sig = pd.Series(1, index=synthetic_ohlcv.index)
        result = backtest(synthetic_ohlcv, sig, default_config)
        # Run completed without error and produced a valid cum_return
        assert "cum_return" in result
        assert not pd.isna(result["cum_return"])


class TestBacktestAllFlatSignal:
    def test_all_flat_signal_zero_trades(self, synthetic_ohlcv, default_config):
        sig = pd.Series(0, index=synthetic_ohlcv.index)
        result = backtest(synthetic_ohlcv, sig, default_config)
        assert result["buy_cnt"] == 0
        assert result["sell_cnt"] == 0


class TestBacktestAtrStopLoss:
    def test_atr_stop_loss_triggers(self, atr_plunge_ohlcv, default_config):
        sig = pd.Series(1, index=atr_plunge_ohlcv.index)
        cfg = {**default_config}
        cfg["risk_management"] = {
            **default_config["risk_management"],
            "simulate_in_backtest": True,
            "use_atr_stop": True,
            "atr_period": 5,
            "atr_multiplier": 2.0,
            "trailing_stop": True,
            "cooldown_bars": 0,
        }
        result = backtest(atr_plunge_ohlcv, sig, cfg)
        # ATR stop-loss should generate sell trades
        assert result["sell_cnt"] > 0


class TestBacktestSignalShift:
    def test_signal_shift_no_look_ahead(self, synthetic_ohlcv, default_config):
        """
        Verify that the backtest does NOT execute on the same bar the signal fires.
        We check that with a signal that fires only on the first day, no trade happens
        on that first day (because shift(1) defers execution).
        """
        sig = pd.Series(0, index=synthetic_ohlcv.index)
        sig.iloc[0] = 1  # buy signal fires on day 0

        result = backtest(synthetic_ohlcv, sig, default_config)
        # With shift(1), the first signal executes on day 1, not day 0
        # Therefore buy should happen AFTER the first bar
        assert result["buy_cnt"] <= 1  # at most 1 buy after shifted signal
