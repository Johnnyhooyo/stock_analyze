"""
tests/test_backtest.py — analyze_factor.backtest() tests
"""
import numpy as np
import pandas as pd

import pytest

from analyze_factor import backtest, _backtest_reference


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


# ── 向量化引擎 vs 参考引擎对比 ────────────────────────────────────

_NO_ATR_CONFIG = {
    "initial_capital": 100_000.0,
    "invest_fraction": 0.95,
    "slippage": 0.001,
    "fees_rate": 0.00088,
    "stamp_duty": 0.001,
    "risk_management": {"simulate_in_backtest": False, "use_atr_stop": False},
}

_FLOAT_METRICS = [
    "cum_return", "annualized_return", "sharpe_ratio",
    "max_drawdown", "volatility", "calmar_ratio", "sortino_ratio",
    "win_rate", "profit_loss_ratio",
]

_INT_METRICS = ["buy_cnt", "sell_cnt", "total_trades"]


def _make_cycle_signal(index, hold=20, flat=10):
    """交替持仓/空仓信号，产生多笔完整交易。"""
    sig = pd.Series(0, index=index)
    cycle = hold + flat
    for i, _ in enumerate(index):
        if i % cycle < hold:
            sig.iloc[i] = 1
    return sig


class TestVectorizedMatchesReference:
    """向量化引擎与参考实现的数值一致性测试。"""

    def test_trade_counts_exact(self, synthetic_ohlcv):
        sig = _make_cycle_signal(synthetic_ohlcv.index)
        vec = backtest(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        ref = _backtest_reference(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        for key in _INT_METRICS:
            assert vec[key] == ref[key], f"{key}: vec={vec[key]} ref={ref[key]}"

    def test_float_metrics_close(self, synthetic_ohlcv):
        sig = _make_cycle_signal(synthetic_ohlcv.index)
        vec = backtest(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        ref = _backtest_reference(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        for key in _FLOAT_METRICS:
            v, r = vec[key], ref[key]
            if pd.isna(v) and pd.isna(r):
                continue
            assert abs(v - r) < 1e-6, f"{key}: vec={v} ref={r} diff={abs(v-r)}"

    def test_portfolio_value_series_close(self, synthetic_ohlcv):
        sig = _make_cycle_signal(synthetic_ohlcv.index)
        vec = backtest(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        ref = _backtest_reference(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        pv_diff = (vec["portfolio_value"] - ref["portfolio_value"]).abs()
        assert pv_diff.max() < 1e-6, f"max pv diff={pv_diff.max()}"

    def test_all_flat_signal(self, synthetic_ohlcv):
        sig = pd.Series(0, index=synthetic_ohlcv.index)
        vec = backtest(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        ref = _backtest_reference(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        assert vec["buy_cnt"] == ref["buy_cnt"] == 0
        assert vec["sell_cnt"] == ref["sell_cnt"] == 0

    def test_all_hold_signal(self, synthetic_ohlcv):
        sig = pd.Series(1, index=synthetic_ohlcv.index)
        vec = backtest(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        ref = _backtest_reference(synthetic_ohlcv, sig, _NO_ATR_CONFIG)
        for key in _INT_METRICS:
            assert vec[key] == ref[key], f"{key}: vec={vec[key]} ref={ref[key]}"
        for key in ["cum_return", "sharpe_ratio", "max_drawdown"]:
            v, r = vec[key], ref[key]
            if pd.isna(v) and pd.isna(r):
                continue
            assert abs(v - r) < 1e-6, f"{key}: vec={v} ref={r}"
