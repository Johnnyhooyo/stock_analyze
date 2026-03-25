"""
tests/test_strategies.py — Strategy run() + predict() regression tests
"""
import importlib
import numpy as np
import pandas as pd
import pytest

# Rule-based strategies (single-stock training)
RULE_STRATEGIES = [
    ("atr_breakout", "strategies.atr_breakout"),
    ("bollinger_breakout", "strategies.bollinger_breakout"),
    ("bollinger_rsi_trend", "strategies.bollinger_rsi_trend"),
    ("kdj_obv", "strategies.kdj_obv"),
    ("kdj_pvt", "strategies.kdj_pvt"),
    ("ma_crossover", "strategies.ma_crossover"),
    ("macd_rsi_combo", "strategies.macd_rsi_combo"),
    ("macd_rsi_trend", "strategies.macd_rsi_trend"),
    ("rsi_divergence", "strategies.rsi_divergence"),
    ("rsi_drawdown_0225", "strategies.rsi_drawdown_0225"),
    ("rsi_obv", "strategies.rsi_obv"),
    ("rsi_pvt", "strategies.rsi_pvt"),
    ("rsi_reversion", "strategies.rsi_reversion"),
    ("stochastic_oscillator", "strategies.stochastic_oscillator"),
    ("volume_price_trend", "strategies.volume_price_trend"),
    ("vwap_momentum", "strategies.vwap_momentum"),
]


class TestRuleStrategiesSignalValues:
    @pytest.mark.parametrize("name,module_path", RULE_STRATEGIES)
    def test_rule_signal_values_strictly_01(self, synthetic_ohlcv, name, module_path):
        mod = importlib.import_module(module_path)
        sig, model, meta = mod.run(synthetic_ohlcv, {})
        # Signal must be 0 or 1 only
        unique = sig.dropna().unique()
        assert set(unique).issubset({0, 1}), f"{name}: unexpected values {unique}"

    @pytest.mark.parametrize("name,module_path", RULE_STRATEGIES)
    def test_rule_predict_interface(self, synthetic_ohlcv, name, module_path):
        mod = importlib.import_module(module_path)
        sig_run, _, meta = mod.run(synthetic_ohlcv, {})
        # Rule strategies' predict() re-runs run() internally
        sig_pred = mod.predict(None, synthetic_ohlcv, {}, meta)
        assert isinstance(sig_pred, pd.Series)
        assert set(sig_pred.dropna().unique()).issubset({0, 1})


class TestRuleStrategiesWithMinimalConfig:
    def test_all_rule_strategies_run_with_empty_config(self, synthetic_ohlcv):
        """Every rule strategy must accept an empty config dict without crashing."""
        for name, module_path in RULE_STRATEGIES:
            mod = importlib.import_module(module_path)
            try:
                sig, model, meta = mod.run(synthetic_ohlcv, {})
                assert isinstance(sig, pd.Series)
            except Exception as e:
                pytest.fail(f"{name} crashed with empty config: {e}")


class TestMlStrategiesSignalValues:
    def test_xgboost_signal_values(self, synthetic_ohlcv):
        try:
            from strategies.xgboost_enhanced import run as xgb_run
        except ImportError:
            pytest.skip("xgboost not available")
        cfg = {
            "test_days": 5,
            "label_period": 1,
            "xgb_n_estimators": 30,
            "xgb_max_depth": 3,
            "xgb_learning_rate": 0.1,
        }
        sig, model, meta = xgb_run(synthetic_ohlcv, cfg)
        unique = sig.dropna().unique()
        assert set(unique).issubset({0, 1}), f"xgboost unexpected values {unique}"

    def test_lightgbm_signal_values(self, synthetic_ohlcv):
        try:
            from strategies.lightgbm_enhanced import run as lgbm_run
        except ImportError:
            pytest.skip("lightgbm not available")
        cfg = {
            "test_days": 5,
            "label_period": 1,
            "lgbm_n_estimators": 30,
            "lgbm_max_depth": 3,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 15,
        }
        sig, model, meta = lgbm_run(synthetic_ohlcv, cfg)
        unique = sig.dropna().unique()
        assert set(unique).issubset({0, 1}), f"lightgbm unexpected values {unique}"
