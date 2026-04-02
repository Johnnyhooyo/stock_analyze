"""
tests/test_signal_aggregator.py — SignalAggregator voting logic tests
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from engine.signal_aggregator import SignalAggregator, AggregatedSignal


class TestSignalAggregatorEmptyFactorsDir:
    def test_empty_factors_dir_returns_zero_confidence(self, synthetic_ohlcv, tmp_path):
        """When no factors exist, aggregator should return default zero-confidence signal."""
        agg = SignalAggregator(factors_dir=tmp_path / "nonexistent")
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert result.consensus_signal == 0
        assert result.confidence_pct == 0.0
        assert result.total_strategies == 0


class TestSignalAggregatorConfidenceBounds:
    def test_confidence_pct_bounded_0_1(self, synthetic_ohlcv, tmp_path):
        """Weighted confidence must always be in [0, 1]."""
        # Create a dummy factor file
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        import joblib
        art = {
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.5,
            "config": {},
        }
        joblib.dump(art, factors_dir / "factor_0001.pkl")

        agg = SignalAggregator(factors_dir=factors_dir)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert 0.0 <= result.confidence_pct <= 1.0


class TestAggregatedSignalProperties:
    def test_signal_label_bullish(self):
        s = AggregatedSignal(ticker="0700.HK", consensus_signal=1)
        assert "看涨" in s.signal_label

    def test_signal_label_bearish(self):
        s = AggregatedSignal(ticker="0700.HK", consensus_signal=0)
        assert "看跌" in s.signal_label

    def test_confidence_label_high(self):
        s = AggregatedSignal(ticker="0700.HK", consensus_signal=1, confidence_pct=0.8)
        assert s.confidence_label == "高"

    def test_confidence_label_medium(self):
        s = AggregatedSignal(ticker="0700.HK", consensus_signal=1, confidence_pct=0.60)
        assert s.confidence_label == "中"

    def test_confidence_label_low(self):
        s = AggregatedSignal(ticker="0700.HK", consensus_signal=1, confidence_pct=0.40)
        assert s.confidence_label == "低"


def test_run_search_strategy_type_filters_strategies(synthetic_ohlcv, default_config):
    """run_search(strategy_type='single') should not include multi-class ML strategies in results."""
    from analyze_factor import run_search, _discover_strategies
    cfg = default_config.copy()
    cfg['max_tries'] = 1
    cfg['min_return'] = -999.0
    _, sorted_results, _ = run_search(synthetic_ohlcv, cfg, strategy_type='single')
    ml_names = {'xgboost_enhanced', 'lightgbm_enhanced',
                'xgboost_enhanced_tsfresh', 'lightgbm_enhanced_tsfresh'}
    result_names = {r.get('strategy_name', '') for r in sorted_results}
    assert result_names.isdisjoint(ml_names), (
        f"strategy_type='single' 搜索结果中出现了 ML 策略: {result_names & ml_names}"
    )
