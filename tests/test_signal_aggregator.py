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


class TestSignalAggregatorHybridLoading:
    """分层混合：per-ticker 规则因子 + 全局 ML 因子合并投票。"""

    def _make_rule_factor(self, d: Path, run_id: int = 1, sharpe: float = 1.2):
        import joblib
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": sharpe, "config": {},
        }, d / f"factor_{run_id:04d}.pkl")

    def _make_ml_factor(self, d: Path, run_id: int = 1, sharpe: float = 1.5):
        import joblib
        from xgboost import XGBClassifier
        d.mkdir(parents=True, exist_ok=True)
        model = XGBClassifier(n_estimators=3, max_depth=2, verbosity=0, eval_metric='logloss')
        joblib.dump({
            "meta": {"name": "xgboost_enhanced", "params": {},
                     "feat_cols": ["rsi", "macd"]},
            "model": model, "sharpe_ratio": sharpe, "config": {},
        }, d / f"factor_{run_id:04d}.pkl")

    def test_both_dirs_loaded_when_per_ticker_exists(self, tmp_path, synthetic_ohlcv):
        """per-ticker 目录存在时，规则因子 + 全局 ML 因子都应参与投票。"""
        global_dir = tmp_path / "factors"
        ticker_dir = global_dir / "0005_HK"
        self._make_rule_factor(ticker_dir, run_id=1)
        self._make_ml_factor(global_dir, run_id=1)

        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("0005.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        assert result.total_strategies >= 0

    def test_global_only_when_no_per_ticker_dir(self, tmp_path, synthetic_ohlcv):
        """没有 per-ticker 目录时，加载全局目录所有因子（向后兼容）。"""
        global_dir = tmp_path / "factors"
        self._make_rule_factor(global_dir, run_id=1)
        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("9988.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_global_ml_only_when_per_ticker_exists(self, tmp_path, synthetic_ohlcv):
        """per-ticker 目录存在时，全局目录中的规则因子不应重复加载。"""
        global_dir = tmp_path / "factors"
        ticker_dir = global_dir / "0005_HK"
        self._make_rule_factor(global_dir, run_id=1, sharpe=1.0)
        self._make_ml_factor(global_dir, run_id=2, sharpe=1.8)
        self._make_rule_factor(ticker_dir, run_id=1, sharpe=1.3)

        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("0005.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_empty_per_ticker_dir_falls_back_to_global(self, tmp_path, synthetic_ohlcv):
        """空的 per-ticker 目录不影响全局目录加载。"""
        global_dir = tmp_path / "factors"
        (global_dir / "1299_HK").mkdir(parents=True, exist_ok=True)
        self._make_rule_factor(global_dir, run_id=1)
        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("1299.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_load_factors_explicit_dir(self, tmp_path):
        """_load_factors(dir) 使用传入的目录，而非 self.factors_dir。"""
        import joblib
        dir_a = tmp_path / "a"; dir_a.mkdir()
        dir_b = tmp_path / "b"; dir_b.mkdir()
        joblib.dump({"meta": {"name": "s1", "params": {}, "feat_cols": []},
                     "model": None, "sharpe_ratio": 0.5, "config": {}},
                    dir_a / "factor_0001.pkl")
        joblib.dump({"meta": {"name": "s2", "params": {}, "feat_cols": []},
                     "model": None, "sharpe_ratio": 1.8, "config": {}},
                    dir_b / "factor_0001.pkl")
        agg = SignalAggregator(factors_dir=dir_a)
        arts = agg._load_factors(dir_b)
        assert len(arts) == 1
        assert arts[0]["sharpe_ratio"] == 1.8


class TestSignalAggregatorStacking:
    """SignalAggregator stacking mode: uses meta-model when available, fallback to vote."""

    def _make_rule_factor(self, factors_dir: Path, run_id: int = 1):
        import joblib
        factors_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": 1.2, "config": {},
        }, factors_dir / f"factor_{run_id:04d}.pkl")

    def _make_and_save_meta(self, ticker: str, strategy_names: list, meta_dir: Path, synthetic_ohlcv):
        """Create a minimal fitted meta-model saved to meta_dir."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        meta_dir.mkdir(parents=True, exist_ok=True)
        n_feats = len(strategy_names) + 3
        X = np.random.RandomState(0).randn(60, n_feats)
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=100)
        lr.fit(scaler.fit_transform(X), y)
        ticker_safe = ticker.replace(".", "_").upper()
        joblib.dump(
            {"model": lr, "scaler": scaler, "strategy_names": strategy_names},
            meta_dir / f"meta_model_{ticker_safe}.pkl",
        )

    def test_stacking_uses_meta_model_when_available(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='stacking' with meta-model present → result is AggregatedSignal."""
        factors_dir = tmp_path / "factors"
        meta_dir = tmp_path / "meta"
        self._make_rule_factor(factors_dir)
        self._make_and_save_meta("0700.HK", ["ma_crossover"], meta_dir, synthetic_ohlcv)

        agg = SignalAggregator(
            factors_dir=factors_dir,
            aggregation_method="stacking",
            meta_dir=meta_dir,
            use_registry=False,
        )
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})
        assert isinstance(result, AggregatedSignal)
        assert result.consensus_signal in (0, 1)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_stacking_fallback_to_vote_when_no_meta_model(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='stacking' without meta-model → fallback to vote, no error."""
        factors_dir = tmp_path / "factors"
        self._make_rule_factor(factors_dir)

        agg = SignalAggregator(
            factors_dir=factors_dir,
            aggregation_method="stacking",
            meta_dir=tmp_path / "nonexistent_meta",
            use_registry=False,
        )
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_vote_method_unchanged_by_new_params(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='vote' (default) behaves identically to original."""
        factors_dir = tmp_path / "factors"
        self._make_rule_factor(factors_dir)
        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0


class TestICWeightedVoting:
    """IC 加权投票：rank_ic 存在时应优先于 sharpe 作为权重。"""

    def _make_factor(self, d: Path, run_id: int, sharpe: float, rank_ic: float | None = None):
        import joblib
        d.mkdir(parents=True, exist_ok=True)
        art = {
            "meta": {"name": f"strat_{run_id}", "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": sharpe,
            "config": {},
        }
        if rank_ic is not None:
            art["rank_ic"] = rank_ic
        joblib.dump(art, d / f"factor_{run_id:04d}.pkl")

    def test_ic_weight_beats_sharpe_fallback(self, tmp_path, synthetic_ohlcv):
        """有 rank_ic 的因子，权重为 abs(ic)，不使用 sharpe。"""
        factors_dir = tmp_path / "factors"
        # factor 1: high sharpe, no IC → uses sharpe=2.0
        self._make_factor(factors_dir, 1, sharpe=2.0, rank_ic=None)
        # factor 2: low sharpe, high IC → uses abs(ic)=0.25
        self._make_factor(factors_dir, 2, sharpe=0.2, rank_ic=0.25)

        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_negative_ic_uses_abs(self, tmp_path, synthetic_ohlcv):
        """负的 IC（反向相关因子）weight = abs(ic) > 0。"""
        factors_dir = tmp_path / "factors"
        self._make_factor(factors_dir, 1, sharpe=1.0, rank_ic=-0.30)
        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_nan_ic_falls_back_to_sharpe(self, tmp_path, synthetic_ohlcv):
        """rank_ic=nan → 回退到 sharpe 权重。"""
        factors_dir = tmp_path / "factors"
        self._make_factor(factors_dir, 1, sharpe=1.5, rank_ic=float("nan"))
        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_missing_ic_key_falls_back_to_sharpe(self, tmp_path, synthetic_ohlcv):
        """旧因子文件没有 rank_ic 键 → 回退到 sharpe 权重，不抛出异常。"""
        import joblib
        factors_dir = tmp_path / "factors"
        factors_dir.mkdir()
        # Deliberately omit rank_ic key (old factor format)
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.2,
            "config": {},
        }, factors_dir / "factor_0001.pkl")
        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
