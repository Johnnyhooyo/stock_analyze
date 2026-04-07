"""tests/test_meta_aggregator.py — MetaAggregator unit tests"""
import numpy as np
import pytest
from pathlib import Path
from engine.meta_aggregator import MetaAggregator


class TestBuildFeatureVector:
    def test_shape_is_n_strategies_plus_3(self, synthetic_ohlcv, tmp_path):
        """Feature vector = n strategy signals + 3 market state indicators."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["s1", "s2", "s3"]
        feat = ma.build_feature_vector({"s1": 1, "s2": 0, "s3": 1}, synthetic_ohlcv)
        assert feat.shape == (6,)   # 3 strategies + 3 market state

    def test_missing_strategy_fills_zero(self, synthetic_ohlcv, tmp_path):
        """Strategy not in base_signals → treated as bearish (0)."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["known", "unknown_in_signals"]
        feat = ma.build_feature_vector({"known": 1}, synthetic_ohlcv)
        assert feat[0] == 1.0
        assert feat[1] == 0.0

    def test_market_state_values_in_reasonable_range(self, synthetic_ohlcv, tmp_path):
        """ADX in [0, 100], ATR rank in [0, 1], volume ratio > 0."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = []
        feat = ma.build_feature_vector({}, synthetic_ohlcv)
        adx, atr_rank, vol_ratio = feat[0], feat[1], feat[2]
        assert 0.0 <= adx <= 100.0
        assert 0.0 <= atr_rank <= 1.0
        assert vol_ratio > 0.0


class TestSaveLoadPredict:
    def _make_trained_meta(self, tmp_path, synthetic_ohlcv):
        """Helper: create a MetaAggregator with a minimal fitted model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["s1", "s2"]
        X = np.random.RandomState(0).randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X_s, y)
        ma._model = lr
        ma._scaler = scaler
        return ma

    def test_save_creates_pkl_file(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        path = ma.save("0700.HK")
        assert path.exists()
        assert path.name == "meta_model_0700_HK.pkl"

    def test_load_returns_none_when_file_missing(self, tmp_path):
        result = MetaAggregator.load("9999.HK", tmp_path)
        assert result is None

    def test_save_load_round_trip_preserves_strategy_names(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma.save("0700.HK")
        loaded = MetaAggregator.load("0700.HK", tmp_path)
        assert loaded is not None
        assert loaded._strategy_names == ["s1", "s2"]

    def test_predict_signal_in_0_or_1(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma._strategy_names = ["s1", "s2"]
        feat = ma.build_feature_vector({"s1": 1, "s2": 0}, synthetic_ohlcv)
        signal, proba = ma.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0

    def test_predict_after_save_load(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma.save("0700.HK")
        loaded = MetaAggregator.load("0700.HK", tmp_path)
        feat = loaded.build_feature_vector({"s1": 1, "s2": 0}, synthetic_ohlcv)
        signal, proba = loaded.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0
