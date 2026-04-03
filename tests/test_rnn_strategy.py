"""
tests/test_rnn_strategy.py — RNN strategy (GRU/LSTM) unit tests (offline, no network)
"""
import importlib
import numpy as np
import pandas as pd
import pytest

RNN_MODULE = "strategies.rnn_trend"

torch = pytest.importorskip("torch", reason="torch not installed")
import strategies.rnn_trend


class TestRNNStrategyImports:
    def test_module_importable(self):
        pass

    def test_name_attribute(self):
        mod = importlib.import_module(RNN_MODULE)
        assert hasattr(mod, "NAME")
        assert mod.NAME == "rnn_trend"


class TestGRURunInterface:
    @pytest.fixture
    def gru_mod(self):
        return importlib.import_module(RNN_MODULE)

    @pytest.fixture
    def minimal_cfg(self):
        return {
            "rnn_window": 20,
            "rnn_label_period": 3,
            "rnn_label_threshold": 0.02,
            "rnn_upper_threshold": 0.60,
            "rnn_lower_threshold": 0.40,
            "rnn_epochs": 5,
            "rnn_batch_size": 64,
            "rnn_lr": 0.001,
            "rnn_hidden_size": 32,
            "rnn_num_layers": 1,
            "rnn_dropout": 0.2,
            "rnn_cell_type": "gru",
        }

    def test_run_returns_tuple(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal, model, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        assert isinstance(signal, pd.Series)
        assert isinstance(meta, dict)

    def test_signal_values_subset_of_01(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal, _, _ = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        unique = set(signal.dropna().unique())
        assert unique.issubset({0, 1})

    def test_signal_length_matches_data(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal, _, _ = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        assert len(signal) == len(synthetic_ohlcv)

    def test_predict_returns_series(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal_run, model, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        signal_pred = gru_mod.predict(model, synthetic_ohlcv, minimal_cfg, meta)
        assert isinstance(signal_pred, pd.Series)
        assert len(signal_pred) == len(synthetic_ohlcv)

    def test_predict_values_subset_of_01(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal_run, model, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        signal_pred = gru_mod.predict(model, synthetic_ohlcv, minimal_cfg, meta)
        unique = set(signal_pred.dropna().unique())
        assert unique.issubset({0, 1})

    def test_meta_contains_required_keys(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        _, _, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        assert "name" in meta
        assert "params" in meta
        assert "feat_cols" in meta
        assert meta["name"] == "rnn_trend"

    def test_meta_state_dict_b64(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        _, model, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        assert "state_dict_b64" in meta
        assert isinstance(meta["state_dict_b64"], str)
        assert len(meta["state_dict_b64"]) > 100

    def test_predict_with_restored_model(self, synthetic_ohlcv, gru_mod, minimal_cfg):
        signal_run, model, meta = gru_mod.run(synthetic_ohlcv, minimal_cfg)
        restored_pred = gru_mod.predict(None, synthetic_ohlcv, minimal_cfg, meta)
        assert isinstance(restored_pred, pd.Series)


class TestGRUNoLookAheadBias:
    @pytest.fixture
    def gru_mod(self):
        return importlib.import_module(RNN_MODULE)

    def test_window_features_use_only_past_data(self, gru_mod):
        cfg = {
            "rnn_window": 10,
            "rnn_label_period": 3,
            "rnn_epochs": 3,
            "rnn_batch_size": 32,
            "rnn_hidden_size": 16,
            "rnn_num_layers": 1,
            "rnn_dropout": 0.2,
        }
        n = 200
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        np.random.seed(42)
        close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
        df = pd.DataFrame(
            {
                "Open": close * (1 + np.random.normal(0, 0.002, n)),
                "High": close * (1 + np.abs(np.random.normal(0, 0.005, n))),
                "Low": close * (1 - np.abs(np.random.normal(0, 0.005, n))),
                "Close": close,
                "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
            },
            index=dates,
        )
        signal, _, meta = gru_mod.run(df, cfg)
        assert isinstance(signal, pd.Series)
        assert len(signal) == n


class TestLSTMCellType:
    @pytest.fixture
    def gru_mod(self):
        return importlib.import_module(RNN_MODULE)

    def test_lstm_cell_type(self, synthetic_ohlcv, gru_mod):
        cfg = {
            "rnn_window": 20,
            "rnn_label_period": 3,
            "rnn_label_threshold": 0.02,
            "rnn_upper_threshold": 0.60,
            "rnn_lower_threshold": 0.40,
            "rnn_epochs": 3,
            "rnn_batch_size": 64,
            "rnn_lr": 0.001,
            "rnn_hidden_size": 32,
            "rnn_num_layers": 1,
            "rnn_dropout": 0.2,
            "rnn_cell_type": "lstm",
        }
        signal, _, meta = gru_mod.run(synthetic_ohlcv, cfg)
        assert meta["model_type"] == "lstm"
        assert isinstance(signal, pd.Series)


class TestFeatureMatrixShape:
    def test_build_feature_matrix_shape(self):
        from strategies.rnn_trend import _build_feature_matrix

        n = 100
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        np.random.seed(42)
        close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
        df = pd.DataFrame(
            {
                "Open": close * 1.001,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
            },
            index=dates,
        )

        window = 30
        X = _build_feature_matrix(df, window=window)
        assert X.shape[0] == n
        assert X.shape[1] == window
        assert X.shape[2] == 13


class TestMultiStockTraining:
    @pytest.fixture
    def gru_mod(self):
        return importlib.import_module(RNN_MODULE)

    def test_run_multi_stock(self, gru_mod):
        n = 300
        dates = pd.bdate_range(end="2025-12-31", periods=n)
        np.random.seed(0)
        frames = []
        for tk in ["0001.HK", "0700.HK"]:
            close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
            df = pd.DataFrame(
                {
                    "Open": close,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": 1e6 * np.ones(n),
                    "ticker": tk,
                },
                index=dates,
            )
            frames.append(df)
        multi_df = pd.concat(frames).sort_index()

        signal, model, meta = gru_mod.run(multi_df, {})
        assert model is not None
        assert "state_dict_b64" in meta
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(multi_df)
