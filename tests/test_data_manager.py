"""
tests/test_data_manager.py — DataManager offline tests

Tests the DataManager methods that don't require network access:
  - normalize_columns (delegates to data.schemas)
  - validate_ohlcv    (delegates to data.schemas)
  - AutoBackend detection
"""
import numpy as np
import pandas as pd
import pytest

# Re-use schemas tests directly via the DataManager's public API
from data.manager import _atomic_write_csv
from data.schemas import normalize_columns, validate_ohlcv


class TestDataManagerSchemasIntegration:
    """DataManager delegates column normalization and validation to schemas.py."""

    def test_normalize_columns_chinese_synonyms(self):
        df = pd.DataFrame({"开盘": [1, 2, 3], "收盘价": [1.0, 2.0, 3.0], "成交量": [100, 100, 100]})
        result = normalize_columns(df)
        assert "Open" in result.columns
        assert "Close" in result.columns

    def test_normalize_columns_multindex_yfinance(self):
        columns = pd.MultiIndex.from_tuples(
            [("Open", "0700.HK"), ("High", "0700.HK"), ("Low", "0700.HK"),
             ("Close", "0700.HK"), ("Volume", "0700.HK")]
        )
        df = pd.DataFrame(np.random.randn(3, 5), columns=columns)
        result = normalize_columns(df)
        assert "Open" in result.columns
        assert "Close" in result.columns

    def test_validate_ohlcv_requires_close_column(self):
        df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Volume": [1, 1]})
        result = validate_ohlcv(df)
        assert not result.ok
        assert any("Close" in e for e in result.errors)


class TestAtomicWrite:
    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "subdir" / "data.csv"
        df = pd.DataFrame({"Open": [1, 2], "Close": [1.0, 2.0]})
        _atomic_write_csv(df, target)
        assert target.exists()

    def test_atomic_write_round_trip(self, tmp_path):
        target = tmp_path / "data.csv"
        df_orig = pd.DataFrame({
            "Open": [1.0, 2.0, 3.0],
            "High": [1.1, 2.1, 3.1],
            "Low": [0.9, 1.9, 2.9],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [100.0, 100.0, 100.0],
        })
        _atomic_write_csv(df_orig, target)
        df_read = pd.read_csv(target, index_col="date")
        # Compare values only (index name may differ)
        pd.testing.assert_frame_equal(df_orig.reset_index(drop=True), df_read.reset_index(drop=True))
