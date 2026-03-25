"""
tests/test_schemas.py — data.schemas column normalization & validation
"""
import numpy as np
import pandas as pd
import pytest

from data.schemas import normalize_columns, validate_ohlcv


class TestNormalizeColumnsChineseSynonyms:
    def test_opens(self):
        df = pd.DataFrame({"开盘": [1, 2, 3], "收盘": [1.0, 2.0, 3.0], "成交量": [100, 100, 100]})
        result = normalize_columns(df)
        assert "Open" in result.columns

    def test_high(self):
        df = pd.DataFrame({"最高": [1, 2, 3], "收盘": [1.0, 2.0, 3.0], "成交量": [100, 100, 100]})
        result = normalize_columns(df)
        assert "High" in result.columns

    def test_low(self):
        df = pd.DataFrame({"最低": [1, 2, 3], "收盘": [1.0, 2.0, 3.0], "成交量": [100, 100, 100]})
        result = normalize_columns(df)
        assert "Low" in result.columns

    def test_close(self):
        df = pd.DataFrame({"收盘价": [1, 2, 3], "开盘": [1, 2, 3]})
        result = normalize_columns(df)
        assert "Close" in result.columns

    def test_volume(self):
        df = pd.DataFrame({"vol": [100, 100, 100], "close": [1.0, 2.0, 3.0]})
        result = normalize_columns(df)
        assert "Volume" in result.columns


class TestNormalizeColumnsMultiIndexYfinance:
    def test_multindex_flatten(self):
        """yfinance >=0.2.31 returns (Price, Ticker) MultiIndex columns."""
        columns = pd.MultiIndex.from_tuples(
            [("Open", "0700.HK"), ("High", "0700.HK"), ("Low", "0700.HK"),
             ("Close", "0700.HK"), ("Volume", "0700.HK")]
        )
        df = pd.DataFrame(np.random.randn(3, 5), columns=columns)
        result = normalize_columns(df)
        assert "Open" in result.columns
        assert "Close" in result.columns

    def test_multindex_swaplevel(self):
        """Price outer level should be swapped to inner."""
        columns = pd.MultiIndex.from_tuples(
            [("0700.HK", "Open"), ("0700.HK", "High"),
             ("0700.HK", "Close"), ("0700.HK", "Volume")]
        )
        df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
        result = normalize_columns(df)
        assert "Open" in result.columns


class TestValidateOhlcvRequiresCloseColumn:
    def test_missing_close_column(self):
        df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Volume": [1, 1]})
        result = validate_ohlcv(df)
        assert not result.ok
        assert any("Close" in e for e in result.errors)

    def test_empty_dataframe(self):
        result = validate_ohlcv(pd.DataFrame())
        assert not result.ok

    def test_none_dataframe(self):
        result = validate_ohlcv(None)
        assert not result.ok


class TestValidateOhlcvWarnsOnBadOslc:
    def test_high_below_open_close(self):
        df = pd.DataFrame({
            "Open": [100.0, 105.0],
            "High": [99.0, 104.0],   # High < Open
            "Low":  [99.0, 104.0],
            "Close": [99.0, 104.0],
            "Volume": [1000, 1000],
        })
        result = validate_ohlcv(df)
        assert any("High < Open" in w for w in result.warnings)

    def test_low_above_open_close(self):
        df = pd.DataFrame({
            "Open": [100.0, 105.0],
            "High": [101.0, 106.0],
            "Low":  [100.0, 104.0],   # Low > Close
            "Close": [99.0, 104.0],
            "Volume": [1000, 1000],
        })
        result = validate_ohlcv(df)
        assert any("Low > Open" in w or "Low > Close" in w for w in result.warnings)


class TestValidateOhlcvWarnsOnNegativeVolume:
    def test_negative_volume(self):
        df = pd.DataFrame({
            "Open": [100.0, 105.0, 110.0],
            "High": [101.0, 106.0, 111.0],
            "Low":  [99.0,  104.0, 109.0],
            "Close": [100.0, 105.0, 110.0],
            "Volume": [-100, 1000, 1000],
        })
        result = validate_ohlcv(df)
        assert any("Volume < 0" in w for w in result.warnings)
