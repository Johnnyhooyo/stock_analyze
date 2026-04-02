"""Portfolio-mode training orchestration tests (all I/O mocked)."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
    dates = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.ones(n) * 1e6,
    }, index=dates)


_OHLCV = _make_ohlcv()
_FACTOR_RESULT = ("/tmp/factor_0001.pkl", {"sharpe_ratio": 1.5, "validated": "double"}, [])
_EMPTY_RESULT  = (None, None, [])


class TestTrainPortfolioTickers:

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_ml_trained_once_rules_per_ticker(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """ML 全局训练 1 次，2 只股票的规则训练各 1 次，共调用 step2_train 3 次。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)
        # 1 次 ML global + 2 次 per-ticker rule = 3
        assert mock_train.call_count == 3
        mock_hsi.assert_called_once()

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_ml_call_uses_multi_strategy_type(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """ML 全局训练调用中 strategy_type='multi'，无 factors_dir_override。"""
        from main import train_portfolio_tickers
        train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        first_call = mock_train.call_args_list[0]
        assert first_call.kwargs.get("strategy_type") == "multi"
        assert first_call.kwargs.get("factors_dir_override") is None

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_rule_calls_use_single_and_per_ticker_dir(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """规则训练调用：strategy_type='single'，factors_dir_override 包含 TICKER_SAFE。"""
        from main import train_portfolio_tickers
        train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        # call index 1 = 0700.HK rule, call index 2 = 0005.HK rule
        for i, expected_safe in enumerate(["0700_HK", "0005_HK"], start=1):
            call = mock_train.call_args_list[i]
            assert call.kwargs.get("strategy_type") == "single"
            dir_arg = call.kwargs.get("factors_dir_override")
            assert dir_arg is not None, f"call {i} 没有 factors_dir_override"
            assert expected_safe in str(dir_arg), (
                f"call {i} 的目录 {dir_arg} 不含 {expected_safe}"
            )

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", side_effect=Exception("network error"))
    @patch("main.step2_train", return_value=_FACTOR_RESULT)
    def test_data_failure_skips_ticker_continues(self, mock_train, mock_step1, mock_hsi):
        """数据下载失败时跳过该 ticker，继续处理其余 ticker，不抛异常。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert len(results) == 2
        # step1 失败，所有 ticker 都 data_failed（ML training 也需要 step1）
        assert all(r["status"] == "data_failed" for r in results)

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, Exception("train error"), _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_rule_train_failure_skips_ticker_continues(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """规则训练失败时跳过该 ticker，其余 ticker 正常完成。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert results[0]["status"] == "train_failed"   # 0700.HK rule 失败
        assert results[1]["status"] == "ok"             # 0005.HK 正常

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_EMPTY_RESULT, _EMPTY_RESULT, _EMPTY_RESULT])
    @patch("main._latest_factor_path", return_value=None)
    def test_no_factor_still_ok_status(self, mock_latest, mock_train, mock_step1, mock_hsi):
        """训练未产生因子时 status=ok 但 factor_path=None。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK"])
        assert results[0]["status"] == "ok"
        assert results[0]["factor_path"] is None
