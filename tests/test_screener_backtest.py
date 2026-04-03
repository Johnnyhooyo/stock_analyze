"""
tests/test_screener_backtest.py — ScreenerBacktester unit tests

Phase 2 Step 5: 选股因子回测验证框架
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from engine.stock_screener import ScreenerResult
from engine.screener_backtest import ScreenerBacktester, ScreenerBacktestResult


def make_ohlcv_for_ticker(
    ticker: str,
    n: int = 500,
    start_price: float = 100.0,
    trend: float = 0.0,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper: generate deterministic OHLCV DataFrame for a ticker."""
    np.random.seed(seed + hash(ticker) % 10000)
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    rets = np.random.normal(trend, vol, n)
    close = start_price * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "Open": close * (1 + np.random.uniform(-0.002, 0.002, n)),
            "High": close * (1 + np.abs(np.random.uniform(0.001, 0.005, n))),
            "Low": close * (1 - np.abs(np.random.uniform(0.001, 0.005, n))),
            "Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def make_multi_stock_data(
    tickers: list[str],
    n: int = 500,
    trend: float = 0.001,
) -> dict[str, pd.DataFrame]:
    """Helper: create historical data dict for multiple tickers."""
    return {t: make_ohlcv_for_ticker(t, n=n, trend=trend) for t in tickers}


class TestScreenerBacktestResult:
    def test_backtest_result_fields(self):
        """ScreenerBacktestResult has expected fields"""
        r = ScreenerBacktestResult(
            hit_rate=0.65,
            avg_return_pct=8.5,
            total_return_pct=42.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=-15.3,
            n_picks=50,
            n_periods=10,
            equity_curve=[],
        )
        assert r.hit_rate == 0.65
        assert r.avg_return_pct == 8.5
        assert r.total_return_pct == 42.0
        assert r.sharpe_ratio == 1.2
        assert r.max_drawdown_pct == -15.3
        assert r.n_picks == 50
        assert r.n_periods == 10


class TestScreenerBacktesterInit:
    def test_default_config(self):
        """ScreenerBacktester initializes with defaults"""
        bt = ScreenerBacktester({})
        assert bt.top_n == 10
        assert bt.rebalance_days == 20
        assert bt.min_hold_days == 5

    def test_custom_config(self):
        """ScreenerBacktester accepts custom config"""
        cfg = {
            "screener": {
                "top_n": 5,
                "rebalance_days": 10,
                "min_hold_days": 3,
            }
        }
        bt = ScreenerBacktester(cfg)
        assert bt.top_n == 5
        assert bt.rebalance_days == 10
        assert bt.min_hold_days == 3


class TestScreenerBacktesterBacktest:
    def test_backtest_returns_result_object(self):
        """backtest() returns ScreenerBacktestResult"""
        tickers = ["0700.HK", "0005.HK", "0988.HK", "1810.HK", "3690.HK"]
        data = make_multi_stock_data(tickers, n=300, trend=0.001)
        bt = ScreenerBacktester({"screener": {"top_n": 3}})
        result = bt.backtest(data)
        assert isinstance(result, ScreenerBacktestResult)

    def test_backtest_hit_rate_between_0_and_1(self):
        """hit_rate is always between 0 and 1"""
        tickers = ["0700.HK", "0005.HK", "0988.HK"]
        data = make_multi_stock_data(tickers, n=300, trend=0.002)
        bt = ScreenerBacktester({"screener": {"top_n": 2}})
        result = bt.backtest(data)
        assert 0.0 <= result.hit_rate <= 1.0

    def test_backtest_with_positive_trend_higher_hit_rate(self):
        """Positive trend stocks have higher hit rate"""
        tickers = ["0700.HK", "0005.HK", "0988.HK", "1810.HK", "3690.HK"]
        data_good = make_multi_stock_data(tickers, n=300, trend=0.003)
        data_bad = make_multi_stock_data(tickers, n=300, trend=-0.003)
        bt = ScreenerBacktester({"screener": {"top_n": 3}})
        r_good = bt.backtest(data_good)
        r_bad = bt.backtest(data_bad)
        assert r_good.hit_rate >= r_bad.hit_rate

    def test_backtest_requires_min_data(self):
        """backtest() raises if data too short"""
        tickers = ["0700.HK"]
        data = make_multi_stock_data(tickers, n=30, trend=0.001)
        bt = ScreenerBacktester({})
        result = bt.backtest(data)
        assert result.n_picks == 0
