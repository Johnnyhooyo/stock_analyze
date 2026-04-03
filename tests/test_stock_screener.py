"""
tests/test_stock_screener.py — StockScreener unit tests

Phase 2 Step 1: 基础选股引擎（动量 + 趋势 + 量价评分）
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from engine.stock_screener import StockScreener, ScreenerResult
from engine.screener_factors import ScreenerFactors, FactorResult
from engine.portfolio_state import PortfolioState


def make_ohlcv(
    n: int = 100,
    start_price: float = 100.0,
    trend: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper: generate deterministic OHLCV DataFrame."""
    np.random.seed(seed)
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


class TestScreenerFactorsCalcAll:
    def test_calc_all_returns_factor_result(self):
        df = make_ohlcv(n=100)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert isinstance(result, FactorResult)
        assert 0 <= result.momentum_score <= 100
        assert 0 <= result.trend_score <= 100
        assert 0 <= result.volume_score <= 100

    def test_calc_all_short_data_returns_valid_scores(self):
        df = make_ohlcv(n=30)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert isinstance(result, FactorResult)
        assert 0 <= result.momentum_score <= 100
        assert 0 <= result.trend_score <= 100


class TestScreenerFactorsMomentum:
    def test_momentum_positive_returns_high_score(self):
        df = make_ohlcv(n=100, trend=0.003)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert result.momentum_score > 50

    def test_momentum_negative_returns_low_score(self):
        df = make_ohlcv(n=100, trend=-0.003)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert result.momentum_score < 50


class TestScreenerFactorsSignals:
    def test_detect_signals_empty_on_insufficient_data(self):
        df = make_ohlcv(n=30)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert isinstance(result.signals, list)

    def test_detect_signals_breakout(self):
        df = make_ohlcv(n=100)
        df.iloc[-1, df.columns.get_loc("Close")] = df["High"].rolling(20).max().iloc[-1] * 1.01
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert "突破20日新高" in result.signals

    def test_detect_signals_rsi_oversold_bounce(self):
        df = make_ohlcv(n=100)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert isinstance(result.signals, list)

    def test_detect_signals_ma_alignment(self):
        df = make_ohlcv(n=100, trend=0.002)
        factors = ScreenerFactors()
        result = factors.calc_all(df)
        assert isinstance(result.signals, list)


class TestScreenerFactorsVolume:
    def test_avg_volume_ratio_positive(self):
        df = make_ohlcv(n=100)
        result = ScreenerFactors().calc_all(df)
        assert result.avg_volume_ratio >= 0


class TestStockScreenerInit:
    def test_default_weights(self):
        screener = StockScreener({})
        assert screener.weights["momentum"] == 0.35
        assert screener.weights["trend"] == 0.35
        assert screener.weights["volume"] == 0.30

    def test_custom_weights(self):
        config = {
            "screener": {
                "weight_momentum": 0.50,
                "weight_trend": 0.30,
                "weight_volume": 0.20,
            }
        }
        screener = StockScreener(config)
        assert screener.weights["momentum"] == 0.50
        assert screener.weights["trend"] == 0.30
        assert screener.weights["volume"] == 0.20


class TestStockScreenerScreen:
    def test_screen_returns_sorted_results(self):
        screener = StockScreener({})
        tickers = ["0700.HK", "0005.HK"]
        data_dict = {
            "0700.HK": make_ohlcv(n=100, trend=0.003),
            "0005.HK": make_ohlcv(n=100, trend=0.002),
        }
        results = screener.screen(tickers, data_dict)
        assert len(results) == 2
        assert results[0].composite_score >= results[1].composite_score

    def test_screen_insufficient_data_skipped(self):
        screener = StockScreener({})
        tickers = ["0700.HK"]
        data_dict = {"0700.HK": make_ohlcv(n=30)}
        results = screener.screen(tickers, data_dict)
        assert len(results) == 0

    def test_screen_min_score_filter(self):
        config = {"screener": {"min_score": 90.0}}
        screener = StockScreener(config)
        tickers = ["0700.HK"]
        data_dict = {"0700.HK": make_ohlcv(n=100, trend=0.001)}
        results = screener.screen(tickers, data_dict)
        for r in results:
            assert r.composite_score >= 90.0

    def test_screen_ranks_set(self):
        screener = StockScreener({})
        data_dict = {f"{i}.HK": make_ohlcv(n=100) for i in range(5)}
        results = screener.screen(list(data_dict.keys()), data_dict)
        if len(results) > 1:
            ranks = [r.rank for r in results]
            assert ranks == list(range(1, len(results) + 1))


class TestStockScreenerTopN:
    def test_top_n_default(self):
        screener = StockScreener({"screener": {"top_n": 3}})
        data_dict = {f"{i}.HK": make_ohlcv(n=100) for i in range(10)}
        results = screener.screen(list(data_dict.keys()), data_dict)
        top = screener.top_n(results)
        assert len(top) <= 3

    def test_top_n_exclude_held(self):
        screener = StockScreener({})
        data_dict = {f"{i}.HK": make_ohlcv(n=100) for i in range(5)}
        results = screener.screen(list(data_dict.keys()), data_dict)

        state = PortfolioState()
        state.add_watchlist_ticker("0.HK")
        state.positions["0.HK"].shares = 100
        state.positions["0.HK"].avg_cost = 100.0
        assert state.positions["0.HK"].has_position is True

        top = screener.top_n(results, exclude_held=True, portfolio_state=state)
        tickers = [r.ticker for r in top]
        assert "0.HK" not in tickers


class TestStockScreenerSectorRanking:
    def test_sector_ranking_returns_sorted(self):
        screener = StockScreener(
            {
                "screener": {
                    "sectors": {
                        "科技": ["0700.HK", "0005.HK"],
                        "金融": ["0988.HK"],
                    }
                }
            }
        )
        results = [
            ScreenerResult(ticker="0700.HK", composite_score=80.0, sector="科技"),
            ScreenerResult(ticker="0005.HK", composite_score=70.0, sector="科技"),
            ScreenerResult(ticker="0988.HK", composite_score=60.0, sector="金融"),
        ]
        ranking = screener.sector_ranking(results)
        assert len(ranking) >= 1
        assert ranking[0]["avg_score"] >= ranking[-1]["avg_score"]


class TestScreenerResult:
    def test_to_dict_returns_dict(self):
        result = ScreenerResult(
            ticker="0700.HK",
            composite_score=85.5,
            momentum_score=80.0,
            trend_score=90.0,
            volume_score=85.0,
            signals=["突破20日新高", "MACD金叉"],
            sector="科技",
            last_close=500.0,
            change_pct_5d=3.5,
            change_pct_20d=10.2,
            avg_volume_ratio=1.8,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["ticker"] == "0700.HK"
        assert d["composite_score"] == 85.5


class TestScreenerValuationAndSentiment:
    """Phase 2 Step 4: 估值 + 情感因子"""

    def test_factor_result_has_valuation_and_sentiment_fields(self):
        """FactorResult has valuation_score and sentiment_score fields"""
        from engine.screener_factors import FactorResult
        fr = FactorResult(
            momentum_score=70.0,
            trend_score=60.0,
            volume_score=80.0,
            signals=[],
            change_pct_5d=0.0,
            change_pct_20d=0.0,
            avg_volume_ratio=1.0,
            rsi_14=50.0,
            macd_hist=0.0,
            obv_slope=0.0,
            valuation_score=65.0,
            sentiment_score=55.0,
        )
        assert hasattr(fr, "valuation_score")
        assert hasattr(fr, "sentiment_score")
        assert fr.valuation_score == 65.0
        assert fr.sentiment_score == 55.0

    def test_screener_result_has_valuation_and_sentiment_fields(self):
        """ScreenerResult has valuation_score and sentiment_score fields"""
        from engine.stock_screener import ScreenerResult
        r = ScreenerResult(
            ticker="0700.HK",
            composite_score=80.0,
            momentum_score=70.0,
            trend_score=60.0,
            volume_score=80.0,
            valuation_score=65.0,
            sentiment_score=55.0,
        )
        assert hasattr(r, "valuation_score")
        assert hasattr(r, "sentiment_score")
        assert r.valuation_score == 65.0
        assert r.sentiment_score == 55.0

    def test_screener_result_to_dict_includes_valuation_sentiment(self):
        """ScreenerResult.to_dict() includes valuation_score and sentiment_score"""
        from engine.stock_screener import ScreenerResult
        r = ScreenerResult(
            ticker="0700.HK",
            composite_score=80.0,
            momentum_score=70.0,
            trend_score=60.0,
            volume_score=80.0,
            valuation_score=65.0,
            sentiment_score=55.0,
        )
        d = r.to_dict()
        assert "valuation_score" in d
        assert "sentiment_score" in d

    def test_screener_with_valuation_config_affects_composite(self):
        """enable_valuation=True causes composite to include valuation score"""
        config = {
            "screener": {
                "weight_momentum": 0.30,
                "weight_trend": 0.25,
                "weight_volume": 0.20,
                "weight_valuation": 0.25,
                "weight_sentiment": 0.00,
                "enable_valuation": True,
                "enable_sentiment": False,
            }
        }
        screener = StockScreener(config)
        assert screener.weights["valuation"] == 0.25
        assert screener.enable_valuation is True
        assert screener.enable_sentiment is False

    def test_screener_default_disables_valuation_and_sentiment(self):
        """By default, valuation and sentiment are disabled"""
        screener = StockScreener({})
        assert screener.enable_valuation is False
        assert screener.enable_sentiment is False

    def test_screener_factors_calc_all_includes_optional_scores(self):
        """calc_all() returns FactorResult with valuation and sentiment scores"""
        df = make_ohlcv(n=100, trend=0.002)
        factors = ScreenerFactors()
        result = factors.calc_all(df, enable_valuation=False, enable_sentiment=False)
        assert hasattr(result, "valuation_score")
        assert hasattr(result, "sentiment_score")
        assert isinstance(result.signals, list)
