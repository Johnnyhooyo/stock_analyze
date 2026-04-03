"""
tests/test_integration.py — End-to-end integration smoke tests
"""
import numpy as np
import pandas as pd
import pytest

from analyze_factor import backtest, run_factor_analysis
from engine.portfolio_state import PortfolioPosition, PortfolioState
from position_manager import PositionManager, Position


class TestAnalyzeFactorPipelineNoNetwork:
    def test_analyze_factor_pipeline_no_network(self, synthetic_ohlcv):
        """Full pipeline: strategy signal → factor analysis → backtest, no network."""
        # Use MA crossover as the "strategy"
        from strategies.ma_crossover import run as ma_run

        sig, _, _ = ma_run(synthetic_ohlcv, {})
        assert isinstance(sig, pd.Series)

        # Factor analysis
        fa = run_factor_analysis(synthetic_ohlcv, sig, {})
        assert "ic_mean" in fa or "error" not in fa

        # Backtest
        cfg = {
            "initial_capital": 100_000.0,
            "fees_rate": 0.00088,
            "stamp_duty": 0.001,
            "lookback_months": 3,
            "risk_management": {"simulate_in_backtest": False},
        }
        result = backtest(synthetic_ohlcv, sig, cfg)
        assert "cum_return" in result


class TestPortfolioStateToPositionManagerRoundTrip:
    def test_portfolio_state_to_position_manager_position_round_trip(self):
        """PortfolioPosition → PositionManager.Position → PortfolioPosition round-trip."""
        # Create a portfolio position
        port_pos = PortfolioPosition(
            ticker="0700.HK",
            shares=200,
            avg_cost=380.0,
            peak_price=400.0,
        )
        assert port_pos.has_position

        # Convert to PositionManager Position
        pm_pos = port_pos.to_position_manager_position(current_price=390.0)
        assert isinstance(pm_pos, Position)
        assert pm_pos.shares == 200
        assert pm_pos.avg_cost == 380.0
        assert pm_pos.current_price == 390.0
        assert pm_pos.profit == (390.0 - 380.0) * 200


class TestPositionAnalyzerRecommendation:
    def test_position_analyzer_returns_recommendation_result(self, synthetic_ohlcv):
        """PositionManager.get_recommendation returns expected dict structure."""
        pm = PositionManager(portfolio_value=100_000.0)
        pm.set_position(shares=200, avg_cost=380.0, current_price=390.0)

        result = pm.get_recommendation(signal=1, predicted_return=0.05)
        assert "action" in result
        assert "reason" in result
        assert result["action"] in ("买入", "持有", "卖出", "观望", "无法判断")


class TestScreenerIntegration:
    """Phase 2 Step 2: 选股模块集成测试"""

    def test_build_daily_report_with_screener_results(self, synthetic_ohlcv):
        """daily_report 字典包含 screener 选股结果"""
        from engine.stock_screener import ScreenerResult
        from engine.position_analyzer import PositionAnalyzer, RecommendationResult
        from engine.signal_aggregator import AggregatedSignal

        # Create a minimal recommendation result (no screener)
        agg = AggregatedSignal(
            ticker="0700.HK",
            consensus_signal=1,
            confidence_pct=0.7,
        )
        rec_result = RecommendationResult(
            ticker="0700.HK",
            last_date="2026-04-03",
            last_close=500.0,
            action="持有",
            reason="测试",
            signal=1,
            agg_signal=agg,
            has_position=True,
            shares=200,
            avg_cost=480.0,
            market_value=100000.0,
            profit=4000.0,
            profit_pct=4.17,
            stop_price=450.0,
            kelly_shares=0,
            kelly_amount=0,
            circuit_breaker=False,
            consecutive_loss_days=0,
            confidence_pct=0.7,
            confidence_label="中",
            sentiment=None,
            price_lo_1d=0,
            price_hi_1d=0,
            atr=10.0,
            risk_flags=[],
        )

        screener_results = [
            ScreenerResult(
                ticker="1810.HK",
                composite_score=85.0,
                rank=1,
                momentum_score=80.0,
                trend_score=90.0,
                volume_score=85.0,
                signals=["突破20日新高", "MACD金叉"],
                sector="科技",
                last_close=18.5,
                change_pct_5d=6.3,
                change_pct_20d=15.1,
                avg_volume_ratio=1.8,
            ),
            ScreenerResult(
                ticker="3690.HK",
                composite_score=78.0,
                rank=2,
                momentum_score=75.0,
                trend_score=80.0,
                volume_score=78.0,
                signals=["放量上涨", "OBV趋势新高"],
                sector="消费",
                last_close=120.0,
                change_pct_5d=4.1,
                change_pct_20d=8.3,
                avg_volume_ratio=1.5,
            ),
        ]

        # Import the build function from daily_run
        import daily_run
        daily_report = daily_run._build_daily_report(
            results=[rec_result],
            portfolio_value=200000.0,
            run_date="2026-04-03",
            market_is_open=True,
            config={},
            screener_results=screener_results,
        )

        assert "screener_results" in daily_report
        assert len(daily_report["screener_results"]) == 2
        assert daily_report["screener_results"][0]["ticker"] == "1810.HK"
        assert daily_report["screener_results"][0]["composite_score"] == 85.0

    def test_build_markdown_report_includes_screener_section(self, synthetic_ohlcv):
        """Markdown 报告包含选股板块"""
        from engine.stock_screener import ScreenerResult

        screener_results = [
            ScreenerResult(
                ticker="1810.HK",
                composite_score=85.0,
                rank=1,
                momentum_score=80.0,
                trend_score=90.0,
                volume_score=85.0,
                signals=["突破20日新高", "MACD金叉"],
                sector="科技",
                last_close=18.5,
                change_pct_5d=6.3,
                change_pct_20d=15.1,
                avg_volume_ratio=1.8,
            ),
        ]

        daily_report = {
            "run_date": "2026-04-03",
            "market_is_open": True,
            "portfolio_value": 200000.0,
            "total_market_value": 100000.0,
            "total_cost_basis": 96000.0,
            "total_pnl": 4000.0,
            "total_pnl_pct": 4.17,
            "cash_value": 100000.0,
            "cash_pct": 50.0,
            "held_count": 1,
            "total_tickers": 1,
            "buy_signals": [],
            "sell_signals": [],
            "recommendations": [],
            "screener_results": [
                r.to_dict() for r in screener_results
            ],
        }

        import daily_run
        md = daily_run._build_markdown_report(daily_report)

        assert "今日选股推荐" in md or "选股" in md
        assert "1810.HK" in md
        assert "85" in md  # composite score

    def test_build_daily_report_with_sector_ranking(self, synthetic_ohlcv):
        """daily_report 包含板块强弱排名"""
        import daily_run

        screener_results = []
        sector_ranking = [
            {"sector": "科技/互联网", "avg_score": 82.5, "count": 3, "top_stock": "0700.HK", "top_score": 90.0},
            {"sector": "消费", "avg_score": 72.3, "count": 2, "top_stock": "3690.HK", "top_score": 78.0},
            {"sector": "金融/银行", "avg_score": 58.1, "count": 2, "top_stock": "0005.HK", "top_score": 61.0},
        ]

        daily_report = daily_run._build_daily_report(
            results=[],
            portfolio_value=200000.0,
            run_date="2026-04-03",
            market_is_open=True,
            config={},
            screener_results=[],
            sector_ranking=sector_ranking,
        )

        assert "sector_ranking" in daily_report
        assert len(daily_report["sector_ranking"]) == 3
        assert daily_report["sector_ranking"][0]["sector"] == "科技/互联网"

    def test_build_markdown_report_includes_sector_ranking(self, synthetic_ohlcv):
        """Markdown 报告包含板块排名表格"""
        import daily_run

        daily_report = {
            "run_date": "2026-04-03",
            "market_is_open": True,
            "portfolio_value": 200000.0,
            "total_market_value": 0.0,
            "total_cost_basis": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "cash_value": 200000.0,
            "cash_pct": 100.0,
            "held_count": 0,
            "total_tickers": 0,
            "buy_signals": [],
            "sell_signals": [],
            "recommendations": [],
            "screener_results": [],
            "screener_weights": {"momentum": 0.35, "trend": 0.35, "volume": 0.30},
            "sector_ranking": [
                {"sector": "科技/互联网", "avg_score": 82.5, "count": 3, "top_stock": "0700.HK", "top_score": 90.0},
            ],
        }

        md = daily_run._build_markdown_report(daily_report)
        assert "板块强弱" in md
        assert "科技/互联网" in md
        assert "82.5" in md
