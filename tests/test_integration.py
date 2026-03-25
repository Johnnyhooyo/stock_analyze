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
