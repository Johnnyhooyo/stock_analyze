"""
tests/test_integration.py — End-to-end integration smoke tests
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

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


class TestFactorRegistryAggregateIntegration:
    """BUG-2: 因子注册表集成 — _save_factor → registry → aggregate 端到端验证。"""

    def _make_factor_file(self, factors_dir: Path, run_id: int = 1, ticker: str = "0700.HK") -> Path:
        """创建最小化因子文件，模拟训练后保存的 artifact。"""
        import joblib
        factors_dir.mkdir(parents=True, exist_ok=True)
        art = {
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.5,
            "config": {"ticker": ticker},
        }
        path = factors_dir / f"factor_{run_id:04d}.pkl"
        joblib.dump(art, path)
        return path

    def test_registry_registered_factor_is_loadable_by_aggregator(self, tmp_path, synthetic_ohlcv):
        """注册表中的因子能被 SignalAggregator 正确加载，不返回空信号。"""
        from data.factor_registry import FactorRegistry
        from engine.signal_aggregator import SignalAggregator

        factors_dir = tmp_path / "factors"
        factor_path = self._make_factor_file(factors_dir, run_id=1)

        registry = FactorRegistry(registry_path=factors_dir / "factor_registry.json")
        registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="ma_crossover",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.12,
            max_drawdown=-0.08,
            total_trades=10,
        )

        agg = SignalAggregator(factors_dir=factors_dir)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})

        assert result.total_strategies >= 1, "注册表过滤后因子不应为空"
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_per_ticker_subdir_factor_survives_registry_filter(self, tmp_path, synthetic_ohlcv):
        """per-ticker 子目录因子在注册表过滤后仍可被加载（subdir 匹配）。"""
        from data.factor_registry import FactorRegistry
        from engine.signal_aggregator import SignalAggregator

        factors_dir = tmp_path / "factors"
        ticker_dir = factors_dir / "0700_HK"
        self._make_factor_file(ticker_dir, run_id=1, ticker="0700.HK")

        registry = FactorRegistry(registry_path=factors_dir / "factor_registry.json")
        registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir="0700_HK",
            strategy_name="ma_crossover",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.12,
            max_drawdown=-0.08,
            total_trades=10,
        )

        agg = SignalAggregator(factors_dir=factors_dir)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})

        assert result.total_strategies >= 1, "per-ticker 注册表因子不应被过滤为空"

    def test_registry_filter_fallback_when_subdir_mismatch(self, tmp_path, synthetic_ohlcv):
        """registry 中 subdir 与磁盘不匹配时，fallback 返回全部磁盘因子（不返回空信号）。"""
        from data.factor_registry import FactorRegistry
        from engine.signal_aggregator import SignalAggregator

        factors_dir = tmp_path / "factors"
        self._make_factor_file(factors_dir, run_id=1)

        # 故意注册错误 subdir，模拟历史数据不一致
        registry = FactorRegistry(registry_path=factors_dir / "factor_registry.json")
        registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir="WRONG_SUBDIR",   # 不匹配
            strategy_name="ma_crossover",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.12,
            max_drawdown=-0.08,
            total_trades=10,
        )

        agg = SignalAggregator(factors_dir=factors_dir)
        # fallback 应触发：过滤后为空但磁盘有文件 → 回退使用全部磁盘因子
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})

        assert result.total_strategies >= 1, "fallback 应保证有因子参与投票，不返回空信号"

    def test_empty_registry_falls_back_to_disk_factors(self, tmp_path, synthetic_ohlcv):
        """空注册表（无 active 记录）时，回退加载磁盘上所有因子（向后兼容）。"""
        from engine.signal_aggregator import SignalAggregator

        factors_dir = tmp_path / "factors"
        self._make_factor_file(factors_dir, run_id=1)

        # 不注册任何因子 → 注册表为空
        agg = SignalAggregator(factors_dir=factors_dir)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})

        assert result.total_strategies >= 1, "空注册表不应导致空信号"


class TestMetaAggregatorEndToEnd:
    """Offline end-to-end: train → save → load → predict."""

    def _make_rule_artifact(self, strategy_name: str) -> dict:
        return {
            "meta": {"name": strategy_name, "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.2,
            "config": {},
        }

    def test_train_save_load_predict_pipeline(self, synthetic_ohlcv, tmp_path):
        """Full offline pipeline: train on synthetic data, save, load, predict."""
        from engine.meta_aggregator import MetaAggregator

        meta_dir = tmp_path / "meta"
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("atr_breakout"),
        ]

        ma = MetaAggregator(meta_dir=meta_dir)
        metrics = ma.train(
            ticker="0700.HK",
            data=synthetic_ohlcv,
            artifacts=artifacts,
            config={},
            n_splits=2,
            label_days=5,
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] > 0

        path = ma.save("0700.HK")
        assert path.exists()

        loaded = MetaAggregator.load("0700.HK", meta_dir)
        assert loaded is not None
        assert loaded._strategy_names == ma._strategy_names

        feat = loaded.build_feature_vector(
            {name: 1 for name in loaded._strategy_names}, synthetic_ohlcv
        )
        signal, proba = loaded.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0

    def test_signal_aggregator_stacking_e2e(self, synthetic_ohlcv, tmp_path):
        """SignalAggregator in stacking mode uses trained meta-model correctly."""
        import joblib
        from engine.meta_aggregator import MetaAggregator
        from engine.signal_aggregator import SignalAggregator, AggregatedSignal

        factors_dir = tmp_path / "factors"
        meta_dir = tmp_path / "meta"
        factors_dir.mkdir()
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": 1.5, "config": {},
        }, factors_dir / "factor_0001.pkl")

        ma = MetaAggregator(meta_dir=meta_dir)
        artifacts = [
            {"meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
             "model": None, "sharpe_ratio": 1.5, "config": {}},
            {"meta": {"name": "atr_breakout", "params": {}, "feat_cols": []},
             "model": None, "sharpe_ratio": 1.2, "config": {}},
        ]
        ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        ma.save("0700.HK")

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
