"""
tests/test_pnl_tracker.py — PnLTracker unit tests
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from data.pnl_tracker import PnLTracker, _is_correct
from engine.position_analyzer import RecommendationResult
from engine.signal_aggregator import AggregatedSignal


class MockRecommenderResult:
    """Build a lightweight mock RecommendationResult for testing."""

    def __init__(
        self,
        ticker="0700.HK",
        action="持有",
        signal=1,
        confidence_pct=0.72,
        confidence_label="高",
        last_close=395.20,
        shares=200,
        avg_cost=380.0,
        has_position=True,
        stop_price=381.50,
        kelly_shares=220,
        profit=3040.0,
        profit_pct=4.0,
        agg_signal=None,
    ):
        self.ticker = ticker
        self.action = action
        self.signal = signal
        self.confidence_pct = confidence_pct
        self.confidence_label = confidence_label
        self.last_close = last_close
        self.shares = shares
        self.avg_cost = avg_cost
        self.has_position = has_position
        self.stop_price = stop_price
        self.kelly_shares = kelly_shares
        self.profit = profit
        self.profit_pct = profit_pct
        self.agg_signal = agg_signal


class TestIsCorrect:
    def test_buy_correct_when_up(self):
        assert _is_correct("买入", 0.02) is True

    def test_buy_incorrect_when_down(self):
        assert _is_correct("买入", -0.01) is False

    def test_hold_correct_when_up(self):
        assert _is_correct("持有", 0.015) is True

    def test_hold_incorrect_when_down(self):
        assert _is_correct("持有", -0.03) is False

    def test_sell_correct_when_down(self):
        assert _is_correct("卖出", -0.025) is True

    def test_sell_incorrect_when_up(self):
        assert _is_correct("卖出", 0.01) is False

    def test_stop_loss_correct_when_down(self):
        assert _is_correct("止损卖出", -0.04) is True

    def test_stop_loss_incorrect_when_up(self):
        assert _is_correct("止损卖出", 0.02) is False

    def test_watch_correct_when_flat(self):
        assert _is_correct("观望", 0.005) is True

    def test_watch_correct_when_small_down(self):
        assert _is_correct("观望", -0.015) is True

    def test_watch_incorrect_when_big_down(self):
        assert _is_correct("观望", -0.03) is False

    def test_circuit_watch_correct_when_flat(self):
        assert _is_correct("熔断观望", 0.001) is True

    def test_circuit_watch_incorrect_when_big_down(self):
        assert _is_correct("熔断观望", -0.025) is False

    def test_unknown_action_always_false(self):
        assert _is_correct("未知", 0.05) is False


class TestPnLTrackerRecordDaily:
    def test_record_daily_idempotent(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", action="买入", last_close=400.0),
        ]
        n1 = tracker.record_daily("2026-04-03", recs)
        n2 = tracker.record_daily("2026-04-03", recs)
        assert n1 == 1
        assert n2 == 0
        lines = (tmp_path / "pnl.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_record_daily_multiple_tickers(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK"),
            MockRecommenderResult(ticker="0005.HK"),
        ]
        n = tracker.record_daily("2026-04-03", recs)
        assert n == 2
        lines = (tmp_path / "pnl.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_record_daily_with_agg_signal(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        agg = AggregatedSignal(
            ticker="0700.HK",
            consensus_signal=1,
            bullish_count=8,
            bearish_count=3,
            total_strategies=11,
            confidence_pct=0.72,
            top_strategy="macd_rsi_trend",
            top_strategy_sharpe=1.85,
            raw_votes=[
                {"strategy_name": "macd_rsi_trend", "signal": 1, "weight": 1.5, "sharpe": 1.5, "is_ml": False},
                {"strategy_name": "rsi_reversion", "signal": 1, "weight": 1.2, "sharpe": 1.2, "is_ml": False},
                {"strategy_name": "bollinger_rsi_trend", "signal": 0, "weight": 1.0, "sharpe": 1.0, "is_ml": False},
                {"strategy_name": "xgboost_enhanced", "signal": 1, "weight": 2.0, "sharpe": 2.0, "is_ml": True},
            ],
        )
        recs = [MockRecommenderResult(ticker="0700.HK", agg_signal=agg)]
        tracker.record_daily("2026-04-03", recs)
        row = json.loads((tmp_path / "pnl.jsonl").read_text().splitlines()[0])
        assert row["ticker"] == "0700.HK"
        assert row["action"] == "持有"
        assert row["bullish_count"] == 8
        assert row["bearish_count"] == 3
        assert row["total_strategies"] == 11
        assert row["top_strategy"] == "macd_rsi_trend"
        assert row["top_strategy_sharpe"] == 1.85
        assert row["strategy_votes"]["macd_rsi_trend"] == 1
        assert row["strategy_votes"]["bollinger_rsi_trend"] == 0

    def test_record_daily_creates_parent_dir(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "subdir" / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK")]
        tracker.record_daily("2026-04-03", recs)
        assert (tmp_path / "subdir" / "pnl.jsonl").exists()


class TestPnLTrackerFillT1Returns:
    def test_fill_t1_returns_basic(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
        ]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0700.HK": 408.0}
        n = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n == 1
        rows = tracker._read_all()
        assert rows[0]["t1_close"] == 408.0
        assert rows[0]["t1_return_pct"] == pytest.approx(2.0, rel=1e-3)
        assert rows[0]["t1_correct"] is True

    def test_fill_t1_returns_buy_wrong_direction(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
        ]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0700.HK": 392.0}
        n = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n == 1
        assert tracker._read_all()[0]["t1_correct"] is False

    def test_fill_t1_returns_multiple_tickers(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
            MockRecommenderResult(ticker="0005.HK", last_close=600.0, action="卖出"),
        ]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0700.HK": 408.0, "0005.HK": 588.0}
        n = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n == 2
        rows = {r["ticker"]: r for r in tracker._read_all()}
        assert rows["0700.HK"]["t1_correct"] is True
        assert rows["0005.HK"]["t1_correct"] is True

    def test_fill_t1_returns_skip_already_filled(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入")]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0700.HK": 408.0}
        tracker.fill_t1_returns("2026-04-03", price_map)
        n2 = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n2 == 0

    def test_fill_t1_returns_unknown_ticker(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=400.0)]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0005.HK": 608.0}
        n = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n == 0

    def test_fill_t1_returns_zero_last_close_skips(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=0.0)]
        tracker.record_daily("2026-04-03", recs)
        price_map = {"0700.HK": 400.0}
        n = tracker.fill_t1_returns("2026-04-03", price_map)
        assert n == 0
        rows = tracker._read_all()
        assert rows[0]["t1_close"] is None


class TestPnLTrackerAttribution:
    def test_attribution_by_strategy(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(
                ticker="0700.HK",
                last_close=400.0,
                action="买入",
                agg_signal=AggregatedSignal(
                    ticker="0700.HK",
                    consensus_signal=1,
                    bullish_count=1,
                    bearish_count=0,
                    total_strategies=1,
                    raw_votes=[{"strategy_name": "macd_rsi_trend", "signal": 1, "weight": 1.0, "sharpe": 1.0, "is_ml": False}],
                ),
            ),
        ]
        tracker.record_daily("2026-04-03", recs)
        tracker.fill_t1_returns("2026-04-03", {"0700.HK": 408.0})
        result = tracker.attribution_by_strategy(period_days=30)
        assert not result.empty
        assert "strategy" in result.columns
        assert "accuracy_pct" in result.columns

    def test_attribution_by_ticker(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
            MockRecommenderResult(ticker="0005.HK", last_close=600.0, action="卖出"),
        ]
        tracker.record_daily("2026-04-03", recs)
        tracker.fill_t1_returns("2026-04-03", {"0700.HK": 408.0, "0005.HK": 588.0})
        result = tracker.attribution_by_ticker(period_days=30)
        assert not result.empty
        assert "ticker" in result.columns
        assert "avg_t1_return_pct" in result.columns

    def test_attribution_by_action(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
            MockRecommenderResult(ticker="0005.HK", last_close=600.0, action="卖出"),
        ]
        tracker.record_daily("2026-04-03", recs)
        tracker.fill_t1_returns("2026-04-03", {"0700.HK": 408.0, "0005.HK": 588.0})
        result = tracker.attribution_by_action(period_days=30)
        assert not result.empty
        assert "action" in result.columns
        assert "accuracy_pct" in result.columns


class TestPnLTrackerSummaryReport:
    def test_summary_report_empty(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        result = tracker.summary_report(period_days=30)
        assert result == {"error": "no data"}

    def test_summary_report_no_filled_records(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入")]
        tracker.record_daily("2026-04-03", recs)
        result = tracker.summary_report(period_days=30)
        assert result == {"total_recommendations": 0}

    def test_summary_report_with_data(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入"),
            MockRecommenderResult(ticker="0005.HK", last_close=600.0, action="买入"),
        ]
        tracker.record_daily("2026-04-03", recs)
        tracker.fill_t1_returns("2026-04-03", {"0700.HK": 408.0, "0005.HK": 588.0})
        result = tracker.summary_report(period_days=30)
        assert result["total_recommendations"] == 2
        assert "overall_accuracy_pct" in result
        assert "avg_t1_return_pct" in result


class TestPnLTrackerExportCsv:
    def test_export_csv(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=400.0, action="买入")]
        tracker.record_daily("2026-04-03", recs)
        csv_path = tracker.export_csv(output_path=tmp_path / "export.csv")
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8-sig")
        assert "date" in content
        assert "ticker" in content
        assert "action" in content


class TestPnLTrackerAtomicWrite:
    def test_atomic_write(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [MockRecommenderResult(ticker="0700.HK", last_close=400.0)]
        tracker.record_daily("2026-04-03", recs)
        tracker.fill_t1_returns("2026-04-03", {"0700.HK": 408.0})
        assert not (tmp_path / "pnl.tmp").exists()
        rows = tracker._read_all()
        assert len(rows) == 1
        assert rows[0]["t1_correct"] is True


class TestPnLTrackerLoadDf:
    def test_load_df_columns(self, tmp_path):
        tracker = PnLTracker(pnl_path=tmp_path / "pnl.jsonl")
        recs = [
            MockRecommenderResult(
                ticker="0700.HK",
                last_close=400.0,
                agg_signal=AggregatedSignal(
                    ticker="0700.HK",
                    consensus_signal=1,
                    bullish_count=5,
                    bearish_count=2,
                    total_strategies=7,
                    raw_votes=[],
                ),
            ),
        ]
        tracker.record_daily("2026-04-03", recs)
        df = tracker.load_df()
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "action" in df.columns
        assert "signal" in df.columns
        assert "confidence_pct" in df.columns
        assert "last_close" in df.columns
        assert "shares" in df.columns
        assert "avg_cost" in df.columns
        assert "has_position" in df.columns
        assert "kelly_shares" in df.columns
        assert "top_strategy" in df.columns
        assert "bullish_count" in df.columns
        assert "bearish_count" in df.columns
        assert "total_strategies" in df.columns
        assert "strategy_votes" in df.columns
        assert "t1_close" in df.columns
        assert "t1_return_pct" in df.columns
        assert "t1_correct" in df.columns