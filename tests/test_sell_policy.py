"""卖出决策中移动止损和低样本共识的回归测试。"""

import pandas as pd

from engine.portfolio_state import PortfolioPosition
from engine.position_analyzer import PositionAnalyzer
from engine.signal_aggregator import AggregatedSignal


def _bars(closes):
    index = pd.bdate_range("2026-01-01", periods=len(closes))
    close = pd.Series(closes, index=index)
    return pd.DataFrame({
        "Open": close,
        "High": close + 0.03,
        "Low": close - 0.03,
        "Close": close,
        "Volume": 1_000_000,
    }, index=index)


def _analyzer(monkeypatch, tmp_path, signal, total_strategies, confidence):
    import position_manager

    monkeypatch.setattr(position_manager, "_STATE_DIR", str(tmp_path))
    analyzer = PositionAnalyzer({
        "risk_management": {
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "min_sell_strategies": 3,
            "min_sell_confidence": 0.55,
        }
    }, enable_sentiment=False)
    analyzer._aggregator.aggregate = lambda *_: AggregatedSignal(
        ticker="TEST.HK",
        consensus_signal=signal,
        total_strategies=total_strategies,
        confidence_pct=confidence,
    )
    return analyzer


def test_peak_rises_and_next_decline_triggers_atr_stop(monkeypatch, tmp_path):
    analyzer = _analyzer(monkeypatch, tmp_path, signal=1, total_strategies=3, confidence=1.0)
    pos = PortfolioPosition(ticker="TEST.HK", shares=100, avg_cost=4.5, peak_price=4.5)
    rising = analyzer.analyze("TEST.HK", _bars([4.5] * 29 + [4.7]), pos)
    assert rising.action == "持有"
    assert rising.peak_price == 4.7

    pos.peak_price = rising.peak_price
    falling = analyzer.analyze("TEST.HK", _bars([4.5] * 28 + [4.7, 4.5]), pos)
    assert falling.stop_price > falling.last_close
    assert falling.action == "止损卖出"


def test_existing_position_recovers_peak_from_historical_closes(monkeypatch, tmp_path):
    analyzer = _analyzer(monkeypatch, tmp_path, signal=1, total_strategies=3, confidence=1.0)
    pos = PortfolioPosition(
        ticker="TEST.HK", shares=100, avg_cost=4.5,
        peak_price=4.5, entry_date="2026-01-01",
    )
    result = analyzer.analyze("TEST.HK", _bars([4.5] * 28 + [4.7, 4.5]), pos)
    assert result.peak_price == 4.7
    assert result.action == "止损卖出"


def test_one_bearish_factor_does_not_force_sell(monkeypatch, tmp_path):
    analyzer = _analyzer(monkeypatch, tmp_path, signal=0, total_strategies=1, confidence=1.0)
    pos = PortfolioPosition(ticker="TEST.HK", shares=100, avg_cost=4.5, peak_price=4.5)
    result = analyzer.analyze("TEST.HK", _bars([4.5] * 30), pos)
    assert result.action == "持有"
    assert "暂缓卖出" in result.reason
