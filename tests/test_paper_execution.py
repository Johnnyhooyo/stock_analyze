"""纸面卖出须按信号后的开盘价结算。"""

import json

import pandas as pd
import pytest

from engine.paper_execution import cancel_pending_buys, queue_paper_signals, settle_paper_signals
from engine.paper_trader import PaperTradingEngine
from engine.portfolio_state import PortfolioState
from engine.position_analyzer import RecommendationResult


class _DataManager:
    def __init__(self, data):
        self.data = data

    def load(self, ticker, period):
        return self.data


def _data(with_next_bar):
    rows = [{"Open": 99.0, "High": 101.0, "Low": 98.0, "Close": 100.0}]
    dates = ["2026-01-05"]
    if with_next_bar:
        rows.append({"Open": 110.0, "High": 112.0, "Low": 109.0, "Close": 111.0})
        dates.append("2026-01-06")
    return pd.DataFrame(rows, index=pd.to_datetime(dates))


def test_sell_waits_for_next_bar_and_uses_its_open(tmp_path):
    config = {"slippage": 0.001, "paper_trading": {"enabled": True}}
    state = PortfolioState(
        portfolio_value=100_000, cash=90_000, initial_capital=100_000,
        path=tmp_path / "portfolio.yaml",
    )
    state.update_position("0700.HK", shares=100, avg_cost=100.0, peak_price=100.0)
    pending = tmp_path / "pending.json"
    result = RecommendationResult("0700.HK", "2026-01-05", 100.0,
                                  action="卖出", confidence_pct=1.0)
    assert queue_paper_signals([result], pending) == 1
    assert queue_paper_signals([result], pending) == 0
    trader = PaperTradingEngine(
        config,
        orders_file=tmp_path / "orders.jsonl",
        assets_file=tmp_path / "assets.jsonl",
    )

    assert settle_paper_signals(config, state, _DataManager(_data(False)), "5y", pending, trader) == []
    assert state.get_position("0700.HK").shares == 100
    trades = settle_paper_signals(config, state, _DataManager(_data(True)), "5y", pending, trader)

    assert len(trades) == 1
    assert trades[0].trade_date == "2026-01-06"
    assert trades[0].price == pytest.approx(110.0 * (1 - 0.001))
    assert state.get_position("0700.HK").shares == 0
    assert json.loads(pending.read_text(encoding="utf-8")) == []


def test_portfolio_stop_cancels_queued_buys_but_keeps_sells(tmp_path):
    path = tmp_path / "pending.json"
    buy = RecommendationResult("0005.HK", "2026-01-05", 50.0,
                               action="买入", confidence_pct=0.8)
    sell = RecommendationResult("0700.HK", "2026-01-05", 100.0,
                                action="止损卖出", confidence_pct=0.0)
    queue_paper_signals([buy, sell], path)

    assert cancel_pending_buys(path) == 1
    remaining = json.loads(path.read_text(encoding="utf-8"))
    assert [(item["ticker"], item["action"]) for item in remaining] == [
        ("0700.HK", "止损卖出")
    ]
