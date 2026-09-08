from types import SimpleNamespace

import pytest

from engine.paper_trader import PaperTradingEngine
from engine.portfolio_state import PortfolioState


def _result(ticker, action, price, confidence=0.8):
    return SimpleNamespace(
        ticker=ticker,
        action=action,
        last_close=price,
        confidence_pct=confidence,
        reason="test signal",
        signal=1 if action == "买入" else 0,
        has_position=False,
        shares=0,
        avg_cost=0.0,
        market_value=0.0,
        profit=0.0,
        profit_pct=0.0,
        peak_price=price,
        kelly_shares=0,
        kelly_amount=0.0,
    )


def _config():
    return {
        "fees_rate": 0.001,
        "stamp_duty": 0.001,
        "slippage": 0.0,
        "risk_management": {"max_position_pct": 0.25},
        "portfolio_risk": {"max_position_ratio": 0.80},
        "paper_trading": {
            "enabled": True,
            "min_confidence": 0.55,
            "max_new_positions_per_day": 3,
            "reserve_cash_pct": 0.05,
            "lot_size": 1,
        },
    }


def test_buys_are_limited_by_cash_and_position_rules(tmp_path):
    state = PortfolioState(portfolio_value=100_000, cash=100_000, initial_capital=100_000)
    results = [
        _result("0001.HK", "买入", 100, 0.90),
        _result("0002.HK", "买入", 50, 0.80),
        _result("0003.HK", "买入", 20, 0.70),
        _result("0004.HK", "买入", 10, 0.60),
    ]
    engine = PaperTradingEngine(
        _config(),
        orders_file=tmp_path / "orders.jsonl",
        assets_file=tmp_path / "assets.jsonl",
    )
    trades = engine.execute(results, state, "2026-09-04")

    assert len(trades) == 3
    assert len(state.held_tickers()) == 3
    assert state.cash >= 5_000
    assert sum(p.shares * results[i].last_close for i, p in enumerate(
        state.positions[t] for t in state.held_tickers()
    )) <= 80_000
    assert (tmp_path / "assets.jsonl").exists()


def test_sells_before_buying_and_sells_full_position(tmp_path):
    state = PortfolioState(portfolio_value=100_000, cash=50_000, initial_capital=100_000)
    state.update_position("0700.HK", shares=100, avg_cost=500.0, peak_price=520.0)
    sell = _result("0700.HK", "卖出", 510.0)
    sell.has_position = True
    sell.shares = 100
    sell.avg_cost = 500.0
    buy = _result("0005.HK", "买入", 50.0)

    engine = PaperTradingEngine(
        _config(),
        orders_file=tmp_path / "orders.jsonl",
        assets_file=tmp_path / "assets.jsonl",
    )
    trades = engine.execute([buy, sell], state, "2026-09-04")

    assert [t.action for t in trades] == ["卖出", "买入"]
    assert trades[0].shares == 100
    assert state.get_position("0700.HK").shares == 0
    assert state.get_position("0005.HK").shares > 0
    assert state.realized_pnl == pytest.approx(1000 - 51000 * 0.002)
