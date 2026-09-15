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
    assert trades[0].avg_cost == pytest.approx(500.0)
    assert state.get_position("0700.HK").shares == 0
    assert state.get_position("0005.HK").shares > 0
    assert trades[0].fee_breakdown["stamp_duty"] > 0
    assert trades[0].fee == pytest.approx(sum(
        value for key, value in trades[0].fee_breakdown.items()
        if key not in ("gross_amount", "total")
    ))
    assert state.realized_pnl == pytest.approx(1000 - trades[0].fee)


def test_buy_uses_ticker_specific_board_lot(tmp_path):
    config = _config()
    config["board_lots"] = {
        "enabled": True,
        "required": True,
        "overrides": {"0700.HK": 100},
    }
    state = PortfolioState(portfolio_value=100_000, cash=100_000, initial_capital=100_000)
    result = _result("0700.HK", "买入", 500.0)
    engine = PaperTradingEngine(
        config,
        orders_file=tmp_path / "orders.jsonl",
        assets_file=tmp_path / "assets.jsonl",
    )

    trades = engine.execute([result], state, "2026-09-09")

    assert trades == []
    assert result.action == "观望"
    assert state.get_position("0700.HK") is None


def test_buy_quantity_is_a_whole_number_of_board_lots(tmp_path):
    config = _config()
    config["board_lots"] = {
        "enabled": True,
        "required": True,
        "overrides": {"0005.HK": 400},
    }
    state = PortfolioState(portfolio_value=500_000, cash=500_000, initial_capital=500_000)
    result = _result("0005.HK", "买入", 160.0)
    engine = PaperTradingEngine(
        config,
        orders_file=tmp_path / "orders.jsonl",
        assets_file=tmp_path / "assets.jsonl",
    )

    trades = engine.execute([result], state, "2026-09-09")

    assert len(trades) == 1
    assert trades[0].shares == 400
    assert trades[0].shares % 400 == 0
