"""
tests/test_portfolio_state.py — PortfolioState & PortfolioPosition tests
"""
import yaml
from pathlib import Path

import pytest

from engine.portfolio_state import (
    PortfolioPosition,
    PortfolioState,
    load_portfolio,
    holding_days,
)


class TestPortfolioPosition:
    def test_has_position_true(self):
        pos = PortfolioPosition(ticker="0700.HK", shares=200, avg_cost=380.0)
        assert pos.has_position is True

    def test_has_position_false_zero_shares(self):
        pos = PortfolioPosition(ticker="0700.HK", shares=0, avg_cost=380.0)
        assert pos.has_position is False

    def test_has_position_false_zero_cost(self):
        pos = PortfolioPosition(ticker="0700.HK", shares=200, avg_cost=0.0)
        assert pos.has_position is False

    def test_to_dict(self):
        pos = PortfolioPosition(
            ticker="0700.HK",
            shares=200,
            avg_cost=380.0,
            peak_price=400.0,
            consecutive_loss_days=2,
            trailing_peak=395.0,
        )
        d = pos.to_dict()
        assert d["shares"] == 200
        assert d["avg_cost"] == 380.0
        assert d["peak_price"] == 400.0
        assert d["consecutive_loss_days"] == 2

    def test_holding_days_counts_entry_as_day_one(self):
        assert holding_days("2026-09-14", "2026-09-15") == 2
        assert holding_days("", "2026-09-15") == 0


class TestPortfolioStateSaveLoad:
    def test_save_load_round_trip(self, tmp_path):
        state = PortfolioState(portfolio_value=200_000.0, path=tmp_path / "portfolio.yaml")
        state.update_position(
            "0700.HK",
            shares=200,
            avg_cost=380.0,
            peak_price=400.0,
            consecutive_loss_days=1,
        )
        state.save()

        loaded = load_portfolio(tmp_path / "portfolio.yaml")
        assert "0700.HK" in loaded.positions
        assert loaded.positions["0700.HK"].shares == 200
        assert loaded.positions["0700.HK"].avg_cost == 380.0

    def test_entry_date_round_trip(self, tmp_path):
        path = tmp_path / "portfolio.yaml"
        state = PortfolioState(path=path)
        state.buy("0700.HK", 100, 400.0, trade_date="2026-09-14")
        state.save()

        loaded = load_portfolio(path)
        assert loaded.get_position("0700.HK").entry_date == "2026-09-14"

    def test_load_portfolio_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        state = load_portfolio(missing)
        assert state.positions == {}
        assert state.path == missing

    def test_load_portfolio_ticker_key_underscore_conversion(self, tmp_path):
        """portfolio.yaml stores keys as "0700_HK" but should load as "0700.HK"."""
        yaml_content = {
            "portfolio_value": 200_000.0,
            "positions": {
                "0700_HK": {
                    "ticker": None,
                    "shares": 100,
                    "avg_cost": 350.0,
                    "peak_price": 0.0,
                    "consecutive_loss_days": 0,
                }
            },
        }
        f = tmp_path / "portfolio.yaml"
        f.write_text(yaml.dump(yaml_content, allow_unicode=True), encoding="utf-8")

        state = load_portfolio(f)
        assert "0700.HK" in state.positions
        assert state.positions["0700.HK"].shares == 100


class TestPortfolioStateUpdatePosition:
    def test_update_existing_position(self, tmp_path):
        state = PortfolioState(path=tmp_path / "p.yaml")
        state.update_position("0700.HK", shares=100, avg_cost=350.0)
        state.update_position("0700.HK", peak_price=400.0)
        assert state.positions["0700.HK"].shares == 100
        assert state.positions["0700.HK"].peak_price == 400.0

    def test_update_creates_if_missing(self, tmp_path):
        state = PortfolioState(path=tmp_path / "p.yaml")
        state.update_position("0700.HK", shares=50, avg_cost=400.0)
        assert "0700.HK" in state.positions
        assert state.positions["0700.HK"].shares == 50

    def test_get_position_case_insensitive(self, tmp_path):
        state = PortfolioState(path=tmp_path / "p.yaml")
        state.update_position("0700.HK", shares=100, avg_cost=350.0)
        assert state.get_position("0700.hk") is not None
        assert state.get_position("0700.HK") is not None

    def test_held_tickers(self, tmp_path):
        state = PortfolioState(path=tmp_path / "p.yaml")
        state.update_position("0700.HK", shares=100, avg_cost=350.0)
        state.update_position("0005.HK", shares=0, avg_cost=0.0)  # empty
        state.update_position("0011.HK", shares=50, avg_cost=100.0)
        held = state.held_tickers()
        assert "0700.HK" in held
        assert "0011.HK" in held
        assert "0005.HK" not in held


class TestPortfolioCashAccounting:
    def test_buy_sell_and_mark_to_market(self, tmp_path):
        state = PortfolioState(
            portfolio_value=100_000.0,
            cash=100_000.0,
            initial_capital=100_000.0,
            path=tmp_path / "p.yaml",
        )
        state.buy("0700.HK", 100, 400.0, fee=40.0, trade_date="2026-09-04")
        assert state.cash == pytest.approx(59_960.0)
        assert state.get_position("0700.HK").avg_cost == pytest.approx(400.0)
        assert state.get_position("0700.HK").buy_fees == pytest.approx(40.0)
        assert state.get_position("0700.HK").entry_date == "2026-09-04"

        state.mark_to_market({"0700.HK": 420.0}, "2026-09-04")
        assert state.portfolio_value == pytest.approx(101_960.0)

        realized = state.sell("0700.HK", 100, 420.0, fee=42.0)
        assert realized == pytest.approx(1_918.0)
        assert state.cash == pytest.approx(101_918.0)
        assert state.held_tickers() == []
        assert state.get_position("0700.HK").entry_date == ""

    def test_cash_fields_round_trip(self, tmp_path):
        path = tmp_path / "portfolio.yaml"
        state = PortfolioState(
            portfolio_value=101_000.0,
            cash=61_000.0,
            initial_capital=100_000.0,
            realized_pnl=500.0,
            last_valuation_date="2026-09-04",
            path=path,
        )
        state.save()
        loaded = load_portfolio(path)
        assert loaded.cash == 61_000.0
        assert loaded.initial_capital == 100_000.0
        assert loaded.realized_pnl == 500.0
        assert loaded.last_valuation_date == "2026-09-04"
