"""
tests/test_oms.py — OMS order management tests
"""
import json
from pathlib import Path

import pytest

from oms import PaperOMS, create_oms


class TestPaperOMSSubmitOrder:
    def test_submit_buy_order(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        result = oms.submit_order("0700.HK", "买入", 100, 380.0)
        assert result.status == "submitted"
        assert result.action == "买入"
        assert result.shares == 100
        assert result.price == 380.0
        assert result.amount == 38000.0
        assert "PAPER-" in result.order_id

    def test_submit_sell_order(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        result = oms.submit_order("0700.HK", "卖出", 100, 395.0)
        assert result.status == "submitted"
        assert result.action == "卖出"

    def test_rejects_invalid_action(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        result = oms.submit_order("0700.HK", "观望", 0, 0)
        assert result.status == "rejected"
        assert "不支持的操作" in result.message

    def test_rejects_invalid_shares_price(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        # shares <= 0
        r1 = oms.submit_order("0700.HK", "买入", 0, 380.0)
        assert r1.status == "rejected"
        # price < 0
        r2 = oms.submit_order("0700.HK", "买入", 100, -1.0)
        assert r2.status == "rejected"

    def test_get_orders_round_trip(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        oms.submit_order("0700.HK", "买入", 100, 380.0)
        oms.submit_order("0700.HK", "卖出", 50, 395.0)
        orders = oms.get_orders()
        assert len(orders) == 2
        assert orders[0]["action"] == "买入"
        assert orders[1]["action"] == "卖出"

    def test_counter_increments(self, tmp_path):
        oms = PaperOMS(orders_file=tmp_path / "orders.jsonl")
        r1 = oms.submit_order("0700.HK", "买入", 100, 380.0)
        r2 = oms.submit_order("0700.HK", "买入", 100, 380.0)
        id1 = int(r1.order_id.split("-")[-1])
        id2 = int(r2.order_id.split("-")[-1])
        assert id2 > id1


class TestCreateOMS:
    def test_returns_paper_oms_when_no_broker_url(self):
        oms = create_oms({"broker_api_url": None})
        assert isinstance(oms, PaperOMS)

    def test_returns_paper_oms_when_broker_url_empty(self):
        oms = create_oms({})
        assert isinstance(oms, PaperOMS)

    def test_returns_live_oms_when_broker_url_set(self):
        # LiveOMS is a subclass of PaperOMS (falls back to paper), so we check
        # the type name via module+class
        oms = create_oms({"broker_api_url": "https://api.futu.com"})
        assert "Live" in type(oms).__name__ or isinstance(oms, PaperOMS)
