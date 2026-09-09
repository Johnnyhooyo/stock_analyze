import pytest

from engine.hk_fees import affordable_hk_shares, calculate_hk_stock_fees


def test_complete_hk_stock_fee_breakdown_for_both_sides():
    config = {"hk_trading_fees": {
        "broker_commission_rate": 0.0008,
        "broker_commission_min": 0,
        "platform_fee": 0,
    }}
    fee = calculate_hk_stock_fees(100_000, "0700.HK", config)

    assert fee.stamp_duty == 100.0
    assert fee.sfc_levy == 2.70
    assert fee.afrc_levy == 0.15
    assert fee.hkex_trading_fee == 5.65
    assert fee.settlement_fee == 4.20
    assert fee.commission == 80.0
    assert fee.total == 192.70


def test_stamp_duty_rounds_up_and_supports_explicit_exemption():
    assert calculate_hk_stock_fees(100.01, "0005.HK").stamp_duty == 1.0
    config = {"hk_trading_fees": {"stamp_duty_exempt_tickers": ["3200.HK"]}}
    assert calculate_hk_stock_fees(100_000, "3200.HK", config).stamp_duty == 0.0


def test_affordable_shares_includes_all_buy_fees():
    config = {"hk_trading_fees": {"broker_commission_rate": 0}}
    shares = affordable_hk_shares(10_000, 10, "0700.HK", config)
    fee = calculate_hk_stock_fees(shares * 10, "0700.HK", config)
    assert shares * 10 + fee.total <= 10_000
    next_fee = calculate_hk_stock_fees((shares + 1) * 10, "0700.HK", config)
    assert (shares + 1) * 10 + next_fee.total > 10_000
