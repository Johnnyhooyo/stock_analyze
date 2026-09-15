from data.hk_board_lots import normalize_hk_ticker
from engine.hk_fees import affordable_hk_shares


def test_normalize_hkex_code_to_project_ticker():
    assert normalize_hk_ticker("00001") == "0001.HK"
    assert normalize_hk_ticker("00700") == "0700.HK"
    assert normalize_hk_ticker("09988") == "9988.HK"


def test_affordable_shares_obeys_configured_board_lot_override():
    config = {
        "board_lots": {
            "enabled": True,
            "required": True,
            "overrides": {"0700.HK": 100},
        }
    }
    shares = affordable_hk_shares(55_000, 500, "0700.HK", config)
    assert shares == 100


def test_affordable_shares_returns_zero_when_one_lot_is_unaffordable():
    config = {
        "board_lots": {
            "enabled": True,
            "required": True,
            "overrides": {"0700.HK": 100},
        }
    }
    assert affordable_hk_shares(49_999, 500, "0700.HK", config) == 0
