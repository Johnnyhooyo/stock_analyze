"""每日建议只能使用预期交易日的已确认 K 线。"""

from datetime import date

import pandas as pd

from daily_run import _analyze_one_ticker


class _Manager:
    def __init__(self, data):
        self.data = data
        self.downloads = 0

    def download(self, ticker, period):
        self.downloads += 1
        return self.data, None


class _Analyzer:
    def __init__(self):
        self.seen = None

    def analyze(self, ticker, data, position):
        self.seen = data
        return ticker


class _Portfolio:
    def get_position(self, ticker):
        return None


def _bars(*dates):
    return pd.DataFrame({"Close": [100.0] * len(dates)}, index=pd.to_datetime(dates))


def test_stale_cache_does_not_produce_recommendation():
    manager = _Manager(_bars("2026-01-05"))
    analyzer = _Analyzer()
    result = _analyze_one_ticker("0700.HK", {}, analyzer, _Portfolio(), manager,
                                 expected_bar_date=date(2026, 1, 6))
    assert result is None
    assert manager.downloads == 1
    assert analyzer.seen is None


def test_future_intraday_bar_is_excluded():
    manager = _Manager(_bars("2026-01-05", "2026-01-06", "2026-01-07"))
    analyzer = _Analyzer()
    price_data = {}
    result = _analyze_one_ticker("0700.HK", {}, analyzer, _Portfolio(), manager,
                                 expected_bar_date=date(2026, 1, 6), price_data=price_data)
    assert result == "0700.HK"
    assert analyzer.seen.index.max().date() == date(2026, 1, 6)
    assert price_data["0700.HK"].index.max().date() == date(2026, 1, 6)
