"""
Alpha Vantage 数据供应商
========================
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor

logger = logging.getLogger(__name__)


class AlphaVantageVendor(DataVendor):
    name = "alpha_vantage"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            from alpha_vantage.timeseries import TimeSeries  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        period: str = "1y",
        timeout: int = 20,
    ) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.info("Alpha Vantage 未提供 API key，跳过")
            return None
        try:
            from alpha_vantage.timeseries import TimeSeries
        except ImportError:
            logger.info("alpha_vantage 库未安装，跳过")
            return None

        def _do_fetch():
            logger.info(f"Alpha Vantage 下载 {ticker}")
            ts = TimeSeries(key=self.api_key, output_format="pandas")
            data, meta = ts.get_daily_adjusted(symbol=ticker, outputsize="compact")
            data = data.rename(columns=lambda s: s.split(". ")[1] if ". " in s else s)
            data.index = pd.to_datetime(data.index)
            return data.sort_index()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_fetch)
                return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.info(f"Alpha Vantage {ticker} 下载超时 ({timeout}s)")
        except Exception as e:
            logger.info(f"Alpha Vantage 下载失败: {e}")
        return None

