"""
pandas_datareader 数据供应商
============================
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor

logger = logging.getLogger(__name__)


class PandasDataReaderVendor(DataVendor):
    name = "pandas_datareader"

    def is_available(self) -> bool:
        try:
            from pandas_datareader import data as pdr  # noqa: F401
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
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            logger.info("pandas_datareader 未安装，跳过")
            return None

        sources = ["yahoo", "stooq"]
        for source in sources:
            try:
                logger.info(f"pandas_datareader/{source} 下载 {ticker}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(pdr.DataReader, ticker, source, start=start, end=end)
                    df = fut.result(timeout=timeout)
                if df is not None and not df.empty:
                    logger.info(f"pandas_datareader/{source} 获取 {ticker} 成功 ({len(df)} 行)")
                    return df.sort_index()
            except concurrent.futures.TimeoutError:
                logger.info(f"pandas_datareader/{source} 超时 ({timeout}s)")
            except Exception as e:
                logger.info(f"pandas_datareader/{source} 失败: {e}")
        return None

