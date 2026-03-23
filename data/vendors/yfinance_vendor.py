"""
yfinance 数据供应商
===================
单次请求逻辑，重试由基类 fetch_with_retry() 统一处理。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor, generate_ticker_variants

logger = logging.getLogger(__name__)


class YFinanceVendor(DataVendor):
    name = "yfinance"

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
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
            import yfinance as yf
        except ImportError:
            logger.info("yfinance 未安装，跳过")
            return None

        variants = generate_ticker_variants(ticker)
        for variant in variants:
            df = self._single_download(yf, variant, start, end)
            if df is not None and not df.empty:
                logger.info(f"yfinance 获取 {variant} 成功 ({len(df)} 行)")
                return df.sort_index()
        return None

    @staticmethod
    def _single_download(yf, ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """单次下载尝试。"""
        try:
            df = yf.download(
                ticker,
                start=start.date() if hasattr(start, 'date') else start,
                end=end.date() if hasattr(end, 'date') else end,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        # 回退: yf.Ticker.history
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return None

