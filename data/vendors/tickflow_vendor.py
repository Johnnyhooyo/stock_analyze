"""
TickFlow 数据供应商
=================
免费版 TickFlow API (无需注册/Key)，支持日K线数据。
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor, parse_period_to_days

logger = logging.getLogger(__name__)


def _normalize_ticker(ticker: str) -> str:
    """
    将 ticker 转换为 TickFlow 所需格式。
    '0700.HK' / '700.HK' → '00700.HK'
    """
    if not isinstance(ticker, str):
        return ticker
    if "." in ticker:
        parts = ticker.split(".")
        code = parts[0].zfill(5)
        return f"{code}.{parts[1].upper()}"
    else:
        return f"{ticker.zfill(5)}.HK"


class TickFlowVendor(DataVendor):
    name = "tickflow"

    def is_available(self) -> bool:
        try:
            import tickflow  # noqa: F401
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
            from tickflow import TickFlow
        except ImportError:
            logger.info("tickflow 未安装，跳过")
            return None

        days = parse_period_to_days(period)
        count = min(days + 10, 5000)
        tf_ticker = _normalize_ticker(ticker)

        def _do_fetch():
            tf = TickFlow.free()
            return tf.klines.get(tf_ticker, period="1d", count=count, as_dataframe=True)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_fetch)
                df = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.info(f"TickFlow {tf_ticker} 下载超时 ({timeout}s)")
            return None
        except Exception as e:
            logger.info(f"TickFlow 获取 {tf_ticker} 失败: {e}")
            return None

        if df is None or df.empty:
            return None

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        if "trade_date" in df.columns:
            df.index = pd.to_datetime(df["trade_date"])
            df.index.name = "date"
            df = df.drop(columns=["trade_date"], errors="ignore")

        drop_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume", "Adj Close")]
        df = df.drop(columns=drop_cols, errors="ignore")

        df = df.sort_index()

        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]

        logger.info(f"TickFlow 获取 {ticker} 成功 ({len(df)} 行)")
        return df
