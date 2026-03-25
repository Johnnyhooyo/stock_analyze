"""
TickFlow 数据供应商
==================
免费版 TickFlow API (无需注册/Key)，支持日K线数据。
"""

from __future__ import annotations

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
        code = parts[0].zfill(5)  # 补零到5位
        return f"{code}.{parts[1].upper()}"
    else:
        # 无后缀，假定港股
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

        tf = TickFlow.free()

        # 计算所需数据天数
        days = parse_period_to_days(period)
        # 加一些缓冲余量
        count = min(days + 10, 5000)

        # 转换 ticker 格式
        tf_ticker = _normalize_ticker(ticker)

        try:
            df = tf.klines.get(tf_ticker, period="1d", count=count, as_dataframe=True)
        except Exception as e:
            logger.info(f"TickFlow 获取 {tf_ticker} 失败: {e}")
            return None

        if df is None or df.empty:
            return None

        # 重命名列以匹配标准 OHLCV schema
        # TickFlow 返回: symbol, name, timestamp, trade_date, trade_time, open, high, low, close, volume, amount
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        # 解析 trade_date 为 datetime index
        if "trade_date" in df.columns:
            df.index = pd.to_datetime(df["trade_date"])
            df.index.name = "date"
            df = df.drop(columns=["trade_date"], errors="ignore")

        # 删除无关列
        drop_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume", "Adj Close")]
        df = df.drop(columns=drop_cols, errors="ignore")

        # 按日期排序
        df = df.sort_index()

        # 根据 start/end 截取
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]

        logger.info(f"TickFlow 获取 {ticker} 成功 ({len(df)} 行)")
        return df
