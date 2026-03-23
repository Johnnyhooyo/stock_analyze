"""
数据供应商抽象基类
=================
所有数据源适配器继承此基类，实现 fetch() 方法。
fetch_with_retry() 模板方法统一提供重试/退避逻辑。
"""

from __future__ import annotations

import logging
import re
import time as _time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parse_period_to_days(period: str) -> int:
    """将 '1y', '6mo', '30d' 等周期字符串转换为天数。"""
    if not isinstance(period, str):
        return 90
    m = re.match(r"(\d+)y", period)
    if m:
        return int(m.group(1)) * 365
    m = re.match(r"(\d+)mo", period)
    if m:
        return int(m.group(1)) * 30
    m = re.match(r"(\d+)d", period)
    if m:
        return int(m.group(1))
    return 90


def generate_ticker_variants(ticker: str) -> list[str]:
    """
    生成 ticker 变体列表用于重试。
    例: '0700' → ['0700', '0700.HK']
        '0700.HK' → ['0700.HK', '0700']
    """
    variants: list[str] = [ticker]
    if not isinstance(ticker, str):
        return variants
    if "." in ticker:
        base = ticker.split(".")[0]
        variants.append(base)
    else:
        variants.append(f"{ticker}.HK")
        if ticker.isdigit() and len(ticker) < 4:
            padded = ticker.zfill(4)
            variants.append(padded)
            variants.append(f"{padded}.HK")
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class DataVendor(ABC):
    """数据供应商抽象基类。"""

    # 子类必须设置
    name: str = "base"

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        period: str = "1y",
        timeout: int = 20,
    ) -> Optional[pd.DataFrame]:
        """
        从数据源获取 OHLCV 数据（单次请求，不含重试逻辑）。

        Args:
            ticker: 股票代码 (如 '0700.HK')
            start: 开始日期
            end: 结束日期
            period: 周期字符串 (部分数据源需要)
            timeout: 超时秒数

        Returns:
            标准化后的 DataFrame，失败返回 None
        """
        ...

    def fetch_with_retry(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        period: str = "1y",
        timeout: int = 20,
        max_attempts: int = 3,
        backoff: float = 1.0,
    ) -> Optional[pd.DataFrame]:
        """
        带重试/退避 + 速率限制的统一下载入口（模板方法）。

        所有 vendor 的 fetch() 只需实现单次请求逻辑，
        重试 / 指数退避 / rate limiting 由此方法统一处理。

        Args:
            max_attempts: 最大尝试次数
            backoff: 退避基数（秒），实际等待 = backoff * 2^attempt
        """
        # 获取 per-vendor 限速器
        try:
            from data.rate_limiter import get_limiter
            limiter = get_limiter(self.name)
        except Exception:
            limiter = None

        last_err: Optional[Exception] = None
        for attempt in range(max_attempts):
            # 速率限制
            if limiter is not None:
                limiter.acquire(timeout=60)

            try:
                df = self.fetch(
                    ticker, start, end,
                    period=period,
                    timeout=timeout,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_err = e
                logger.info(
                    f"{self.name} {ticker} 尝试 {attempt + 1}/{max_attempts} 失败: {e}"
                )
            if attempt < max_attempts - 1:
                wait = backoff * (2 ** attempt)
                logger.debug(f"{self.name} 等待 {wait:.1f}s 后重试…")
                _time.sleep(wait)

        if last_err:
            logger.info(f"{self.name} {ticker} 重试耗尽: {last_err}")
        return None

    def is_available(self) -> bool:
        """检查该数据源的依赖是否可用。"""
        return True

    def __repr__(self) -> str:
        return f"<DataVendor: {self.name}>"

