"""
yahooquery 数据供应商
====================
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor, generate_ticker_variants

logger = logging.getLogger(__name__)


class YahooQueryVendor(DataVendor):
    name = "yahooquery"

    def is_available(self) -> bool:
        try:
            from yahooquery import Ticker  # noqa: F401
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
            from yahooquery import Ticker as YQ
        except ImportError:
            logger.info("yahooquery 未安装，跳过")
            return None

        variants = generate_ticker_variants(ticker)
        for variant in variants:
            df = self._fetch_one(YQ, variant, start, end, timeout)
            if df is not None and not df.empty:
                logger.info(f"yahooquery 获取 {variant} 成功 ({len(df)} 行)")
                return df.sort_index()
        return None

    def _fetch_one(self, YQ, ticker: str, start: datetime, end: datetime, timeout: int = 20) -> Optional[pd.DataFrame]:
        def _do_fetch():
            yq = YQ(ticker)
            hist = yq.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if hist is None or (hasattr(hist, "empty") and hist.empty):
                return None

            if isinstance(hist.index, pd.MultiIndex):
                try:
                    df = hist.xs(ticker, level=0)
                except KeyError:
                    base = ticker.split(".")[0]
                    lvl0 = hist.index.get_level_values(0)
                    if base in lvl0:
                        df = hist.xs(base, level=0)
                    else:
                        df = hist.droplevel(0)
            else:
                df = hist

            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
            return df

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_fetch)
                return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.info(f"yahooquery {ticker} 下载超时 ({timeout}s)")
        except Exception as e:
            logger.info(f"yahooquery {ticker} 失败: {e}")
        return None

