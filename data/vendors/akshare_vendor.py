"""
akshare 数据供应商
=================
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.vendor_base import DataVendor
from data.schemas import _DATE_CANDIDATES

logger = logging.getLogger(__name__)


class AkShareVendor(DataVendor):
    name = "akshare"

    def is_available(self) -> bool:
        try:
            import akshare  # noqa: F401
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
            import akshare as ak
        except ImportError:
            logger.info("akshare 未安装，跳过")
            return None

        symbol = ticker.split(".")[0] if "." in ticker else ticker
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        def _do_fetch():
            logger.info(f"akshare 下载 {ticker} (symbol={symbol})")

            df = None
            is_hk = ticker.upper().endswith(".HK")
            ak_funcs = []
            if is_hk:
                ak_funcs = [
                    ("stock_hk_hist", {"symbol": symbol, "period": "daily", "start_date": start_str, "end_date": end_str, "adjust": "qfq"}),
                    ("stock_zh_a_hist", {"symbol": symbol, "period": "daily", "start_date": start_str, "end_date": end_str, "adjust": "qfq"}),
                ]
            else:
                ak_funcs = [
                    ("stock_zh_a_hist", {"symbol": symbol, "period": "daily", "start_date": start_str, "end_date": end_str, "adjust": "qfq"}),
                ]
            for func_name, kwargs in ak_funcs:
                fn = getattr(ak, func_name, None)
                if fn is None:
                    continue
                try:
                    df = fn(**kwargs)
                    if df is not None and not df.empty:
                        logger.info(f"ak.{func_name} 获取成功 ({len(df)} 行)")
                        break
                except Exception as e:
                    logger.info(f"ak.{func_name} 失败: {e}")
                    df = None

            if df is None or df.empty:
                logger.info("akshare 所有接口均无数据")
                return None

            if pd.api.types.is_datetime64_any_dtype(df.index):
                return df.sort_index()

            col_map = {}
            cn_mapping = {
                "open": ["open", "开盘", "开盘价"],
                "high": ["high", "最高", "最高价"],
                "low": ["low", "最低", "最低价"],
                "close": ["close", "收盘", "收盘价", "close*"],
                "volume": ["volume", "成交量", "volume_traded"],
                "date": ["date", "time", "交易日期", "date_time", "日期"],
            }
            lower_cols = {str(c).lower(): c for c in df.columns}
            for std, variants in cn_mapping.items():
                for v in variants:
                    if v in lower_cols:
                        col_map[lower_cols[v]] = std.capitalize()
                        break
            df = df.rename(columns=col_map)

            date_col = None
            for cand in _DATE_CANDIDATES + ["Date"]:
                if cand in df.columns:
                    date_col = cand
                    break
            if date_col is None:
                for c in df.columns:
                    sample = df[c].dropna().astype(str)
                    if not sample.empty:
                        try:
                            pd.to_datetime(sample.iloc[0])
                            date_col = c
                            break
                        except Exception:
                            continue
            if date_col is not None:
                try:
                    df.index = pd.to_datetime(df[date_col])
                    df = df.drop(columns=[date_col], errors="ignore")
                except Exception as e:
                    logger.info(f"日期列 {date_col} 解析失败: {e}")

            if not pd.api.types.is_datetime64_any_dtype(df.index):
                first_col = df.columns[0] if len(df.columns) > 0 else None
                if first_col is not None:
                    try:
                        df.index = pd.to_datetime(df[first_col])
                        df = df.drop(columns=[first_col], errors="ignore")
                    except Exception:
                        pass

            return df.sort_index()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_fetch)
                return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.info(f"akshare {ticker} 下载超时 ({timeout}s)")
        except Exception as e:
            logger.info(f"akshare {ticker} 失败: {e}")
        return None

