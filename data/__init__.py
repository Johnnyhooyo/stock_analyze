"""
数据模块 (data package)
=====================
提供标准化的量化数据获取、校验、存储能力。

核心组件:
  - manager     DataManager — 统一数据获取入口
  - vendors/    各数据源适配器 (yfinance, akshare, yahooquery …)
  - schemas     OHLCV 数据模型 & 列归一化 & schema 校验
  - calendar    港股交易日历 (基于 exchange_calendars)
  - quality     数据质量管道 (缺口/异常/复权检测 + 修复)
  - config      数据模块配置
  - storage     存储后端 (CSV / Parquet)

Usage::

    from data import DataManager
    mgr = DataManager()
    df, path = mgr.download("0700.HK", period="3y")

    # 纯读取（不触发网络请求）
    df = mgr.load("0700.HK", period="3y")
"""

from data.manager import DataManager  # noqa: F401
from data.config import DataConfig, get_config  # noqa: F401
from data.schemas import normalize_columns, validate_ohlcv, STANDARD_COLUMNS  # noqa: F401
from data.quality import check_quality, repair_quality, QualityReport  # noqa: F401
from data.calendar import (  # noqa: F401
    is_trading_day,
    prev_trading_day,
    next_trading_day,
    latest_expected_trading_day,
)
from data.hsi_stocks import HSI_STOCKS, get_hsi_stocks  # noqa: F401

__all__ = [
    # 核心
    "DataManager",
    "DataConfig",
    "get_config",
    # Schema
    "normalize_columns",
    "validate_ohlcv",
    "STANDARD_COLUMNS",
    # 质量
    "check_quality",
    "repair_quality",
    "QualityReport",
    # 日历
    "is_trading_day",
    "prev_trading_day",
    "next_trading_day",
    "latest_expected_trading_day",
    # 成分股
    "HSI_STOCKS",
    "get_hsi_stocks",
]
