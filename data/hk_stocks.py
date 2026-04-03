"""
全量港股股票列表
================
从多个数据源实时获取全部港股主板股票代码，按日缓存。
当日内多次调用只请求一次网络，第二天重新获取。

数据源优先级：akshare → yfinance → 本地历史文件 → HSI fallback
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from log_config import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache"
_BLACKLIST_FILE = Path(__file__).parent / "hk_stocks_blacklist.json"


def _get_cache_date_path(fetch_date: date) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"hk_stocks_{fetch_date.isoformat()}.json"


def _fetch_from_akshare() -> Optional[list[str]]:
    """从 akshare 获取全量港股主板代码。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_hk_spot_em()
        if df is None or df.empty:
            logger.warning("[hk_stocks] akshare 返回空数据")
            return None

        col_code = None
        for col in ["代码", "代码名称", "symbol", "代码/名称"]:
            if col in df.columns:
                col_code = col
                break
        if col_code is None:
            logger.warning(f"[hk_stocks] akshare 列名未知: {df.columns.tolist()}")
            return None

        tickers = []
        for code in df[col_code].dropna():
            code_str = str(code).strip()
            if not code_str.endswith(".HK"):
                code_str = code_str + ".HK"
            if code_str != ".HK":
                tickers.append(code_str)

        logger.info(f"[hk_stocks] akshare 获取到 {len(tickers)} 只港股")
        return tickers
    except Exception as e:
        logger.warning(f"[hk_stocks] akshare 获取失败: {e}")
        return None


def _fetch_from_yfinance() -> Optional[list[str]]:
    """从 yfinance 获取港股主板代码。失败返回 None。"""
    try:
        import yfinance as yf

        # yfinance 港股代码格式: XXXX.HK
        # 通过获取 HSI 成分股 + 常见的其他股票代码段来构建
        # yfinance 没有直接的"所有港股"接口，用 PKing 板块来扫描
        # 更可靠的方式：直接查 SPDR 可投资港股指数的成分
        tickers = set()

        # 方法1：扫描港交所主板可交易的ETF成分
        # iShares MSCI Hong Kong ETF (EWH) 成分股
        try:
            ehk = yf.Ticker("EWH")
            hist = ehk.history(period="1d")
            # EWH 持有很多大蓝筹，但不够全面
        except Exception:
            pass

        # 方法2：直接从 NASDAQ-香港板块扫描（通过NASDAQ的上市列表）
        # 更可靠：用 yfinance 的 ticker 搜索接口
        # yfinance 没有 search，但我们可以枚举已知范围
        # 港股代码 0001.HK - 9999.HK 遍历太慢，改用已知大蓝筹
        # 用 HSI 成分股 + 主要板块代表作为种子
        # 这个方法不太行，改用其他方式

        # 方法3：用 pandas_datareader 的 WorldTR 指数成分
        try:
            import pandas_datareader as pdr
            # 没有直接的港股成分，改用宏观指数
        except Exception:
            pass

        logger.warning("[hk_stocks] yfinance 无法获取全量港股列表（无直接接口）")
        return None
    except Exception as e:
        logger.warning(f"[hk_stocks] yfinance 获取失败: {e}")
        return None


def _normalize_ticker(ticker: str) -> str:
    """标准化股票代码格式: 00001.HK → 0001.HK，0005.HK → 0005.HK"""
    try:
        code, suffix = ticker.rsplit(".", 1)
        code = code.lstrip("0") or "0"
        if len(code) < 4:
            code = code.zfill(4)
        return f"{code}.{suffix}"
    except Exception:
        return ticker


def _fetch_from_tickflow() -> Optional[list[str]]:
    """从 tickflow 获取港股全量股票列表（免费接口）。失败返回 None。"""
    try:
        from tickflow import TickFlow

        tf = TickFlow.free()
        detail = tf.universes.get("HK_Equity")
        if not detail:
            logger.warning("[hk_stocks] tickflow 未找到 HK_Equity universe")
            return None

        symbol_count = detail.get("symbol_count", 0)
        logger.info(f"[hk_stocks] tickflow HK_Equity 包含 {symbol_count} 只股票")

        raw_symbols = detail.get("symbols", [])
        all_tickers = []
        for s in raw_symbols:
            ticker = str(s).strip()
            if not ticker.endswith(".HK"):
                ticker = ticker + ".HK"
            # 标准化: 5位码如 00001.HK → 4位 0001.HK
            ticker = _normalize_ticker(ticker)
            if ticker:
                all_tickers.append(ticker)

        all_tickers = sorted(set(all_tickers))
        logger.info(f"[hk_stocks] tickflow 获取到 {len(all_tickers)} 只港股")
        return all_tickers
    except Exception as e:
        logger.warning(f"[hk_stocks] tickflow 获取失败: {e}")
        return None


def _fetch_from_tushare() -> Optional[list[str]]:
    """从 tushare 获取港股列表（需要 token）。失败返回 None。"""
    try:
        import os
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            logger.warning("[hk_stocks] TUSHARE_TOKEN 环境变量未设置")
            return None

        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()

        df = pro.hk_basic(exchange='SH', list_status='L')
        if df is None or df.empty:
            df = pro.hk_basic(exchange='SZ', list_status='L')

        if df is None or df.empty:
            logger.warning("[hk_stocks] tushare 返回空数据")
            return None

        tickers = []
        for code in df['ts_code'].dropna():
            code_str = str(code).strip()
            if not code_str.endswith('.HK'):
                code_str = code_str + '.HK'
            tickers.append(code_str)

        logger.info(f"[hk_stocks] tushare 获取到 {len(tickers)} 只港股")
        return tickers
    except Exception as e:
        logger.warning(f"[hk_stocks] tushare 获取失败: {e}")
        return None


def _fetch_all_sources() -> Optional[list[str]]:
    """依次尝试多个数据源，返回第一个成功的。"""
    sources = [
        ("tickflow", _fetch_from_tickflow),
        ("akshare", _fetch_from_akshare),
        ("tushare", _fetch_from_tushare),
        ("yfinance", _fetch_from_yfinance),
    ]

    for name, fetch_fn in sources:
        logger.info(f"[hk_stocks] 尝试数据源: {name}")
        result = fetch_fn()
        if result:
            return result

    return None


def _load_from_cache(cache_path: Path) -> Optional[list[str]]:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        logger.info(f"[hk_stocks] 从缓存加载 {len(tickers)} 只: {cache_path.name}")
        return tickers
    except Exception:
        return None


def _save_to_cache(cache_path: Path, tickers: list[str]) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"tickers": tickers, "date": date.today().isoformat()}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[hk_stocks] 缓存写入失败: {e}")


def _load_blacklist() -> dict:
    """加载黑名单。返回 {ticker: {reason, added_at, failed_checks}}。"""
    try:
        if not _BLACKLIST_FILE.exists():
            return {}
        with open(_BLACKLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("blacklist", {})
    except Exception as e:
        logger.warning(f"[hk_stocks] 黑名单加载失败: {e}")
        return {}


def _save_blacklist(blacklist: dict) -> None:
    """保存黑名单到文件。"""
    try:
        with open(_BLACKLIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"blacklist": blacklist, "updated_at": date.today().isoformat()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[hk_stocks] 黑名单保存失败: {e}")


def add_to_blacklist(ticker: str, reason: str, failed_checks: list[str]) -> None:
    """
    将股票加入黑名单。

    Args:
        ticker: 股票代码（如 0001.HK）
        reason: 加入原因描述
        failed_checks: 失败的检查项列表
    """
    blacklist = _load_blacklist()
    ticker = ticker.upper()
    if ticker in blacklist:
        return
    blacklist[ticker] = {
        "reason": reason,
        "added_at": date.today().isoformat(),
        "failed_checks": failed_checks,
    }
    _save_blacklist(blacklist)
    logger.info(f"[hk_stocks] 加入黑名单: {ticker} ({reason})")


def is_blacklisted(ticker: str) -> bool:
    """检查股票是否在黑名单中。"""
    blacklist = _load_blacklist()
    return ticker.upper() in blacklist


def _filter_blacklist(tickers: list[str]) -> list[str]:
    """从列表中移除黑名单股票，返回过滤后的列表。"""
    blacklist = _load_blacklist()
    if not blacklist:
        return tickers
    bl_set = set(blacklist.keys())
    before = len(tickers)
    filtered = [t for t in tickers if t.upper() not in bl_set]
    removed = before - len(filtered)
    if removed > 0:
        logger.info(f"[hk_stocks] 黑名单过滤掉 {removed} 只: {list(bl_set)[:10]}")
    return filtered


def get_all_hk_stocks() -> list[str]:
    """
    获取全部港股主板股票代码列表。

    - 同一天内多次调用只请求一次网络，后续返回缓存
    - 第二天首次调用重新从数据源获取
    - 黑名单中的股票会被自动过滤
    """
    today = date.today()
    cache_path = _get_cache_date_path(today)

    if cache_path.exists():
        tickers = _load_from_cache(cache_path)
        if tickers:
            return _filter_blacklist(tickers)

    tickers = _fetch_all_sources()
    if tickers:
        _save_to_cache(cache_path, tickers)
        return _filter_blacklist(tickers)

    logger.warning("[hk_stocks] 所有数据源失败，从本地历史数据扫描股票列表")
    local_tickers = _scan_local_stocks()
    if local_tickers:
        _save_to_cache(cache_path, local_tickers)
        return _filter_blacklist(local_tickers)

    logger.warning("[hk_stocks] 本地也无数据，回退到 HSI 成分股列表")
    from data.hsi_stocks import get_hsi_stocks
    return _filter_blacklist(get_hsi_stocks())


def _scan_local_stocks() -> Optional[list[str]]:
    """扫描本地 historical 目录，从文件名提取股票代码。"""
    try:
        historical_dir = Path(__file__).parent.parent / "data" / "historical"
        if not historical_dir.is_dir():
            return None
        tickers = set()
        for csv_file in historical_dir.glob("*.csv"):
            name = csv_file.stem
            for sep in ("_", "-"):
                if sep in name:
                    ticker = name.split(sep)[0] + ".HK"
                    tickers.add(ticker)
                    break
        result = sorted(tickers)
        logger.info(f"[hk_stocks] 本地扫描到 {len(result)} 只股票")
        return result if result else None
    except Exception as e:
        logger.warning(f"[hk_stocks] 本地扫描失败: {e}")
        return None


def clear_cache() -> None:
    """清除所有缓存文件（测试用）"""
    try:
        for f in _CACHE_DIR.glob("hk_stocks_*.json"):
            f.unlink()
    except Exception as e:
        logger.warning(f"[hk_stocks] 清除缓存失败: {e}")
