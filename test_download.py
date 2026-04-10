#!/usr/bin/env python3
"""
test_download.py — 指定数据源下载指定股票的独立测试脚本

用法:
    python3 test_download.py                           # 默认: 0700.HK, 所有可用数据源
    python3 test_download.py --ticker 0005.HK          # 指定股票
    python3 test_download.py --source yfinance         # 指定单一数据源
    python3 test_download.py --ticker 0700.HK 0005.HK --source yfinance yahooquery
    python3 test_download.py --period 6mo --force      # 强制重下（忽略缓存）
    python3 test_download.py --list-sources            # 列出所有数据源
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional

# ── 日志配置 ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_download")

# 静默第三方库噪音
for noisy in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── 可用数据源 ──────────────────────────────────────────────────

ALL_SOURCES = ["tickflow", "yahooquery", "yfinance", "pandas_datareader", "akshare", "alpha_vantage"]


def _check_vendor_availability() -> dict[str, bool]:
    """检查各数据源依赖是否安装。"""
    from data.manager import _build_vendor_registry
    from data.config import DataConfig

    cfg = DataConfig()
    try:
        registry = _build_vendor_registry(cfg)
        return {name: v.is_available() for name, v in registry.items()}
    except Exception as e:
        logger.warning(f"构建 vendor 注册表失败: {e}")
        return {}


# ── 单数据源下载测试 ─────────────────────────────────────────────

def test_vendor_direct(
    ticker: str,
    source: str,
    period: str = "3mo",
    timeout: int = 30,
) -> dict:
    """
    直接调用指定 vendor 下载单只股票，绕过缓存。

    Returns:
        {"source": str, "ticker": str, "ok": bool, "rows": int,
         "columns": list, "date_range": str, "elapsed": float, "error": str}
    """
    from data.manager import _build_vendor_registry
    from data.config import DataConfig
    from data.schemas import normalize_columns
    from data.vendor_base import parse_period_to_days

    result = {
        "source": source,
        "ticker": ticker,
        "ok": False,
        "rows": 0,
        "columns": [],
        "date_range": "",
        "elapsed": 0.0,
        "error": "",
    }

    cfg = DataConfig()
    registry = _build_vendor_registry(cfg)
    vendor = registry.get(source)

    if vendor is None:
        result["error"] = f"未知数据源: {source}"
        return result

    if not vendor.is_available():
        result["error"] = f"数据源 {source} 不可用（依赖未安装）"
        return result

    days = parse_period_to_days(period)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    t0 = time.monotonic()
    try:
        df = vendor.fetch_with_retry(
            ticker,
            start_date,
            end_date,
            period=period,
            timeout=timeout,
            max_attempts=2,
            backoff=1.0,
        )
        elapsed = time.monotonic() - t0

        if df is None or df.empty:
            result["error"] = "返回数据为空"
            result["elapsed"] = elapsed
            return result

        df = normalize_columns(df)
        if df is None or df.empty:
            result["error"] = "标准化后数据为空"
            result["elapsed"] = elapsed
            return result

        idx = df.index
        result.update(
            ok=True,
            rows=len(df),
            columns=list(df.columns),
            date_range=f"{idx.min().date()} → {idx.max().date()}",
            elapsed=elapsed,
        )

    except Exception as e:
        result["error"] = str(e)
        result["elapsed"] = time.monotonic() - t0

    return result


# ── DataManager 集成下载测试 ─────────────────────────────────────

def test_manager_download(
    ticker: str,
    sources: Optional[List[str]],
    period: str = "3mo",
    force: bool = False,
) -> dict:
    """
    通过 DataManager（含缓存/合并/保存逻辑）下载。

    Returns:
        {"ticker": str, "ok": bool, "rows": int, "file": str,
         "date_range": str, "elapsed": float, "error": str}
    """
    from data.manager import DataManager

    result = {
        "ticker": ticker,
        "ok": False,
        "rows": 0,
        "file": "",
        "date_range": "",
        "elapsed": 0.0,
        "error": "",
    }

    mgr = DataManager()
    t0 = time.monotonic()
    try:
        df, path = mgr.download(
            ticker,
            period=period,
            sources_override=sources,
            force=force,
        )
        elapsed = time.monotonic() - t0

        if df is None or df.empty:
            result["error"] = "下载结果为空"
            result["elapsed"] = elapsed
            return result

        idx = df.index
        result.update(
            ok=True,
            rows=len(df),
            file=str(path) if path else "",
            date_range=f"{idx.min().date()} → {idx.max().date()}",
            elapsed=elapsed,
        )
    except Exception as e:
        result["error"] = str(e)
        result["elapsed"] = time.monotonic() - t0

    return result


# ── 输出格式化 ──────────────────────────────────────────────────

def _print_result(r: dict, mode: str = "vendor") -> None:
    status = "✅ OK" if r["ok"] else "❌ FAIL"
    ticker = r["ticker"]

    if mode == "vendor":
        src = r["source"]
        print(f"  [{status}] {src:<20} {ticker}")
    else:
        print(f"  [{status}] {ticker}")

    if r["ok"]:
        print(f"         行数: {r['rows']}")
        print(f"         日期: {r['date_range']}")
        print(f"         耗时: {r['elapsed']:.2f}s")
        if mode == "vendor":
            cols = ", ".join(r["columns"])
            print(f"         列名: {cols}")
        else:
            print(f"         文件: {r['file']}")
    else:
        print(f"         错误: {r['error']}")
        print(f"         耗时: {r['elapsed']:.2f}s")


# ── 主入口 ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="测试从指定数据源下载指定股票数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", nargs="+", default=["0700.HK"],
        metavar="TICKER",
        help="股票代码（默认: 0700.HK）",
    )
    parser.add_argument(
        "--source", nargs="+", default=None,
        metavar="SOURCE",
        choices=ALL_SOURCES + ["all"],
        help=f"数据源（默认: 全部可用）。可选: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument(
        "--period", default="3mo",
        help="数据周期（默认: 3mo）。支持: 1y, 6mo, 3mo, 30d …",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载，忽略本地缓存",
    )
    parser.add_argument(
        "--mode", choices=["vendor", "manager", "both"], default="vendor",
        help="测试模式: vendor=直接调用各vendor, manager=通过DataManager, both=两者都测（默认: vendor）",
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="列出所有数据源及其可用状态，然后退出",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="单次下载超时秒数（默认: 30）",
    )
    args = parser.parse_args()

    # ── 列出数据源 ──────────────────────────────────────────────
    if args.list_sources:
        print("\n可用数据源检测:")
        avail = _check_vendor_availability()
        for src in ALL_SOURCES:
            ok = avail.get(src, False)
            mark = "✅" if ok else "❌"
            print(f"  {mark} {src}")
        return 0

    # ── 确定测试的数据源 ────────────────────────────────────────
    if args.source is None or "all" in (args.source or []):
        avail = _check_vendor_availability()
        sources = [s for s in ALL_SOURCES if avail.get(s, False)]
        print(f"\n自动检测可用数据源: {sources}")
    else:
        sources = args.source

    tickers = args.ticker
    period = args.period

    print(f"\n{'='*60}")
    print(f"  股票: {', '.join(tickers)}")
    print(f"  数据源: {', '.join(sources)}")
    print(f"  周期: {period}")
    print(f"  模式: {args.mode}")
    print(f"  强制刷新: {args.force}")
    print(f"{'='*60}\n")

    total = 0
    passed = 0

    # ── 模式 1: 直接 vendor 测试 ────────────────────────────────
    if args.mode in ("vendor", "both"):
        print("【直接 Vendor 测试】绕过缓存，直接测试每个数据源的下载能力\n")
        for ticker in tickers:
            print(f"  股票: {ticker}")
            for src in sources:
                r = test_vendor_direct(ticker, src, period=period, timeout=args.timeout)
                _print_result(r, mode="vendor")
                total += 1
                if r["ok"]:
                    passed += 1
                print()

    # ── 模式 2: DataManager 集成测试 ───────────────────────────
    if args.mode in ("manager", "both"):
        if args.mode == "both":
            print("\n" + "─" * 60)
        print("【DataManager 集成测试】含缓存判断、数据合并、保存逻辑\n")
        for ticker in tickers:
            print(f"  股票: {ticker}")
            r = test_manager_download(
                ticker,
                sources=sources if args.source else None,
                period=period,
                force=args.force,
            )
            _print_result(r, mode="manager")
            total += 1
            if r["ok"]:
                passed += 1
            print()

    # ── 汇总 ────────────────────────────────────────────────────
    print("─" * 60)
    failed = total - passed
    print(f"结果: {passed}/{total} 通过，{failed} 失败")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
