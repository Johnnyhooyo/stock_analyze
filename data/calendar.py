"""
港股交易日历模块
===============
优先使用 exchange_calendars 库 (XHKG) 作为数据源。
若库不可用，则回退到内置港股假期表（覆盖 2024-2027 年）。
支持通过 keys.yaml 追加临时停市日（如台风信号）。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 内置港股公众假期备用表（exchange_calendars 不可用时使用）──────
# 仅含周一至周五中的假期（周末已由 weekday 判断排除）
# 覆盖范围：2024-01-01 至 2027-04-16
_BUILTIN_HK_HOLIDAYS: frozenset[date] = frozenset(
    date.fromisoformat(d)
    for d in [
        # 2024
        "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-29",
        "2024-04-01", "2024-04-04", "2024-05-01", "2024-05-15",
        "2024-06-10", "2024-07-01", "2024-09-06", "2024-09-18",
        "2024-10-01", "2024-10-11", "2024-12-25", "2024-12-26",
        # 2025
        "2025-01-01", "2025-01-29", "2025-01-30", "2025-01-31",
        "2025-04-04", "2025-04-18", "2025-04-21", "2025-05-01",
        "2025-05-05", "2025-07-01", "2025-10-01", "2025-10-07",
        "2025-10-29", "2025-12-25", "2025-12-26",
        # 2026
        "2026-01-01", "2026-02-17", "2026-02-18", "2026-02-19",
        "2026-04-03", "2026-04-06", "2026-04-07", "2026-05-01",
        "2026-05-25", "2026-06-19", "2026-07-01", "2026-10-01",
        "2026-10-19", "2026-12-25",
        # 2027
        "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-26",
        "2027-03-29", "2027-04-05",
    ]
)

# ── 加载 exchange_calendars (XHKG) ──────────────────────────────
try:
    import exchange_calendars as xcals
    _xcal: Any = xcals.get_calendar("XHKG")
    logger.debug("exchange_calendars (XHKG) 加载成功")
except Exception as exc:
    _xcal = None
    logger.warning(
        f"exchange_calendars 不可用 ({exc})，"
        f"将使用内置港股假期表（覆盖 2024-2027）作为备用。"
        f"如需完整支持，请安装: pip install exchange_calendars>=4.5"
    )

# ── 从 keys.yaml 追加额外假期 ───────────────────────────────────
_extra_holidays: set[date] = set()
try:
    _keys_path = Path(__file__).resolve().parent.parent / "keys.yaml"
    if _keys_path.exists():
        import yaml
        with open(_keys_path, encoding="utf-8") as _f:
            _keys = yaml.safe_load(_f) or {}
        for _d in _keys.get("extra_hk_holidays", []):
            try:
                _extra_holidays.add(date.fromisoformat(str(_d)))
            except Exception:
                pass
    if _extra_holidays:
        logger.debug(f"额外假期已加载: {sorted(_extra_holidays)}")
except Exception:
    pass


# ── 公开 API ────────────────────────────────────────────────────

def is_trading_day(d: date) -> bool:
    """判断某天是否为港股交易日（非周末且非公众假期）。"""
    if d.weekday() >= 5:
        return False
    if d in _extra_holidays:
        return False
    if _xcal is not None:
        try:
            import pandas as pd
            ts = pd.Timestamp(d)
            return _xcal.is_session(ts)
        except Exception:
            pass
    # exchange_calendars 不可用时，使用内置港股假期表
    return d not in _BUILTIN_HK_HOLIDAYS


def prev_trading_day(ref: date) -> date:
    """返回 *ref* 之前（不含 ref）最近一个港股交易日。"""
    d = ref - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(ref: date) -> date:
    """返回 *ref* 之后（不含 ref）最近一个港股交易日。"""
    d = ref + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def latest_expected_trading_day() -> date:
    """
    根据当前时间判断最新一个应该有数据的交易日。
    - 18:00 HKT 前：上一个交易日（当天数据尚未收盘确认）
    - 18:00 HKT 后：如果今天是交易日则为今天，否则上一个交易日
    """
    from datetime import datetime, time as dtime
    now = datetime.now()
    today = now.date()
    if now.time() < dtime(18, 0):
        # 当天数据尚未确认，回退到前一个交易日
        # prev_trading_day(ref) 返回 ref 之前（不含 ref）的最近交易日
        # 所以传 today 即可得到「今天之前」的最近交易日
        return prev_trading_day(today)
    else:
        return today if is_trading_day(today) else prev_trading_day(today + timedelta(days=1))

